'''Shared code for the fixed-architecture GNN experiments (GIN / PNA / GAT / Transformer).

Everything that must be identical across operators lives here and is imported by the
operator notebooks, which only set OPERATOR, the batch of knob configs and paths.
'''
import gc
import json
import os
import random
import shutil
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import psutil
import torch
import torch.nn as nn
from sklearn.metrics import (average_precision_score, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)
from torch_geometric.data import Data
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.nn import GATv2Conv, GINConv, GINEConv
from torch_geometric.utils import degree
from tqdm.auto import tqdm

# ---------------- fixed recipe (CLAUDE.md §3) — identical for every run, never tuned ----------------
HIDDEN        = 128
NUM_LAYERS    = 2
DROPOUT       = 0.3
HEADS         = 4              # GATv2 / Transformer: 4 heads x 32 = 128
NUM_NEIGHBORS = [100, 100]     # neighbours sampled per hop, one entry per layer
BATCH_SIZE    = 8192           # seed edges per optimizer step
ACCUM_STEPS   = 1              # >1: the 8192 seeds arrive as ACCUM_STEPS smaller sampled micro-batches, one step per 8192
                               # (same update, smaller subgraph in memory — for GATv2 on Kaggle if it runs out of GPU RAM)
EPOCHS        = 20
LR            = 1e-3
WEIGHT_DECAY  = 1e-5           # Adam L2 term; every run since the first batch used it, so it is part of the recipe
POS_WEIGHT    = 8.0            # weight of the laundering class in the loss
SEED          = 42

OPERATORS   = ('gin', 'pna', 'gat', 'transformer')
SPLITS      = ('train', 'val', 'test')
METRICS     = ('f1', 'precision', 'recall', 'pr_auc', 'roc_auc', 'precision_at_5pct', 'recall_at_5pct')
RUN_FIELDS  = ('threshold', 'best_epoch', 'params', 'invariant_params')
METRIC_COLS = [f'{split}_{metric}' for split in SPLITS for metric in METRICS]
# compact view of batch_summary.csv for the notebook comparison table
DISPLAY_COLS = ['run', 'best_epoch', 'threshold', 'val_f1', 'test_f1', 'test_precision', 'test_recall',
                'test_pr_auc', 'test_precision_at_5pct', 'test_recall_at_5pct', 'params', 'invariant_params']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ==================================================================================================
# data
# ==================================================================================================

def set_seed(seed):
    '''Seed python, numpy and torch so the invariant weights start identical in every run.'''
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_graphs(data_dir):
    '''Load the three cumulative snapshots and the feature column groups.

    Returns graphs {split: Data}, col_sets {none|base|gfp|full: column indices}, node_dim.
    '''
    graphs = {split: torch.load(f'{data_dir}/{split}_graph.pt', weights_only=False) for split in SPLITS}
    meta = json.load(open(f'{data_dir}/feature_meta.json'))

    all_cols = meta['EDGE_FEAT_COLS']
    base_idx = [all_cols.index(c) for c in meta['BASE_EDGE_COLS']]   # 0..19
    gfp_idx  = [all_cols.index(c) for c in meta['GFP_FEAT_COLS']]    # 20..80
    col_sets = {'none': [], 'base': base_idx, 'gfp': gfp_idx, 'full': base_idx + gfp_idx}
    node_dim = graphs['train'].x.shape[1]
    assert graphs['train'].edge_attr.shape[1] == len(all_cols) == 81

    for split, g in graphs.items():
        n_eval = int(g.eval_mask.sum())
        n_pos  = int((g.y[g.eval_mask] == 1).sum())
        print(f'{split:<5} nodes={g.num_nodes:,} edges={g.edge_index.shape[1]:,} '
              f'evaluated={n_eval:,} laundering={n_pos:,} ({n_pos / n_eval:.4%})')
    return graphs, col_sets, node_dim


def in_degree_histogram(graph):
    '''In-degree histogram of a graph (PNA needs it from the training graph).'''
    deg = degree(graph.edge_index[1], num_nodes=graph.num_nodes, dtype=torch.long)
    return torch.bincount(deg)


# ==================================================================================================
# model — the fixed template; only make_conv() differs between operators
# ==================================================================================================

def gin_mlp(hidden):
    '''The GIN node MLP: Linear -> BatchNorm -> ReLU -> Linear, identical in every configuration.'''
    return nn.Sequential(nn.Linear(hidden, hidden), nn.BatchNorm1d(hidden), nn.ReLU(),
                         nn.Linear(hidden, hidden))


def make_conv(operator, hidden, mp_edge_dim, deg_hist=None):
    '''One graph convolution for the operator; the only place operators differ.

    mp_edge_dim = 0 -> edge features stay out of message passing.
    '''
    if operator == 'gin':
        if mp_edge_dim > 0:
            return GINEConv(gin_mlp(hidden), train_eps=True, edge_dim=mp_edge_dim)
        return GINConv(gin_mlp(hidden), train_eps=True)
    if operator == 'pna':
        raise NotImplementedError('pna: PNAConv(hidden, hidden, aggregators, scalers, deg=deg_hist, '
                                  'edge_dim=mp_edge_dim or None, towers=1) — needs the training-graph '
                                  'in-degree histogram (in_degree_histogram)')
    if operator == 'gat':
        # 4 heads x 32 concatenated -> 128; lin_edge (edge_dim) is the only width-dependent tensor
        if mp_edge_dim > 0:
            return GATv2Conv(hidden, hidden // HEADS, heads=HEADS, edge_dim=mp_edge_dim)
        return GATv2Conv(hidden, hidden // HEADS, heads=HEADS)
    if operator == 'transformer':
        raise NotImplementedError('transformer: TransformerConv(hidden, hidden // HEADS, heads=HEADS, '
                                  'edge_dim=mp_edge_dim or None)')
    raise ValueError(f'unknown operator {operator!r}; choose from {OPERATORS}')


class MPLayer(nn.Module):
    '''One message-passing step with a residual: h <- h + Dropout(ReLU(conv(h, edges))).

    conv_in aggregates the accounts that PAID a node; bidirectional adds conv_out over
    reversed edges ("whom I paid") and merges the two 128-dim views back to 128.
    '''

    def __init__(self, operator, hidden, mp_edge_dim, bidirectional, dropout, deg_hist=None):
        super().__init__()
        self.use_edges = mp_edge_dim > 0
        self.bidirectional = bidirectional
        self.conv_in = make_conv(operator, hidden, mp_edge_dim, deg_hist)
        if bidirectional:
            self.conv_out = make_conv(operator, hidden, mp_edge_dim, deg_hist)
            self.merge = nn.Linear(2 * hidden, hidden)
        self.dropout = nn.Dropout(dropout)

    def apply_conv(self, conv, h, edge_index, edge_attr):
        '''Route the call: convs with edge features take edge_attr, the others do not.'''
        if self.use_edges:
            return conv(h, edge_index, edge_attr)
        return conv(h, edge_index)

    def forward(self, h, edge_index, edge_attr):
        out = self.apply_conv(self.conv_in, h, edge_index, edge_attr)
        if self.bidirectional:
            # flip(0) swaps senders and receivers; a transaction's features describe it both ways
            out_rev = self.apply_conv(self.conv_out, h, edge_index.flip(0), edge_attr)
            out = self.merge(torch.cat([out, out_rev], dim=1))
        return h + self.dropout(torch.relu(out))


#   node features x [N x node_dim] -> node_proj Linear -> 128, ReLU
#   NUM_LAYERS x MPLayer: conv (+ W_e . e_uv when MP edge features are on), ReLU, Dropout, residual
#   readout: concat [h_src || h_dst || e_seed] -> Linear 128 -> ReLU -> Dropout -> Linear 1 (logit)
#   Only W_e (edge width) and the readout's first Linear (seed width) change between configs.
class EdgeClassifier(nn.Module):
    '''Fixed-architecture edge classifier: node_proj -> NUM_LAYERS message-passing layers -> readout MLP.

    Module names (node_proj, layers.i.conv_in/conv_out/merge, classifier) are the ones the
    saved best.pt checkpoints use — do not rename them.
    '''

    def __init__(self, operator, node_dim, mp_edge_dim, readout_edge_dim, bidirectional, deg_hist=None):
        super().__init__()
        self.node_proj = nn.Linear(node_dim, HIDDEN)
        self.layers = nn.ModuleList(
            [MPLayer(operator, HIDDEN, mp_edge_dim, bidirectional, DROPOUT, deg_hist) for _ in range(NUM_LAYERS)])
        self.classifier = nn.Sequential(
            nn.Linear(2 * HIDDEN + readout_edge_dim, HIDDEN), nn.ReLU(), nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN, 1))

    def forward(self, x, edge_index, edge_attr, seed_index, seed_edge_feats):
        '''x [N, node_dim], edge_index [2, E], edge_attr [E, mp_dim] or None,
        seed_index [2, B], seed_edge_feats [B, ro_dim] -> logits [B].'''
        h = torch.relu(self.node_proj(x))
        for layer in self.layers:
            h = layer(h, edge_index, edge_attr)
        z = torch.cat([h[seed_index[0]], h[seed_index[1]], seed_edge_feats], dim=1)
        return self.classifier(z).squeeze(-1)


def build_model(operator, node_dim, mp_edge_dim, readout_edge_dim, bidirectional=False, deg_hist=None):
    '''Build the fixed template for one operator and feature configuration, on the device.'''
    assert operator in OPERATORS, f'unknown operator {operator!r}'
    model = EdgeClassifier(operator, node_dim, mp_edge_dim, readout_edge_dim, bidirectional, deg_hist)
    return model.to(device)


def count_params(model):
    '''Total number of trainable parameters.'''
    return sum(p.numel() for p in model.parameters())


def invariant_params(model):
    '''Parameters whose shape does not depend on the edge-feature widths.

    67,587 for the GIN family, 67,585 for the GATv2 family (lin_l / lin_r / att / bias count,
    lin_edge does not). Excluded: the edge projection inside the conv (.lin. / lin_edge /
    edge_encoder) and the readout's first Linear (classifier.0.). Must be identical across
    feature configs within a family.
    '''
    width_dependent = ('.lin.', 'lin_edge', 'edge_encoder', 'classifier.0.')
    total = 0
    for name, p in model.named_parameters():
        if not any(tag in name for tag in width_dependent):
            total += p.numel()
    return total


# ==================================================================================================
# loaders and prediction
# ==================================================================================================

def make_loader(graph, shuffle, mp_idx, readout_idx, temporal=False):
    '''Neighbour-sampling loader over a split's evaluated edges + their readout features.

    Built once per run and reused by every epoch (rebuilding per epoch exhausted Kaggle RAM).
    '''
    mp_edge_attr = graph.edge_attr[:, mp_idx] if len(mp_idx) > 0 else None
    mp_graph = Data(x=graph.x, edge_index=graph.edge_index, edge_attr=mp_edge_attr,
                    edge_time=graph.edge_time, num_nodes=graph.num_nodes)

    seed_mask  = graph.eval_mask
    seed_index = graph.edge_index[:, seed_mask]
    seed_y     = graph.y[seed_mask].float()
    seed_feats = graph.edge_attr[seed_mask][:, readout_idx]

    # the shuffled (train) loader yields micro-batches of BATCH_SIZE // ACCUM_STEPS seeds; val/test predict at full size
    assert BATCH_SIZE % ACCUM_STEPS == 0, f'BATCH_SIZE {BATCH_SIZE} must be divisible by ACCUM_STEPS {ACCUM_STEPS}'
    batch_size = BATCH_SIZE // ACCUM_STEPS if shuffle else BATCH_SIZE
    kwargs = dict(num_neighbors=NUM_NEIGHBORS, batch_size=batch_size, edge_label_index=seed_index,
                  edge_label=seed_y, shuffle=shuffle)
    if temporal:
        # only edges earlier than the seed are sampled (needs a recent pyg-lib)
        kwargs.update(time_attr='edge_time', edge_label_time=graph.edge_time[seed_mask])
    return LinkNeighborLoader(mp_graph, **kwargs), seed_feats


def make_loaders(graphs, mp_idx, readout_idx, temporal=False):
    '''One (loader, seed_feats) pair per split; train is shuffled, val/test are not.'''
    return {split: make_loader(graphs[split], split == 'train', mp_idx, readout_idx, temporal)
            for split in SPLITS}


def forward_batch(model, batch, seed_feats):
    '''One forward pass; batch.input_id locates this batch's seed edges in seed_feats.'''
    batch = batch.to(device)
    feats = seed_feats[batch.input_id.cpu()].to(device)
    logits = model(batch.x, batch.edge_index, batch.edge_attr, batch.edge_label_index, feats)
    return logits, batch.edge_label


@torch.no_grad()
def predict(model, loader, seed_feats, name):
    '''Labels, probabilities and seed positions (batch.input_id) for every seed edge of the loader.

    Positions map a shuffled loader's rows back to the split's evaluated edges.
    '''
    model.eval()
    labels, probs, positions = [], [], []
    for batch in tqdm(loader, desc=f'predict {name}', leave=False):
        logits, y = forward_batch(model, batch, seed_feats)
        probs.append(torch.sigmoid(logits).cpu().numpy())
        labels.append(y.cpu().numpy())
        positions.append(batch.input_id.cpu().numpy())
    return np.concatenate(labels).astype(int), np.concatenate(probs), np.concatenate(positions)


# ==================================================================================================
# metrics (CLAUDE.md §4)
# ==================================================================================================

def best_f1_threshold(y, p):
    '''Threshold that maximises F1 on the given (validation) predictions -> (threshold, f1).'''
    precision, recall, thresholds = precision_recall_curve(y, p)
    f1 = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-12)
    return float(thresholds[f1.argmax()]), float(f1.max())


def weighted_bce(y, p):
    '''The training objective computed from eval-mode probabilities (comparable train/val loss curves).'''
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return float(-np.mean(POS_WEIGHT * y * np.log(p) + (1 - y) * np.log(1 - p)))


def top_k_metrics(y, p, frac=0.05):
    '''Precision and recall when flagging the top `frac` highest-scored edges.'''
    k = max(1, int(np.ceil(frac * len(p))))
    top = np.argsort(-p, kind='stable')[:k]
    hits = int(y[top].sum())
    pct = int(round(frac * 100))
    return {f'precision_at_{pct}pct': hits / k, f'recall_at_{pct}pct': hits / max(1, int(y.sum()))}


def compute_metrics(y, p, threshold):
    '''The seven §4 metrics for one split; thresholded ones use the validation-chosen threshold.'''
    pred = (p >= threshold).astype(int)
    metrics = {
        'f1':        float(f1_score(y, pred)),
        'precision': float(precision_score(y, pred, zero_division=0)),
        'recall':    float(recall_score(y, pred)),
        'pr_auc':    float(average_precision_score(y, p)),
        'roc_auc':   float(roc_auc_score(y, p)),
    }
    metrics.update(top_k_metrics(y, p))
    assert tuple(metrics) == METRICS
    return metrics


def score_all_splits(model, loaders, graphs, threshold):
    '''Score train, val and test once at the given threshold.

    Returns {split: metrics} and {split: {'edge_id', 'y', 'prob'}}, where edge_id is the edge's
    position in the graph edge list = its row in Data/edge_features.csv.
    '''
    set_seed(SEED)   # same neighbour samples whether scoring after training or re-scoring
    metrics, preds = {}, {}
    for split in SPLITS:
        loader, seed_feats = loaders[split]
        y, p, positions = predict(model, loader, seed_feats, split)
        eval_positions = graphs[split].eval_mask.nonzero().squeeze(1).numpy()
        metrics[split] = compute_metrics(y, p, threshold)
        preds[split] = {'edge_id': eval_positions[positions], 'y': y, 'prob': p}
    return metrics, preds


def save_predictions(out_dir, graphs, preds, threshold):
    '''predictions.csv: one row per evaluated edge of every split
    (split, edge_id, src, dst, edge_time, y, prob, pred) with pred = prob >= the val-chosen threshold.'''
    frames = []
    for split in SPLITS:
        graph, edge_id = graphs[split], torch.as_tensor(preds[split]['edge_id'])
        prob = preds[split]['prob'].astype(np.float32)
        frames.append(pd.DataFrame({
            'split':     split,
            'edge_id':   edge_id.numpy(),
            'src':       graph.edge_index[0, edge_id].numpy(),
            'dst':       graph.edge_index[1, edge_id].numpy(),
            'edge_time': graph.edge_time[edge_id].numpy(),
            'y':         preds[split]['y'],
            'prob':      prob,
            'pred':      (prob >= threshold).astype(int),   # same comparison as compute_metrics
        }))
    table = pd.concat(frames).sort_values('edge_id')
    assert table['edge_id'].is_unique, 'edge_id must identify every scored edge once'
    table.to_csv(f'{out_dir}/predictions.csv', index=False)
    print(f'predictions: {len(table):,} rows -> {out_dir}/predictions.csv')


def flatten_metrics(metrics):
    '''{split: {metric: v}} -> {'<split>_<metric>': v}.'''
    return {f'{split}_{metric}': value for split in SPLITS for metric, value in metrics[split].items()}


# ==================================================================================================
# training
# ==================================================================================================

def train_one_run(model, loaders, out_dir):
    '''The fixed training loop: per-epoch validation threshold sweep, best-val-F1 checkpoint.

    Writes best.pt and history.csv to out_dir; returns (history rows, best_epoch, best_threshold).
    '''
    train_loader, train_feats = loaders['train']
    val_loader,   val_feats   = loaders['val']
    # summed (not mean) loss: micro-batches add up and the gradient is divided by the seed count at
    # step time -> exactly the mean-loss gradient over the BATCH_SIZE seeds, whatever ACCUM_STEPS is
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([POS_WEIGHT], device=device), reduction='sum')
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    if ACCUM_STEPS > 1:
        print(f'gradient accumulation: {ACCUM_STEPS} micro-batches of {BATCH_SIZE // ACCUM_STEPS} seeds per optimizer step')

    history = []
    best_val_f1, best_epoch, best_threshold = -1.0, 0, 0.5
    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        model.train()
        optimizer.zero_grad()
        n_seen = 0
        for step, batch in enumerate(tqdm(train_loader, desc='train', leave=False), start=1):
            logits, y = forward_batch(model, batch, train_feats)
            criterion(logits, y).backward()
            n_seen += len(y)
            if step % ACCUM_STEPS == 0 or step == len(train_loader):
                for p in model.parameters():
                    if p.grad is not None:
                        p.grad.div_(n_seen)
                optimizer.step()
                optimizer.zero_grad()
                n_seen = 0

        # overfitting diagnostics: train and val in eval mode; train F1 reuses the val threshold
        y_train, p_train, _ = predict(model, train_loader, train_feats, 'train')
        y_val,   p_val,   _ = predict(model, val_loader,   val_feats,   'val')
        threshold, val_f1 = best_f1_threshold(y_val, p_val)
        train_f1 = f1_score(y_train, (p_train >= threshold).astype(int))

        is_best = val_f1 > best_val_f1
        if is_best:
            best_val_f1, best_epoch, best_threshold = val_f1, epoch, threshold
            torch.save(model.state_dict(), f'{out_dir}/best.pt')

        history.append({
            'epoch':        epoch,
            'train_loss':   weighted_bce(y_train, p_train),
            'val_loss':     weighted_bce(y_val, p_val),
            'train_f1':     train_f1,
            'val_f1':       val_f1,
            'train_pr_auc': average_precision_score(y_train, p_train),
            'val_pr_auc':   average_precision_score(y_val, p_val),
            'val_roc_auc':  roc_auc_score(y_val, p_val),
            'threshold':    threshold,
            'seconds':      time.time() - t0,
        })
        r = history[-1]
        ram_gb = psutil.Process().memory_info().rss / 2**30
        print(f"epoch {epoch:02d} | loss {r['train_loss']:.4f} / {r['val_loss']:.4f} | "
              f"F1 {r['train_f1']:.4f} / {r['val_f1']:.4f} @ {threshold:.3f} | "
              f"PR-AUC {r['train_pr_auc']:.4f} / {r['val_pr_auc']:.4f} | "
              f"{r['seconds']:.0f}s | RAM {ram_gb:.1f}GB   (train / val)"
              + ('   * new best -> best.pt' if is_best else ''))
        scheduler.step()

    pd.DataFrame(history).to_csv(f'{out_dir}/history.csv', index=False)
    print(f'best validation F1 {best_val_f1:.4f} at epoch {best_epoch} (threshold {best_threshold:.3f})')
    return history, best_epoch, best_threshold


def best_epoch_from_history(history):
    '''best_epoch and its threshold from history.csv (first epoch with the highest val F1).'''
    best = int(np.argmax(history['val_f1'].to_numpy()))
    return int(history['epoch'].iloc[best]), float(history['threshold'].iloc[best])


# ==================================================================================================
# plots, saving, naming
# ==================================================================================================

def plot_curves(history, y_test, p_test, test_pr_auc, best_epoch, run_name, path):
    '''curves.png: train/val loss, F1, PR-AUC per epoch (dashed line at best_epoch) + test PR curve.'''
    df = pd.DataFrame(history)
    fig, ax = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle(run_name, fontweight='bold')
    panels = [(ax[0, 0], 'loss', 'weighted BCE loss'),
              (ax[0, 1], 'f1', 'F1 (at the validation-chosen threshold)'),
              (ax[1, 0], 'pr_auc', 'PR-AUC')]
    for axis, key, title in panels:
        axis.plot(df['epoch'], df[f'train_{key}'], marker='o', label='train')
        axis.plot(df['epoch'], df[f'val_{key}'], marker='s', label='validation')
        axis.axvline(best_epoch, color='gray', linestyle='--', label=f'best epoch = {best_epoch}')
        axis.set_title(title)
        axis.set_xlabel('epoch')
        axis.legend()
    precision, recall, _ = precision_recall_curve(y_test, p_test)
    ax[1, 1].plot(recall, precision, label=f'test AP = {test_pr_auc:.3f}')
    ax[1, 1].axhline(y_test.mean(), color='gray', linestyle='--', label='random')
    ax[1, 1].set_title('test precision-recall')
    ax[1, 1].set_xlabel('recall')
    ax[1, 1].set_ylabel('precision')
    ax[1, 1].legend()
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.show()
    plt.close(fig)


def run_name(operator, cfg, mp_direction='in', temporal=False):
    '''<model>_mp-<mp>_readout-<ro>_dir-<dir>[_temporal] (CLAUDE.md §6).'''
    name = f"{operator}_mp-{cfg['mp']}_readout-{cfg['readout']}_dir-{mp_direction}"
    if temporal:
        name += '_temporal'
    return name


def result_row(name, operator, cfg, mp_direction, temporal, best_epoch, threshold,
               n_params, n_invariant, metrics):
    '''The flat result row shared by results.json and batch_summary.csv.'''
    row = {'run': name, 'operator': operator, 'mp': cfg['mp'], 'readout': cfg['readout'],
           'mp_direction': mp_direction, 'temporal_sampling': temporal, 'seed': SEED,
           'accum_steps': ACCUM_STEPS, 'best_epoch': best_epoch, 'threshold': threshold,
           'params': n_params, 'invariant_params': n_invariant}
    row.update(flatten_metrics(metrics))
    return row


def save_run(out_dir, row):
    '''Write results.json (the flat row plus the fixed-architecture block) and print the per-split table.'''
    results = dict(row)
    results['architecture'] = {'hidden': HIDDEN, 'layers': NUM_LAYERS, 'dropout': DROPOUT,
                               'neighbors': NUM_NEIGHBORS}
    json.dump(results, open(f'{out_dir}/results.json', 'w'), indent=2)
    table = {split: {metric: row[f'{split}_{metric}'] for metric in METRICS} for split in SPLITS}
    print(pd.DataFrame(table).round(4))


def print_model_header(name, mp_idx, readout_idx, n_params, n_invariant):
    '''One-line description of the run and its parameter fingerprint.'''
    print(f'=== {name} ===')
    print(f'MP edge features: {len(mp_idx)} | readout edge features: {len(readout_idx)}')
    print(f'total params: {n_params:,} | architecture-invariant: {n_invariant:,}')


# ==================================================================================================
# one run, one re-score, one batch summary
# ==================================================================================================

def run_experiment(operator, cfg, graphs, col_sets, node_dim, out_root, mp_direction='in', temporal=False):
    '''Train and evaluate one knob configuration end-to-end; returns the flat result row.'''
    name = run_name(operator, cfg, mp_direction, temporal)
    out_dir = f'{out_root}/{name}'
    os.makedirs(out_dir, exist_ok=True)
    mp_idx, readout_idx = col_sets[cfg['mp']], col_sets[cfg['readout']]

    set_seed(SEED)
    deg_hist = in_degree_histogram(graphs['train']) if operator == 'pna' else None
    model = build_model(operator, node_dim, len(mp_idx), len(readout_idx),
                        mp_direction == 'bidirectional', deg_hist)
    n_params, n_invariant = count_params(model), invariant_params(model)
    print_model_header(name, mp_idx, readout_idx, n_params, n_invariant)

    loaders = make_loaders(graphs, mp_idx, readout_idx, temporal)
    history, best_epoch, threshold = train_one_run(model, loaders, out_dir)

    # final evaluation: best checkpoint, one validation-chosen threshold, test scored once
    model.load_state_dict(torch.load(f'{out_dir}/best.pt', map_location=device))
    metrics, preds = score_all_splits(model, loaders, graphs, threshold)
    row = result_row(name, operator, cfg, mp_direction, temporal, best_epoch, threshold,
                     n_params, n_invariant, metrics)
    save_run(out_dir, row)
    save_predictions(out_dir, graphs, preds, threshold)
    plot_curves(history, preds['test']['y'], preds['test']['prob'], metrics['test']['pr_auc'],
                best_epoch, name, f'{out_dir}/curves.png')

    del model, loaders
    gc.collect()
    torch.cuda.empty_cache()
    return row


# def rescore_run(operator, cfg, graphs, col_sets, node_dim, src_root, out_root,
#                 mp_direction='in', temporal=False):
#     '''Rebuild results.json and curves.png of a finished run from its best.pt + history.csv; no training.

#     best_epoch and threshold come from history.csv; best.pt and history.csv are copied to out_root.
#     '''
#     name = run_name(operator, cfg, mp_direction, temporal)
#     src_dir, out_dir = f'{src_root}/{name}', f'{out_root}/{name}'
#     os.makedirs(out_dir, exist_ok=True)
#     mp_idx, readout_idx = col_sets[cfg['mp']], col_sets[cfg['readout']]

#     history = pd.read_csv(f'{src_dir}/history.csv')
#     best_epoch, threshold = best_epoch_from_history(history)

#     set_seed(SEED)
#     deg_hist = in_degree_histogram(graphs['train']) if operator == 'pna' else None
#     model = build_model(operator, node_dim, len(mp_idx), len(readout_idx),
#                         mp_direction == 'bidirectional', deg_hist)
#     model.load_state_dict(torch.load(f'{src_dir}/best.pt', map_location=device))   # strict: keys must match
#     n_params, n_invariant = count_params(model), invariant_params(model)
#     print_model_header(name, mp_idx, readout_idx, n_params, n_invariant)
#     print(f're-scoring from {src_dir}: best epoch {best_epoch}, threshold {threshold:.3f}')

#     loaders = make_loaders(graphs, mp_idx, readout_idx, temporal)
#     metrics, preds = score_all_splits(model, loaders, graphs, threshold)
#     row = result_row(name, operator, cfg, mp_direction, temporal, best_epoch, threshold,
#                      n_params, n_invariant, metrics)
#     save_run(out_dir, row)
#     save_predictions(out_dir, graphs, preds, threshold)
#     plot_curves(history, preds['test']['y'], preds['test']['prob'], metrics['test']['pr_auc'],
#                 best_epoch, name, f'{out_dir}/curves.png')

#     for fname in ('best.pt', 'history.csv'):
#         if os.path.abspath(f'{src_dir}/{fname}') != os.path.abspath(f'{out_dir}/{fname}'):
#             shutil.copy(f'{src_dir}/{fname}', f'{out_dir}/{fname}')

#     del model, loaders
#     gc.collect()
#     torch.cuda.empty_cache()
#     return row


def summarize_batch(rows, out_root):
    '''Write batch_summary.csv; check every §4 column is present and invariant_params is identical.'''
    summary = pd.DataFrame(rows)
    missing = [c for c in METRIC_COLS + list(RUN_FIELDS) if c not in summary.columns]
    assert not missing, f'batch_summary.csv is missing columns: {missing}'
    assert summary['invariant_params'].nunique() == 1, \
        f"invariant_params differ across runs: {summary['invariant_params'].tolist()}"
    summary.to_csv(f'{out_root}/batch_summary.csv', index=False)
    print(f'{len(summary)} runs -> {out_root}/batch_summary.csv | invariant_params = '
          f"{int(summary['invariant_params'].iloc[0]):,} in every row")
    return summary
