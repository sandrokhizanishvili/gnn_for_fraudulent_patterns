"""Threshold, metrics, neighbour loaders and the curves."""
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (average_precision_score, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)
from torch_geometric.data import Data
from torch_geometric.loader import LinkNeighborLoader
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

from gnn_frauds.dataset.hi_small import set_seed
from gnn_frauds.hyperparameters.gnns import METRICS, POS_WEIGHT, SEED, SENSITIVITY_DELTAS, SPLITS, full_recipe
from gnn_frauds.model.gnns.classifier import device

def make_loader(graph, shuffle, mp_idx, readout_idx, recipe=None):
    '''Neighbour-sampling loader over a split's evaluated edges + their readout features; built once per run.

    Neighbours per hop and the batch size come from the recipe (None = v1). recipe num_workers > 0 samples in
    that many non-persistent worker processes (seeded from the main RNG, so a resumed run samples the same
    neighbours).
    '''
    r = full_recipe(recipe)
    mp_edge_attr = graph.edge_attr[:, mp_idx] if len(mp_idx) > 0 else None
    mp_graph = Data(x=graph.x, edge_index=graph.edge_index, edge_attr=mp_edge_attr,
                    edge_time=graph.edge_time, num_nodes=graph.num_nodes)

    seed_mask  = graph.eval_mask
    seed_index = graph.edge_index[:, seed_mask]
    seed_y     = graph.y[seed_mask].float()
    seed_feats = graph.edge_attr[seed_mask][:, readout_idx]

    kwargs = dict(num_neighbors=r['num_neighbors'], batch_size=r['batch_size'], edge_label_index=seed_index,
                  edge_label=seed_y, shuffle=shuffle)
    if r['num_workers'] > 0:
        # Workers die with the loader, so a resumed run can reseed them instead of keeping the old sample stream.
        kwargs.update(num_workers=int(r['num_workers']), persistent_workers=False)
    return LinkNeighborLoader(mp_graph, **kwargs), seed_feats


def make_loaders(graphs, mp_idx, readout_idx, recipe=None, splits=SPLITS):
    '''One (loader, seed_feats) pair per split; train is shuffled, val / test are not. Tuning passes
    splits=('train', 'val'), so no test loader is built there.'''
    return {split: make_loader(graphs[split], split == 'train', mp_idx, readout_idx, recipe) for split in splits}


def forward_batch(model, batch, seed_feats):
    '''One forward pass; batch.input_id locates this batch's seed edges in seed_feats.'''
    batch = batch.to(device)
    feats = seed_feats[batch.input_id.cpu()].to(device)  # the full seed-feature matrix stays on CPU
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
    # The last PR point has no threshold: sklearn appends precision 1, recall 0.
    f1 = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-12)
    return float(thresholds[f1.argmax()]), float(f1.max())


def weighted_bce(y, p, pos_weight=None):
    '''The training objective computed from eval-mode probabilities (comparable train/val loss curves).'''
    if pos_weight is None:
        pos_weight = POS_WEIGHT
    p = np.clip(p, 1e-7, 1 - 1e-7)  # a probability of 0 or 1 makes the log undefined
    return float(-np.mean(pos_weight * y * np.log(p) + (1 - y) * np.log(1 - p)))


def top_k_metrics(y, p, frac=0.05):
    '''Precision and recall when flagging the top `frac` highest-scored edges.'''
    k = max(1, int(np.ceil(frac * len(p))))
    top = np.argsort(-p, kind='stable')[:k]  # stable: tied scores keep edge order, so the cut does not depend on the sort
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


def threshold_sensitivity(y, p, threshold):
    '''Test F1 / precision / recall at the headline threshold +-0.05, +-0.10 and at 0.5 — a robustness table, never used to choose.'''
    rows = []
    for delta in SENSITIVITY_DELTAS:
        rows.append(('frozen', threshold + delta, delta))
    rows.append(('paper 0.5', 0.5, 0.5 - threshold))
    table = []
    for label, t, delta in rows:
        t = float(min(max(t, 1e-6), 1 - 1e-6))
        pred = (p >= t).astype(int)
        table.append({'point': label, 'threshold': t, 'delta': float(delta),
                      'f1': float(f1_score(y, pred)), 'precision': float(precision_score(y, pred, zero_division=0)),
                      'recall': float(recall_score(y, pred))})
    return table


def score_all_splits(model, loaders, graphs, threshold, seed=None):
    '''Score train, val and test once at the given threshold.

    Returns {split: metrics} and {split: {'edge_id', 'y', 'prob'}}, where edge_id is the edge's
    position in the graph edge list = its row in Data/edge_features.csv.
    '''
    set_seed(SEED if seed is None else seed)   # same neighbour samples every time a model is scored
    metrics, preds = {}, {}
    for split in SPLITS:
        loader, seed_feats = loaders[split]
        y, p, positions = predict(model, loader, seed_feats, split)
        eval_positions = graphs[split].eval_mask.nonzero().squeeze(1).numpy()
        metrics[split] = compute_metrics(y, p, threshold)
        # input_id indexes evaluated edges only; edge_id is the row in the full edge list.
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
    n_eval = sum(int(graphs[split].eval_mask.sum()) for split in SPLITS)
    assert len(table) == n_eval, f'predictions cover {len(table):,} edges, the splits evaluate {n_eval:,}'
    table.to_csv(f'{out_dir}/predictions.csv', index=False)
    print(f'predictions: {len(table):,} rows -> {out_dir}/predictions.csv')


def flatten_metrics(metrics):
    '''{split: {metric: v}} -> {'<split>_<metric>': v}.'''
    return {f'{split}_{metric}': value for split in SPLITS for metric, value in metrics[split].items()}


# ==================================================================================================
# curves
# ==================================================================================================

def plot_curves(history, y_test, p_test, test_pr_auc, best_epoch, run_name, path, select_metric='f1'):
    '''curves.png: train / val loss, F1, PR-AUC per epoch (dashed line at best_epoch) + test PR curve.

    Train points may be sparse (eval_train_every > 1): NaN rows are skipped per series. y_test=None (tuning
    runs, no test) replaces the PR panel by the validation threshold per epoch; select_metric labels the line.
    '''
    df = pd.DataFrame(history)
    fig, ax = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle(run_name, fontweight='bold')
    panels = [(ax[0, 0], 'loss', 'weighted BCE loss'),
              (ax[0, 1], 'f1', 'F1 (at the validation-chosen threshold)'),
              (ax[1, 0], 'pr_auc', 'PR-AUC')]
    best_label = 'best epoch' if select_metric == 'f1' else 'best epoch (val PR-AUC)'
    for axis, key, title in panels:
        for split, marker in (('train', 'o'), ('val', 's')):
            series = df[['epoch', f'{split}_{key}']].dropna()
            axis.plot(series['epoch'], series[f'{split}_{key}'], marker=marker,
                      label='validation' if split == 'val' else 'train')
        axis.axvline(best_epoch, color='gray', linestyle='--', label=f'{best_label} = {best_epoch}')
        axis.set_title(title)
        axis.set_xlabel('epoch')
        axis.legend()
    if y_test is None:
        ax[1, 1].plot(df['epoch'], df['threshold'], marker='d', color='tab:purple')
        ax[1, 1].axvline(best_epoch, color='gray', linestyle='--')
        ax[1, 1].set_title('validation-chosen threshold per epoch')
        ax[1, 1].set_xlabel('epoch')
    else:
        precision, recall, _ = precision_recall_curve(y_test, p_test)
        ax[1, 1].plot(recall, precision, label=f'test AP = {test_pr_auc:.3f}')
        ax[1, 1].axhline(y_test.mean(), color='gray', linestyle='--', label='random')  # random precision equals the positive rate
        ax[1, 1].set_title('test precision-recall')
        ax[1, 1].set_xlabel('recall')
        ax[1, 1].set_ylabel('precision')
        ax[1, 1].legend()
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.show()
    plt.close(fig)


def plot_final_curves(history, y_test, p_test, test_pr_auc, run_name, path):
    '''curves.png of a final retrain: train-mode batch loss and learning rate per epoch + the test PR curve.

    When the history has eval-mode train / val columns (train_fixed_epochs with eval_loaders) a loss and a
    PR-AUC panel show them, val labelled "seen" (it is training data in this phase).
    '''
    df = pd.DataFrame(history)
    eval_curves = 'train_pr_auc' in df
    if eval_curves:
        fig, grid = plt.subplots(2, 2, figsize=(13, 8))
        ax = [grid[0, 0], grid[1, 0], grid[1, 1], grid[0, 1]]   # loss, LR, test PR, PR-AUC
    else:
        fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    fig.suptitle(run_name, fontweight='bold')
    ax[0].plot(df['epoch'], df['train_loss_batches'], marker='o', label='train, mini-batch mean (dropout on)')
    ax[0].set_title('train loss (mini-batch mean, dropout on)')
    ax[0].set_xlabel('epoch')
    if eval_curves:
        ax[0].plot(df['epoch'], df['train_loss'], marker='o', label='train (eval mode)')
        ax[0].plot(df['epoch'], df['val_loss'], marker='s', label='val (eval mode, seen in training)')
        ax[0].set_title('weighted BCE loss')
        ax[0].legend()
        ax[3].plot(df['epoch'], df['train_pr_auc'], marker='o', label='train')
        ax[3].plot(df['epoch'], df['val_pr_auc'], marker='s', label='val (seen in training)')
        ax[3].set_title('PR-AUC (eval mode)')
        ax[3].set_xlabel('epoch')
        ax[3].legend()
    ax[1].plot(df['epoch'], df['lr'], marker='o', color='tab:orange')
    ax[1].set_title('learning rate')
    ax[1].set_xlabel('epoch')
    precision, recall, _ = precision_recall_curve(y_test, p_test)
    ax[2].plot(recall, precision, label=f'test AP = {test_pr_auc:.3f}')
    ax[2].axhline(y_test.mean(), color='gray', linestyle='--', label='random')  # random precision equals the positive rate
    ax[2].set_title('test precision-recall')
    ax[2].set_xlabel('recall')
    ax[2].set_ylabel('precision')
    ax[2].legend()
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.show()
    plt.close(fig)


