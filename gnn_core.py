'''Shared code for the fixed-architecture GNN experiments (GIN / PNA / GAT / Transformer).

Everything that must be identical across operators lives here and is imported by the
operator notebooks, which only set OPERATOR, the batch of knob configs and paths.
Knobs handled here: mp / readout edge columns, MP_DIRECTION, TEMPORAL_SAMPLING, NODE_ENC
(none | rwpe8 | rwpe16: pre-computed node encodings appended to the 6 entity columns) and
GFP_VARIANT (v0 | tuned: which GFP sheet fills the GFP block of edge_attr, swapped at load time).

Recipe tuning (EXPERIMENTS.md section 2b): the training recipe is a dict (hidden, layers, dropout,
norm, ffn, neighbours, batch size, epochs = cosine length, lr, weight decay, pos_weight). recipe=None
means the v1 recipe below and reproduces every finished run exactly. Tuning runs (run_tuning_trial),
the frozen recipe.json (save_recipe / load_recipe) and the train+val retrain (run_final) live at the
end of this file.
'''
import gc
import glob
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
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.nn import GATv2Conv, GINConv, GINEConv, PNAConv, TransformerConv
from torch_geometric.utils import degree
from tqdm.auto import tqdm

# ---------------- v1 recipe (CLAUDE.md §3) — identical for every run before tuning; a tuned run passes a recipe dict ----------------
HIDDEN        = 128
NUM_LAYERS    = 2
DROPOUT       = 0.3
HEADS         = 4              # GATv2 / Transformer: 4 heads x 32 = 128
NUM_NEIGHBORS = [100, 100]     # neighbours sampled per hop, one entry per layer
BATCH_SIZE    = 8192           # seed edges per mini-batch
EPOCHS        = 20
LR            = 1e-3
WEIGHT_DECAY  = 1e-5           # Adam L2 term; every run since the first batch used it, so it is part of the recipe
POS_WEIGHT    = 8.0            # weight of the laundering class in the loss
SEED          = 42

OPERATORS     = ('gin', 'pna', 'gat', 'transformer')
NODE_ENCS     = ('none', 'rwpe8', 'rwpe16')   # NODE_ENC knob: node encoding appended to the entity one-hots
BASE_NODE_DIM = 6                             # the entity one-hots every run has; encodings come after them
GFP_VARIANTS  = ('v0', 'tuned')               # GFP_VARIANT knob: which GFP sheet fills edge_attr after the 20 baseline columns
GFP_WIDTH     = {'v0': 61, 'tuned': 64}       # v0 = the sheet inside the graph files; tuned = Data/gfp_variants/tuned.npy
BASE_EDGE_DIM = 20                            # baseline edge columns, always edge_attr[:, :20]
SPLITS      = ('train', 'val', 'test')
METRICS     = ('f1', 'precision', 'recall', 'pr_auc', 'roc_auc', 'precision_at_5pct', 'recall_at_5pct')
RUN_FIELDS  = ('threshold', 'best_epoch', 'params', 'invariant_params')
METRIC_COLS = [f'{split}_{metric}' for split in SPLITS for metric in METRICS]
# compact view of batch_summary.csv for the notebook comparison table
DISPLAY_COLS = ['run', 'best_epoch', 'threshold', 'val_f1', 'test_f1', 'test_precision', 'test_recall',
                'test_pr_auc', 'test_precision_at_5pct', 'test_recall_at_5pct', 'params', 'invariant_params']

# ---------------- recipe tuning (EXPERIMENTS.md section 2b) ----------------
NORMS = ('none', 'batch', 'layer')            # normalisation after each message-passing layer
RECIPE_KNOBS = ('hidden', 'num_layers', 'dropout', 'norm', 'ffn', 'num_neighbors', 'batch_size',
                'epochs', 'lr', 'weight_decay', 'pos_weight')
# neighbours per hop by depth: hops 1-2 are always [100, 100], deeper models only add hops
NEIGHBOURS_BY_DEPTH = {2: [100, 100], 3: [100, 100, 50], 4: [100, 100, 50, 50],
                       6: [100, 100, 50, 50, 25, 25], 8: [100, 100, 50, 50, 25, 25, 25, 25]}
NEIGHBOURS_FALLBACK = {3: [100, 100, 25], 4: [100, 100, 25, 25],
                       6: [100, 100, 25, 25, 10, 10], 8: [100, 100, 25, 25, 10, 10, 10, 10]}
# edge rule: a numeric winner at the edge of its grid gets one trial one step beyond
EDGE_BEYOND = {'dropout': {0.2: 0.1, 0.4: 0.5}, 'num_layers': {2: 3, 6: 8},
               'pos_weight': {8.0: 4.0, 16.0: 32.0}, 'lr': {5e-4: 1e-4, 3e-3: 1e-2}}
SENSITIVITY_DELTAS = (-0.10, -0.05, 0.0, 0.05, 0.10)   # threshold-sensitivity table of the final retrain
VAL_PREVALENCE_NOTE = ('train - val gap: validation has 0.107 % laundering vs 0.075 % on train, '
                       'so part of the gap is prevalence, not underfitting')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ==================================================================================================
# recipe
# ==================================================================================================

def default_recipe():
    '''The v1 recipe from the module constants (read at call time, so a scratch patch of a constant still applies).'''
    return dict(hidden=HIDDEN, num_layers=NUM_LAYERS, dropout=DROPOUT, norm='none', ffn=False,
                num_neighbors=list(NUM_NEIGHBORS), batch_size=BATCH_SIZE, epochs=EPOCHS,
                lr=LR, weight_decay=WEIGHT_DECAY, pos_weight=POS_WEIGHT)


def full_recipe(recipe=None):
    '''The v1 recipe overlaid with the given knobs; unknown keys or a depth / neighbour mismatch are errors.'''
    out = default_recipe()
    extra = set(recipe or {}) - set(out)
    assert not extra, f'unknown recipe keys {sorted(extra)}; knobs are {RECIPE_KNOBS}'
    out.update(recipe or {})
    out['num_neighbors'] = [int(n) for n in out['num_neighbors']]
    assert out['norm'] in NORMS, f"norm must be one of {NORMS}, got {out['norm']!r}"
    assert out['hidden'] % HEADS == 0, 'hidden must split into the attention heads'
    assert len(out['num_neighbors']) == out['num_layers'], 'one neighbour count per message-passing layer'
    assert out['num_layers'] >= 1 and out['epochs'] >= 1
    return out


def recipe_tag(recipe, base=None):
    '''Trial folder name = the knobs that differ from `base` (default: the v1 recipe), e.g. d0.2_bn_l4; 'base' if none.'''
    full, ref = full_recipe(recipe), full_recipe(base)
    short = {'dropout': 'd', 'hidden': 'h', 'pos_weight': 'pw', 'lr': 'lr', 'weight_decay': 'wd',
             'batch_size': 'bs', 'epochs': 'ep'}
    parts = []
    for knob in ('dropout', 'norm', 'num_layers', 'num_neighbors', 'hidden', 'pos_weight', 'lr', 'ffn',
                 'weight_decay', 'batch_size', 'epochs'):
        if full[knob] == ref[knob]:
            continue
        if knob == 'norm':
            parts.append({'batch': 'bn', 'layer': 'ln', 'none': 'nonorm'}[full['norm']])
        elif knob == 'ffn':
            parts.append('ffn' if full['ffn'] else 'noffn')
        elif knob == 'num_layers':
            parts.append(f"l{full['num_layers']}")
        elif knob == 'num_neighbors':
            # implied by the depth unless the depth is unchanged (a fallback neighbour list)
            if full['num_layers'] == ref['num_layers']:
                parts.append('nb' + '-'.join(str(n) for n in full['num_neighbors']))
        else:
            parts.append(f'{short[knob]}{full[knob]:g}')
    return '_'.join(parts) or 'base'


def with_knob(recipe, knob, value, neighbours_by_depth=None):
    '''Copy of the recipe with one knob changed; num_layers also sets the neighbour list for that depth.

    Returns None when the depth has no neighbour list (dropped by the probe).
    '''
    out = dict(full_recipe(recipe))
    if knob == 'num_layers':
        table = neighbours_by_depth if neighbours_by_depth is not None else NEIGHBOURS_BY_DEPTH
        if table.get(int(value)) is None:
            return None
        out['num_layers'], out['num_neighbors'] = int(value), list(table[int(value)])
    else:
        out[knob] = value
    return full_recipe(out)


def edge_value(knob, value, grid):
    '''The edge rule: the value one step beyond the grid when `value` sits at its edge, else None.'''
    beyond = EDGE_BEYOND.get(knob, {})
    if value in beyond and value in (min(grid), max(grid)):
        return beyond[value]
    return None


# ==================================================================================================
# data
# ==================================================================================================

def set_seed(seed):
    '''Seed python, numpy and torch so the invariant weights start identical in every run.'''
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def normalise_gfp_sheet(values, cols, n_train):
    '''The GFP normalisation of Data_preparation.ipynb section 6, fit on the first n_train rows only.

    Pattern bins (*_bins_*) untouched; fan/deg/sum/avg/var/kurtosis: log1p -> clip at train p1/p99
    -> StandardScaler; ratio/skew: clip -> StandardScaler. values [E, len(cols)] float32 -> float32.
    '''
    assert values.shape == (values.shape[0], len(cols)) and values.dtype == np.float32
    assert not np.isnan(values).any(), 'GFP sheet contains NaN'
    pattern_cols = [c for c in cols if '_bins_' in c]
    vertex_cols  = [c for c in cols if c not in pattern_cols]
    log_cols = [c for c in vertex_cols if any(s in c for s in ['fan', 'deg', 'sum', 'avg', 'var', 'kurtosis'])]
    std_cols = [c for c in vertex_cols if c not in log_cols]
    print(f'GFP normalisation: untouched pattern bins {len(pattern_cols)} | log+scale {len(log_cols)} '
          f'| scale-only {len(std_cols)} | fit on the first {n_train:,} rows')

    out = values.copy()
    for group, use_log in [(log_cols, True), (std_cols, False)]:
        if not group:
            continue
        idx = [cols.index(c) for c in group]
        vals = values[:, idx]
        if use_log:
            vals = np.log1p(np.clip(vals, 0, None))
        p01 = np.percentile(vals[:n_train], 1, axis=0)
        p99 = np.percentile(vals[:n_train], 99, axis=0)
        vals = np.clip(vals, p01, p99)
        scaler = StandardScaler().fit(vals[:n_train])
        out[:, idx] = scaler.transform(vals).astype(np.float32)
        del vals
    if vertex_cols:
        vertex = out[:, [cols.index(c) for c in vertex_cols]]
        print(f'vertex stats after normalisation: min={vertex.min():.2f} max={vertex.max():.2f} '
              f'mean={vertex.mean():.3f} std={vertex.std():.3f}')
    assert not np.isnan(out).any()
    return out


def load_graphs(data_dir, gfp_variant='v0', gfp_dir=None):
    '''Load the three cumulative snapshots and the feature column groups for the GFP_VARIANT knob.

    v0 -> edge_attr as saved (20 baseline + 61 GFP); tuned -> the GFP block is replaced by
    <gfp_dir>/tuned.npy (64 columns), normalised like v0 on the train rows. Returns graphs
    {split: Data}, col_sets {none|base|gfp|full: column indices}, node_dim.
    '''
    assert gfp_variant in GFP_VARIANTS, f'unknown GFP variant {gfp_variant!r}; choose from {GFP_VARIANTS}'
    graphs = {split: torch.load(f'{data_dir}/{split}_graph.pt', weights_only=False) for split in SPLITS}
    meta = json.load(open(f'{data_dir}/feature_meta.json'))
    base_cols, gfp_cols = meta['BASE_EDGE_COLS'], meta['GFP_FEAT_COLS']
    assert len(base_cols) == BASE_EDGE_DIM and len(gfp_cols) == GFP_WIDTH['v0']
    assert graphs['train'].edge_attr.shape[1] == len(meta['EDGE_FEAT_COLS']) == BASE_EDGE_DIM + GFP_WIDTH['v0']

    if gfp_variant == 'tuned':
        gfp_cols = json.load(open(f'{gfp_dir}/tuned_cols.json'))
        sheet = np.load(f'{gfp_dir}/tuned.npy')
        n_train, n_all = graphs['train'].edge_index.shape[1], graphs['test'].edge_index.shape[1]
        assert sheet.shape == (n_all, GFP_WIDTH['tuned']) and len(gfp_cols) == GFP_WIDTH['tuned'], \
            f'tuned sheet {sheet.shape} must have one row per edge of the test snapshot ({n_all:,}) and 64 columns'
        print(f'GFP variant tuned: sheet {sheet.shape} from {gfp_dir}')
        sheet = torch.from_numpy(normalise_gfp_sheet(sheet, gfp_cols, n_train))
        for split, g in graphs.items():
            n_edges = g.edge_index.shape[1]
            g.edge_attr = torch.cat([g.edge_attr[:, :BASE_EDGE_DIM], sheet[:n_edges]], dim=1)
        del sheet
        gc.collect()

    all_cols = base_cols + gfp_cols
    base_idx = [all_cols.index(c) for c in base_cols]   # 0..19
    gfp_idx  = [all_cols.index(c) for c in gfp_cols]    # 20..80 (v0) or 20..83 (tuned)
    col_sets = {'none': [], 'base': base_idx, 'gfp': gfp_idx, 'full': base_idx + gfp_idx}
    node_dim = graphs['train'].x.shape[1]
    edge_dim = BASE_EDGE_DIM + GFP_WIDTH[gfp_variant]
    assert all(g.edge_attr.shape[1] == len(all_cols) == edge_dim for g in graphs.values())
    print(f'edge_attr width {edge_dim} (base {BASE_EDGE_DIM} + gfp {GFP_WIDTH[gfp_variant]}, variant {gfp_variant})')

    for split, g in graphs.items():
        n_eval = int(g.eval_mask.sum())
        n_pos  = int((g.y[g.eval_mask] == 1).sum())
        print(f'{split:<5} nodes={g.num_nodes:,} edges={g.edge_index.shape[1]:,} '
              f'evaluated={n_eval:,} laundering={n_pos:,} ({n_pos / n_eval:.4%})')
    return graphs, col_sets, node_dim


def set_node_encoding(graphs, node_enc, enc_dir=None):
    '''Set graph.x of every snapshot to [entity one-hots || node encoding] for the NODE_ENC knob.

    none -> the 6 entity columns only (as every run so far); rwpe<k> -> that snapshot's k RWPE
    columns from <enc_dir>/rwpe_k<k>_<split>.pt appended. Returns the node dim (6 or 6 + k).
    '''
    assert node_enc in NODE_ENCS, f'unknown node encoding {node_enc!r}; choose from {NODE_ENCS}'
    for split, graph in graphs.items():
        if 'x_base' not in graph:
            graph.x_base = graph.x        # remembered once, so the knob can change between runs of a batch
        if node_enc == 'none':
            graph.x = graph.x_base
            continue
        k = int(node_enc[len('rwpe'):])
        enc = torch.load(f'{enc_dir}/rwpe_k{k}_{split}.pt', weights_only=True).float()
        assert enc.shape == (graph.num_nodes, k), f'{split}: encoding {tuple(enc.shape)} != ({graph.num_nodes}, {k})'
        assert not torch.isnan(enc).any() and enc.min() >= 0 and enc.max() <= 1
        graph.x = torch.cat([graph.x_base, enc], dim=1)
    node_dim = graphs['train'].x.shape[1]
    assert all(g.x.shape[1] == node_dim for g in graphs.values())
    return node_dim


def make_trainval_graph(graphs):
    '''The final-retrain training graph: the val snapshot with every edge labelled and evaluated.

    Labels of the first n_train rows come from the train snapshot (the val snapshot holds -1 there).
    Call after set_node_encoding (so x already carries the val-snapshot encoding) and never put the
    result into `graphs`. x / edge_index / edge_attr / edge_time are shared views, y and eval_mask new.
    '''
    train, val = graphs['train'], graphs['val']
    n_train, n_all = train.edge_index.shape[1], val.edge_index.shape[1]
    assert torch.equal(val.edge_index[:, :n_train], train.edge_index), 'val snapshot must start with the train edges'
    assert torch.equal(val.edge_attr[:n_train], train.edge_attr), 'edge features of the train rows must match'
    assert bool((val.y[:n_train] == -1).all()) and bool(train.eval_mask.all())
    assert not bool(val.eval_mask[:n_train].any()) and bool(val.eval_mask[n_train:].all())
    y = val.y.clone()
    y[:n_train] = train.y
    assert bool(((y == 0) | (y == 1)).all()), 'every train+val edge must carry a 0/1 label'
    fit = Data(x=val.x, edge_index=val.edge_index, edge_attr=val.edge_attr, edge_time=val.edge_time, y=y,
               eval_mask=torch.ones(n_all, dtype=torch.bool), num_nodes=val.num_nodes)
    print(f'train+val graph: edges={n_all:,} (train {n_train:,} + val {n_all - n_train:,}) '
          f'laundering={int((y == 1).sum()):,} | all edges are training seeds')
    return fit


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
    transformer: root_weight (the node's own state through lin_skip) is part of the standard operator.
    '''
    if operator == 'gin':
        if mp_edge_dim > 0:
            return GINEConv(gin_mlp(hidden), train_eps=True, edge_dim=mp_edge_dim)
        return GINConv(gin_mlp(hidden), train_eps=True)
    if operator == 'pna':
        # 4 aggregators x 3 scalers -> post-MLP -> hidden; scalers calibrated on the train in-degree histogram
        assert deg_hist is not None, 'pna needs the training-graph in-degree histogram (in_degree_histogram)'
        pna_kwargs = dict(aggregators=['mean', 'max', 'min', 'std'],
                          scalers=['identity', 'amplification', 'attenuation'], deg=deg_hist, towers=1)
        if mp_edge_dim > 0:
            return PNAConv(hidden, hidden, edge_dim=mp_edge_dim, **pna_kwargs)
        return PNAConv(hidden, hidden, **pna_kwargs)
    if operator == 'gat':
        # HEADS heads concatenated -> hidden; lin_edge (edge_dim) is the only width-dependent tensor.
        # add_self_loops: node i also attends to itself; fill_value: its self-loop gets the mean of
        # i's incoming edge features. Both are PyG's defaults, written out to be visible.
        if mp_edge_dim > 0:
            return GATv2Conv(hidden, hidden // HEADS, heads=HEADS, edge_dim=mp_edge_dim,
                             add_self_loops=True, fill_value='mean', bias=True)
        return GATv2Conv(hidden, hidden // HEADS, heads=HEADS,
                         add_self_loops=True, fill_value='mean', bias=True)
    if operator == 'transformer':
        # HEADS heads concatenated -> hidden; lin_edge (edge_dim) is the only width-dependent tensor.
        # beta, dropout and root_weight are PyG's defaults, written out to be visible.
        if mp_edge_dim > 0:
            return TransformerConv(hidden, hidden // HEADS, heads=HEADS, concat=True,
                                   edge_dim=mp_edge_dim, beta=False, dropout=0.0, root_weight=True)
        return TransformerConv(hidden, hidden // HEADS, heads=HEADS, concat=True,
                               beta=False, dropout=0.0, root_weight=True)
    raise ValueError(f'unknown operator {operator!r}; choose from {OPERATORS}')


def make_norm(norm, hidden):
    '''none -> Identity (no parameters, so v1 checkpoints still load); batch -> BatchNorm1d; layer -> LayerNorm.'''
    if norm == 'none':
        return nn.Identity()
    if norm == 'batch':
        return nn.BatchNorm1d(hidden)
    if norm == 'layer':
        return nn.LayerNorm(hidden)
    raise ValueError(f'unknown norm {norm!r}; choose from {NORMS}')


class MPLayer(nn.Module):
    '''One message-passing step with a residual: h <- h + Dropout(ReLU(Norm(conv(h, edges)))).

    conv_in aggregates the accounts that PAID a node; bidirectional adds conv_out over
    reversed edges ("whom I paid") and merges the two hidden-dim views back to hidden.
    norm='none' (v1) is the identity; ffn adds the GNN+ block h <- Norm(h + W2(Dropout(ReLU(W1 h))))
    after the residual (expansion 2), where the FFN's Norm is the layer norm choice, or LayerNorm when
    that choice is 'none' (the FFN block always ends in a normalisation). Neither registers parameters when off.
    '''

    def __init__(self, operator, hidden, mp_edge_dim, bidirectional, dropout, deg_hist=None, norm='none', ffn=False):
        super().__init__()
        self.use_edges = mp_edge_dim > 0
        self.bidirectional = bidirectional
        self.conv_in = make_conv(operator, hidden, mp_edge_dim, deg_hist)
        if bidirectional:
            self.conv_out = make_conv(operator, hidden, mp_edge_dim, deg_hist)
            self.merge = nn.Linear(2 * hidden, hidden)
        self.dropout = nn.Dropout(dropout)
        self.norm = make_norm(norm, hidden)
        self.ffn = None
        if ffn:
            self.ffn = nn.Sequential(nn.Linear(hidden, 2 * hidden), nn.ReLU(), nn.Dropout(dropout),
                                     nn.Linear(2 * hidden, hidden))
            self.ffn_norm = make_norm(norm if norm != 'none' else 'layer', hidden)

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
        h = h + self.dropout(torch.relu(self.norm(out)))
        if self.ffn is not None:
            h = self.ffn_norm(h + self.ffn(h))
        return h


#   node features x [N x node_dim] -> node_proj Linear -> hidden, ReLU
#   num_layers x MPLayer: conv (+ W_e . e_uv when MP edge features are on), Norm, ReLU, Dropout, residual, (FFN)
#   readout: concat [h_src || h_dst || e_seed] -> Linear hidden -> ReLU -> Dropout -> Linear 1 (logit)
#   Only W_e (edge width) and the readout's first Linear (seed width) change between feature configs.
class EdgeClassifier(nn.Module):
    '''Fixed-architecture edge classifier: node_proj -> num_layers message-passing layers -> readout MLP.

    Module names (node_proj, layers.i.conv_in/conv_out/merge, classifier) are the ones the
    saved best.pt checkpoints use — do not rename them. recipe=None -> the v1 recipe.
    '''

    def __init__(self, operator, node_dim, mp_edge_dim, readout_edge_dim, bidirectional, deg_hist=None, recipe=None):
        super().__init__()
        r = full_recipe(recipe)
        hidden = r['hidden']
        self.node_proj = nn.Linear(node_dim, hidden)
        self.layers = nn.ModuleList(
            [MPLayer(operator, hidden, mp_edge_dim, bidirectional, r['dropout'], deg_hist, r['norm'], r['ffn'])
             for _ in range(r['num_layers'])])
        self.classifier = nn.Sequential(
            nn.Linear(2 * hidden + readout_edge_dim, hidden), nn.ReLU(), nn.Dropout(r['dropout']),
            nn.Linear(hidden, 1))

    def forward(self, x, edge_index, edge_attr, seed_index, seed_edge_feats):
        '''x [N, node_dim], edge_index [2, E], edge_attr [E, mp_dim] or None,
        seed_index [2, B], seed_edge_feats [B, ro_dim] -> logits [B].'''
        h = torch.relu(self.node_proj(x))
        for layer in self.layers:
            h = layer(h, edge_index, edge_attr)
        z = torch.cat([h[seed_index[0]], h[seed_index[1]], seed_edge_feats], dim=1)
        return self.classifier(z).squeeze(-1)


def build_model(operator, node_dim, mp_edge_dim, readout_edge_dim, bidirectional=False, deg_hist=None, recipe=None):
    '''Build the template for one operator, feature configuration and recipe (None = v1), on the device.'''
    assert operator in OPERATORS, f'unknown operator {operator!r}'
    model = EdgeClassifier(operator, node_dim, mp_edge_dim, readout_edge_dim, bidirectional, deg_hist, recipe)
    return model.to(device)


def count_params(model):
    '''Total number of trainable parameters.'''
    return sum(p.numel() for p in model.parameters())


def invariant_params(model):
    '''Parameters whose shape does not depend on the edge-feature widths.

    node_proj.weight is counted at its base width (hidden x BASE_NODE_DIM): a node encoding
    (NODE_ENC) only widens it, exactly like edge features widen the edge projections.
    67,587 for the GIN family, 67,585 for the GATv2 family (lin_l / lin_r / att / bias count,
    lin_edge does not), 427,265 for the PNA family (its message MLP input widens from 2 x 128
    to 3 x 128 once edge features enter, so its first Linear pre_nns.0.0 is counted with the
    edge projections; PNAConv's output `lin` shares the GINE tag and is left out — it is
    constant, so the check is unaffected), 133,121 for the Transformer family (lin_key /
    lin_query / lin_value / lin_skip count, lin_edge does not) — all at the v1 recipe; a tuned
    recipe (hidden, layers, norm, FFN) gives its own constant. Excluded: the edge projection
    inside the conv (.lin. / lin_edge / edge_encoder / pre_nns.0.0.) and the readout's first
    Linear (classifier.0.). Must be identical across feature configs within a family and recipe.
    '''
    width_dependent = ('.lin.', 'lin_edge', 'edge_encoder', 'pre_nns.0.0.', 'classifier.0.')
    total = 0
    for name, p in model.named_parameters():
        if name == 'node_proj.weight':
            total += model.node_proj.out_features * BASE_NODE_DIM   # the entity columns only; encoding columns are width-dependent
        elif not any(tag in name for tag in width_dependent):
            total += p.numel()
    return total


# ==================================================================================================
# loaders and prediction
# ==================================================================================================

def make_loader(graph, shuffle, mp_idx, readout_idx, temporal=False, recipe=None):
    '''Neighbour-sampling loader over a split's evaluated edges + their readout features.

    Built once per run and reused by every epoch (rebuilding per epoch exhausted Kaggle RAM).
    Neighbours per hop and the batch size come from the recipe (None = v1).
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
    if temporal:
        # only edges earlier than the seed are sampled (needs a recent pyg-lib)
        kwargs.update(time_attr='edge_time', edge_label_time=graph.edge_time[seed_mask])
    return LinkNeighborLoader(mp_graph, **kwargs), seed_feats


def make_loaders(graphs, mp_idx, readout_idx, temporal=False, recipe=None, splits=SPLITS):
    '''One (loader, seed_feats) pair per split; train is shuffled, val/test are not.

    splits=('train', 'val') is what tuning uses: no test loader is ever built there.
    '''
    return {split: make_loader(graphs[split], split == 'train', mp_idx, readout_idx, temporal, recipe)
            for split in splits}


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


def weighted_bce(y, p, pos_weight=None):
    '''The training objective computed from eval-mode probabilities (comparable train/val loss curves).'''
    if pos_weight is None:
        pos_weight = POS_WEIGHT
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return float(-np.mean(pos_weight * y * np.log(p) + (1 - y) * np.log(1 - p)))


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


def threshold_sensitivity(y, p, threshold):
    '''Test F1 / precision / recall at the frozen threshold +-0.05, +-0.10 and at 0.5 — a robustness table, never used to choose.'''
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
    set_seed(SEED if seed is None else seed)   # same neighbour samples whether scoring after training or re-scoring
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

def train_epoch(model, loader, feats, optimizer, criterion):
    '''One pass over the loader in train mode; returns the mean mini-batch loss (train mode, dropout on).'''
    model.train()
    total, count = 0.0, 0
    for batch in tqdm(loader, desc='train', leave=False):
        optimizer.zero_grad()
        logits, y = forward_batch(model, batch, feats)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total, count = total + float(loss.detach()) * len(y), count + len(y)
    return total / max(1, count)


def smoothed(values, window=3):
    '''Centred moving average (window 3 by default, shorter at the ends) of a 1-d sequence.'''
    return pd.Series(np.asarray(values, dtype=float)).rolling(window, center=True, min_periods=1).mean().to_numpy()


def train_one_run(model, loaders, out_dir, recipe=None, stop_epoch=None, eval_train_every=1,
                  patience=None, min_epochs=20, keep_epochs=False):
    '''The training loop: per-epoch validation threshold sweep, best-val-F1 checkpoint.

    recipe=None -> v1 (20 epochs, cosine T_max 20, lr 1e-3, pos_weight 8). recipe['epochs'] is the
    cosine length; the loop runs to stop_epoch (default: all epochs) or stops early when
    `patience` is set and the 3-epoch moving average of val F1 has not improved for that many
    epochs (never before min_epochs). The train split is scored every eval_train_every epochs
    (+ the last); keep_epochs saves epoch_<e>.pt every epoch (tuning: scored later, then deleted).
    Writes best.pt and history.csv to out_dir; returns (history rows, best_epoch, best_threshold).
    '''
    r = full_recipe(recipe)
    train_loader, train_feats = loaders['train']
    val_loader,   val_feats   = loaders['val']
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([r['pos_weight']], device=device))
    optimizer = torch.optim.Adam(model.parameters(), lr=r['lr'], weight_decay=r['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=r['epochs'])
    last_epoch = r['epochs'] if stop_epoch is None else int(stop_epoch)
    assert 1 <= last_epoch <= r['epochs'], f'stop_epoch {last_epoch} must lie inside the schedule (1..{r["epochs"]})'

    history = []
    best_val_f1, best_epoch, best_threshold = -1.0, 0, 0.5
    best_smoothed, epochs_without_gain = -1.0, 0
    for epoch in range(1, last_epoch + 1):
        t0 = time.time()
        lr_now = optimizer.param_groups[0]['lr']
        loss_batches = train_epoch(model, train_loader, train_feats, optimizer, criterion)

        # overfitting diagnostics: train and val in eval mode; train F1 reuses the val threshold
        score_train = (epoch % eval_train_every == 0) or epoch == last_epoch
        if score_train:
            y_train, p_train, _ = predict(model, train_loader, train_feats, 'train')
        y_val, p_val, _ = predict(model, val_loader, val_feats, 'val')
        threshold, val_f1 = best_f1_threshold(y_val, p_val)
        train_f1 = f1_score(y_train, (p_train >= threshold).astype(int)) if score_train else float('nan')

        is_best = val_f1 > best_val_f1
        if is_best:
            best_val_f1, best_epoch, best_threshold = val_f1, epoch, threshold
            torch.save(model.state_dict(), f'{out_dir}/best.pt')
        if keep_epochs:
            torch.save(model.state_dict(), f'{out_dir}/epoch_{epoch:03d}.pt')

        history.append({
            'epoch':        epoch,
            'train_loss':   weighted_bce(y_train, p_train, r['pos_weight']) if score_train else float('nan'),
            'val_loss':     weighted_bce(y_val, p_val, r['pos_weight']),
            'train_f1':     train_f1,
            'val_f1':       val_f1,
            'train_pr_auc': average_precision_score(y_train, p_train) if score_train else float('nan'),
            'val_pr_auc':   average_precision_score(y_val, p_val),
            'val_roc_auc':  roc_auc_score(y_val, p_val),
            'threshold':    threshold,
            'seconds':      time.time() - t0,
            'train_loss_batches': loss_batches,
            'lr':           lr_now,
        })
        row = history[-1]
        ram_gb = psutil.Process().memory_info().rss / 2**30
        train_str = f"{row['train_loss']:.4f}" if score_train else '  -   '
        f1_str = f"{row['train_f1']:.4f}" if score_train else '  -   '
        pr_str = f"{row['train_pr_auc']:.4f}" if score_train else '  -   '
        print(f"epoch {epoch:02d} | loss {train_str} / {row['val_loss']:.4f} | "
              f"F1 {f1_str} / {row['val_f1']:.4f} @ {threshold:.3f} | "
              f"PR-AUC {pr_str} / {row['val_pr_auc']:.4f} | "
              f"{row['seconds']:.0f}s | RAM {ram_gb:.1f}GB   (train / val)"
              + ('   * new best -> best.pt' if is_best else ''))
        scheduler.step()

        if patience is not None:
            # early stopping on the smoothed val F1 curve, never before min_epochs
            current = smoothed([h['val_f1'] for h in history])[-1]
            if current > best_smoothed:
                best_smoothed, epochs_without_gain = current, 0
            else:
                epochs_without_gain += 1
            if epoch >= min_epochs and epochs_without_gain >= patience:
                print(f'early stop at epoch {epoch}: smoothed val F1 has not improved for {patience} epochs')
                break

    pd.DataFrame(history).to_csv(f'{out_dir}/history.csv', index=False)
    print(f'best validation F1 {best_val_f1:.4f} at epoch {best_epoch} (threshold {best_threshold:.3f}) '
          f'| {len(history)} of {r["epochs"]} epochs run')
    return history, best_epoch, best_threshold


def train_fixed_epochs(model, fit_loader, out_dir, recipe, stop_epoch):
    '''Final retrain: stop_epoch epochs on the train+val loader with the cosine schedule of recipe['epochs']; no validation.

    Writes final.pt and history.csv (epoch, train_loss_batches, lr, seconds); returns the history rows.
    '''
    r = full_recipe(recipe)
    assert 1 <= stop_epoch <= r['epochs'], f'stop_epoch {stop_epoch} must lie inside the schedule (1..{r["epochs"]})'
    loader, feats = fit_loader
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([r['pos_weight']], device=device))
    optimizer = torch.optim.Adam(model.parameters(), lr=r['lr'], weight_decay=r['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=r['epochs'])
    history = []
    for epoch in range(1, stop_epoch + 1):
        t0 = time.time()
        lr_now = optimizer.param_groups[0]['lr']
        loss = train_epoch(model, loader, feats, optimizer, criterion)
        history.append({'epoch': epoch, 'train_loss_batches': loss, 'lr': lr_now, 'seconds': time.time() - t0})
        ram_gb = psutil.Process().memory_info().rss / 2**30
        print(f'epoch {epoch:02d}/{stop_epoch} | batch loss {loss:.4f} | lr {lr_now:.2e} | '
              f'{history[-1]["seconds"]:.0f}s | RAM {ram_gb:.1f}GB')
        scheduler.step()
    torch.save(model.state_dict(), f'{out_dir}/final.pt')
    pd.DataFrame(history).to_csv(f'{out_dir}/history.csv', index=False)
    return history


def best_epoch_from_history(history):
    '''best_epoch and its threshold from history.csv (first epoch with the highest val F1).'''
    best = int(np.argmax(history['val_f1'].to_numpy()))
    return int(history['epoch'].iloc[best]), float(history['threshold'].iloc[best])


# ==================================================================================================
# plots, saving, naming
# ==================================================================================================

def plot_curves(history, y_test, p_test, test_pr_auc, best_epoch, run_name, path):
    '''curves.png: train/val loss, F1, PR-AUC per epoch (dashed line at best_epoch) + test PR curve.

    Train points may be sparse (eval_train_every > 1): NaN rows are skipped per series.
    y_test=None (tuning runs, no test) replaces the PR panel by the validation threshold per epoch.
    '''
    df = pd.DataFrame(history)
    fig, ax = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle(run_name, fontweight='bold')
    panels = [(ax[0, 0], 'loss', 'weighted BCE loss'),
              (ax[0, 1], 'f1', 'F1 (at the validation-chosen threshold)'),
              (ax[1, 0], 'pr_auc', 'PR-AUC')]
    for axis, key, title in panels:
        for split, marker in (('train', 'o'), ('val', 's')):
            series = df[['epoch', f'{split}_{key}']].dropna()
            axis.plot(series['epoch'], series[f'{split}_{key}'], marker=marker,
                      label='validation' if split == 'val' else 'train')
        axis.axvline(best_epoch, color='gray', linestyle='--', label=f'best epoch = {best_epoch}')
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
        ax[1, 1].axhline(y_test.mean(), color='gray', linestyle='--', label='random')
        ax[1, 1].set_title('test precision-recall')
        ax[1, 1].set_xlabel('recall')
        ax[1, 1].set_ylabel('precision')
        ax[1, 1].legend()
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.show()
    plt.close(fig)


def plot_final_curves(history, y_test, p_test, test_pr_auc, run_name, path):
    '''curves.png of a final retrain: train-mode batch loss and learning rate per epoch + the test PR curve.'''
    df = pd.DataFrame(history)
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    fig.suptitle(run_name, fontweight='bold')
    ax[0].plot(df['epoch'], df['train_loss_batches'], marker='o')
    ax[0].set_title('train loss (mini-batch mean, dropout on)')
    ax[0].set_xlabel('epoch')
    ax[1].plot(df['epoch'], df['lr'], marker='o', color='tab:orange')
    ax[1].set_title('learning rate (cosine, frozen T_max)')
    ax[1].set_xlabel('epoch')
    precision, recall, _ = precision_recall_curve(y_test, p_test)
    ax[2].plot(recall, precision, label=f'test AP = {test_pr_auc:.3f}')
    ax[2].axhline(y_test.mean(), color='gray', linestyle='--', label='random')
    ax[2].set_title('test precision-recall')
    ax[2].set_xlabel('recall')
    ax[2].set_ylabel('precision')
    ax[2].legend()
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.show()
    plt.close(fig)


def run_name(operator, cfg, mp_direction='in', temporal=False, node_enc='none', gfp_variant='v0', seed=None):
    '''<model>_mp-<mp>_readout-<ro>_dir-<dir>[_enc-<enc>][_gfp-<variant>][_temporal][_seed-<s>] (CLAUDE.md §6).

    The seed suffix appears only for an explicit seed other than 42; tuned phases keep the seed as a sub-folder.
    '''
    name = f"{operator}_mp-{cfg['mp']}_readout-{cfg['readout']}_dir-{mp_direction}"
    if node_enc != 'none':
        name += f'_enc-{node_enc}'
    if gfp_variant != 'v0':
        name += f'_gfp-{gfp_variant}'
    if temporal:
        name += '_temporal'
    if seed is not None and seed != SEED:
        name += f'_seed-{seed}'
    return name


def result_row(name, operator, cfg, mp_direction, temporal, best_epoch, threshold,
               n_params, n_invariant, metrics, node_enc='none', gfp_variant='v0', seed=None):
    '''The flat result row shared by results.json and batch_summary.csv.'''
    row = {'run': name, 'operator': operator, 'mp': cfg['mp'], 'readout': cfg['readout'],
           'mp_direction': mp_direction, 'temporal_sampling': temporal, 'node_enc': node_enc,
           'gfp_variant': gfp_variant, 'seed': SEED if seed is None else seed,
           'best_epoch': best_epoch, 'threshold': threshold,
           'params': n_params, 'invariant_params': n_invariant}
    row.update(flatten_metrics(metrics))
    return row


def save_run(out_dir, row, recipe=None, extra=None):
    '''Write results.json (the flat row, the architecture block, the full recipe, extras) and print the per-split table.'''
    r = full_recipe(recipe)
    results = dict(row)
    results['architecture'] = {'hidden': r['hidden'], 'layers': r['num_layers'], 'dropout': r['dropout'],
                               'neighbors': r['num_neighbors']}
    results['recipe'] = r
    results.update(extra or {})
    json.dump(results, open(f'{out_dir}/results.json', 'w'), indent=2)
    table = {split: {metric: row[f'{split}_{metric}'] for metric in METRICS} for split in SPLITS}
    print(pd.DataFrame(table).round(4))


def print_model_header(name, mp_idx, readout_idx, n_params, n_invariant, node_dim=BASE_NODE_DIM):
    '''One-line description of the run and its parameter fingerprint.'''
    print(f'=== {name} ===')
    print(f'MP edge features: {len(mp_idx)} | readout edge features: {len(readout_idx)} | node features: {node_dim}')
    print(f'total params: {n_params:,} | architecture-invariant: {n_invariant:,}')


# ==================================================================================================
# one run, one batch summary
# ==================================================================================================

def run_experiment(operator, cfg, graphs, col_sets, node_dim, out_root, mp_direction='in', temporal=False,
                   node_enc='none', enc_dir=None, gfp_variant='v0', recipe=None, seed=None, eval_train_every=1):
    '''Train and evaluate one knob configuration end-to-end; returns the flat result row.

    node_enc = 'none' leaves graph.x and node_dim as loaded; 'rwpe<k>' appends the k RWPE columns
    from enc_dir, so only node_proj widens (6 + k -> 128). gfp_variant must be the one the graphs
    were loaded with (load_graphs); it only labels the run, the swap happened at load time.
    recipe=None and seed=None reproduce the v1 runs exactly.
    '''
    assert len(col_sets['gfp']) == GFP_WIDTH[gfp_variant], \
        f"graphs hold a {len(col_sets['gfp'])}-column GFP block, not the {gfp_variant!r} sheet; reload with load_graphs"
    name = run_name(operator, cfg, mp_direction, temporal, node_enc, gfp_variant, seed)
    out_dir = f'{out_root}/{name}'
    os.makedirs(out_dir, exist_ok=True)
    mp_idx, readout_idx = col_sets[cfg['mp']], col_sets[cfg['readout']]
    node_dim = set_node_encoding(graphs, node_enc, enc_dir)   # 6 for 'none', 6 + k for 'rwpe<k>'

    set_seed(SEED if seed is None else seed)
    deg_hist = in_degree_histogram(graphs['train']) if operator == 'pna' else None
    model = build_model(operator, node_dim, len(mp_idx), len(readout_idx),
                        mp_direction == 'bidirectional', deg_hist, recipe)
    n_params, n_invariant = count_params(model), invariant_params(model)
    print_model_header(name, mp_idx, readout_idx, n_params, n_invariant, node_dim)

    loaders = make_loaders(graphs, mp_idx, readout_idx, temporal, recipe)
    history, best_epoch, threshold = train_one_run(model, loaders, out_dir, recipe, eval_train_every=eval_train_every)

    # final evaluation: best checkpoint, one validation-chosen threshold, test scored once
    model.load_state_dict(torch.load(f'{out_dir}/best.pt', map_location=device))
    metrics, preds = score_all_splits(model, loaders, graphs, threshold, seed)
    row = result_row(name, operator, cfg, mp_direction, temporal, best_epoch, threshold,
                     n_params, n_invariant, metrics, node_enc, gfp_variant, seed)
    save_run(out_dir, row, recipe)
    save_predictions(out_dir, graphs, preds, threshold)
    plot_curves(history, preds['test']['y'], preds['test']['prob'], metrics['test']['pr_auc'],
                best_epoch, name, f'{out_dir}/curves.png')

    del model, loaders
    gc.collect()
    torch.cuda.empty_cache()
    return row


def summarize_batch(rows, out_root, filename='batch_summary.csv', check_invariant=True):
    '''Write the batch summary CSV; check every §4 column is present and (by default) invariant_params is identical.'''
    summary = pd.DataFrame(rows)
    missing = [c for c in METRIC_COLS + list(RUN_FIELDS) if c not in summary.columns]
    assert not missing, f'batch_summary.csv is missing columns: {missing}'
    if check_invariant:
        assert summary['invariant_params'].nunique() == 1, \
            f"invariant_params differ across runs: {summary['invariant_params'].tolist()}"
    summary.to_csv(f'{out_root}/{filename}', index=False)
    print(f'{len(summary)} runs -> {out_root}/{filename} | invariant_params = '
          f"{int(summary['invariant_params'].iloc[0]):,} in the first row"
          + (' (identical in every row)' if check_invariant else ''))
    return summary


def seed_summary(rows, out_root, filename='seed_summary.csv'):
    '''Mean and std over the seeds of every metric column (one configuration) -> <out_root>/<filename>; returns the one-row table.'''
    summary = pd.DataFrame(rows)
    out = {'run': summary['run'].iloc[0], 'operator': summary['operator'].iloc[0], 'mp': summary['mp'].iloc[0],
           'readout': summary['readout'].iloc[0], 'n_seeds': len(summary),
           'seeds': ' '.join(str(s) for s in summary['seed'])}
    extra = [c for c in summary.columns if c.startswith('seen_') or c == 'val_seen_threshold']   # final retrain only
    for col in list(METRIC_COLS) + ['best_epoch', 'threshold'] + extra:
        out[f'{col}_mean'] = float(summary[col].mean())
        out[f'{col}_std'] = float(summary[col].std(ddof=1)) if len(summary) > 1 else 0.0
    table = pd.DataFrame([out])
    table.to_csv(f'{out_root}/{filename}', index=False)
    print(f'{len(summary)} seeds -> {out_root}/{filename}')
    return table


# ==================================================================================================
# recipe tuning — trials on the train / val graphs only (EXPERIMENTS.md section 2b)
# ==================================================================================================

def run_tuning_seed(operator, cfg, graphs, col_sets, node_dim, trial_dir, recipe, seed, mp_direction='in',
                    temporal=False, eval_train_every=5, patience=10, min_epochs=20):
    '''One (trial, seed): train on the train graph, validate every epoch, keep a checkpoint per epoch.

    No test loader, no predictions. Writes seed-<s>/history.csv, curves.png, best.pt, epoch_<e>.pt.
    '''
    out_dir = f'{trial_dir}/seed-{seed}'
    os.makedirs(out_dir, exist_ok=True)
    mp_idx, readout_idx = col_sets[cfg['mp']], col_sets[cfg['readout']]
    set_seed(seed)
    deg_hist = in_degree_histogram(graphs['train']) if operator == 'pna' else None
    model = build_model(operator, node_dim, len(mp_idx), len(readout_idx), mp_direction == 'bidirectional',
                        deg_hist, recipe)
    n_params, n_invariant = count_params(model), invariant_params(model)
    name = f'{os.path.basename(trial_dir)} seed {seed}'
    print_model_header(name, mp_idx, readout_idx, n_params, n_invariant, node_dim)
    loaders = make_loaders(graphs, mp_idx, readout_idx, temporal, recipe, splits=('train', 'val'))
    history, best_epoch, _ = train_one_run(model, loaders, out_dir, recipe, eval_train_every=eval_train_every,
                                           patience=patience, min_epochs=min_epochs, keep_epochs=True)
    plot_curves(history, None, None, None, best_epoch, name, f'{out_dir}/curves.png')
    del model, loaders
    gc.collect()
    torch.cuda.empty_cache()
    return {'params': n_params, 'invariant_params': n_invariant, 'epochs_run': len(history),
            'seconds_per_epoch': float(np.mean([h['seconds'] for h in history]))}


def seed_mean_best_epoch(histories, window=3):
    '''Best epoch of a trial = argmax of the smoothed seed-mean val F1 over the epochs every seed reached.

    histories: {seed: history DataFrame}. Returns (best_epoch, mean curve, smoothed curve).
    '''
    n_common = min(len(h) for h in histories.values())
    f1 = np.stack([h['val_f1'].to_numpy()[:n_common] for h in histories.values()])
    mean = f1.mean(axis=0)
    smooth = smoothed(mean, window)
    return int(np.argmax(smooth)) + 1, mean, smooth


def score_trial(operator, cfg, graphs, col_sets, node_dim, trial_dir, recipe, seeds, stage, seed_info,
                mp_direction='in', temporal=False):
    '''After the last seed: score each seed's checkpoint at the seed-mean best epoch on train and val, write trial.json.

    Each seed uses its own validation threshold of that epoch. The per-epoch checkpoints are deleted
    afterwards; trial.json is what marks the trial as finished (resume skips it).
    '''
    r = full_recipe(recipe)
    tag = os.path.basename(trial_dir)
    histories = {s: pd.read_csv(f'{trial_dir}/seed-{s}/history.csv') for s in seeds}
    best_epoch, mean_curve, smooth_curve = seed_mean_best_epoch(histories)
    mp_idx, readout_idx = col_sets[cfg['mp']], col_sets[cfg['readout']]
    deg_hist = in_degree_histogram(graphs['train']) if operator == 'pna' else None

    per_seed = {}
    for s in seeds:
        h = histories[s]
        threshold = float(h.loc[h['epoch'] == best_epoch, 'threshold'].iloc[0])
        set_seed(s)
        model = build_model(operator, node_dim, len(mp_idx), len(readout_idx), mp_direction == 'bidirectional',
                            deg_hist, recipe)
        model.load_state_dict(torch.load(f'{trial_dir}/seed-{s}/epoch_{best_epoch:03d}.pt', map_location=device))
        loaders = make_loaders(graphs, mp_idx, readout_idx, temporal, recipe, splits=('train', 'val'))
        metrics = {}
        for split in ('train', 'val'):
            loader, feats = loaders[split]
            y, p, _ = predict(model, loader, feats, f'{split} seed {s} @ epoch {best_epoch}')
            metrics[split] = compute_metrics(y, p, threshold)
            metrics[split]['loss'] = weighted_bce(y, p, r['pos_weight'])
        per_seed[str(s)] = {'threshold': threshold, 'epochs_run': int(len(h)),
                            'reached_max': bool(len(h) >= r['epochs']),
                            'best_epoch_own': int(h['epoch'].iloc[int(np.argmax(h['val_f1'].to_numpy()))]),
                            **{f'{split}_{m}': v for split in ('train', 'val') for m, v in metrics[split].items()}}
        del model, loaders
        gc.collect()
        torch.cuda.empty_cache()

    def agg(key):
        vals = np.array([per_seed[str(s)][key] for s in seeds], dtype=float)
        return float(vals.mean()), (float(vals.std(ddof=1)) if len(vals) > 1 else 0.0)

    trial = {'trial': tag, 'stage': stage, 'operator': operator, 'cfg': cfg, 'recipe': r, 'seeds': list(seeds),
             'best_epoch': best_epoch, 'epochs_common': int(len(mean_curve)),
             'epochs_run': {str(s): per_seed[str(s)]['epochs_run'] for s in seeds},
             'reached_max': any(per_seed[str(s)]['reached_max'] for s in seeds),
             'thresholds': {str(s): per_seed[str(s)]['threshold'] for s in seeds},
             'params': seed_info['params'], 'invariant_params': seed_info['invariant_params'],
             'seconds_per_epoch': seed_info['seconds_per_epoch'],
             'val_f1_mean_curve': [float(v) for v in mean_curve],
             'val_f1_smoothed_curve': [float(v) for v in smooth_curve],
             'per_seed': per_seed, 'gap_note': VAL_PREVALENCE_NOTE}
    for key in ['train_f1', 'val_f1', 'train_precision', 'val_precision', 'train_recall', 'val_recall',
                'train_pr_auc', 'val_pr_auc', 'train_loss', 'val_loss']:
        trial[f'{key}_mean'], trial[f'{key}_std'] = agg(key)
    trial['gap_f1'] = trial['train_f1_mean'] - trial['val_f1_mean']
    trial['gap_pr_auc'] = trial['train_pr_auc_mean'] - trial['val_pr_auc_mean']
    json.dump(trial, open(f'{trial_dir}/trial.json', 'w'), indent=2)

    for s in seeds:
        for path in glob.glob(f'{trial_dir}/seed-{s}/epoch_*.pt'):
            os.remove(path)
    print(f"trial {tag}: best epoch {best_epoch} (seed-mean, smoothed) | val F1 {trial['val_f1_mean']:.4f} "
          f"± {trial['val_f1_std']:.4f} | train F1 {trial['train_f1_mean']:.4f} | gap {trial['gap_f1']:+.4f} "
          f"| epochs run {trial['epochs_run']} | checkpoints deleted")
    return trial


def run_tuning_trial(operator, cfg, graphs, col_sets, node_dim, stage_dir, recipe, seeds, base_recipe=None,
                     mp_direction='in', temporal=False, eval_train_every=5, patience=10, min_epochs=20, reuse=None):
    '''One trial = both seeds back to back, then the post-trial scoring; skipped when trial.json exists.

    reuse = the incumbent's trial dict: when the recipe is identical, its scoring is copied into this
    stage (trial.json with `reused_from`) instead of training the same recipe again.
    '''
    stage = os.path.basename(stage_dir)
    tag = recipe_tag(recipe, base_recipe)
    trial_dir = f'{stage_dir}/{tag}'
    if os.path.exists(f'{trial_dir}/trial.json'):
        trial = json.load(open(f'{trial_dir}/trial.json'))
        print(f"=== {stage}/{tag} === already finished (val F1 {trial['val_f1_mean']:.4f} ± {trial['val_f1_std']:.4f}, "
              f"best epoch {trial['best_epoch']}), skipping")
        return trial
    if reuse is not None and full_recipe(reuse['recipe']) == full_recipe(recipe):
        os.makedirs(trial_dir, exist_ok=True)
        trial = dict(reuse, stage=stage, reused_from=f"{reuse['stage']}/{reuse['trial']}")
        json.dump(trial, open(f'{trial_dir}/trial.json', 'w'), indent=2)
        print(f"=== {stage}/{tag} === same recipe as the incumbent {trial['reused_from']}: scoring copied, not retrained")
        return trial
    print('=' * 90)
    print(f'=== {stage}/{tag} === recipe {full_recipe(recipe)}')
    seed_info = None
    for seed in seeds:
        seed_info = run_tuning_seed(operator, cfg, graphs, col_sets, node_dim, trial_dir, recipe, seed,
                                    mp_direction, temporal, eval_train_every, patience, min_epochs)
    return score_trial(operator, cfg, graphs, col_sets, node_dim, trial_dir, recipe, seeds, stage, seed_info,
                       mp_direction, temporal)


def choose_winner(trials, incumbent_tag, min_gain=0.01):
    '''The stage winner: a trial replaces the incumbent only if its seed-mean val F1 beats it by more than
    max(min_gain, the larger of the two seed stds); otherwise the incumbent (the simpler option) stays.'''
    by_tag = {t['trial']: t for t in trials}
    assert incumbent_tag in by_tag, f'incumbent {incumbent_tag!r} not among the trials {sorted(by_tag)}'
    incumbent = by_tag[incumbent_tag]
    winner = incumbent
    for trial in sorted(trials, key=lambda t: -t['val_f1_mean']):
        if trial['trial'] == incumbent_tag:
            continue
        margin = max(min_gain, trial['val_f1_std'], incumbent['val_f1_std'])
        if trial['val_f1_mean'] > incumbent['val_f1_mean'] + margin:
            winner = trial
            break
    print(f"winner: {winner['trial']} (val F1 {winner['val_f1_mean']:.4f} ± {winner['val_f1_std']:.4f}); "
          f"incumbent {incumbent_tag} {incumbent['val_f1_mean']:.4f} ± {incumbent['val_f1_std']:.4f}"
          + ('' if winner is incumbent else ' -> replaced'))
    return winner


def trials_table(trials, winners=None):
    '''One row per trial with every knob, best / stopped epochs, train & val F1 / PR-AUC (mean ± std), the gap and a winner flag.'''
    winners = winners or {}
    rows = []
    for t in trials:
        r = t['recipe']
        rows.append({
            'stage': t['stage'], 'trial': t['trial'], 'winner': winners.get(t['stage']) == t['trial'],
            'dropout': r['dropout'], 'norm': r['norm'], 'num_layers': r['num_layers'],
            'num_neighbors': '-'.join(str(n) for n in r['num_neighbors']), 'hidden': r['hidden'],
            'pos_weight': r['pos_weight'], 'lr': r['lr'], 'ffn': r['ffn'], 'epochs_T_max': r['epochs'],
            'best_epoch': t['best_epoch'],
            'epochs_run': '/'.join(str(t['epochs_run'][str(s)]) for s in t['seeds']),
            'reached_max': t['reached_max'],
            'val_f1_mean': t['val_f1_mean'], 'val_f1_std': t['val_f1_std'],
            'train_f1_mean': t['train_f1_mean'], 'train_f1_std': t['train_f1_std'], 'gap_f1': t['gap_f1'],
            'val_pr_auc_mean': t['val_pr_auc_mean'], 'val_pr_auc_std': t['val_pr_auc_std'],
            'train_pr_auc_mean': t['train_pr_auc_mean'], 'gap_pr_auc': t['gap_pr_auc'],
            'thresholds': '/'.join(f"{t['thresholds'][str(s)]:.3f}" for s in t['seeds']),
            'params': t['params'], 'invariant_params': t['invariant_params'],
            'seconds_per_epoch': t['seconds_per_epoch'],
        })
    return pd.DataFrame(rows)


def load_trials(root):
    '''Every finished trial.json under root (a stage folder or the tuning root), in path order.'''
    paths = sorted(glob.glob(f'{root}/trial.json')) + sorted(glob.glob(f'{root}/*/trial.json')) \
        + sorted(glob.glob(f'{root}/*/*/trial.json'))
    return [json.load(open(p)) for p in paths]


def summarize_stage(stage_dir, winner_tag):
    '''stage_summary.csv of one stage (all finished trials in the folder, winner flagged); returns the table.'''
    trials = load_trials(stage_dir)
    table = trials_table(trials, {os.path.basename(stage_dir): winner_tag})
    table.to_csv(f'{stage_dir}/stage_summary.csv', index=False)
    print(f'{len(table)} trials -> {stage_dir}/stage_summary.csv | {VAL_PREVALENCE_NOTE}')
    return table


def summarize_tuning(out_root, winners):
    '''trials_summary.csv + tuning_curves.png over every finished trial under out_root; returns the table.'''
    trials = load_trials(out_root)
    table = trials_table(trials, winners)
    table.to_csv(f'{out_root}/trials_summary.csv', index=False)
    plot_val_curves(trials, winners, f'{out_root}/tuning_curves.png')
    print(f'{len(table)} trials -> {out_root}/trials_summary.csv | {VAL_PREVALENCE_NOTE}')
    return table


def plot_val_curves(trials, winners, path):
    '''tuning_curves.png: seed-mean val F1 per epoch, one line per trial (winners bold), a dot at each best epoch.'''
    winners = winners or {}
    fig, ax = plt.subplots(figsize=(12, 6))
    for t in trials:
        curve = t['val_f1_mean_curve']
        epochs = np.arange(1, len(curve) + 1)
        is_winner = winners.get(t['stage']) == t['trial']
        line, = ax.plot(epochs, curve, linewidth=2.2 if is_winner else 1.0, alpha=1.0 if is_winner else 0.7,
                        label=f"{t['stage']}/{t['trial']} ({t['val_f1_mean']:.3f} @ {t['best_epoch']})")
        ax.plot(t['best_epoch'], curve[t['best_epoch'] - 1], 'o', color=line.get_color())
    ax.set_xlabel('epoch')
    ax.set_ylabel('validation F1 (seed mean)')
    ax.set_title('recipe tuning: seed-mean validation F1 per trial')
    ax.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.show()
    plt.close(fig)


def probe_depths(operator, cfg, graphs, col_sets, node_dim, recipe, depths, out_path, n_warmup=3, n_timed=20,
                 max_ratio=3.0, mp_direction='in', temporal=False, reference_seconds=None):
    '''Memory / time probe per depth at the heaviest setting: warm-up + timed train batches, then one full val prediction pass.

    For each depth the primary neighbour list is tried first, then the fallback; a depth passes when
    both fit in memory and its epoch estimate is <= max_ratio x the reference (the first depth probed,
    or reference_seconds from an earlier probe). Returns {depth: {'neighbors': list or None, ...}} and
    writes it to out_path (probe.json, with 'reference_seconds').
    '''
    mp_idx, readout_idx = col_sets[cfg['mp']], col_sets[cfg['readout']]
    deg_hist = in_degree_histogram(graphs['train']) if operator == 'pna' else None
    results, base_epoch_seconds = {}, reference_seconds
    for depth in depths:
        candidates = [('primary', NEIGHBOURS_BY_DEPTH.get(depth))]
        if depth in NEIGHBOURS_FALLBACK:
            candidates.append(('fallback', NEIGHBOURS_FALLBACK[depth]))
        results[depth] = {'neighbors': None, 'status': 'dropped', 'attempts': []}
        for label, neighbours in candidates:
            if neighbours is None:
                continue
            r = full_recipe(dict(recipe, num_layers=depth, num_neighbors=neighbours))
            attempt = {'list': label, 'neighbors': neighbours}
            print(f'--- probe: {depth} layers, {label} neighbours {neighbours}')
            try:
                set_seed(SEED)
                model = build_model(operator, node_dim, len(mp_idx), len(readout_idx), mp_direction == 'bidirectional',
                                    deg_hist, r)
                loaders = make_loaders(graphs, mp_idx, readout_idx, temporal, r, splits=('train', 'val'))
                train_loader, train_feats = loaders['train']
                criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([r['pos_weight']], device=device))
                optimizer = torch.optim.Adam(model.parameters(), lr=r['lr'], weight_decay=r['weight_decay'])
                if device.type == 'cuda':
                    torch.cuda.reset_peak_memory_stats()
                model.train()
                seconds, done = 0.0, 0
                for i, batch in enumerate(train_loader):
                    t0 = time.time()
                    optimizer.zero_grad()
                    logits, y = forward_batch(model, batch, train_feats)
                    loss = criterion(logits, y)
                    loss.backward()
                    optimizer.step()
                    if device.type == 'cuda':
                        torch.cuda.synchronize()
                    if i >= n_warmup:
                        seconds, done = seconds + time.time() - t0, done + 1
                    if done >= n_timed:
                        break
                attempt['train_s_per_batch'] = seconds / max(1, done)
                attempt['train_batches_per_epoch'] = len(train_loader)
                attempt['train_epoch_s'] = attempt['train_s_per_batch'] * len(train_loader)
                attempt['train_peak_gb'] = torch.cuda.max_memory_allocated() / 2**30 if device.type == 'cuda' else float('nan')
                if device.type == 'cuda':
                    torch.cuda.reset_peak_memory_stats()
                t0 = time.time()
                val_loader, val_feats = loaders['val']
                predict(model, val_loader, val_feats, f'probe val {depth}L')
                attempt['val_predict_s'] = time.time() - t0
                attempt['val_peak_gb'] = torch.cuda.max_memory_allocated() / 2**30 if device.type == 'cuda' else float('nan')
                attempt['epoch_estimate_s'] = attempt['train_epoch_s'] + attempt['val_predict_s']
                attempt['params'] = count_params(model)
                del model, loaders, optimizer
                gc.collect()
                torch.cuda.empty_cache()
            except torch.cuda.OutOfMemoryError as err:   # the one place an error is caught: the probe exists to find this
                attempt['error'] = f'CUDA out of memory: {str(err)[:120]}'
                gc.collect()
                torch.cuda.empty_cache()
                results[depth]['attempts'].append(attempt)
                print(f'    out of memory with {label} neighbours')
                continue
            if base_epoch_seconds is None:
                base_epoch_seconds = attempt['epoch_estimate_s']
            attempt['ratio_to_first_depth'] = attempt['epoch_estimate_s'] / base_epoch_seconds
            results[depth]['attempts'].append(attempt)
            print(f"    train {attempt['train_s_per_batch']:.2f} s/batch x {len(train_loader)} = {attempt['train_epoch_s'] / 60:.1f} min, "
                  f"peak {attempt['train_peak_gb']:.1f} GB | val pass {attempt['val_predict_s'] / 60:.1f} min, "
                  f"peak {attempt['val_peak_gb']:.1f} GB | epoch ~{attempt['epoch_estimate_s'] / 60:.1f} min "
                  f"({attempt['ratio_to_first_depth']:.2f}x the first depth)")
            if attempt['ratio_to_first_depth'] <= max_ratio:
                results[depth].update(neighbors=neighbours, status=f'ok ({label})')
                break
            print(f'    too slow: {attempt["ratio_to_first_depth"]:.2f}x > {max_ratio}x')
    summary = {str(d): v for d, v in results.items()}
    summary['max_ratio'] = max_ratio
    summary['reference_seconds'] = base_epoch_seconds
    json.dump(summary, open(out_path, 'w'), indent=2)
    print('probe:', {d: v['status'] for d, v in results.items()}, '->', out_path)
    return summary


def neighbours_from_probe(probe, table=None):
    '''Add the depths that passed the probe to {depth: neighbour list} (depth 2 always keeps its primary list).'''
    table = dict(table) if table else {2: list(NEIGHBOURS_BY_DEPTH[2])}
    for depth, result in probe.items():
        if depth.isdigit() and result['neighbors'] is not None:
            table[int(depth)] = list(result['neighbors'])
    return table


# ==================================================================================================
# freeze and final retrain (train + val graph, frozen epoch and threshold, test once)
# ==================================================================================================

def save_recipe(path, operator, cfg, trial, retrain_seeds=(42, 43, 44, 45)):
    '''recipe.json: the frozen knobs, best epoch, per-seed thresholds and the validation F1 that chose them.

    One threshold for every retrain seed: the mean of the tuning seeds' validation thresholds at the best epoch
    (the per-seed tuning thresholds are kept in 'tuning_thresholds' for the record).
    '''
    tuned = {str(s): float(trial['thresholds'][str(s)]) for s in trial['seeds']}
    mean_threshold = float(np.mean(list(tuned.values())))
    thresholds = {str(s): mean_threshold for s in retrain_seeds}
    frozen = {'operator': operator, 'cfg': cfg, 'recipe': full_recipe(trial['recipe']),
              'best_epoch': int(trial['best_epoch']), 'tuning_seeds': list(trial['seeds']),
              'retrain_seeds': list(retrain_seeds), 'thresholds': thresholds, 'tuning_thresholds': tuned,
              'threshold_source': 'one threshold for every retrain seed: the mean of the tuning seeds\' '
                                  'validation thresholds at best_epoch',
              'val_f1_mean': trial['val_f1_mean'], 'val_f1_std': trial['val_f1_std'],
              'train_f1_mean': trial['train_f1_mean'], 'epochs_run': trial['epochs_run'],
              'source_stage': trial['stage'], 'source_trial': trial['trial']}
    json.dump(frozen, open(path, 'w'), indent=2)
    print(f'frozen recipe -> {path}\n{json.dumps(frozen, indent=1)}')
    return frozen


def load_recipe(path):
    '''Read recipe.json and check it is complete; a missing file stops with a clear message.'''
    assert os.path.exists(path), (f'{path} not found: run the tuning notebook to the end (save_recipe) and '
                                  f'commit Outputs/TUNING/<FAMILY>/recipe.json before the final retrain')
    frozen = json.load(open(path))
    frozen['recipe'] = full_recipe(frozen['recipe'])
    assert 1 <= frozen['best_epoch'] <= frozen['recipe']['epochs'], 'best_epoch must lie inside the cosine schedule'
    assert set(frozen['thresholds']) == {str(s) for s in frozen['retrain_seeds']}, 'one threshold per retrain seed'
    assert all(0 < t < 1 for t in frozen['thresholds'].values())
    print(f"frozen recipe {frozen['operator']} {frozen['cfg']}: {frozen['recipe']} | best epoch {frozen['best_epoch']} "
          f"| thresholds {frozen['thresholds']} | val F1 {frozen['val_f1_mean']:.4f} ± {frozen['val_f1_std']:.4f}")
    return frozen


def run_final(operator, cfg, graphs, col_sets, node_dim, out_root, frozen, seed, mp_direction='in', temporal=False,
              node_enc='none', enc_dir=None, gfp_variant='v0'):
    '''Final retrain of one seed: train+val graph, stop at the frozen epoch, frozen threshold, every split scored once.

    Writes <out_root>/<run>/seed-<s>/: results.json (+ phase, stop_epoch, frozen_threshold, val_seen, and the
    secondary view val_seen_threshold / seen_<split>_<f1|precision|recall>: the threshold re-swept on the val rows,
    which were trained on here, and the thresholded metrics at it — reported next to the frozen threshold, never
    the headline), history.csv, curves.png, sensitivity.csv, predictions.csv, final.pt. Skipped when results.json exists.
    '''
    recipe, stop_epoch = frozen['recipe'], int(frozen['best_epoch'])
    threshold = float(frozen['thresholds'][str(seed)])
    name = run_name(operator, cfg, mp_direction, temporal, node_enc, gfp_variant)
    out_dir = f'{out_root}/{name}/seed-{seed}'
    if os.path.exists(f'{out_dir}/results.json'):
        results = json.load(open(f'{out_dir}/results.json'))
        print(f'=== {name} seed {seed} === already finished (results.json found), skipping')
        return {k: v for k, v in results.items() if k not in ('architecture', 'recipe', 'threshold_sensitivity')}
    os.makedirs(out_dir, exist_ok=True)
    mp_idx, readout_idx = col_sets[cfg['mp']], col_sets[cfg['readout']]
    node_dim = set_node_encoding(graphs, node_enc, enc_dir)
    fit_graph = make_trainval_graph(graphs)

    set_seed(seed)
    deg_hist = in_degree_histogram(fit_graph) if operator == 'pna' else None   # the graph trained on
    model = build_model(operator, node_dim, len(mp_idx), len(readout_idx), mp_direction == 'bidirectional',
                        deg_hist, recipe)
    n_params, n_invariant = count_params(model), invariant_params(model)
    print_model_header(f'{name} seed {seed} (train+val retrain, stop at epoch {stop_epoch}, threshold {threshold:.3f})',
                       mp_idx, readout_idx, n_params, n_invariant, node_dim)
    fit_loader = make_loader(fit_graph, True, mp_idx, readout_idx, temporal, recipe)
    history = train_fixed_epochs(model, fit_loader, out_dir, recipe, stop_epoch)
    del fit_loader, fit_graph
    gc.collect()

    # scoring: the three usual loaders (val is training data here), the frozen threshold, test once
    loaders = make_loaders(graphs, mp_idx, readout_idx, temporal, recipe)
    metrics, preds = score_all_splits(model, loaders, graphs, threshold, seed)
    row = result_row(name, operator, cfg, mp_direction, temporal, stop_epoch, threshold,
                     n_params, n_invariant, metrics, node_enc, gfp_variant, seed)
    row.update(phase='retrain', trained_on='train+val', stop_epoch=stop_epoch, frozen_threshold=threshold,
               val_seen=True)
    sensitivity = threshold_sensitivity(preds['test']['y'], preds['test']['prob'], threshold)
    # secondary view, never the headline: the threshold re-swept on the val rows (seen in training here) and every
    # split scored at it; thresholded metrics only (PR-AUC, ROC-AUC and the top-5 % metrics do not move)
    seen_threshold, _ = best_f1_threshold(preds['val']['y'], preds['val']['prob'])
    seen = {split: compute_metrics(preds[split]['y'], preds[split]['prob'], seen_threshold) for split in SPLITS}
    row['val_seen_threshold'] = seen_threshold
    row.update({f'seen_{split}_{m}': seen[split][m] for split in SPLITS for m in ('f1', 'precision', 'recall')})
    sensitivity.append({'point': 'val seen sweep', 'threshold': seen_threshold, 'delta': float(seen_threshold - threshold),
                        'f1': seen['test']['f1'], 'precision': seen['test']['precision'], 'recall': seen['test']['recall']})
    print(f'threshold re-swept on the seen val rows: {seen_threshold:.3f} (frozen {threshold:.3f}) -> '
          f'test F1 {seen["test"]["f1"]:.4f} vs {metrics["test"]["f1"]:.4f} at the frozen threshold')
    save_run(out_dir, row, recipe, extra={'threshold_sensitivity': sensitivity,
                                          'threshold_source': frozen['threshold_source']})
    pd.DataFrame(sensitivity).to_csv(f'{out_dir}/sensitivity.csv', index=False)
    print(pd.DataFrame(sensitivity).round(4))
    save_predictions(out_dir, graphs, preds, threshold)
    plot_final_curves(history, preds['test']['y'], preds['test']['prob'], metrics['test']['pr_auc'],
                      f'{name} seed {seed}', f'{out_dir}/curves.png')

    del model, loaders
    gc.collect()
    torch.cuda.empty_cache()
    return row
