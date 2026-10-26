'''Shared code for the fixed-architecture GNN experiments (GIN / PNA / GATv2 / Transformer).

Everything that must be identical across operators lives here and is imported by the operator notebooks,
which only set OPERATOR, the batch of knob configs and paths. Knobs: mp / readout edge columns, NODE_ENC
(none | rwpe8 | rwpe16: a pre-computed node encoding appended to the 6 entity columns) and GFP_VARIANT
(v0 | tuned: which GFP sheet fills the GFP block of edge_attr, swapped at load time).

The training recipe is a dict of knobs (hidden, layers, dropout, norm, ...); recipe=None is the v1 recipe
below and reproduces every v1 run exactly. The file ends with the recipe-tuning code: the one-pass tuning
(one knob at a time: run_tuning_trial, choose_winner, probe_depths), the frozen recipe.json (save_recipe /
load_recipe), the train + val retrain (run_final) and the Luo et al. (2024) random search that cineca/ runs
(draw_luo_trials, probe_memory, finalise_trial).
'''
import gc
import glob
import itertools
import json
import os
import random
import time
from pathlib import Path

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

# ---------------- v1 recipe (CLAUDE.md section 3): the same for every run; a tuned run passes a recipe dict ----------------
HIDDEN        = 128
NUM_LAYERS    = 2
DROPOUT       = 0.3
HEADS         = 4              # GATv2 / Transformer: 4 heads x 32 = 128
NUM_NEIGHBORS = [100, 100]     # neighbours sampled per hop, one entry per layer
BATCH_SIZE    = 8192           # seed edges per mini-batch
EPOCHS        = 20
LR            = 1e-3
WEIGHT_DECAY  = 1e-5           # Adam L2 term, part of the v1 recipe
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
TRIAL_METRIC_KEYS = [f'{split}_{m}' for split in ('train', 'val') for m in METRICS + ('loss',)]   # trial.json mean / std
# compact view of batch_summary.csv for the notebook comparison table
DISPLAY_COLS = ['run', 'best_epoch', 'threshold', 'val_f1', 'test_f1', 'test_precision', 'test_recall',
                'test_pr_auc', 'test_precision_at_5pct', 'test_recall_at_5pct', 'params', 'invariant_params']

# ---------------- recipe tuning (EXPERIMENTS.md section 2b) ----------------
NORMS = ('none', 'batch', 'layer')            # normalisation after each message-passing layer
RESIDUALS = ('identity', 'linear', 'none')    # identity = v1 (h + layer(h)); linear = Luo et al. 2024 Eq. 7
LR_SCHEDULES = ('cosine', 'constant')
SELECT_METRICS = ('f1', 'pr_auc')             # validation metric that picks the best epoch (and the winning trial)
# neighbours per hop by depth: hops 1-2 are always 100, hops 3-4 50, hops 5+ 25 (fallback: 25 and 10)
NEIGHBOURS_BY_DEPTH = {1: [100], 2: [100, 100], 3: [100, 100, 50], 4: [100, 100, 50, 50],
                       5: [100, 100, 50, 50, 25], 6: [100, 100, 50, 50, 25, 25],
                       7: [100, 100, 50, 50, 25, 25, 25], 8: [100, 100, 50, 50, 25, 25, 25, 25],
                       9: [100, 100, 50, 50, 25, 25, 25, 25, 25], 10: [100, 100, 50, 50, 25, 25, 25, 25, 25, 25]}
NEIGHBOURS_FALLBACK = {3: [100, 100, 25], 4: [100, 100, 25, 25], 5: [100, 100, 25, 25, 10],
                       6: [100, 100, 25, 25, 10, 10], 7: [100, 100, 25, 25, 10, 10, 10],
                       8: [100, 100, 25, 25, 10, 10, 10, 10], 9: [100, 100, 25, 25, 10, 10, 10, 10, 10],
                       10: [100, 100, 25, 25, 10, 10, 10, 10, 10, 10]}
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
    '''The v1 recipe, read from the module constants at call time. self_loops is GATv2's add_self_loops (the other
    operators ignore it); the knobs after it (residual ... select_metric) were added for the Luo tuning with v1 defaults.'''
    return dict(hidden=HIDDEN, num_layers=NUM_LAYERS, dropout=DROPOUT, norm='none', ffn=False,
                num_neighbors=list(NUM_NEIGHBORS), batch_size=BATCH_SIZE, epochs=EPOCHS,
                lr=LR, weight_decay=WEIGHT_DECAY, pos_weight=POS_WEIGHT, self_loops=True,
                residual='identity', readout_layers=2, readout_width=1.0, lr_schedule='cosine',
                num_workers=0, select_metric='f1')


def full_recipe(recipe=None):
    '''The v1 recipe overlaid with the given knobs; unknown keys or a depth / neighbour mismatch are errors.'''
    out = default_recipe()
    extra = set(recipe or {}) - set(out)
    assert not extra, f'unknown recipe keys {sorted(extra)}; knobs are {sorted(out)}'
    out.update(recipe or {})
    out['num_neighbors'] = [int(n) for n in out['num_neighbors']]
    assert out['norm'] in NORMS, f"norm must be one of {NORMS}, got {out['norm']!r}"
    assert out['hidden'] % HEADS == 0, 'hidden must split into the attention heads'
    assert len(out['num_neighbors']) == out['num_layers'], 'one neighbour count per message-passing layer'
    assert out['num_layers'] >= 1 and out['epochs'] >= 1
    assert isinstance(out['self_loops'], bool), 'self_loops is a flag'
    assert out['residual'] in RESIDUALS, f"residual must be one of {RESIDUALS}, got {out['residual']!r}"
    assert out['readout_layers'] in (1, 2, 3) and out['readout_width'] > 0
    assert out['lr_schedule'] in LR_SCHEDULES, f"lr_schedule must be one of {LR_SCHEDULES}"
    assert int(out['num_workers']) == out['num_workers'] >= 0
    assert out['select_metric'] in SELECT_METRICS, f"select_metric must be one of {SELECT_METRICS}"
    return out


def recipe_tag(recipe, base=None):
    '''Trial folder name = the knobs that differ from `base` (default: the v1 recipe), e.g. d0.2_bn_l4; 'base' if none.'''
    full, ref = full_recipe(recipe), full_recipe(base)
    short = {'dropout': 'd', 'hidden': 'h', 'pos_weight': 'pw', 'lr': 'lr', 'weight_decay': 'wd',
             'batch_size': 'bs', 'epochs': 'ep'}
    parts = []
    for knob in ('dropout', 'norm', 'num_layers', 'num_neighbors', 'hidden', 'pos_weight', 'lr', 'ffn',
                 'self_loops', 'weight_decay', 'batch_size', 'epochs'):
        if full[knob] == ref[knob]:
            continue
        if knob == 'norm':
            parts.append({'batch': 'bn', 'layer': 'ln', 'none': 'nonorm'}[full['norm']])
        elif knob == 'ffn':
            parts.append('ffn' if full['ffn'] else 'noffn')
        elif knob == 'self_loops':
            parts.append('sl' if full['self_loops'] else 'nosl')
        elif knob == 'num_layers':
            parts.append(f"l{full['num_layers']}")
        elif knob == 'num_neighbors':
            # implied by the depth unless the depth is unchanged (a fallback neighbour list)
            if full['num_layers'] == ref['num_layers']:
                parts.append('nb' + '-'.join(str(n) for n in full['num_neighbors']))
        else:
            parts.append(f'{short[knob]}{full[knob]:g}')
    # num_workers and select_metric are run settings, not part of the tag
    if full['residual'] != ref['residual']:
        parts.append({'identity': 'resid', 'linear': 'reslin', 'none': 'nores'}[full['residual']])
    if full['readout_layers'] != ref['readout_layers']:
        parts.append(f"ro{full['readout_layers']}")
    if full['readout_width'] != ref['readout_width'] and full['readout_layers'] > 1:
        parts.append(f"rw{full['readout_width']:g}")
    if full['lr_schedule'] != ref['lr_schedule']:
        parts.append({'cosine': 'lrcos', 'constant': 'lrconst'}[full['lr_schedule']])
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

    Pattern bins (*_bins_*) untouched; fan / deg / sum / avg / var / kurtosis: log1p -> clip at train p1 / p99
    -> StandardScaler; ratio / skew: clip -> StandardScaler. values [E, len(cols)] float32 -> float32.
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


def load_graphs(data_dir, gfp_variant='v0', gfp_dir=None, splits=SPLITS):
    '''Load the three cumulative snapshots and the feature column groups for the GFP_VARIANT knob.

    v0 -> edge_attr as saved (20 baseline + 61 GFP); tuned -> the GFP block is replaced by
    <gfp_dir>/tuned.npy (64 columns), normalised like v0 on the train rows. Returns graphs
    {split: Data}, col_sets {none|base|gfp|full: column indices}, node_dim.
    splits=('train', 'val') is what tuning loads: the test snapshot is never read there.
    '''
    assert gfp_variant in GFP_VARIANTS, f'unknown GFP variant {gfp_variant!r}; choose from {GFP_VARIANTS}'
    assert gfp_variant == 'v0' or tuple(splits) == SPLITS, 'the tuned GFP sheet is aligned with all three snapshots'
    assert 'train' in splits, 'the train snapshot is always needed'
    graphs = {split: torch.load(f'{data_dir}/{split}_graph.pt', weights_only=False) for split in splits}
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


def make_conv(operator, hidden, mp_edge_dim, deg_hist=None, self_loops=True):
    '''One graph convolution for the operator; the only place operators differ.

    mp_edge_dim = 0 -> edge features stay out of message passing. self_loops is GATv2's add_self_loops
    (v1 = True); the other operators ignore it. PyG defaults are written out where they matter.
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
        # HEADS heads concatenated -> hidden; a self-loop gets the mean of the node's incoming edge features
        if mp_edge_dim > 0:
            return GATv2Conv(hidden, hidden // HEADS, heads=HEADS, edge_dim=mp_edge_dim,
                             add_self_loops=self_loops, fill_value='mean', bias=True)
        return GATv2Conv(hidden, hidden // HEADS, heads=HEADS,
                         add_self_loops=self_loops, fill_value='mean', bias=True)
    if operator == 'transformer':
        # HEADS heads concatenated -> hidden; root_weight = the node's own state through lin_skip
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

    conv_in aggregates the accounts that paid a node (the module name is fixed by the saved checkpoints).
    norm='none' (v1) is the identity. ffn adds the GNN+ block h <- Norm(h + W2(Dropout(ReLU(W1 h)))) after the
    residual; its Norm is the layer's norm choice, or LayerNorm when that is 'none'. residual: 'identity' (v1, the
    formula above), 'linear' (Luo et al. 2024 Eq. 7: h <- Dropout(ReLU(Norm(conv(h) + W_r h)))) or 'none' (no skip).
    Only ffn and the linear residual add parameters, so v1 checkpoints load unchanged.
    '''

    def __init__(self, operator, hidden, mp_edge_dim, dropout, deg_hist=None, norm='none', ffn=False,
                 self_loops=True, residual='identity'):
        super().__init__()
        self.use_edges = mp_edge_dim > 0
        self.conv_in = make_conv(operator, hidden, mp_edge_dim, deg_hist, self_loops)
        self.dropout = nn.Dropout(dropout)
        self.norm = make_norm(norm, hidden)
        self.ffn = None
        if ffn:
            self.ffn = nn.Sequential(nn.Linear(hidden, 2 * hidden), nn.ReLU(), nn.Dropout(dropout),
                                     nn.Linear(2 * hidden, hidden))
            self.ffn_norm = make_norm(norm if norm != 'none' else 'layer', hidden)
        self.residual = residual
        if residual == 'linear':
            self.residual_proj = nn.Linear(hidden, hidden)   # registered last, so v1 layers initialise unchanged

    def forward(self, h, edge_index, edge_attr):
        out = self.conv_in(h, edge_index, edge_attr) if self.use_edges else self.conv_in(h, edge_index)
        if self.residual == 'linear':
            out = out + self.residual_proj(h)                 # the paper's residual enters before the norm
        out = self.dropout(torch.relu(self.norm(out)))
        h = h + out if self.residual == 'identity' else out
        if self.ffn is not None:
            h = self.ffn_norm(h + self.ffn(h))
        return h


class EdgeClassifier(nn.Module):
    '''Fixed-architecture edge classifier: node_proj -> num_layers message-passing layers -> readout MLP.

    x [N, node_dim] -> node_proj Linear -> hidden, ReLU -> MPLayers -> readout on the seed edge:
    concat [h_src || h_dst || e_seed] -> (readout_layers - 1) x (Linear -> ReLU -> Dropout) -> Linear 1 (logit).
    Between feature configs only the conv's edge projection and the readout's first Linear change width.
    Module names (node_proj, layers.i.conv_in, classifier) are fixed by the saved checkpoints. recipe=None -> v1.
    bidirectional (message passing over reversed edges too) is not implemented; the parameter stays because the
    notebooks pass it (always False).
    '''

    def __init__(self, operator, node_dim, mp_edge_dim, readout_edge_dim, bidirectional=False, deg_hist=None, recipe=None):
        super().__init__()
        assert not bidirectional, 'only message passing over incoming edges is implemented'
        r = full_recipe(recipe)
        hidden = r['hidden']
        self.node_proj = nn.Linear(node_dim, hidden)
        self.layers = nn.ModuleList(
            [MPLayer(operator, hidden, mp_edge_dim, r['dropout'], deg_hist, r['norm'], r['ffn'], r['self_loops'], r['residual'])
             for _ in range(r['num_layers'])])
        width = max(1, int(round(r['readout_width'] * hidden)))   # neurons per readout hidden layer
        in_dim, readout = 2 * hidden + readout_edge_dim, []
        for _ in range(r['readout_layers'] - 1):
            readout += [nn.Linear(in_dim, width), nn.ReLU(), nn.Dropout(r['dropout'])]
            in_dim = width
        self.classifier = nn.Sequential(*readout, nn.Linear(in_dim, 1))

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
    '''Parameters whose shape does not depend on the feature widths; identical across the feature configs of a family.

    Left out: the edge projection inside the conv (.lin. / lin_edge / edge_encoder, and PNA's message MLP input
    pre_nns.0.0., which widens when edge features enter) and the readout's first Linear (classifier.0.).
    node_proj.weight is counted at the base node width 6, so a node encoding widens it without changing the count.
    v1 values: GIN 67,587 · GATv2 67,585 · PNA 427,265 · Transformer 133,121; a tuned recipe has its own constant.
    '''
    width_dependent = ('.lin.', 'lin_edge', 'edge_encoder', 'pre_nns.0.0.', 'classifier.0.')
    total = 0
    for name, p in model.named_parameters():
        if name == 'node_proj.weight':
            total += model.node_proj.out_features * BASE_NODE_DIM
        elif not any(tag in name for tag in width_dependent):
            total += p.numel()
    return total


# ==================================================================================================
# loaders and prediction
# ==================================================================================================

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
        kwargs.update(num_workers=int(r['num_workers']), persistent_workers=False)
    return LinkNeighborLoader(mp_graph, **kwargs), seed_feats


def make_loaders(graphs, mp_idx, readout_idx, recipe=None, splits=SPLITS):
    '''One (loader, seed_feats) pair per split; train is shuffled, val / test are not. Tuning passes
    splits=('train', 'val'), so no test loader is built there.'''
    return {split: make_loader(graphs[split], split == 'train', mp_idx, readout_idx, recipe) for split in splits}


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


def make_scheduler(optimizer, recipe):
    '''cosine (v1): CosineAnnealingLR over recipe['epochs']; constant: the LR never moves (Luo et al. 2024, tunedGNN).'''
    r = full_recipe(recipe)
    if r['lr_schedule'] == 'constant':
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 1.0)
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=r['epochs'])


def write_json_atomic(path, obj):
    '''json.dump through a temporary file + os.replace, so a killed job never leaves a half-written file.'''
    tmp = f'{path}.tmp{os.getpid()}'
    with open(tmp, 'w') as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def rng_state():
    '''Every random-number state a run consumes: python, numpy, torch CPU (neighbour sampling, shuffling), CUDA.'''
    return {'python': random.getstate(), 'numpy': np.random.get_state(), 'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def set_rng_state(state):
    '''Inverse of rng_state(); the torch states must be CPU ByteTensors.'''
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if state['cuda'] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state['cuda'])


def save_resume(path, model, optimizer, scheduler, loop_state):
    '''resume.pt: model, optimizer, scheduler, RNG and the loop state, written atomically at the end of an epoch.'''
    state = dict(loop_state, model=model.state_dict(), optimizer=optimizer.state_dict(),
                 scheduler=scheduler.state_dict(), rng=rng_state())
    tmp = f'{path}.tmp{os.getpid()}'
    torch.save(state, tmp)
    os.replace(tmp, path)


def load_resume(path, model, optimizer, scheduler):
    '''Restore model, optimizer, scheduler and RNG from resume.pt; returns the loop state (history, best, ...).'''
    state = torch.load(path, map_location='cpu', weights_only=False)   # our own file; RNG states must stay on the CPU
    model.load_state_dict(state['model'])
    optimizer.load_state_dict(state['optimizer'])
    scheduler.load_state_dict(state['scheduler'])
    set_rng_state(state['rng'])
    return state


def add_full_metrics(row, split, y, p, threshold):
    '''Every section-4 metric of one split at the epoch's validation threshold, added to a history row
    (columns the row already has, e.g. the sweep's val_f1, are kept).'''
    for name, value in compute_metrics(y, p, threshold).items():
        row.setdefault(f'{split}_{name}', value)


def train_one_run(model, loaders, out_dir, recipe=None, stop_epoch=None, eval_train_every=1,
                  patience=None, min_epochs=20, keep_epochs=False, resume_path=None, full_metrics=False):
    '''The training loop: per-epoch validation threshold sweep, best-validation checkpoint (best.pt), history.csv.

    recipe=None -> v1. recipe['epochs'] is the schedule length; the loop runs to stop_epoch (default: every
    epoch) or stops early when `patience` is set and the 3-epoch moving average of the validation metric has not
    improved for that many epochs (never before min_epochs). recipe['select_metric'] (val F1 = v1, or val PR-AUC)
    picks the best epoch; the returned threshold is always that epoch's validation-F1 sweep. The train split is
    scored every eval_train_every epochs (+ the last). keep_epochs saves epoch_<e>.pt every epoch; resume_path
    writes resume.pt after every epoch and continues from it when it exists; full_metrics adds every section-4
    metric of train and val to each row. Non-finite predictions (a diverged run) stop the loop.
    Returns (history rows, best_epoch, best_threshold).
    '''
    r = full_recipe(recipe)
    metric = r['select_metric']
    metric_name = {'f1': 'F1', 'pr_auc': 'PR-AUC'}[metric]
    train_loader, train_feats = loaders['train']
    val_loader,   val_feats   = loaders['val']
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([r['pos_weight']], device=device))
    optimizer = torch.optim.Adam(model.parameters(), lr=r['lr'], weight_decay=r['weight_decay'])
    scheduler = make_scheduler(optimizer, r)
    last_epoch = r['epochs'] if stop_epoch is None else int(stop_epoch)
    assert 1 <= last_epoch <= r['epochs'], f'stop_epoch {last_epoch} must lie inside the schedule (1..{r["epochs"]})'

    history = []
    best_score, best_epoch, best_threshold = -1.0, 0, 0.5
    best_smoothed, epochs_without_gain, stopped = -1.0, 0, False
    if resume_path is not None and os.path.exists(resume_path):
        state = load_resume(resume_path, model, optimizer, scheduler)
        history, stopped = state['history'], state['stopped']
        best_score, best_epoch, best_threshold = state['best']
        best_smoothed, epochs_without_gain = state['patience']
        print(f'resumed from {resume_path}: {len(history)} epochs done' + (' (the run had stopped)' if stopped else ''))
    for epoch in range(len(history) + 1, last_epoch + 1):
        if stopped:
            break
        t0 = time.time()
        lr_now = optimizer.param_groups[0]['lr']
        loss_batches = train_epoch(model, train_loader, train_feats, optimizer, criterion)

        # train and val scored in eval mode; train F1 reuses the val threshold (overfitting diagnostics)
        score_train = (epoch % eval_train_every == 0) or epoch == last_epoch
        if score_train:
            y_train, p_train, _ = predict(model, train_loader, train_feats, 'train')
        y_val, p_val, _ = predict(model, val_loader, val_feats, 'val')
        if not (np.isfinite(p_val).all() and (not score_train or np.isfinite(p_train).all())):
            print(f'non-finite predictions at epoch {epoch}: the run diverged, training stops here')
            stopped = True
        else:
            threshold, val_f1 = best_f1_threshold(y_val, p_val)
            train_f1 = f1_score(y_train, (p_train >= threshold).astype(int)) if score_train else float('nan')
            val_pr_auc = average_precision_score(y_val, p_val)

            score = val_f1 if metric == 'f1' else val_pr_auc
            is_best = score > best_score
            if is_best:
                best_score, best_epoch, best_threshold = score, epoch, threshold
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
                'val_pr_auc':   val_pr_auc,
                'val_roc_auc':  roc_auc_score(y_val, p_val),
                'threshold':    threshold,
                'seconds':      time.time() - t0,
                'train_loss_batches': loss_batches,
                'lr':           lr_now,
            })
            row = history[-1]
            if full_metrics:
                add_full_metrics(row, 'val', y_val, p_val, threshold)
                if score_train:
                    add_full_metrics(row, 'train', y_train, p_train, threshold)
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
                # early stopping on the smoothed validation curve of the selection metric, never before min_epochs
                current = smoothed([h[f'val_{metric}'] for h in history])[-1]
                if current > best_smoothed:
                    best_smoothed, epochs_without_gain = current, 0
                else:
                    epochs_without_gain += 1
                if epoch >= min_epochs and epochs_without_gain >= patience:
                    print(f'early stop at epoch {epoch}: smoothed val {metric_name} has not improved for {patience} epochs')
                    stopped = True
        if resume_path is not None:
            save_resume(resume_path, model, optimizer, scheduler,
                        dict(history=history, stopped=stopped, best=(best_score, best_epoch, best_threshold),
                             patience=(best_smoothed, epochs_without_gain)))
        if stopped:
            break

    if history:
        pd.DataFrame(history).to_csv(f'{out_dir}/history.csv', index=False)
    print(f'best validation {metric_name} {best_score:.4f} at epoch {best_epoch} (threshold {best_threshold:.3f}) '
          f'| {len(history)} of {r["epochs"]} epochs run')
    return history, best_epoch, best_threshold


def train_fixed_epochs(model, fit_loader, out_dir, recipe, stop_epoch, resume_path=None, eval_loaders=None):
    '''Final retrain: stop_epoch epochs on the train + val loader with the recipe's LR schedule; no model selection.

    Writes final.pt and history.csv (epoch, train_loss_batches, lr, seconds); returns the history rows.
    eval_loaders {split: (loader, feats)} adds eval-mode <split>_loss and <split>_pr_auc per epoch, for the curves
    only (val is training data here). resume_path: resume.pt after every epoch, continued from when it exists.
    '''
    r = full_recipe(recipe)
    assert 1 <= stop_epoch <= r['epochs'], f'stop_epoch {stop_epoch} must lie inside the schedule (1..{r["epochs"]})'
    loader, feats = fit_loader
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([r['pos_weight']], device=device))
    optimizer = torch.optim.Adam(model.parameters(), lr=r['lr'], weight_decay=r['weight_decay'])
    scheduler = make_scheduler(optimizer, r)
    history = []
    if resume_path is not None and os.path.exists(resume_path):
        history = load_resume(resume_path, model, optimizer, scheduler)['history']
        print(f'resumed from {resume_path}: {len(history)} of {stop_epoch} epochs done')
    for epoch in range(len(history) + 1, stop_epoch + 1):
        t0 = time.time()
        lr_now = optimizer.param_groups[0]['lr']
        loss = train_epoch(model, loader, feats, optimizer, criterion)
        assert np.isfinite(loss), f'non-finite training loss at epoch {epoch}: the retrain diverged'
        row = {'epoch': epoch, 'train_loss_batches': loss, 'lr': lr_now}
        for split, (eval_loader, eval_feats) in (eval_loaders or {}).items():
            y, p, _ = predict(model, eval_loader, eval_feats, split)
            assert np.isfinite(p).all(), f'non-finite {split} predictions at epoch {epoch}: the retrain diverged'
            row[f'{split}_loss'] = weighted_bce(y, p, r['pos_weight'])
            row[f'{split}_pr_auc'] = average_precision_score(y, p)
        row['seconds'] = time.time() - t0
        history.append(row)
        ram_gb = psutil.Process().memory_info().rss / 2**30
        evals = ''.join(f" | {s} loss {row[f'{s}_loss']:.4f} PR-AUC {row[f'{s}_pr_auc']:.4f}" for s in (eval_loaders or {}))
        print(f'epoch {epoch:02d}/{stop_epoch} | batch loss {loss:.4f} | lr {lr_now:.2e}{evals} | '
              f'{row["seconds"]:.0f}s | RAM {ram_gb:.1f}GB')
        scheduler.step()
        if resume_path is not None:
            save_resume(resume_path, model, optimizer, scheduler, dict(history=history))
    torch.save(model.state_dict(), f'{out_dir}/final.pt')
    pd.DataFrame(history).to_csv(f'{out_dir}/history.csv', index=False)
    return history


# ==================================================================================================
# plots, saving, naming
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
    ax[2].axhline(y_test.mean(), color='gray', linestyle='--', label='random')
    ax[2].set_title('test precision-recall')
    ax[2].set_xlabel('recall')
    ax[2].set_ylabel('precision')
    ax[2].legend()
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.show()
    plt.close(fig)


def run_name(operator, cfg, mp_direction='in', temporal=False, node_enc='none', gfp_variant='v0'):
    '''<model>_mp-<mp>_readout-<ro>_dir-in[_enc-<enc>][_gfp-<variant>] (CLAUDE.md section 6); the optional parts
    appear only when the knob is not at its default.

    Message passing is always over incoming edges without temporal sampling; mp_direction and temporal stay in the
    signature because the notebooks pass them, and any other value is an error.
    '''
    assert mp_direction == 'in' and not temporal, 'only mp_direction="in" without temporal sampling is implemented'
    name = f"{operator}_mp-{cfg['mp']}_readout-{cfg['readout']}_dir-in"
    if node_enc != 'none':
        name += f'_enc-{node_enc}'
    if gfp_variant != 'v0':
        name += f'_gfp-{gfp_variant}'
    return name


def result_row(name, operator, cfg, best_epoch, threshold, n_params, n_invariant, metrics,
               node_enc='none', gfp_variant='v0', seed=None):
    '''The flat result row shared by results.json and batch_summary.csv (mp_direction / temporal_sampling are
    constant columns, kept so old and new result files have the same layout).'''
    row = {'run': name, 'operator': operator, 'mp': cfg['mp'], 'readout': cfg['readout'],
           'mp_direction': 'in', 'temporal_sampling': False, 'node_enc': node_enc,
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
                   node_enc='none', enc_dir=None, gfp_variant='v0', recipe=None, eval_train_every=1):
    '''Train and evaluate one knob configuration end-to-end (seed 42); returns the flat result row.

    node_enc = 'none' leaves graph.x and node_dim as loaded; 'rwpe<k>' appends the k RWPE columns from enc_dir,
    so only node_proj widens (6 + k -> 128). gfp_variant must be the one the graphs were loaded with
    (load_graphs); it only labels the run, the swap happened at load time. recipe=None reproduces the v1 runs.
    '''
    assert len(col_sets['gfp']) == GFP_WIDTH[gfp_variant], \
        f"graphs hold a {len(col_sets['gfp'])}-column GFP block, not the {gfp_variant!r} sheet; reload with load_graphs"
    name = run_name(operator, cfg, mp_direction, temporal, node_enc, gfp_variant)
    out_dir = f'{out_root}/{name}'
    os.makedirs(out_dir, exist_ok=True)
    mp_idx, readout_idx = col_sets[cfg['mp']], col_sets[cfg['readout']]
    node_dim = set_node_encoding(graphs, node_enc, enc_dir)   # 6 for 'none', 6 + k for 'rwpe<k>'

    set_seed(SEED)
    deg_hist = in_degree_histogram(graphs['train']) if operator == 'pna' else None
    model = build_model(operator, node_dim, len(mp_idx), len(readout_idx), deg_hist=deg_hist, recipe=recipe)
    n_params, n_invariant = count_params(model), invariant_params(model)
    print_model_header(name, mp_idx, readout_idx, n_params, n_invariant, node_dim)

    loaders = make_loaders(graphs, mp_idx, readout_idx, recipe)
    history, best_epoch, threshold = train_one_run(model, loaders, out_dir, recipe, eval_train_every=eval_train_every)

    # final evaluation: best checkpoint, one validation-chosen threshold, test scored once
    model.load_state_dict(torch.load(f'{out_dir}/best.pt', map_location=device))
    metrics, preds = score_all_splits(model, loaders, graphs, threshold)
    row = result_row(name, operator, cfg, best_epoch, threshold, n_params, n_invariant, metrics, node_enc, gfp_variant)
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
    extra = [c for c in summary.columns
             if c.startswith(('seen_', 'transfer_')) or c == 'val_seen_threshold']   # final retrain only
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

def run_tuning_seed(operator, cfg, graphs, col_sets, node_dim, trial_dir, recipe, seed, eval_train_every=5,
                    patience=10, min_epochs=20, resume=False, full_metrics=False):
    '''One (trial, seed): train on the train graph, validate every epoch, keep a checkpoint per epoch.

    No test loader, no predictions. Writes seed-<s>/history.csv, curves.png, best.pt, epoch_<e>.pt.
    resume=True continues from seed-<s>/resume.pt; full_metrics = every section-4 metric per epoch.
    Returns params, invariant params, epochs run and seconds per epoch.
    '''
    out_dir = f'{trial_dir}/seed-{seed}'
    os.makedirs(out_dir, exist_ok=True)
    mp_idx, readout_idx = col_sets[cfg['mp']], col_sets[cfg['readout']]
    set_seed(seed)
    deg_hist = in_degree_histogram(graphs['train']) if operator == 'pna' else None
    model = build_model(operator, node_dim, len(mp_idx), len(readout_idx), deg_hist=deg_hist, recipe=recipe)
    n_params, n_invariant = count_params(model), invariant_params(model)
    name = f'{os.path.basename(trial_dir)} seed {seed}'
    print_model_header(name, mp_idx, readout_idx, n_params, n_invariant, node_dim)
    loaders = make_loaders(graphs, mp_idx, readout_idx, recipe, splits=('train', 'val'))
    history, best_epoch, _ = train_one_run(model, loaders, out_dir, recipe, eval_train_every=eval_train_every,
                                           patience=patience, min_epochs=min_epochs, keep_epochs=True,
                                           resume_path=f'{out_dir}/resume.pt' if resume else None,
                                           full_metrics=full_metrics)
    if history:
        plot_curves(history, None, None, None, best_epoch, name, f'{out_dir}/curves.png',
                    full_recipe(recipe)['select_metric'])
    del model, loaders
    gc.collect()
    torch.cuda.empty_cache()
    return {'params': n_params, 'invariant_params': n_invariant, 'epochs_run': len(history),
            'seconds_per_epoch': float(np.mean([h['seconds'] for h in history])) if history else float('nan')}


def seed_mean_best_epoch(histories, window=3, column='val_f1'):
    '''Best epoch of a trial = argmax of the smoothed seed-mean `column` over the epochs every seed reached.

    histories: {seed: history DataFrame}; window=1 is the raw curve (Luo tuning: column val_pr_auc).
    Returns (best_epoch, mean curve, smoothed curve); np.argmax keeps the first epoch on ties.
    '''
    n_common = min(len(h) for h in histories.values())
    values = np.stack([h[column].to_numpy()[:n_common] for h in histories.values()])
    mean = values.mean(axis=0)
    smooth = smoothed(mean, window)
    return int(np.argmax(smooth)) + 1, mean, smooth


def score_trial(operator, cfg, graphs, col_sets, node_dim, trial_dir, recipe, seeds, stage, seed_info,
                window=3, from_history=False, extra=None):
    '''After the last seed: score each seed's checkpoint at the seed-mean best epoch on train and val, write trial.json.

    The best epoch comes from the validation curve of recipe['select_metric'] smoothed over `window` epochs; each
    seed uses its own validation threshold of that epoch. The per-epoch checkpoints are deleted, then trial.json
    is written: it marks the trial as finished (a rerun skips it). from_history=True reads the per-seed metrics
    from each seed's history.csv at that epoch instead of re-scoring the checkpoints (graphs / col_sets / node_dim
    may be None then). `extra` is merged into trial.json.
    '''
    r = full_recipe(recipe)
    metric = r['select_metric']
    tag = os.path.basename(trial_dir)
    histories = {s: pd.read_csv(f'{trial_dir}/seed-{s}/history.csv') for s in seeds}
    best_epoch, mean_curve, smooth_curve = seed_mean_best_epoch(histories, window, f'val_{metric}')
    if not from_history:
        mp_idx, readout_idx = col_sets[cfg['mp']], col_sets[cfg['readout']]
        deg_hist = in_degree_histogram(graphs['train']) if operator == 'pna' else None

    per_seed = {}
    for s in seeds:
        h = histories[s]
        threshold = float(h.loc[h['epoch'] == best_epoch, 'threshold'].iloc[0])
        if from_history:
            row = h.loc[h['epoch'] == best_epoch].iloc[0]
            metrics = {split: {m: float(row[f'{split}_{m}']) for m in METRICS + ('loss',)} for split in ('train', 'val')}
        else:
            set_seed(s)
            model = build_model(operator, node_dim, len(mp_idx), len(readout_idx), deg_hist=deg_hist, recipe=recipe)
            model.load_state_dict(torch.load(f'{trial_dir}/seed-{s}/epoch_{best_epoch:03d}.pt', map_location=device))
            loaders = make_loaders(graphs, mp_idx, readout_idx, recipe, splits=('train', 'val'))
            metrics = {}
            for split in ('train', 'val'):
                loader, feats = loaders[split]
                y, p, _ = predict(model, loader, feats, f'{split} seed {s} @ epoch {best_epoch}')
                metrics[split] = compute_metrics(y, p, threshold)
                metrics[split]['loss'] = weighted_bce(y, p, r['pos_weight'])
            del model, loaders
            gc.collect()
            torch.cuda.empty_cache()
        per_seed[str(s)] = {'threshold': threshold, 'epochs_run': int(len(h)),
                            'reached_max': bool(len(h) >= r['epochs']),
                            'best_epoch_own': int(h['epoch'].iloc[int(np.argmax(h[f'val_{metric}'].to_numpy()))]),
                            **{f'{split}_{m}': v for split in ('train', 'val') for m, v in metrics[split].items()}}

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
             f'val_{metric}_mean_curve': [float(v) for v in mean_curve],
             f'val_{metric}_smoothed_curve': [float(v) for v in smooth_curve],
             'per_seed': per_seed, 'gap_note': VAL_PREVALENCE_NOTE}
    for key in TRIAL_METRIC_KEYS:
        trial[f'{key}_mean'], trial[f'{key}_std'] = agg(key)
    trial['gap_f1'] = trial['train_f1_mean'] - trial['val_f1_mean']
    trial['gap_pr_auc'] = trial['train_pr_auc_mean'] - trial['val_pr_auc_mean']
    trial.update(extra or {})

    for s in seeds:
        for path in glob.glob(f'{trial_dir}/seed-{s}/epoch_*.pt'):
            os.remove(path)
    write_json_atomic(f'{trial_dir}/trial.json', trial)
    print(f"trial {tag}: best epoch {best_epoch} (seed-mean val {metric}, window {window}) | val F1 "
          f"{trial['val_f1_mean']:.4f} ± {trial['val_f1_std']:.4f} | val PR-AUC {trial['val_pr_auc_mean']:.4f} | "
          f"train F1 {trial['train_f1_mean']:.4f} | gap {trial['gap_f1']:+.4f} | epochs run {trial['epochs_run']} "
          f"| checkpoints deleted")
    return trial


def run_tuning_trial(operator, cfg, graphs, col_sets, node_dim, stage_dir, recipe, seeds, base_recipe=None,
                     mp_direction='in', temporal=False, eval_train_every=5, patience=10, min_epochs=20, reuse=None):
    '''One trial = every seed back to back, then the post-trial scoring; skipped when trial.json exists.

    reuse = the incumbent's trial dict: when the recipe is identical, its scoring is copied into this stage
    (trial.json with `reused_from`) instead of training the same recipe again. mp_direction / temporal are fixed
    ('in' / False, see run_name).
    '''
    assert mp_direction == 'in' and not temporal, 'only mp_direction="in" without temporal sampling is implemented'
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
                                    eval_train_every, patience, min_epochs)
    return score_trial(operator, cfg, graphs, col_sets, node_dim, trial_dir, recipe, seeds, stage, seed_info)


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
    '''One row per trial: every knob, best / stopped epochs, train and val metrics (mean ± std, and per seed as
    <metric>_s<seed> where trial.json has them), the gap, the status (ok | diverged | oom) and a winner flag.'''
    winners = winners or {}
    rows = []
    for t in trials:
        r = full_recipe(t['recipe'])
        row = {
            'stage': t['stage'], 'trial': t['trial'], 'winner': winners.get(t['stage']) == t['trial'],
            'dropout': r['dropout'], 'norm': r['norm'], 'num_layers': r['num_layers'],
            'num_neighbors': '-'.join(str(n) for n in r['num_neighbors']), 'hidden': r['hidden'],
            'pos_weight': r['pos_weight'], 'lr': r['lr'], 'ffn': r['ffn'], 'self_loops': r['self_loops'],
            'epochs_T_max': r['epochs'],
            'best_epoch': t.get('best_epoch'),
            'epochs_run': '/'.join(str(t['epochs_run'][str(s)]) for s in t['seeds']) if 'epochs_run' in t else '',
            'reached_max': t.get('reached_max'),
        }
        for key in ('val_f1', 'train_f1'):
            row[f'{key}_mean'], row[f'{key}_std'] = t.get(f'{key}_mean', np.nan), t.get(f'{key}_std', np.nan)
        row['gap_f1'] = t.get('gap_f1', np.nan)
        row['val_pr_auc_mean'], row['val_pr_auc_std'] = t.get('val_pr_auc_mean', np.nan), t.get('val_pr_auc_std', np.nan)
        row['train_pr_auc_mean'], row['gap_pr_auc'] = t.get('train_pr_auc_mean', np.nan), t.get('gap_pr_auc', np.nan)
        row['thresholds'] = '/'.join(f"{t['thresholds'][str(s)]:.3f}" for s in t['seeds']) if 'thresholds' in t else ''
        row.update(params=t.get('params'), invariant_params=t.get('invariant_params'),
                   seconds_per_epoch=t.get('seconds_per_epoch'))
        row.update(status=t.get('status', 'ok'), diverged_at=t.get('diverged_at'), residual=r['residual'],
                   readout_layers=r['readout_layers'], readout_width=r['readout_width'],
                   readout_width_used=r['readout_layers'] > 1, weight_decay=r['weight_decay'],
                   lr_schedule=r['lr_schedule'], num_workers=r['num_workers'], select_metric=r['select_metric'])
        for key in TRIAL_METRIC_KEYS:
            if f'{key}_mean' in t:
                row[f'{key}_mean'], row[f'{key}_std'] = t[f'{key}_mean'], t[f'{key}_std']
            for s in t['seeds']:
                if key in t.get('per_seed', {}).get(str(s), {}):
                    row[f'{key}_s{s}'] = t['per_seed'][str(s)][key]
        rows.append(row)
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


def plot_val_curves(trials, winners, path, max_labels=None, select_metric='f1'):
    '''tuning_curves.png: seed-mean val F1 (or PR-AUC) per epoch, one line per trial (winners bold), a dot at each best epoch.

    max_labels: only the winners and the top max_labels trials get a colour and a legend entry (100-trial searches).
    '''
    winners = winners or {}
    key = f'val_{select_metric}'
    name = {'f1': 'F1', 'pr_auc': 'PR-AUC'}[select_metric]
    plotted = [t for t in trials if f'{key}_mean_curve' in t]
    ranked = sorted(plotted, key=lambda t: -t[f'{key}_mean'])
    labelled = {t['trial'] for t in (ranked[:max_labels] if max_labels else ranked)}
    fig, ax = plt.subplots(figsize=(12, 6))
    for t in plotted:
        curve = t[f'{key}_mean_curve']
        epochs = np.arange(1, len(curve) + 1)
        is_winner = winners.get(t['stage']) == t['trial']
        shown = is_winner or t['trial'] in labelled
        line, = ax.plot(epochs, curve, linewidth=2.2 if is_winner else 1.0,
                        alpha=1.0 if is_winner else (0.7 if shown else 0.25), color=None if shown else 'gray',
                        label=f"{t['stage']}/{t['trial']} ({t[f'{key}_mean']:.3f} @ {t['best_epoch']})" if shown else None)
        ax.plot(t['best_epoch'], curve[t['best_epoch'] - 1], 'o', color=line.get_color())
    ax.set_xlabel('epoch')
    ax.set_ylabel(f'validation {name} (seed mean)')
    ax.set_title(f'recipe tuning: seed-mean validation {name} per trial'
                 + (f' (coloured: winner + top {max_labels})' if max_labels else ''))
    ax.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.show()
    plt.close(fig)


def is_out_of_memory(err):
    '''A CUDA out-of-memory error, also when it surfaces as a RuntimeError (cuBLAS / cuSPARSE allocation failure).'''
    text = str(err)
    return isinstance(err, torch.cuda.OutOfMemoryError) or 'out of memory' in text.lower() or 'ALLOC_FAILED' in text


def process_rss_gb():
    '''Resident memory of this process plus its children (loader workers) in GB; shared pages count twice (upper bound).'''
    proc = psutil.Process()
    total = proc.memory_info().rss
    for child in proc.children(recursive=True):
        try:
            total += child.memory_info().rss
        except psutil.NoSuchProcess:   # a loader worker that exited between listing and reading
            pass
    return total / 2**30


def probe_attempt(operator, cfg, graphs, col_sets, node_dim, recipe, n_warmup=3, n_timed=20, deg_hist=None,
                  val_pass=True):
    '''One probe attempt: warm-up + timed training batches, then (val_pass) one full val prediction pass.

    Returns GPU-only and wall-clock (sampling included) seconds per batch, peak allocated / reserved GPU memory and
    peak CPU memory. Its own function, so every tensor it made is freed once an out-of-memory error has left it.
    '''
    mp_idx, readout_idx = col_sets[cfg['mp']], col_sets[cfg['readout']]
    cuda = device.type == 'cuda'
    set_seed(SEED)
    model = build_model(operator, node_dim, len(mp_idx), len(readout_idx), deg_hist=deg_hist, recipe=recipe)
    loaders = make_loaders(graphs, mp_idx, readout_idx, recipe, splits=('train', 'val'))
    train_loader, train_feats = loaders['train']
    r = full_recipe(recipe)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([r['pos_weight']], device=device))
    optimizer = torch.optim.Adam(model.parameters(), lr=r['lr'], weight_decay=r['weight_decay'])
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    model.train()
    seconds, wall, done, peak_rss = 0.0, 0.0, 0, process_rss_gb()
    batches = iter(train_loader)
    for i in range(len(train_loader)):
        t_fetch = time.time()
        batch = next(batches)
        t0 = time.time()
        optimizer.zero_grad()
        logits, y = forward_batch(model, batch, train_feats)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        if cuda:
            torch.cuda.synchronize()
        if i >= n_warmup:
            seconds, wall, done = seconds + time.time() - t0, wall + time.time() - t_fetch, done + 1
            peak_rss = max(peak_rss, process_rss_gb())
        if done >= n_timed:
            break
    del batches
    out = {'train_s_per_batch': seconds / max(1, done), 'train_wall_s_per_batch': wall / max(1, done),
           'train_batches_per_epoch': len(train_loader)}
    out['train_epoch_s'] = out['train_s_per_batch'] * len(train_loader)
    out['train_epoch_wall_s'] = out['train_wall_s_per_batch'] * len(train_loader)
    out['train_peak_gb'] = torch.cuda.max_memory_allocated() / 2**30 if cuda else float('nan')
    out['train_peak_reserved_gb'] = torch.cuda.max_memory_reserved() / 2**30 if cuda else float('nan')
    if val_pass:
        if cuda:
            torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        val_loader, val_feats = loaders['val']
        predict(model, val_loader, val_feats, f'probe val {r["num_layers"]}L')
        out['val_predict_s'] = time.time() - t0
        out['val_peak_gb'] = torch.cuda.max_memory_allocated() / 2**30 if cuda else float('nan')
        out['val_peak_reserved_gb'] = torch.cuda.max_memory_reserved() / 2**30 if cuda else float('nan')
        out['epoch_estimate_s'] = out['train_epoch_s'] + out['val_predict_s']
        # an epoch that scores train AND val: one train pass + predictions on train and val
        out['epoch_wall_estimate_s'] = out['train_epoch_wall_s'] + out['val_predict_s'] * (1 + len(train_loader) / len(val_loader))
    out['peak_rss_gb'] = max(peak_rss, process_rss_gb())
    out['params'] = count_params(model)
    return out


def neighbour_candidates(depth):
    '''The neighbour lists a probe tries at a depth: the primary list, then the fallback if there is one.'''
    return [(label, table[depth]) for label, table in (('primary', NEIGHBOURS_BY_DEPTH), ('fallback', NEIGHBOURS_FALLBACK))
            if depth in table]


def probe_one(operator, cfg, graphs, col_sets, node_dim, recipe, depth, label, neighbours, deg_hist, n_warmup=3,
              n_timed=20):
    '''One probe attempt at one depth and neighbour list; an out-of-memory error is recorded in the attempt, not raised.'''
    r = full_recipe(dict(recipe, num_layers=depth, num_neighbors=neighbours))
    attempt = {'list': label, 'neighbors': neighbours}
    print(f'--- probe: {depth} layers, {label} neighbours {neighbours}')
    try:   # running out of memory is the probe's answer, not a failure
        attempt.update(probe_attempt(operator, cfg, graphs, col_sets, node_dim, r, n_warmup, n_timed, deg_hist))
    except (torch.cuda.OutOfMemoryError, RuntimeError) as err:
        if not is_out_of_memory(err):
            raise
        attempt['error'] = f'CUDA out of memory: {str(err)[:120]}'
    gc.collect()                # outside the except block, so the failed attempt's tensors are released
    torch.cuda.empty_cache()
    if 'error' in attempt:
        print(f'    out of memory with {label} neighbours')
    else:
        print(f"    train {attempt['train_s_per_batch']:.2f} s/batch x {attempt['train_batches_per_epoch']} = "
              f"{attempt['train_epoch_s'] / 60:.1f} min, peak {attempt['train_peak_gb']:.1f} GB | val pass "
              f"{attempt['val_predict_s'] / 60:.1f} min, peak {attempt['val_peak_gb']:.1f} GB | "
              f"CPU RAM peak {attempt['peak_rss_gb']:.1f} GB")
    return attempt


def probe_depths(operator, cfg, graphs, col_sets, node_dim, recipe, depths, out_path, n_warmup=3, n_timed=20,
                 max_ratio=3.0, mp_direction='in', temporal=False, reference_seconds=None):
    '''Memory / time probe per depth at the heaviest setting (the one-pass tuning notebooks).

    For each depth the primary neighbour list is tried first, then the fallback; a depth passes when it fits in
    memory and its epoch estimate is <= max_ratio x the reference (the first depth probed, or reference_seconds
    from an earlier probe). Returns {depth: {'neighbors': list or None, ...}} and writes it to out_path
    (probe.json). mp_direction / temporal are fixed ('in' / False, see run_name).
    '''
    assert mp_direction == 'in' and not temporal, 'only mp_direction="in" without temporal sampling is implemented'
    deg_hist = in_degree_histogram(graphs['train']) if operator == 'pna' else None
    results, base_epoch_seconds = {}, reference_seconds
    for depth in depths:
        results[depth] = {'neighbors': None, 'status': 'dropped', 'attempts': []}
        for label, neighbours in neighbour_candidates(depth):
            attempt = probe_one(operator, cfg, graphs, col_sets, node_dim, recipe, depth, label, neighbours, deg_hist,
                                n_warmup, n_timed)
            results[depth]['attempts'].append(attempt)
            if 'error' in attempt:
                continue
            if base_epoch_seconds is None:
                base_epoch_seconds = attempt['epoch_estimate_s']
            attempt['ratio_to_first_depth'] = attempt['epoch_estimate_s'] / base_epoch_seconds
            print(f"    epoch ~{attempt['epoch_estimate_s'] / 60:.1f} min ({attempt['ratio_to_first_depth']:.2f}x the first depth)")
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


def probe_memory(operator, cfg, graphs, col_sets, node_dim, recipe, depths, out_path, mem_fraction=0.85):
    '''Memory probe of the Luo tuning: which depths fit in GPU memory at the heaviest setting (no time rule).

    A depth fits when nothing ran out of memory and the peak reserved GPU memory stays <= mem_fraction of the GPU
    (the margin: the timed batches do not see the largest subgraph); the fallback list is tried when the primary
    does not fit. The probe stops at the first depth that fits with neither list (memory never shrinks with depth).
    probe.json is saved after every depth and the depths already in it are skipped, so a killed probe resumes.
    '''
    deg_hist = in_degree_histogram(graphs['train']) if operator == 'pna' else None
    cuda = device.type == 'cuda'
    gpu_gb = torch.cuda.get_device_properties(0).total_memory / 2**30 if cuda else float('nan')
    probe = json.load(open(out_path)) if os.path.exists(out_path) else {}
    probe.update(rule='memory', mem_fraction=mem_fraction, gpu_total_gb=gpu_gb, cfg=cfg, recipe=full_recipe(recipe),
                 gpu_name=torch.cuda.get_device_name(0) if cuda else 'cpu')
    for depth in depths:
        if str(depth) not in probe:
            entry = {'neighbors': None, 'status': 'dropped', 'attempts': []}
            for label, neighbours in neighbour_candidates(depth):
                attempt = probe_one(operator, cfg, graphs, col_sets, node_dim, recipe, depth, label, neighbours,
                                    deg_hist)
                entry['attempts'].append(attempt)
                if 'error' in attempt:
                    continue
                reserved = max(attempt['train_peak_reserved_gb'], attempt['val_peak_reserved_gb'])
                if not reserved > mem_fraction * gpu_gb:     # NaN on a CPU run -> fits
                    entry.update(neighbors=neighbours, status=f'ok ({label})')
                    break
                print(f'    too close to the memory limit: {reserved:.1f} GB reserved > {mem_fraction:.0%} of {gpu_gb:.1f} GB')
            probe[str(depth)] = entry
            write_json_atomic(out_path, probe)
        if probe[str(depth)]['neighbors'] is None:
            print(f'probe: {depth} layers fit with neither list; deeper models need more memory, so the probe stops here')
            break
    print('probe:', {d: v['status'] for d, v in probe.items() if d.isdigit()}, '->', out_path)
    return probe


def time_loader_workers(operator, cfg, graphs, col_sets, node_dim, recipe, depth_neighbours, settings=(0, 4, 6),
                        n_warmup=3, n_timed=20):
    '''Wall-clock seconds per training batch (sampling + step) and peak CPU memory for each loader worker count.

    depth_neighbours: {depth: neighbour list}. Returns {depth: {workers: {'wall_s_per_batch', 'peak_rss_gb'}}}.
    '''
    deg_hist = in_degree_histogram(graphs['train']) if operator == 'pna' else None
    out = {}
    for depth, neighbours in depth_neighbours.items():
        out[str(depth)] = {}
        for workers in settings:
            r = full_recipe(dict(recipe, num_layers=depth, num_neighbors=neighbours, num_workers=workers))
            numbers = probe_attempt(operator, cfg, graphs, col_sets, node_dim, r, n_warmup, n_timed, deg_hist,
                                    val_pass=False)
            out[str(depth)][str(workers)] = {'wall_s_per_batch': numbers['train_wall_s_per_batch'],
                                             'peak_rss_gb': numbers['peak_rss_gb']}
            print(f"--- loader timing: {depth} layers, {workers} workers: {numbers['train_wall_s_per_batch']:.3f} s/batch, "
                  f"CPU RAM peak {numbers['peak_rss_gb']:.1f} GB")
            gc.collect()
            torch.cuda.empty_cache()
    return out


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

def save_recipe(path, operator, cfg, trial, retrain_seeds=(42, 43, 44, 45), extra=None):
    '''recipe.json: the frozen knobs, best epoch, per-seed thresholds and the validation F1 that chose them.

    One threshold for every retrain seed: the mean of the tuning seeds' validation thresholds at the best epoch
    (the per-seed tuning thresholds are kept in 'tuning_thresholds' for the record). `extra` is merged in.
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
    frozen.update(extra or {})
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


def final_scoring(preds, threshold, transfer_threshold):
    '''Metrics of a final run from its predictions: every split at `threshold` (the headline), the thresholded metrics
    at `transfer_threshold` (the tuned config's cut, secondary; None skips this view) and at the threshold re-swept
    on the val rows (seen in training here, secondary), and the test sensitivity table.
    Returns (metrics, extra row fields, sensitivity rows).'''
    metrics = {split: compute_metrics(preds[split]['y'], preds[split]['prob'], threshold) for split in SPLITS}
    sensitivity = threshold_sensitivity(preds['test']['y'], preds['test']['prob'], threshold)
    extra = {}
    if transfer_threshold is not None:
        extra['transfer_threshold'] = float(transfer_threshold)
        tr = {split: compute_metrics(preds[split]['y'], preds[split]['prob'], transfer_threshold) for split in SPLITS}
        extra.update({f'transfer_{split}_{m}': tr[split][m] for split in SPLITS for m in ('f1', 'precision', 'recall')})
        sensitivity.append({'point': 'tuned-config threshold', 'threshold': float(transfer_threshold),
                            'delta': float(transfer_threshold - threshold), 'f1': tr['test']['f1'],
                            'precision': tr['test']['precision'], 'recall': tr['test']['recall']})
    seen_threshold, _ = best_f1_threshold(preds['val']['y'], preds['val']['prob'])
    seen = {split: compute_metrics(preds[split]['y'], preds[split]['prob'], seen_threshold) for split in SPLITS}
    extra['val_seen_threshold'] = seen_threshold
    extra.update({f'seen_{split}_{m}': seen[split][m] for split in SPLITS for m in ('f1', 'precision', 'recall')})
    sensitivity.append({'point': 'val seen sweep', 'threshold': seen_threshold, 'delta': float(seen_threshold - threshold),
                        'f1': seen['test']['f1'], 'precision': seen['test']['precision'], 'recall': seen['test']['recall']})
    return metrics, extra, sensitivity


def run_threshold_selection(operator, cfg, graphs, col_sets, node_dim, out_root, frozen, seed, mp_direction='in',
                            temporal=False, eval_train_every=5):
    '''The validation-chosen threshold of one config under the frozen recipe: train on the train graph only for the
    frozen number of epochs (no early stopping), validate every epoch, take the validation-F1 threshold at the last
    epoch. Writes <out_root>/<run>/seed-<s>/{history.csv, curves.png, selection.json}; skipped when selection.json exists.
    Returns {'run', 'seed', 'epoch', 'threshold', 'val_f1', 'val_pr_auc', 'source'}.'''
    recipe, stop_epoch = frozen['recipe'], int(frozen['best_epoch'])
    name = run_name(operator, cfg, mp_direction, temporal)
    out_dir = f'{out_root}/{name}/seed-{seed}'
    if os.path.exists(f'{out_dir}/selection.json'):
        sel = json.load(open(f'{out_dir}/selection.json'))
        print(f"=== threshold selection {name} seed {seed} === already finished (threshold {sel['threshold']:.3f}), skipping")
        return sel
    os.makedirs(out_dir, exist_ok=True)
    mp_idx, readout_idx = col_sets[cfg['mp']], col_sets[cfg['readout']]
    set_seed(seed)
    deg_hist = in_degree_histogram(graphs['train']) if operator == 'pna' else None
    model = build_model(operator, node_dim, len(mp_idx), len(readout_idx), deg_hist=deg_hist, recipe=recipe)
    print_model_header(f'threshold selection {name} seed {seed} (train graph, {stop_epoch} epochs, frozen recipe)',
                       mp_idx, readout_idx, count_params(model), invariant_params(model), node_dim)
    loaders = make_loaders(graphs, mp_idx, readout_idx, recipe, splits=('train', 'val'))
    history, _, _ = train_one_run(model, loaders, out_dir, recipe, stop_epoch=stop_epoch, eval_train_every=eval_train_every)
    last = history[-1]
    assert last['epoch'] == stop_epoch
    plot_curves(history, None, None, None, stop_epoch, f'{name} threshold selection seed {seed}', f'{out_dir}/curves.png')
    sel = {'run': name, 'operator': operator, 'cfg': cfg, 'seed': seed, 'epoch': stop_epoch,
           'threshold': float(last['threshold']), 'val_f1': float(last['val_f1']), 'val_pr_auc': float(last['val_pr_auc']),
           'source': f'validation sweep at epoch {stop_epoch} of a train-only run with the frozen recipe, seed {seed}'}
    json.dump(sel, open(f'{out_dir}/selection.json', 'w'), indent=2)
    if os.path.exists(f'{out_dir}/best.pt'):
        os.remove(f'{out_dir}/best.pt')
    print(f"threshold selection {name}: {sel['threshold']:.3f} (val F1 {sel['val_f1']:.4f} at epoch {stop_epoch})")
    del model, loaders
    gc.collect()
    torch.cuda.empty_cache()
    return sel


def save_config_thresholds(path, entries):
    '''config_thresholds.json: {run: {threshold, source, ...}}, one validation-chosen cut per feature config.'''
    json.dump(entries, open(path, 'w'), indent=2)
    print('config thresholds -> ' + path + ': ' + ', '.join(f"{k}={v['threshold']:.3f}" for k, v in entries.items()))


def run_final(operator, cfg, graphs, col_sets, node_dim, out_root, frozen, seed, mp_direction='in', temporal=False,
              node_enc='none', enc_dir=None, gfp_variant='v0', cfg_threshold=None, resume=False, eval_curves=False):
    '''Final retrain of one seed: train + val graph, stop at the frozen epoch, every split scored once.

    The headline threshold is cfg_threshold (this config's validation-chosen cut, see run_threshold_selection) or,
    when None, the frozen tuning threshold. Writes <out_root>/<run>/seed-<s>/: results.json (+ phase, stop_epoch,
    threshold_source, val_seen and the secondary views: transfer_* at the tuned config's threshold when cfg_threshold
    is given, seen_* at the threshold re-swept on the val rows that were trained on here), history.csv, curves.png,
    sensitivity.csv, predictions.csv, final.pt. results.json is written last, so it marks a finished run, and a
    finished run is skipped. resume=True continues a killed retrain from seed-<s>/resume.pt; eval_curves=True
    scores train and val (seen) every epoch for the curves.
    '''
    recipe, stop_epoch = frozen['recipe'], int(frozen['best_epoch'])
    frozen_threshold = float(frozen['thresholds'][str(seed)])
    threshold = frozen_threshold if cfg_threshold is None else float(cfg_threshold)
    transfer_threshold = None if cfg_threshold is None else frozen_threshold   # 2nd view only beside a config cut
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
    model = build_model(operator, node_dim, len(mp_idx), len(readout_idx), deg_hist=deg_hist, recipe=recipe)
    n_params, n_invariant = count_params(model), invariant_params(model)
    print_model_header(f'{name} seed {seed} (train+val retrain, stop at epoch {stop_epoch}, threshold {threshold:.3f})',
                       mp_idx, readout_idx, n_params, n_invariant, node_dim)
    fit_loader = make_loader(fit_graph, True, mp_idx, readout_idx, recipe)
    eval_loaders = make_loaders(graphs, mp_idx, readout_idx, recipe, splits=('train', 'val')) if eval_curves else None
    history = train_fixed_epochs(model, fit_loader, out_dir, recipe, stop_epoch,
                                 resume_path=f'{out_dir}/resume.pt' if resume else None, eval_loaders=eval_loaders)
    del fit_loader, fit_graph, eval_loaders
    gc.collect()

    # scoring: the three usual loaders (val is training data here), test once; headline = this config's threshold,
    # secondary = the tuned config's threshold and the seen-val sweep (PR-AUC, ROC-AUC and top-5 % do not move)
    loaders = make_loaders(graphs, mp_idx, readout_idx, recipe)
    _, preds = score_all_splits(model, loaders, graphs, threshold, seed)
    metrics, extra, sensitivity = final_scoring(preds, threshold, transfer_threshold)
    row = result_row(name, operator, cfg, stop_epoch, threshold, n_params, n_invariant, metrics, node_enc, gfp_variant, seed)
    row.update(phase='retrain', trained_on='train+val', stop_epoch=stop_epoch,
               threshold_source='config' if cfg_threshold is not None else 'tuning', val_seen=True, **extra)
    transfer = '' if transfer_threshold is None else \
        f"{extra['transfer_test_f1']:.4f} at the tuned config's {transfer_threshold:.3f} | "
    print(f"test F1 {metrics['test']['f1']:.4f} at this config's threshold {threshold:.3f} | {transfer}"
          f"{extra['seen_test_f1']:.4f} at the seen-val sweep {extra['val_seen_threshold']:.3f}")
    pd.DataFrame(sensitivity).to_csv(f'{out_dir}/sensitivity.csv', index=False)
    print(pd.DataFrame(sensitivity).round(4))
    save_predictions(out_dir, graphs, preds, threshold)
    plot_final_curves(history, preds['test']['y'], preds['test']['prob'], metrics['test']['pr_auc'],
                      f'{name} seed {seed}', f'{out_dir}/curves.png')
    save_run(out_dir, row, recipe, extra={'threshold_sensitivity': sensitivity})   # last: results.json = finished

    del model, loaders
    gc.collect()
    torch.cuda.empty_cache()
    return row


# ==================================================================================================
# Luo et al. (2024) tuning: random search per operator and config, two seeds per trial (EXPERIMENTS.md section 2b);
# run on Cineca through cineca/ (probe.py, draw_trials.py, tune.py, aggregate.py, final.py)
# ==================================================================================================

LUO_SPACE = {                          # Luo et al. 2024 section 4 + Tables 8-10, tunedGNN, and our additions
    'lr':             [0.0005, 0.001, 0.003, 0.005, 0.01],
    'hidden':         [64, 128, 256, 512],              # 128 is ours: the size of every earlier run
    'norm':           ['none', 'batch', 'layer'],
    'residual':       ['linear', 'none'],               # the paper's linear residual, or none
    'dropout':        [0.2, 0.3, 0.5, 0.7],
    'weight_decay':   [0.0, 5e-5, 5e-4, 1e-3, 1e-2],    # tunedGNN's run scripts (the paper's text names none)
    'pos_weight':     [4.0, 8.0, 16.0],                 # ours
    'readout_layers': [1, 2, 3],                        # ours: the paper does not tune the head
    'readout_width':  [0.5, 1.0, 2.0],                  # ours; drawn but unused when readout_layers = 1
}
LUO_DEPTHS = tuple(range(1, 11))       # minus the depths that do not fit in GPU memory (probe)
LUO_MAX_EPOCHS = 100
LUO_DRAW_SEED = 20261022
LUO_TUNING_SEEDS = (42, 43)
LUO_FINAL_SEEDS = (42, 43, 44, 45, 46)
LUO_STAGE = 'luo2024'
# every trial: 100 epochs at a constant LR, no early stopping, best epoch = raw argmax of the seed-mean val PR-AUC
LUO_PROTOCOL = dict(epochs=LUO_MAX_EPOCHS, lr_schedule='constant', select_metric='pr_auc')
LUO_HEAVIEST = dict(hidden=512, norm='batch', residual='linear', readout_layers=3, readout_width=2.0)   # probe setting
LUO_MEM_FRACTION = 0.85                # a depth fits when its peak reserved GPU memory is <= 85 % of the GPU
LOADER_WORKER_SETTINGS = (0, 4, 6)     # timed by the probe; non-persistent workers keep a resumed run exact


def luo_recipe(knobs=None, num_layers=2, num_neighbors=None, num_workers=0):
    '''Full recipe of one Luo trial: the protocol (100 epochs, constant LR, PR-AUC selection) + the drawn knobs.

    knobs=None is trial 0: the v1 knobs (hidden 128, 2 layers, identity residual, ...) under the same protocol.
    '''
    recipe = dict(LUO_PROTOCOL, num_workers=int(num_workers), **(knobs or {}))
    if knobs is not None:
        recipe.update(num_layers=int(num_layers), num_neighbors=[int(n) for n in num_neighbors])
    return full_recipe(recipe)


def draw_luo_trials(n_trials, allowed_depths, neighbours, num_workers, seed=LUO_DRAW_SEED):
    '''The pre-registered draw: trial 0 = the v1 knobs; trials 1..n-1 = cells of LUO_SPACE + a depth per operator.

    The cells come from a partial Fisher-Yates shuffle of the 64,800 cells, the same for every operator; each
    trial's depth comes from a second stream of uniforms mapped onto that operator's allowed depths. Both streams
    use random.Random(seed).random(), whose sequence Python keeps across versions, and trial i never depends on
    n_trials. allowed_depths {op: [depth]}, neighbours {op: {depth: list}}, num_workers {op: int}.
    Returns {op: [{'id', 'tag', 'recipe', 'readout_width_used'}]} in plain Python types.
    '''
    cells = list(itertools.product(*LUO_SPACE.values()))
    assert 1 <= n_trials <= len(cells) + 1, f'n_trials must lie in 1..{len(cells) + 1}'
    order = list(range(len(cells)))
    rng = random.Random(seed)
    for i in range(n_trials - 1):            # partial Fisher-Yates: position i is fixed by the i-th draw only
        j = i + int(rng.random() * (len(order) - i))
        order[i], order[j] = order[j], order[i]
    depth_rng = random.Random(seed + 1)
    depth_u = [depth_rng.random() for _ in range(n_trials - 1)]
    trials = {}
    for op, depths in allowed_depths.items():
        depths = sorted(int(d) for d in depths)
        base = luo_recipe(num_workers=num_workers[op])
        rows = [{'id': 0, 'tag': 'v1', 'recipe': base, 'readout_width_used': True}]
        for i in range(1, n_trials):
            knobs = dict(zip(LUO_SPACE, cells[order[i - 1]]))
            depth = depths[int(depth_u[i - 1] * len(depths))]
            recipe = luo_recipe(knobs, depth, neighbours[op][depth], num_workers[op])
            rows.append({'id': i, 'tag': recipe_tag(recipe, base), 'recipe': recipe,
                         'readout_width_used': knobs['readout_layers'] > 1})
        trials[op] = rows
    return trials


def choose_best_trial(trials, select_metric='pr_auc'):
    '''Winner of one (operator, config): highest mean val <select_metric>, tie-break the other metric, then the id.
    Trials without metrics (out of memory, or diverged before their first epoch) cannot win.'''
    other = 'f1' if select_metric == 'pr_auc' else 'pr_auc'
    scored = [t for t in trials if f'val_{select_metric}_mean' in t]
    assert scored, 'no scored trial to choose from'
    return sorted(scored, key=lambda t: (-t[f'val_{select_metric}_mean'], -t[f'val_{other}_mean'], t['trial']))[0]


def delete_trial_checkpoints(trial_dir, seeds, keep_best=True):
    '''Remove the epoch checkpoints, resume.pt and leftover temporary files of every seed (best.pt stays unless
    keep_best=False).'''
    for s in seeds:
        leftovers = glob.glob(f'{trial_dir}/seed-{s}/epoch_*.pt') + glob.glob(f'{trial_dir}/seed-{s}/*.tmp*')
        for path in leftovers + [f'{trial_dir}/seed-{s}/resume.pt']:
            Path(path).unlink(missing_ok=True)
        if not keep_best:
            Path(f'{trial_dir}/seed-{s}/best.pt').unlink(missing_ok=True)


def finalise_trial(operator, cfg, trial_dir, recipe, seeds=LUO_TUNING_SEEDS, stage=LUO_STAGE, extra=None):
    '''Once every seed has its done.json, write trial.json from the seeds' histories (CPU only).

    Best epoch = argmax of the seed-mean val <select_metric> (raw, first epoch on ties) over the epochs every seed
    reached; per-seed metrics and thresholds = that history row (score_trial from_history). That epoch's checkpoint
    becomes seed-<s>/best.pt, every other checkpoint and resume.pt is deleted, the curves are re-drawn at the
    trial's epoch (per seed, and the seed means), trial.json is written last. A seed that ran out of memory, or
    stopped before its first epoch, leaves a trial without metrics ('oom' / 'diverged', cannot win).
    Returns the trial dict, or None while a seed is not done. Rerunning after an interruption is safe; two
    processes at once are not (cineca/tune.py lets only one finalise a trial).
    '''
    tag = os.path.basename(trial_dir)
    if os.path.exists(f'{trial_dir}/trial.json'):
        return json.load(open(f'{trial_dir}/trial.json'))
    done_paths = {s: f'{trial_dir}/seed-{s}/done.json' for s in seeds}
    if not all(os.path.exists(p) for p in done_paths.values()):
        return None
    done = {s: json.load(open(p)) for s, p in done_paths.items()}
    r = full_recipe(recipe)
    diverged = {str(s): done[s]['diverged_at'] for s in seeds if done[s]['diverged_at']}
    info = dict(extra or {}, diverged_at=diverged or None)

    if any(done[s]['status'] == 'oom' or not done[s]['epochs_run'] for s in seeds):
        status = 'oom' if any(done[s]['status'] == 'oom' for s in seeds) else 'diverged'
        delete_trial_checkpoints(trial_dir, seeds, keep_best=False)
        trial = dict(info, trial=tag, stage=stage, operator=operator, cfg=cfg, recipe=r, seeds=list(seeds), status=status)
        write_json_atomic(f'{trial_dir}/trial.json', trial)
        print(f'trial {tag}: {status}, no metrics (cannot win)')
        return trial

    metric = r['select_metric']
    histories = {s: pd.read_csv(f'{trial_dir}/seed-{s}/history.csv') for s in seeds}
    best_epoch, _, _ = seed_mean_best_epoch(histories, 1, f'val_{metric}')
    for s in seeds:
        kept = f'{trial_dir}/seed-{s}/epoch_{best_epoch:03d}.pt'
        if os.path.exists(kept):              # a rerun after a killed finalisation finds it moved already
            os.replace(kept, f'{trial_dir}/seed-{s}/best.pt')
        assert os.path.exists(f'{trial_dir}/seed-{s}/best.pt'), f'{tag} seed {s}: no checkpoint of epoch {best_epoch}'
        plot_curves(histories[s], None, None, None, best_epoch, f'{tag} seed {s} (trial best epoch)',
                    f'{trial_dir}/seed-{s}/curves.png', metric)
    delete_trial_checkpoints(trial_dir, seeds)
    n_common = min(len(h) for h in histories.values())
    mean_history = pd.concat([h.iloc[:n_common] for h in histories.values()]).groupby('epoch').mean().reset_index()
    plot_curves(mean_history, None, None, None, best_epoch, f'{tag} seed mean {list(seeds)}', f'{trial_dir}/curves.png',
                metric)
    seed_info = {'params': done[seeds[0]]['params'], 'invariant_params': done[seeds[0]]['invariant_params'],
                 'seconds_per_epoch': float(np.mean([done[s]['seconds_per_epoch'] for s in seeds]))}
    return score_trial(operator, cfg, None, None, None, trial_dir, r, seeds, stage, seed_info, window=1,
                       from_history=True, extra=dict(info, status='diverged' if diverged else 'ok'))
