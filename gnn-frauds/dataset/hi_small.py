"""Load the HI-Small snapshots named by conf/dataset/hi_small.yaml."""
import gc
import json
import random

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import degree

from gnn_frauds.hyperparameters.gnns import BASE_EDGE_DIM, GFP_VARIANTS, GFP_WIDTH, SPLITS
from gnn_frauds.preprocessing.gfp import normalise_gfp_sheet

def set_seed(seed):
    '''Seed python, numpy and torch so the invariant weights start identical in every run.'''
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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
            # Snapshots are cumulative, so this split is the first n_edges rows of the full sheet.
            g.edge_attr = torch.cat([g.edge_attr[:, :BASE_EDGE_DIM], sheet[:n_edges]], dim=1)
        del sheet
        gc.collect()  # the sheet is a second copy of every edge; release it before training

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

