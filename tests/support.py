"""Helpers shared by the tests. Fixtures live in conftest.py."""
import numpy as np
import torch
from torch_geometric.data import Data


def tiny_recipe(**overrides):
    """v1 knobs shrunk so one epoch fits on CPU. num_neighbors must match num_layers."""
    from gnn_frauds.hyperparameters.gnns import full_recipe
    knobs = dict(hidden=8, num_layers=1, num_neighbors=[2], dropout=0.0, batch_size=16,
                 epochs=1, weight_decay=0.0, pos_weight=1.0)
    knobs.update(overrides)
    return full_recipe(knobs)


def make_bundle(n_nodes=24, n_train=48, n_val=24, n_test=24, edge_dim=81, node_dim=6, seed=0):
    """Three cumulative snapshots: train, then val, then test, with disjoint evaluated edges."""
    rng = np.random.default_rng(seed)
    n_edges = n_train + n_val + n_test
    cycle_src = np.arange(n_nodes)
    cycle_dst = (cycle_src + 1) % n_nodes
    extra = n_edges - n_nodes
    src = np.concatenate([cycle_src, rng.integers(0, n_nodes, extra)])
    dst = np.concatenate([cycle_dst, rng.integers(0, n_nodes, extra)])
    edge_index = torch.tensor(np.stack([src, dst]), dtype=torch.long)
    edge_attr = torch.tensor(rng.normal(size=(n_edges, edge_dim)), dtype=torch.float32)
    edge_time = torch.arange(n_edges, dtype=torch.long)
    x = torch.zeros(n_nodes, node_dim, dtype=torch.float32)
    x[torch.arange(n_nodes), torch.arange(n_nodes) % node_dim] = 1.0

    labels = torch.zeros(n_edges, dtype=torch.long)
    for index in (2, 15, 33, 50, 63, 80, 94):
        labels[index] = 1

    def snapshot(end, context):
        y = labels[:end].clone()
        y[:context] = -1
        mask = torch.zeros(end, dtype=torch.bool)
        mask[context:] = True
        return Data(x=x.clone(), edge_index=edge_index[:, :end].clone(), edge_attr=edge_attr[:end].clone(),
                    edge_time=edge_time[:end].clone(), y=y, eval_mask=mask, num_nodes=n_nodes)

    graphs = {
        'train': snapshot(n_train, 0),
        'val': snapshot(n_train + n_val, n_train),
        'test': snapshot(n_edges, n_train + n_val),
    }
    base = list(range(20))
    gfp = list(range(20, edge_dim))
    col_sets = {'none': [], 'base': base, 'gfp': gfp, 'full': base + gfp}
    return {'graphs': graphs, 'col_sets': col_sets, 'node_dim': node_dim}
