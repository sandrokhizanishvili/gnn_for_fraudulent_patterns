"""GFP normalisation, RWPE, graph loading, and the train+val snapshot."""
import json

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.preprocessing import StandardScaler

from gnn_frauds.dataset.hi_small import in_degree_histogram, load_graphs, make_trainval_graph, set_seed
from gnn_frauds.preprocessing.gfp import normalise_gfp_sheet
from gnn_frauds.preprocessing.node_embedding import set_node_encoding
from rwpe_compute import compute_rwpe, peak_ram_gb


def test_set_seed_repeats_torch_and_numpy():
    set_seed(123)
    a_torch, a_numpy = torch.rand(4), np.random.rand(4)
    set_seed(123)
    assert torch.equal(a_torch, torch.rand(4))
    assert np.allclose(a_numpy, np.random.rand(4))


def test_in_degree_histogram_counts_incoming_edges():
    from torch_geometric.data import Data
    edge_index = torch.tensor([[0, 1, 0], [2, 2, 1]], dtype=torch.long)
    graph = Data(edge_index=edge_index, num_nodes=4)
    hist = in_degree_histogram(graph)
    assert hist[0].item() == 2  # nodes 0 and 3 have no incoming edge
    assert hist[1].item() == 1
    assert hist[2].item() == 1


def test_normalise_gfp_sheet_fits_on_train_and_keeps_bins():
    cols = ['fan_out', 'ratio_x', 'cycle_bins_0', 'deg_in']
    rng = np.random.default_rng(0)
    values = (rng.random((80, 4)) * 10).astype(np.float32)
    values[:, 2] = rng.integers(0, 5, size=80).astype(np.float32)
    values[60:, 0] = 1e6
    out = normalise_gfp_sheet(values, cols, n_train=50)
    assert np.array_equal(out[:, 2], values[:, 2])
    assert np.isfinite(out).all()
    assert abs(float(out[:50, 0].mean())) < 0.2
    assert float(out[60:, 0].max()) < 10
    changed = values.copy()
    changed[60:, 0] = 0
    again = normalise_gfp_sheet(changed, cols, n_train=50)
    assert np.allclose(out[:50], again[:50])
    bad = values.copy()
    bad[0, 0] = np.nan
    with pytest.raises(AssertionError):
        normalise_gfp_sheet(bad, cols, n_train=50)


def test_tail_normalise_uses_the_stored_scaler():
    from gnn_frauds.model.tabular.tail import _normalise_gfp
    frame = pd.DataFrame({'a': [0.0, 1.0, 3.0], 'b': [1.0, 2.0, 4.0]})
    logged = np.log1p(frame[['a']].to_numpy())
    scaler = {
        'log_group': {'cols': ['a'], 'log': True, 'p01': float(logged.min()), 'p99': float(logged.max()),
                      'scaler': StandardScaler().fit(logged)},
        'std_group': {'cols': ['b'], 'log': False, 'p01': 1.0, 'p99': 4.0,
                      'scaler': StandardScaler().fit(frame[['b']])},
    }
    out = _normalise_gfp(frame, scaler)
    assert abs(out['a'].mean()) < 1e-5
    assert abs(out['b'].mean()) < 1e-5
    assert list(frame['a']) == [0.0, 1.0, 3.0]  # the input frame is copied, not scaled in place


def test_set_node_encoding_appends_rwpe_and_can_be_cleared(bundle, tmp_path):
    graphs = bundle['graphs']
    assert set_node_encoding(graphs, 'none') == 6
    assert graphs['train'].x.shape[1] == 6
    for split, graph in graphs.items():
        torch.save(torch.rand(graph.num_nodes, 8), tmp_path / f'rwpe_k8_{split}.pt')
    assert set_node_encoding(graphs, 'rwpe8', str(tmp_path)) == 14
    assert graphs['val'].x.shape[1] == 14
    assert set_node_encoding(graphs, 'none') == 6
    with pytest.raises(AssertionError):
        set_node_encoding(graphs, 'node2vec')
    torch.save(torch.full((graphs['train'].num_nodes, 8), 2.0), tmp_path / 'rwpe_k8_train.pt')
    with pytest.raises(AssertionError):
        set_node_encoding(graphs, 'rwpe8', str(tmp_path))


def test_rwpe_on_a_cycle_is_a_return_probability():
    src = np.array([0, 1, 2, 0])
    dst = np.array([1, 2, 0, 0])  # a 3-cycle plus a self-loop on node 0; node 3 is isolated
    plain = compute_rwpe(src, dst, num_nodes=4, walk_length=3, keep_self_loops=False, log=lambda _line: None)
    assert plain.shape == (4, 3)
    assert plain.min() >= 0 and plain.max() <= 1
    assert plain[:3, 0].max() == 0  # self-loops are dropped, so step 1 is 0
    assert np.allclose(plain[:3, 2], 1.0)
    assert plain[3].sum() == 0
    with_loops = compute_rwpe(src, dst, num_nodes=4, walk_length=1, keep_self_loops=True, log=lambda _line: None)
    assert with_loops[0, 0] > 0
    assert peak_ram_gb() > 0


def _write_snapshots(directory, bundle):
    directory.mkdir(parents=True, exist_ok=True)
    for split, graph in bundle['graphs'].items():
        torch.save(graph, directory / f'{split}_graph.pt')
    base = [f'base_{i}' for i in range(20)]
    gfp = [f'gfp_{i}' for i in range(61)]
    meta = {'BASE_EDGE_COLS': base, 'GFP_FEAT_COLS': gfp, 'EDGE_FEAT_COLS': base + gfp,
            'NODE_FEAT_COLS': [f'node_{i}' for i in range(6)]}
    (directory / 'feature_meta.json').write_text(json.dumps(meta), encoding='utf-8')


def test_load_graphs_v0_and_tuned(bundle, tmp_path):
    data_dir = tmp_path / 'data'
    _write_snapshots(data_dir, bundle)
    graphs, col_sets, node_dim = load_graphs(str(data_dir))
    assert node_dim == 6
    assert len(col_sets['base']) == 20 and len(col_sets['gfp']) == 61 and col_sets['none'] == []
    assert graphs['train'].edge_attr.shape[1] == 81
    assert int(graphs['val'].eval_mask.sum()) == 24
    with pytest.raises(AssertionError):
        load_graphs(str(data_dir), gfp_variant='missing')
    with pytest.raises(AssertionError):
        load_graphs(str(data_dir), splits=('val',))

    gfp_dir = tmp_path / 'gfp'
    gfp_dir.mkdir()
    n_all = bundle['graphs']['test'].edge_index.shape[1]
    cols = [f'fan_{i}' for i in range(32)] + [f'ratio_{i}' for i in range(16)] + [f'pat_bins_{i}' for i in range(16)]
    (gfp_dir / 'tuned_cols.json').write_text(json.dumps(cols), encoding='utf-8')
    sheet = np.random.default_rng(1).random((n_all, 64)).astype(np.float32)
    np.save(gfp_dir / 'tuned.npy', sheet)
    tuned, tuned_cols, _ = load_graphs(str(data_dir), gfp_variant='tuned', gfp_dir=str(gfp_dir))
    assert tuned['test'].edge_attr.shape[1] == 84
    assert len(tuned_cols['gfp']) == 64 and len(tuned_cols['full']) == 84
    with pytest.raises(AssertionError):
        load_graphs(str(data_dir), gfp_variant='tuned', gfp_dir=str(gfp_dir), splits=('train', 'val'))


def test_make_trainval_graph_labels_every_edge(bundle):
    graphs = bundle['graphs']
    fit = make_trainval_graph(graphs)
    n_train = graphs['train'].edge_index.shape[1]
    assert bool(fit.eval_mask.all())
    assert torch.equal(fit.y[:n_train], graphs['train'].y)
    assert set(fit.y.unique().tolist()) <= {0, 1}
    graphs['val'].y[0] = 0
    with pytest.raises(AssertionError):
        make_trainval_graph(graphs)
