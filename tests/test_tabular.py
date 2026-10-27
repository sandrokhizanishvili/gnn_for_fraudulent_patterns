"""MLP and LightGBM on a few dozen rows: shapes, the search pieces, one short fit."""
import json

import numpy as np
import pandas as pd
import pytest
import torch

from gnn_frauds.model.tabular import lightgbm as gbt
from gnn_frauds.model.tabular import mlp
from gnn_frauds.model.tabular.lightgbm import LightGBM
from gnn_frauds.model.tabular.mlp import MLPModel


def _rows(n=36):
    rng = np.random.default_rng(0)
    matrix = rng.normal(size=(n, 6)).astype(np.float32)
    labels = np.zeros(n, dtype=np.int64)
    labels[[1, 5, 14, 18, 25, 30]] = 1
    return {
        'X': matrix, 'y': labels, 'edge_id': np.arange(n),
        'src': np.zeros(n, dtype=np.int64), 'dst': np.ones(n, dtype=np.int64),
        'edge_time': np.arange(n),
        'rows': {'train': slice(0, 12), 'val': slice(12, 24), 'test': slice(24, 36)},
    }


def test_mlp_shapes_scaling_and_draw():
    assert mlp.layer_widths(64, 3, 'flat') == [64, 64, 64]
    assert mlp.layer_widths(64, 3, 'funnel') == [64, 32, 16]
    assert mlp.layer_widths(64, 5, 'funnel') == [64, 32, 16, 16, 16]
    assert mlp.layer_widths(32, 2, 'resnet') == [32, 32]
    with pytest.raises(AssertionError):
        mlp.layer_widths(32, 2, 'pyramid')
    assert mlp.widths_text(93, [64, 32], 'funnel') == '93 -> 64 -> 32 -> 1'
    assert mlp.widths_text(93, [32, 32], 'resnet') == '93 -> 32 (2 residual blocks) -> 1'
    knobs = dict(hidden=16, depth=2, shape='flat', norm='batch', dropout=0.0)
    block = mlp.ResBlock(8, 0.0, 'layer')
    assert block(torch.randn(4, 8)).shape == (4, 8)
    for shape in ('flat', 'funnel', 'resnet'):
        model = mlp.MLP(10, dict(knobs, shape=shape, norm='none'))
        assert model(torch.randn(5, 10)).shape == (5,)

    wide = np.zeros((40, 93), dtype=np.float32)
    wide[:, 0] = np.arange(40)
    wide[:, 81:] = 0
    wide[:20, 81] = 1
    onehot = wide[:, 81:].copy()
    data = {'X': wide, 'rows': {'train': slice(0, 20)}}
    stats = mlp.standardise(data)
    assert abs(float(data['X'][:20, 0].mean())) < 1e-4
    assert np.array_equal(data['X'][:, 81:], onehot)
    assert stats['std'][0] > 1
    tensors, labels = mlp.to_tensors({'X': data['X'], 'y': np.zeros(40, dtype=np.float32)})
    assert tensors.shape == (40, 93) and labels.shape == (40,)

    first = mlp.draw_trials(4, seed=1)
    second = mlp.draw_trials(4, seed=1)
    assert first[0] == mlp.DEFAULT_KNOBS and first == second
    assert len({json.dumps(knob, sort_keys=True) for knob in first.values()}) == 4


def test_mlp_train_run_and_final(tmp_path, monkeypatch):
    data = _rows()
    x = torch.from_numpy(data['X'])
    y = torch.from_numpy(data['y']).float()
    knobs = dict(mlp.DEFAULT_KNOBS, hidden=16, depth=1, shape='flat', norm='none', dropout=0.0,
                 batch_size=8, pos_weight=1.0)
    model, history, best = mlp.train_run(knobs, x, y, data['rows'], seed=1, epochs=2, patience=2)
    assert len(history) == 2 and best['epoch'] in (1, 2)
    assert mlp.predict(model, x, data['rows']['val']).shape == (12,)
    assert mlp.run_name_mlp(4) == 'mlp_readout-full_seed-4'
    original = mlp.train_run

    def short(knobs, x, y, rows, seed, epochs=60, patience=10):
        return original(knobs, x, y, rows, seed, epochs=2, patience=2)

    monkeypatch.setattr(mlp, 'train_run', short)
    row = mlp.train_final(knobs, data, x, y, seed=4, out_root=str(tmp_path))
    assert row['operator'] == 'mlp' and row['invariant_params'] == 0 and row['model_file'] == 'best.pt'
    assert np.isfinite(row['test_f1']) and np.isfinite(row['test_pr_auc'])
    out = tmp_path / row['run']
    assert (out / 'best.pt').exists() and (out / 'predictions.csv').exists() and (out / 'curves.png').exists()
    again = mlp.train_final(knobs, data, x, y, seed=4, out_root=str(tmp_path))
    assert again['test_f1'] == row['test_f1']
    assert MLPModel().name == 'mlp'


def test_lightgbm_search_pieces_and_one_fit(tmp_path):
    rng = np.random.default_rng(0)
    for _ in range(20):
        params = gbt.sample_params(rng)
        for name, (low, high, kind) in gbt.SEARCH_SPACE.items():
            assert low <= params[name] <= high
            if kind == 'int':
                assert isinstance(params[name], int)
    assert gbt.halving_schedule(100, 2, 0.1) == [(100, 0.1), (50, 0.2), (25, 0.4), (12, 0.8), (6, 1.0)]
    assert gbt.halving_schedule(4, 2, 0.5) == [(4, 0.5), (2, 1.0)]
    data = _rows()
    x_train, y_train = gbt.split_arrays(data, 'train')
    x_val, y_val = gbt.split_arrays(data, 'val')
    assert x_train.shape == (12, 6) and y_val.shape == (12,)
    dataset = gbt.make_dataset(x_train, y_train)
    val_dataset = gbt.make_val_dataset(x_val, y_val, dataset)
    params = dict(num_round=8, num_leaves=4, learning_rate=0.1, min_data_in_leaf=1,
                  feature_fraction=1.0, bagging_fraction=1.0, lambda_l1=0.0, lambda_l2=0.0,
                  scale_pos_weight=1.0)
    booster = gbt.fit_booster(params, dataset, val_dataset, seed=1)
    prob = gbt.predict_booster(booster, x_val, num_iteration=booster.best_iteration)
    assert prob.shape == (12,) and prob.min() >= 0 and prob.max() <= 1
    assert gbt.run_name_gbt(3) == 'lightgbm_readout-full_seed-3'
    row = gbt.train_final(params, data, seed=3, out_root=str(tmp_path), check_every=4)
    assert row['operator'] == 'lightgbm' and row['invariant_params'] == 0 and row['mp'] == 'none'
    assert row['params'] == row['best_epoch'] and row['model_file'] == 'best.txt'
    assert np.isfinite(row['val_pr_auc']) and np.isfinite(row['test_f1'])
    out = tmp_path / row['run']
    saved = json.loads((out / 'results.json').read_text(encoding='utf-8'))
    assert 'hyperparams' in saved and 'paper_comparison' in saved
    predictions = pd.read_csv(out / 'predictions.csv')
    assert len(predictions) == 36 and predictions['edge_id'].is_unique
    again = gbt.train_final(params, data, seed=3, out_root=str(tmp_path), check_every=4)
    assert again['test_f1'] == row['test_f1']
    assert LightGBM().name == 'lightgbm'
    assert callable(LightGBM().tune) and callable(MLPModel().retrain)
