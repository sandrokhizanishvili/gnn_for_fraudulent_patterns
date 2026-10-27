"""Threshold, the seven metrics, neighbour loaders, and the curve files."""
import numpy as np
import pandas as pd
import pytest
import torch

from gnn_frauds.experiment.metrics import (
    best_f1_threshold, compute_metrics, flatten_metrics, forward_batch, make_loader, plot_curves,
    plot_final_curves, predict, save_predictions, threshold_sensitivity, top_k_metrics, weighted_bce,
)
from gnn_frauds.hyperparameters.gnns import METRICS, SPLITS
from gnn_frauds.model.gnns.classifier import build_model
from tests.support import tiny_recipe


def test_best_f1_threshold_separates_a_clean_score():
    y = np.array([0, 0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
    threshold, f1 = best_f1_threshold(y, p)
    assert threshold == 0.8 and f1 == pytest.approx(1.0)


def test_weighted_bce_penalises_a_missed_positive():
    y = np.array([1.0, 0.0])
    missed = weighted_bce(y, np.array([0.2, 0.2]), pos_weight=8)
    caught = weighted_bce(y, np.array([0.9, 0.2]), pos_weight=8)
    assert caught < missed
    assert np.isfinite(weighted_bce(y, np.array([0.0, 1.0])))  # 0 and 1 are clipped before the log


def test_top_k_and_compute_metrics_use_one_threshold():
    y = np.zeros(20, dtype=int)
    y[5] = 1
    p = np.zeros(20)
    p[5] = 0.9
    top = top_k_metrics(y, p)
    assert top == {'precision_at_5pct': 1.0, 'recall_at_5pct': 1.0}
    y = np.array([0, 0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
    metrics = compute_metrics(y, p, threshold=0.8)
    assert tuple(metrics) == METRICS
    assert metrics['f1'] == metrics['precision'] == metrics['recall'] == 1.0
    assert metrics['roc_auc'] == 1.0


def test_threshold_sensitivity_and_flatten():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.7, 0.9])
    table = threshold_sensitivity(y, p, threshold=0.5)
    assert [row['point'] for row in table] == ['frozen'] * 5 + ['paper 0.5']
    assert table[2]['delta'] == 0.0 and table[2]['threshold'] == 0.5
    metrics = {split: {metric: float(i) for i, metric in enumerate(METRICS)} for split in SPLITS}
    flat = flatten_metrics(metrics)
    assert len(flat) == 21
    assert flat['train_f1'] == 0.0 and flat['test_recall_at_5pct'] == 6.0


def test_loader_forward_and_predictions(bundle, tmp_path):
    graphs, col_sets = bundle['graphs'], bundle['col_sets']
    recipe = tiny_recipe()
    model = build_model('gin', bundle['node_dim'], len(col_sets['base']), len(col_sets['base']), recipe=recipe)
    loader, seed_feats = make_loader(graphs['val'], shuffle=False, mp_idx=col_sets['base'],
                                     readout_idx=col_sets['base'], recipe=recipe)
    assert seed_feats.shape[0] == int(graphs['val'].eval_mask.sum())
    batch = next(iter(loader))
    logits, labels = forward_batch(model, batch, seed_feats)
    assert logits.shape == labels.shape
    assert torch.isfinite(logits).all()
    y, p, positions = predict(model, loader, seed_feats, 'val')
    assert len(y) == len(p) == len(positions) == seed_feats.shape[0]
    assert p.min() >= 0 and p.max() <= 1

    preds = {}
    for split, graph in graphs.items():
        n = int(graph.eval_mask.sum())
        edge_id = graph.eval_mask.nonzero().squeeze(1).numpy()
        prob = np.linspace(0.1, 0.9, n).astype(np.float32)
        preds[split] = {'edge_id': edge_id, 'y': graph.y[graph.eval_mask].numpy(), 'prob': prob}
    save_predictions(str(tmp_path), graphs, preds, threshold=0.5)
    table = pd.read_csv(tmp_path / 'predictions.csv')
    assert list(table.columns) == ['split', 'edge_id', 'src', 'dst', 'edge_time', 'y', 'prob', 'pred']
    assert table['edge_id'].is_unique and len(table) == sum(int(g.eval_mask.sum()) for g in graphs.values())
    assert (table['pred'] == (table['prob'] >= 0.5).astype(int)).all()


def test_curve_files(tmp_path):
    history = [{'epoch': 1, 'train_loss': 0.5, 'val_loss': 0.4, 'train_f1': 0.2, 'val_f1': 0.3,
                'train_pr_auc': 0.2, 'val_pr_auc': 0.25, 'threshold': 0.4},
               {'epoch': 2, 'train_loss': 0.3, 'val_loss': 0.35, 'train_f1': 0.4, 'val_f1': 0.5,
                'train_pr_auc': 0.4, 'val_pr_auc': 0.45, 'threshold': 0.55}]
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.3, 0.6, 0.8])
    path = tmp_path / 'curves.png'
    plot_curves(history, y, p, 0.5, best_epoch=2, run_name='unit', path=str(path))
    assert path.stat().st_size > 1000
    tuning = tmp_path / 'tuning.png'
    plot_curves(history, None, None, None, 2, 'unit', str(tuning), select_metric='pr_auc')
    assert tuning.stat().st_size > 1000
    final = [{'epoch': 1, 'train_loss_batches': 0.4, 'lr': 1e-3},
             {'epoch': 2, 'train_loss_batches': 0.3, 'lr': 5e-4}]
    final_path = tmp_path / 'final.png'
    plot_final_curves(final, y, p, 0.5, 'unit', str(final_path))
    assert final_path.stat().st_size > 1000
    for row in final:
        row.update(train_loss=0.4, val_loss=0.3, train_pr_auc=0.2, val_pr_auc=0.3)
    seen = tmp_path / 'seen.png'
    plot_final_curves(final, y, p, 0.5, 'unit', str(seen))
    assert seen.stat().st_size > 1000
