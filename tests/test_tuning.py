"""Luo draw, trial scoring, the frozen recipe, and one short retrain."""
import json

import numpy as np
import pandas as pd
import pytest

from gnn_frauds.experiment.final import final_scoring, load_recipe, run_final, save_recipe
from gnn_frauds.hyperparameters.gnns import METRICS
from gnn_frauds.optimization.luo import (
    choose_best_trial, choose_winner, delete_trial_checkpoints, draw_luo_trials, finalise_trial,
    is_out_of_memory, load_trials, luo_recipe, neighbour_candidates, neighbours_from_probe,
    plot_val_curves, probe_attempt, process_rss_gb, run_tuning_seed, score_trial, summarize_stage,
    trials_table,
)
from tests.support import tiny_recipe


def test_luo_recipe_and_draw_are_registered():
    base = luo_recipe()
    assert base['epochs'] == 100 and base['lr_schedule'] == 'constant' and base['select_metric'] == 'pr_auc'
    assert base['hidden'] == 128 and base['residual'] == 'identity'
    allowed = {'gin': [1, 2], 'gat': [2]}
    neighbours = {'gin': {1: [100], 2: [100, 100]}, 'gat': {2: [100, 100]}}
    workers = {'gin': 0, 'gat': 0}
    first = draw_luo_trials(3, allowed, neighbours, workers, seed=1)
    second = draw_luo_trials(3, allowed, neighbours, workers, seed=1)
    assert [row['id'] for row in first['gin']] == [0, 1, 2]
    assert first['gin'][0]['tag'] == 'v1'
    assert first['gin'][0]['recipe']['select_metric'] == 'pr_auc'
    assert first['gin'][1]['recipe'] == second['gin'][1]['recipe']
    assert first['gin'][1]['recipe']['hidden'] == first['gat'][1]['recipe']['hidden']
    with pytest.raises(AssertionError):
        draw_luo_trials(0, allowed, neighbours, workers)


def test_winner_rules_and_neighbour_lists():
    incumbent = {'trial': 'base', 'val_f1_mean': 0.50, 'val_f1_std': 0.01}
    close = {'trial': 'close', 'val_f1_mean': 0.505, 'val_f1_std': 0.0}
    clear = {'trial': 'clear', 'val_f1_mean': 0.53, 'val_f1_std': 0.0}
    assert choose_winner([incumbent, close], 'base')['trial'] == 'base'
    assert choose_winner([incumbent, clear], 'base')['trial'] == 'clear'
    scored = [
        {'trial': 'low', 'val_pr_auc_mean': 0.2, 'val_f1_mean': 0.9},
        {'trial': 'high', 'val_pr_auc_mean': 0.5, 'val_f1_mean': 0.1},
        {'trial': 'oom'},
    ]
    assert choose_best_trial(scored)['trial'] == 'high'
    tied = [
        {'trial': 'b', 'val_pr_auc_mean': 0.5, 'val_f1_mean': 0.4},
        {'trial': 'a', 'val_pr_auc_mean': 0.5, 'val_f1_mean': 0.4},
    ]
    assert choose_best_trial(tied)['trial'] == 'a'
    with pytest.raises(AssertionError):
        choose_best_trial([{'trial': 'oom'}])
    assert [label for label, _ in neighbour_candidates(2)] == ['primary']
    assert [label for label, _ in neighbour_candidates(4)] == ['primary', 'fallback']
    table = neighbours_from_probe({'2': {'neighbors': [100, 100]}, '4': {'neighbors': [10, 10, 10, 10]},
                                   '6': {'neighbors': None}, 'note': {}})
    assert table[2] == [100, 100] and table[4] == [10, 10, 10, 10] and 6 not in table


def test_memory_helpers_and_checkpoint_cleanup(tmp_path):
    assert is_out_of_memory(RuntimeError('CUDA out of memory'))
    assert is_out_of_memory(RuntimeError('ALLOC_FAILED'))
    assert not is_out_of_memory(ValueError('nope'))
    assert 0 < process_rss_gb() < 100
    seed_dir = tmp_path / 'seed-1'
    seed_dir.mkdir()
    for name in ('epoch_001.pt', 'resume.pt', 'best.pt', 'keep.tmp1'):
        (seed_dir / name).write_bytes(b'x')
    delete_trial_checkpoints(str(tmp_path), seeds=(1,))
    assert not (seed_dir / 'epoch_001.pt').exists()
    assert not (seed_dir / 'resume.pt').exists()
    assert (seed_dir / 'best.pt').exists()


def _history(peaks, threshold):
    rows = []
    for epoch, score in peaks.items():
        row = {'epoch': epoch, 'threshold': threshold}
        for split in ('train', 'val'):
            for metric in list(METRICS) + ['loss']:
                row[f'{split}_{metric}'] = score if metric == 'f1' else 0.5
        rows.append(row)
    return pd.DataFrame(rows)


def test_score_trial_freezes_the_mean_threshold(tmp_path):
    trial_dir = tmp_path / 'd0.2'
    peaks = {1: 0.2, 2: 0.9, 3: 0.4}
    for seed, threshold in ((7, 0.3), (8, 0.5)):
        seed_dir = trial_dir / f'seed-{seed}'
        seed_dir.mkdir(parents=True)
        _history(peaks, threshold).to_csv(seed_dir / 'history.csv', index=False)
        (seed_dir / 'epoch_002.pt').write_bytes(b'x')
    trial = score_trial('gin', {'mp': 'base', 'readout': 'base'}, None, None, None, str(trial_dir),
                        None, (7, 8), 'stage', {'params': 10, 'invariant_params': 9, 'seconds_per_epoch': 0.1},
                        window=1, from_history=True)
    assert trial['best_epoch'] == 2 and trial['val_f1_mean'] == pytest.approx(0.9)
    assert not (trial_dir / 'seed-7' / 'epoch_002.pt').exists()
    frozen = save_recipe(str(tmp_path / 'recipe.json'), 'gin', {'mp': 'base', 'readout': 'base'}, trial,
                         retrain_seeds=(7, 8))
    loaded = load_recipe(str(tmp_path / 'recipe.json'))
    assert loaded['best_epoch'] == 2
    assert loaded['thresholds']['7'] == loaded['thresholds']['8'] == pytest.approx(0.4)
    assert frozen['recipe']['hidden'] == 128
    with pytest.raises(AssertionError):
        load_recipe(str(tmp_path / 'missing.json'))


def test_finalise_records_oom_and_waits_for_both_seeds(tmp_path):
    trial_dir = tmp_path / 't000'
    (trial_dir / 'seed-42').mkdir(parents=True)
    (trial_dir / 'seed-42' / 'done.json').write_text(
        json.dumps({'status': 'oom', 'diverged_at': None, 'epochs_run': 0}), encoding='utf-8')
    assert finalise_trial('gin', {'mp': 'base', 'readout': 'base'}, str(trial_dir), tiny_recipe()) is None
    (trial_dir / 'seed-43').mkdir()
    (trial_dir / 'seed-43' / 'done.json').write_text(
        json.dumps({'status': 'ok', 'diverged_at': None, 'epochs_run': 0}), encoding='utf-8')
    trial = finalise_trial('gin', {'mp': 'base', 'readout': 'base'}, str(trial_dir), tiny_recipe())
    assert trial['status'] == 'oom' and 'val_f1_mean' not in trial
    again = finalise_trial('gin', {'mp': 'base', 'readout': 'base'}, str(trial_dir), tiny_recipe())
    assert again['status'] == 'oom'


def test_trials_table_and_curves(tmp_path):
    trial = {'stage': 's0', 'trial': 'base', 'recipe': tiny_recipe(), 'seeds': [1], 'best_epoch': 1,
             'epochs_run': {'1': 1}, 'reached_max': False, 'thresholds': {'1': 0.4},
             'val_f1_mean': 0.5, 'val_f1_std': 0.0, 'train_f1_mean': 0.6, 'train_f1_std': 0.0,
             'val_pr_auc_mean': 0.4, 'val_pr_auc_std': 0.0, 'gap_f1': 0.1, 'status': 'ok',
             'val_f1_mean_curve': [0.2, 0.5], 'params': 10, 'invariant_params': 9}
    stage = tmp_path / 's0'
    stage.mkdir()
    (stage / 'trial.json').write_text(json.dumps(trial), encoding='utf-8')
    table = trials_table(load_trials(str(stage)), {'s0': 'base'})
    assert bool(table.loc[0, 'winner']) and table.loc[0, 'hidden'] == 8
    summary = summarize_stage(str(stage), 'base')
    assert len(summary) == 1 and (stage / 'stage_summary.csv').exists()
    path = tmp_path / 'curves.png'
    plot_val_curves([trial], {'s0': 'base'}, str(path))
    assert path.stat().st_size > 1000


def test_probe_and_tuning_seed_and_final_retrain(bundle, tmp_path):
    graphs, col_sets = bundle['graphs'], bundle['col_sets']
    cfg = {'mp': 'base', 'readout': 'base'}
    recipe = tiny_recipe()
    numbers = probe_attempt('gin', cfg, graphs, col_sets, bundle['node_dim'], recipe, n_warmup=0, n_timed=1)
    assert numbers['params'] > 0 and numbers['peak_rss_gb'] > 0
    assert 'val_predict_s' in numbers
    info = run_tuning_seed('gin', cfg, graphs, col_sets, bundle['node_dim'], str(tmp_path / 'trial'),
                           recipe, seed=3, eval_train_every=1, patience=None, min_epochs=1)
    assert info['epochs_run'] == 1 and info['invariant_params'] > 0
    assert (tmp_path / 'trial' / 'seed-3' / 'history.csv').exists()
    assert not (tmp_path / 'trial' / 'seed-3' / 'predictions.csv').exists()
    frozen = {'recipe': recipe, 'best_epoch': 1, 'thresholds': {'5': 0.4}}
    row = run_final('gin', cfg, graphs, col_sets, bundle['node_dim'], str(tmp_path), frozen, seed=5,
                    cfg_threshold=0.55)
    assert row['phase'] == 'retrain' and row['threshold'] == 0.55
    assert row['transfer_threshold'] == 0.4 and 'seen_test_f1' in row and row['model_file'] == 'final.pt'
    out = tmp_path / row['run'] / 'seed-5'
    assert (out / 'final.pt').exists() and (out / 'sensitivity.csv').exists()
    history = pd.read_csv(out / 'history.csv')
    assert list(history['epoch']) == [1] and 'train_loss_batches' in history and 'lr' in history
    again = run_final('gin', cfg, graphs, col_sets, bundle['node_dim'], str(tmp_path), frozen, seed=5,
                      cfg_threshold=0.55)
    assert again['test_f1'] == row['test_f1']


def test_final_scoring_views():
    preds = {}
    for split in ('train', 'val', 'test'):
        preds[split] = {'y': np.array([0, 0, 1, 1]), 'prob': np.array([0.1, 0.2, 0.7, 0.9])}
    metrics, extra, sensitivity = final_scoring(preds, threshold=0.5, transfer_threshold=0.6)
    assert metrics['test']['f1'] == 1.0
    assert extra['transfer_threshold'] == 0.6 and 'seen_test_f1' in extra
    assert sensitivity[-1]['point'] == 'val seen sweep'
    assert any(row['point'] == 'tuned-config threshold' for row in sensitivity)
    _, extra_only, _ = final_scoring(preds, 0.5, None)
    assert 'transfer_threshold' not in extra_only and 'val_seen_threshold' in extra_only
