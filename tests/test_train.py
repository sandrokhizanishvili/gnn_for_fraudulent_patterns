"""The selecting loop, resume, run names, and one end-to-end grid run."""
import json

import numpy as np
import pandas as pd
import pytest
import torch

from gnn_frauds.experiment.metrics import make_loaders
from gnn_frauds.experiment.train import (
    add_full_metrics, load_resume, make_scheduler, result_row, rng_state, run_experiment, run_name,
    save_resume, save_run, seed_summary, set_rng_state, smoothed, summarize_batch, train_one_run,
    write_json_atomic,
)
from gnn_frauds.hyperparameters.gnns import METRICS, METRIC_COLS
from gnn_frauds.model.gnns.classifier import build_model, count_params, invariant_params
from tests.support import tiny_recipe


def test_smoothed_and_schedulers():
    flat = smoothed([1.0, 1.0, 1.0])
    assert np.allclose(flat, 1.0) and len(smoothed([1.0, 2.0, 3.0, 4.0])) == 4
    recipe = tiny_recipe(epochs=2, lr_schedule='cosine', lr=0.1)
    cosine = torch.optim.Adam([torch.nn.Parameter(torch.zeros(1))], lr=0.1)
    cosine_sched = make_scheduler(cosine, recipe)
    cosine_sched.step()
    assert cosine.param_groups[0]['lr'] < 0.1
    constant = torch.optim.Adam([torch.nn.Parameter(torch.zeros(1))], lr=0.1)
    constant_sched = make_scheduler(constant, tiny_recipe(lr_schedule='constant', lr=0.1))
    constant_sched.step()
    assert constant.param_groups[0]['lr'] == 0.1


def test_atomic_json_and_rng_roundtrip(tmp_path):
    path = tmp_path / 'row.json'
    write_json_atomic(str(path), {'a': 1})
    assert json.loads(path.read_text(encoding='utf-8')) == {'a': 1}
    assert not list(tmp_path.glob('*.tmp*'))
    torch.manual_seed(1)
    state = rng_state()
    expected = torch.rand(3)
    set_rng_state(state)
    assert torch.equal(expected, torch.rand(3))


def test_resume_file_restores_weights(tmp_path):
    model = torch.nn.Linear(4, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    scheduler = make_scheduler(optimizer, tiny_recipe(epochs=2))
    before = model.weight.detach().clone()
    path = tmp_path / 'resume.pt'
    save_resume(str(path), model, optimizer, scheduler, {'history': [{'epoch': 1}], 'stopped': False})
    model.weight.data.add_(1)
    state = load_resume(str(path), model, optimizer, scheduler)
    assert state['history'] == [{'epoch': 1}]
    assert torch.equal(model.weight.detach(), before)


def test_run_name_result_row_and_summaries(tmp_path):
    assert run_name('gin', {'mp': 'base', 'readout': 'full'}) == 'gin_mp-base_readout-full_dir-in'
    assert run_name('gin', {'mp': 'base', 'readout': 'full'}, node_enc='rwpe16', gfp_variant='tuned') == \
        'gin_mp-base_readout-full_dir-in_enc-rwpe16_gfp-tuned'
    with pytest.raises(AssertionError):
        run_name('gin', {'mp': 'base', 'readout': 'base'}, temporal=True)
    metrics = {split: {metric: 0.1 for metric in METRICS} for split in ('train', 'val', 'test')}
    row = result_row('run', 'gin', {'mp': 'base', 'readout': 'base'}, 3, 0.4, 10, 9, metrics, seed=7)
    assert row['seed'] == 7 and row['invariant_params'] == 9 and row['test_f1'] == 0.1
    out = tmp_path / 'run'
    out.mkdir()
    save_run(str(out), row, tiny_recipe())
    saved = json.loads((out / 'results.json').read_text(encoding='utf-8'))
    assert saved['recipe']['hidden'] == 8 and saved['architecture']['layers'] == 1
    row.pop('val_precision')
    add_full_metrics(row, 'val', np.array([0, 1]), np.array([0.2, 0.8]), 0.5)
    assert row['val_f1'] == 0.1  # the value already stored by the sweep is kept
    assert row['val_precision'] == 1.0
    rows = [dict(row, invariant_params=9), dict(row, run='other', invariant_params=9)]
    summary = summarize_batch(rows, str(tmp_path), filename='batch.csv')
    assert list(summary['invariant_params'].unique()) == [9]
    assert (tmp_path / 'batch.csv').exists()
    with pytest.raises(AssertionError):
        summarize_batch([dict(row, invariant_params=1), dict(row, invariant_params=2)], str(tmp_path))
    seeds = [dict(row, seed=1, test_f1=0.2), dict(row, seed=2, test_f1=0.4)]
    for extra in METRIC_COLS:
        for item, value in zip(seeds, (0.2, 0.4)):
            item.setdefault(extra, value)
    table = seed_summary(seeds, str(tmp_path), filename='seeds.csv')
    assert table.loc[0, 'n_seeds'] == 2 and table.loc[0, 'test_f1_mean'] == pytest.approx(0.3)


def test_train_one_run_resumes(bundle, tmp_path):
    graphs, col_sets = bundle['graphs'], bundle['col_sets']
    recipe = tiny_recipe(epochs=2)
    model = build_model('gin', bundle['node_dim'], len(col_sets['base']), len(col_sets['full']), recipe=recipe)
    loaders = make_loaders(graphs, col_sets['base'], col_sets['full'], recipe)
    out = tmp_path / 'select'
    out.mkdir()
    resume = str(out / 'resume.pt')
    first, _, _ = train_one_run(model, loaders, str(out), recipe, stop_epoch=1, resume_path=resume, full_metrics=True)
    assert [row['epoch'] for row in first] == [1]
    assert 'val_precision' in first[0]
    second, best_epoch, threshold = train_one_run(
        model, loaders, str(out), recipe, stop_epoch=2, resume_path=resume, full_metrics=True)
    assert [row['epoch'] for row in second] == [1, 2]
    assert best_epoch in (1, 2) and 0 < threshold < 1
    assert (out / 'best.pt').exists() and (out / 'history.csv').exists()


def test_run_experiment_scores_every_split_once(bundle, tmp_path):
    cfg = {'mp': 'base', 'readout': 'full'}
    row = run_experiment('gin', cfg, bundle['graphs'], bundle['col_sets'], bundle['node_dim'], str(tmp_path),
                         recipe=tiny_recipe(), seed=7)
    assert row['run'] == 'gin_mp-base_readout-full_dir-in'
    assert row['seed'] == 7 and row['best_epoch'] == 1 and row['model_file'] == 'best.pt'
    assert row['params'] == count_params(build_model(
        'gin', bundle['node_dim'], 20, 81, recipe=tiny_recipe()))
    assert row['invariant_params'] == invariant_params(build_model(
        'gin', bundle['node_dim'], 20, 81, recipe=tiny_recipe()))
    for split in ('train', 'val', 'test'):
        assert np.isfinite(row[f'{split}_f1']) and np.isfinite(row[f'{split}_pr_auc'])
    out = tmp_path / row['run']
    assert (out / 'results.json').exists() and (out / 'history.csv').exists()
    assert (out / 'curves.png').stat().st_size > 1000 and (out / 'best.pt').exists()
    predictions = pd.read_csv(out / 'predictions.csv')
    n_eval = sum(int(graph.eval_mask.sum()) for graph in bundle['graphs'].values())
    assert len(predictions) == n_eval and predictions['edge_id'].is_unique
