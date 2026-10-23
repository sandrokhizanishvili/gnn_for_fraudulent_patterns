'''Step "aggregate" (login node; pack_results.sh runs it too): finalise pending trials, check every finished trial and
final run with asserts (the first failed check stops the script with its message), pick each pair's winner once all
its trials are in, and write the summaries.

Per (operator, config): trials_summary.csv and tuning_curves.png (every finished trial), recipe.json (the winner =
highest mean val PR-AUC, tie-break mean val F1, written once all N trials are in; before that the pair prints
"k/N trials, no winner yet"), and the final summaries batch_summary_final.csv / seed_summary_final.csv.
'''
import glob
import json
import math
import os

import pandas as pd

import common
from common import core
from tune import finalise_pending

torch = core.torch


def model_params(op, cfg_id, recipe):
    '''(params, invariant params) of the recipe's model, built on the meta device (no memory, no data needed).'''
    width = {'base': core.BASE_EDGE_DIM, 'full': core.BASE_EDGE_DIM + core.GFP_WIDTH['v0']}
    cfg = common.CONFIGS[cfg_id]
    deg_hist = torch.tensor([0, 10, 5, 2]) if op == 'pna' else None   # the degree values never change the count
    with torch.device('meta'):
        model = core.EdgeClassifier(op, core.BASE_NODE_DIM, width[cfg['mp']], width[cfg['readout']], False, deg_hist, recipe)
    return core.count_params(model), core.invariant_params(model)


def check_trial(trial, space, op, cfg_id):
    '''Every check of one finished trial; an AssertionError names the trial.'''
    tid = trial['trial_id']
    name = f'{op} config {cfg_id} t{tid:03d}'
    tdir = common.trial_dir(op, cfg_id, tid)
    recipe = core.full_recipe(trial['recipe'])
    assert recipe == core.full_recipe(space['trials'][op][tid]['recipe']), f'{name}: recipe differs from trials_space.json'
    for s in core.LUO_TUNING_SEEDS:
        assert json.load(open(f'{tdir}/seed-{s}/done.json'))['test_split_loaded'] is False, f'{name} seed {s}: test loaded'
    if 'val_pr_auc_mean' not in trial:        # out of memory, or diverged before its first epoch: nothing to score
        assert trial['status'] in ('oom', 'diverged'), f"{name}: no metrics but status {trial['status']!r}"
        return
    diverged = trial['diverged_at'] or {}
    histories = {}
    for s in core.LUO_TUNING_SEEDS:
        h = pd.read_csv(f'{tdir}/seed-{s}/history.csv')
        expected = diverged[str(s)] - 1 if str(s) in diverged else core.LUO_MAX_EPOCHS
        assert len(h) == expected, f'{name} seed {s}: {len(h)} epochs in history.csv, expected {expected}'
        assert (h['lr'] == recipe['lr']).all(), f"{name} seed {s}: the learning rate is not constant at {recipe['lr']}"
        assert not h[core.TRIAL_METRIC_KEYS + ['threshold']].isna().any().any(), f'{name} seed {s}: NaN in history.csv'
        histories[s] = h
    best, _, _ = core.seed_mean_best_epoch(histories, 1, 'val_pr_auc')
    assert best == trial['best_epoch'], f"{name}: best epoch {trial['best_epoch']} != {best} (argmax of the seed-mean val PR-AUC)"
    for s, h in histories.items():
        row = h.loc[h['epoch'] == best].iloc[0]
        for key in core.TRIAL_METRIC_KEYS + ['threshold']:
            assert math.isclose(trial['per_seed'][str(s)][key], float(row[key]), abs_tol=1e-12), \
                f"{name} seed {s}: trial.json {key} = {trial['per_seed'][str(s)][key]}, history.csv {row[key]} at epoch {best}"
        assert os.path.exists(f'{tdir}/seed-{s}/best.pt'), f'{name} seed {s}: best.pt missing'
        assert not glob.glob(f'{tdir}/seed-{s}/epoch_*.pt'), f'{name} seed {s}: epoch checkpoints left'
    for key in core.TRIAL_METRIC_KEYS:
        assert math.isfinite(trial[f'{key}_mean']), f'{name}: {key}_mean is not finite'
    params = model_params(op, cfg_id, recipe)
    assert (trial['params'], trial['invariant_params']) == params, \
        f"{name}: params {trial['params']} / {trial['invariant_params']}, the recipe builds {params}"


def check_final_run(results, history, frozen, name, seed):
    '''Every check of one finished final seed.'''
    assert seed in core.LUO_FINAL_SEEDS and frozen['retrain_seeds'] == list(core.LUO_FINAL_SEEDS), f'{name}: seeds'
    assert results['stop_epoch'] == frozen['best_epoch'], f'{name} seed {seed}: stop epoch differs from recipe.json'
    assert results['threshold'] == frozen['thresholds'][str(seed)], f'{name} seed {seed}: threshold differs from recipe.json'
    assert len(history) == frozen['best_epoch'], f'{name} seed {seed}: {len(history)} epochs in history.csv'
    assert not [c for c in history if c.startswith('test')], f'{name} seed {seed}: test scored during training'
    assert (history['lr'] == frozen['recipe']['lr']).all(), f'{name} seed {seed}: the learning rate is not constant'
    assert all(math.isfinite(results[c]) for c in core.METRIC_COLS), f'{name} seed {seed}: a metric is missing or not finite'


def tuning_pair(space, op, cfg_id):
    '''Checks, summary and (once all trials are in) the winner of one (operator, config); returns its trials.'''
    rdir = common.run_dir(op, cfg_id)
    trials = [json.load(open(p)) for p in sorted(glob.glob(f'{rdir}/trials/t*/trial.json'))]
    for trial in trials:
        check_trial(trial, space, op, cfg_id)
    print(f"{op} config {cfg_id}: {len(trials)}/{space['n_trials']} trials, checks passed, "
          f"statuses {pd.Series([t['status'] for t in trials], dtype=object).value_counts().to_dict()}")
    if not trials:
        return
    trial0 = [t for t in trials if t['trial_id'] == 0 and 'val_pr_auc_mean' in t]
    v1 = f'{common.CODE_DIR}/reference_v1/{common.FAMILY[op]}/{core.run_name(op, common.CONFIGS[cfg_id])}/results.json'
    if trial0 and os.path.exists(v1):
        print(f"  trial 0 vs the v1 run (warning only, protocols differ): val PR-AUC {trial0[0]['val_pr_auc_mean']:.4f} "
              f"(2 seeds, 100 epochs, constant LR) vs {json.load(open(v1))['val_pr_auc']:.4f} (1 seed, 20-epoch cosine)")

    recipe_path = f'{rdir}/recipe.json'
    winner = core.choose_best_trial(trials, 'pr_auc') if len(trials) == space['n_trials'] else None
    if winner is None:
        print(f"  {len(trials)}/{space['n_trials']} trials, no winner yet")
    elif os.path.exists(recipe_path):
        assert json.load(open(recipe_path))['source_trial'] == winner['trial'], \
            f"{op} config {cfg_id}: recipe.json is not the trial with the highest mean val PR-AUC ({winner['trial']})"
    else:
        core.save_recipe(recipe_path, op, common.CONFIGS[cfg_id], winner, retrain_seeds=core.LUO_FINAL_SEEDS,
                         extra={'n_trials': space['n_trials'], 'select_metric': 'pr_auc', 'trial_id': winner['trial_id'],
                                'val_pr_auc_mean': winner['val_pr_auc_mean'], 'val_pr_auc_std': winner['val_pr_auc_std'],
                                'best_epoch_in_last_10': winner['best_epoch'] > core.LUO_MAX_EPOCHS - 10})
        if winner['best_epoch'] > core.LUO_MAX_EPOCHS - 10:
            print(f"  WARNING: the winner's best epoch {winner['best_epoch']} is in the last 10 of {core.LUO_MAX_EPOCHS}")
    winners = {core.LUO_STAGE: winner['trial']} if winner else {}
    core.trials_table(trials, winners).to_csv(f'{rdir}/trials_summary.csv', index=False)
    core.plot_val_curves(trials, winners, f'{rdir}/tuning_curves.png', max_labels=10, select_metric='pr_auc')


def final_pair(op, cfg_id):
    '''Checks and summaries of the final retrain seeds of one (operator, config) finished so far.'''
    recipe_path = f'{common.run_dir(op, cfg_id)}/recipe.json'
    if not os.path.exists(recipe_path):
        return
    frozen = core.load_recipe(recipe_path)
    pair_dir = f'{common.final_root(op)}/{core.run_name(op, common.CONFIGS[cfg_id])}'
    rows = []
    for seed in core.LUO_FINAL_SEEDS:
        if os.path.exists(f'{pair_dir}/seed-{seed}/results.json'):
            results = json.load(open(f'{pair_dir}/seed-{seed}/results.json'))
            check_final_run(results, pd.read_csv(f'{pair_dir}/seed-{seed}/history.csv'), frozen, f'{op} config {cfg_id}', seed)
            rows.append({k: v for k, v in results.items() if k not in ('architecture', 'recipe', 'threshold_sensitivity')})
    print(f'  final retrain: {len(rows)}/{len(core.LUO_FINAL_SEEDS)} seeds done, checks passed')
    if rows:
        core.summarize_batch(rows, pair_dir, 'batch_summary_final.csv')
        core.seed_summary(rows, pair_dir, 'seed_summary_final.csv')


def main():
    space = common.load_space()
    finalise_pending(space)
    for op in common.OPERATORS:
        for cfg_id in common.CONFIGS:
            tuning_pair(space, op, cfg_id)
            final_pair(op, cfg_id)
    print('\nall checks passed')


if __name__ == '__main__':
    main()
