"""Luo search: the registered grid shuffle, the memory probe, and one-trial finalisation.

The candidate lists are read from conf/hyperparameters/gnns/luo.yaml.
n_trials, the draw seed and the seeds are read from conf/optimization/luo.yaml.
"""
import gc
import glob
import itertools
import json
import os
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
import torch
import torch.nn as nn

from gnn_frauds.dataset.hi_small import in_degree_histogram
from gnn_frauds.experiment.final import save_recipe
from gnn_frauds.experiment.metrics import (
    compute_metrics, forward_batch, plot_curves, predict, weighted_bce,
)
from gnn_frauds.experiment.train import (
    add_full_metrics, build_model, count_params, full_recipe, invariant_params, make_loaders,
    print_model_header, result_row, run_name, save_run, seed_summary, set_seed, smoothed,
    train_one_run, write_json_atomic,
)
from gnn_frauds.hyperparameters.gnns import (
    EDGE_BEYOND, GFP_WIDTH, METRICS, NEIGHBOURS_BY_DEPTH, NEIGHBOURS_FALLBACK, NORMS, OPERATORS,
    RESIDUALS, SEED, SPLITS, TRIAL_METRIC_KEYS, VAL_PREVALENCE_NOTE, recipe_tag, with_knob,
)
from gnn_frauds.hyperparameters.gnns import conf_file
from gnn_frauds.model.gnns.classifier import device

import yaml as _yaml

_hp = _yaml.safe_load(conf_file('hyperparameters/gnns/luo.yaml').read_text(encoding='utf-8'))
_opt = _yaml.safe_load(conf_file('optimization/luo.yaml').read_text(encoding='utf-8'))
LUO_SPACE = {key: list(values) for key, values in _hp['space'].items()}
LUO_DEPTHS = tuple(int(d) for d in _hp['depths'])
LUO_MAX_EPOCHS = int(_hp['fixed']['epochs'])
LUO_DRAW_SEED = int(_opt['draw_seed'])
LUO_TUNING_SEEDS = tuple(int(s) for s in _opt['tuning_seeds'])
LUO_FINAL_SEEDS = tuple(int(s) for s in _opt['final_seeds'])
LUO_STAGE = 'luo2024'
LUO_PROTOCOL = dict(epochs=LUO_MAX_EPOCHS, lr_schedule=_hp['fixed']['lr_schedule'],
                    select_metric=_hp['fixed']['select_metric'])
LUO_HEAVIEST = dict(_opt['probe'])
LUO_MEM_FRACTION = float(_opt['mem_fraction'])
LOADER_WORKER_SETTINGS = tuple(int(w) for w in _opt['loader_workers'])
LUO_N_TRIALS = int(_opt['n_trials'])

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
    depth_rng = random.Random(seed + 1)  # own stream, so the cell order does not depend on an operator's allowed depths
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
    best_epoch, _, _ = seed_mean_best_epoch(histories, 1, f'val_{metric}')  # window 1: the raw mean, first epoch on a tie
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


class LuoSearch:
    """The registered grid shuffle. Hydra builds this from conf/optimization/luo.yaml."""

    def __init__(self, n_trials, draw_seed, tuning_seeds, final_seeds, mem_fraction, loader_workers, probe):
        self.n_trials = int(n_trials)
        self.draw_seed = int(draw_seed)
        self.tuning_seeds = tuple(int(s) for s in tuning_seeds)
        self.final_seeds = tuple(int(s) for s in final_seeds)
        self.mem_fraction = float(mem_fraction)
        self.loader_workers = tuple(int(w) for w in loader_workers)
        self.probe = dict(probe)

    def trial(self, run):
        """Trial 0 is the fixed point. A later index is that cell of the registered draw."""
        from omegaconf import OmegaConf
        fixed = OmegaConf.to_container(run.hyperparameters.fixed, resolve=True)
        index = getattr(run, 'trial_index', None)
        if index in (None, 0):
            return dict(fixed)
        raise AssertionError('a drawn trial is read from trials_space.json by cineca/luo.py, not redrawn here')
