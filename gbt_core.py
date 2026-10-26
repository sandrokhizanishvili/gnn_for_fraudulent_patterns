'''Shared code for the GBT cross-check: LightGBM on the same inputs as the GNNs, no message passing.

How a number is computed (threshold sweep, the seven metrics, the loss diagnostic, the curves, the batch
summary) is imported from gnn_core, so GBT and GNN rows are comparable by construction. This file adds the
GBT-specific parts: the 93-column row built from the graph files, the successive-halving search (Altman et al.
2023, Appendix C) with early stopping on the weighted validation loss, the final fits with boosting-round
checkpoints, the result files of CLAUDE.md section 6, and the tail-window side check at the end.
'''
import gc
import json
import os
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

import gnn_core as core

# ---------------- fixed recipe of the cross-check (EXPERIMENTS.md appendix), never tuned per run ----------------
SEEDS        = (42, 43, 44, 45)       # final models; 42 is the run compared with the single-seed GNN runs
X0, ETA, R0  = 100, 2, 0.1            # successive halving: start configs, keep 1/ETA per round, first train fraction
CHECK_EVERY  = 25                     # boosting rounds between validation checkpoints of a final fit
THREADS      = os.cpu_count()          # all logical cores
FEATURE_SET  = 'full'                 # the only readout set of the GBT: all 81 edge + 6 + 6 node columns
N_EDGES, N_TRAIN, N_VAL_END = 5_077_237, 3_046_342, 4_061_789   # Progress_Report.md section 4
POSITIVES    = {'train': 2_297, 'val': 1_082, 'test': 1_143}
PAPER_THRESHOLD = 0.5                 # the paper's decision rule, stored for comparison only

# every fit stops when LightGBM's binary_logloss on the validation Dataset (laundering rows weighted POS_WEIGHT,
# so its minimum is gnn_core.weighted_bce's) has not improved for EARLY_STOPPING rounds and keeps its lowest-loss
# round; search trials are ranked by SELECT at that round (threshold-free)
EARLY_STOPPING = 100
SELECT         = 'val_pr_auc'

# search space (name: low, high, how to sample): small trees, many rows per leaf, strong L2, slow learning;
# num_round is only a cap, early stopping decides the kept round
SEARCH_SPACE = {
    'num_round':        (2000, 2000,  'int'),
    'num_leaves':       (7,    63,    'int'),
    'learning_rate':    (0.01, 0.05,  'log'),
    'min_data_in_leaf': (200,  2000,  'int'),
    'feature_fraction': (0.3,  0.8,   'float'),
    'bagging_fraction': (0.5,  1.0,   'float'),
    'lambda_l1':        (0.01, 10.0,  'log'),
    'lambda_l2':        (1.0,  100.0, 'log'),
    'scale_pos_weight': (1.0,  10.0,  'float'),   # the paper's range
}
FIXED_PARAMS = {'objective': 'binary', 'bagging_freq': 1, 'verbose': -1, 'num_threads': THREADS}
DATASET_PARAMS = {'feature_pre_filter': False}   # lets one constructed Dataset serve trials with different min_data_in_leaf
TRIAL_COLS = ['trial', 'round', 'fraction', 'n_rows', 'val_f1', 'threshold', 'val_pr_auc', 'rounds', 'seconds', 'params']


# ==================================================================================================
# data: one row per edge, built from the graph files
# ==================================================================================================

def load_gbt_data(data_dir):
    '''The GBT matrix: edge_attr (81) || x[src] (6) || x[dst] (6) for every edge, plus labels and ids.

    Features, src/dst, time come from test_graph.pt (all edges, time order); the labels of the train
    and val windows come from their own snapshots (context edges carry -1). Returns a dict with
    X, y, edge_id, src, dst, edge_time, cols, rows {split: slice}.
    '''
    meta = json.load(open(f'{data_dir}/feature_meta.json'))
    edge_cols, node_cols = meta['EDGE_FEAT_COLS'], meta['NODE_FEAT_COLS']
    assert len(edge_cols) == 81 and len(node_cols) == 6
    rows = {'train': slice(0, N_TRAIN), 'val': slice(N_TRAIN, N_VAL_END), 'test': slice(N_VAL_END, N_EDGES)}

    # labels: each snapshot carries the true label only on its evaluated rows
    y = np.full(N_EDGES, -1, dtype=np.int64)
    for split in ('train', 'val'):
        graph = torch.load(f'{data_dir}/{split}_graph.pt', weights_only=False)
        eval_rows = graph.eval_mask.nonzero().squeeze(1).numpy()
        assert eval_rows[0] == rows[split].start and eval_rows[-1] == rows[split].stop - 1
        y[rows[split]] = graph.y[eval_rows].numpy()
        del graph
        gc.collect()

    graph = torch.load(f'{data_dir}/test_graph.pt', weights_only=False)
    n_edges = graph.edge_index.shape[1]
    assert n_edges == N_EDGES and int(n_edges * 0.6) == N_TRAIN and int(n_edges * 0.8) == N_VAL_END
    assert graph.edge_attr.shape[1] == len(edge_cols) and graph.x.shape[1] == len(node_cols)
    eval_rows = graph.eval_mask.nonzero().squeeze(1).numpy()
    assert eval_rows[0] == N_VAL_END and len(eval_rows) == N_EDGES - N_VAL_END
    y[rows['test']] = graph.y[eval_rows].numpy()
    assert (y >= 0).all(), 'every edge must carry a 0/1 label'

    src, dst = graph.edge_index[0].numpy(), graph.edge_index[1].numpy()
    x = graph.x.numpy().astype(np.float32)
    X = np.empty((n_edges, len(edge_cols) + 2 * len(node_cols)), dtype=np.float32)
    X[:, :81] = graph.edge_attr.numpy()
    X[:, 81:87] = x[src]
    X[:, 87:93] = x[dst]
    assert not np.isnan(X).any()
    cols = edge_cols + [f'src_{c}' for c in node_cols] + [f'dst_{c}' for c in node_cols]
    data = {'X': X, 'y': y, 'edge_id': np.arange(n_edges), 'src': src, 'dst': dst,
            'edge_time': graph.edge_time.numpy(), 'cols': cols, 'rows': rows}
    del graph, x
    gc.collect()

    print(f'GBT matrix {X.shape} float32 ({X.nbytes / 2**30:.2f} GB): {len(edge_cols)} edge + '
          f'{len(node_cols)} src + {len(node_cols)} dst columns')
    for split, rows_ in rows.items():
        n_pos = int(y[rows_].sum())
        assert n_pos == POSITIVES[split], f'{split}: {n_pos} laundering rows, expected {POSITIVES[split]}'
        print(f'{split:<5} rows {rows_.start:>9,} – {rows_.stop - 1:>9,}  n={rows_.stop - rows_.start:,}  '
              f'laundering={n_pos:,} ({n_pos / (rows_.stop - rows_.start):.4%})')
    return data


def split_arrays(data, split):
    '''(X, y) of one split, as views on the full matrix.'''
    rows = data['rows'][split]
    return data['X'][rows], data['y'][rows]


# ==================================================================================================
# LightGBM wrappers
# ==================================================================================================

def sample_params(rng, space=SEARCH_SPACE):
    '''One random configuration from the search space (ints uniform, floats uniform, log ranges log-uniform).'''
    params = {}
    for name, (low, high, kind) in space.items():
        if kind == 'int':
            params[name] = int(rng.integers(low, high + 1))
        elif kind == 'log':
            params[name] = float(10 ** rng.uniform(np.log10(low), np.log10(high)))
        else:
            params[name] = float(rng.uniform(low, high))
    return params


def make_dataset(X, y):
    '''A constructed LightGBM Dataset (binned once, reused by every fit on the same rows).'''
    return lgb.Dataset(X, label=y, params=dict(DATASET_PARAMS), free_raw_data=False).construct()


def make_val_dataset(X_val, y_val, train_dataset):
    '''The validation Dataset for early stopping: laundering rows weighted by POS_WEIGHT, so LightGBM's
    binary_logloss on it has the same minimum as gnn_core.weighted_bce (the loss the curves show).'''
    weight = np.where(y_val == 1, core.POS_WEIGHT, 1.0)
    return lgb.Dataset(X_val, label=y_val, weight=weight, reference=train_dataset, params=dict(DATASET_PARAMS),
                       free_raw_data=False).construct()


def fit_booster(params, dataset, val_dataset, seed, callbacks=None):
    '''lgb.train with the fixed recipe + one sampled configuration, early stopping on the weighted
    validation logloss (patience EARLY_STOPPING); num_round (from params) is the cap.
    booster.best_iteration is the round with the lowest validation loss.'''
    lgb_params = dict(FIXED_PARAMS)
    lgb_params.update({k: v for k, v in params.items() if k != 'num_round'})
    lgb_params.update(DATASET_PARAMS)
    lgb_params['seed'] = seed
    lgb_params['metric'] = 'binary_logloss'
    callbacks = list(callbacks or []) + [lgb.early_stopping(EARLY_STOPPING, first_metric_only=True, verbose=False)]
    return lgb.train(lgb_params, dataset, num_boost_round=params['num_round'], valid_sets=[val_dataset],
                     callbacks=callbacks)


def predict_booster(booster, X, num_iteration=None):
    '''Probabilities of the positive class from the first num_iteration trees (all if None).'''
    return booster.predict(X, num_iteration=num_iteration, num_threads=THREADS).astype(np.float32)


# ==================================================================================================
# successive halving (Altman et al. 2023, Appendix C) — train rows only, scored on validation
# ==================================================================================================

def halving_schedule(x0, eta, r0):
    '''[(configs alive, train fraction)] per round: x0 @ r0, x0/eta @ r0*eta, ... until the fraction is 1.'''
    rounds, n_alive, fraction = [], x0, r0
    while True:
        rounds.append((n_alive, min(fraction, 1.0)))
        if fraction >= 1.0 or n_alive // eta < 1:
            break
        n_alive, fraction = n_alive // eta, fraction * eta
    return rounds


def successive_halving(data, out_dir, x0=X0, eta=ETA, r0=R0, seed=core.SEED):
    '''Pick the LightGBM configuration: random configs, each round keeps the best 1/eta on more train rows.

    Every round fits on the most recent `fraction` of the train rows with early stopping on the
    weighted validation loss and scores the full validation set at the kept round (test is never
    touched); trials are ranked by SELECT (validation PR-AUC). One line per (trial, round) goes to
    <out_dir>/trials.csv as soon as it finishes, so a stopped search resumes where it was. Returns
    (best params, trials table); best = highest SELECT of the last round, ties -> lower trial id.
    '''
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    configs = {trial: sample_params(rng) for trial in range(x0)}
    config_path = f'{out_dir}/configs.json'
    if os.path.exists(config_path):
        saved = {int(k): v for k, v in json.load(open(config_path)).items()}
        assert saved == configs, 'configs.json does not match the seeded sample; delete the search folder to restart'
    else:
        json.dump(configs, open(config_path, 'w'), indent=1)

    log_path = f'{out_dir}/trials.csv'
    trials = pd.read_csv(log_path) if os.path.exists(log_path) else pd.DataFrame(columns=TRIAL_COLS)
    X_train, y_train = split_arrays(data, 'train')
    X_val, y_val = split_arrays(data, 'val')
    n_train = len(y_train)
    schedule = halving_schedule(x0, eta, r0)
    print(f'successive halving: x0={x0} eta={eta} r0={r0} -> rounds {schedule} | ranked by {SELECT}'
          f' | early stopping on weighted val loss, patience {EARLY_STOPPING}'
          f' | {len(trials)} trial rows already logged')

    alive = list(range(x0))
    for round_no, (n_alive, fraction) in enumerate(schedule, 1):
        alive = alive[:n_alive]
        n_rows = int(round(fraction * n_train))
        start = n_train - n_rows                     # the most recent rows of the train window
        todo = [t for t in alive if not ((trials['trial'] == t) & (trials['round'] == round_no)).any()]
        print(f'--- round {round_no}: {n_alive} configs on the last {n_rows:,} train rows '
              f'({fraction:.0%}, {int(y_train[start:].sum()):,} laundering) | {len(todo)} to fit')
        dataset = val_dataset = None
        if todo:
            dataset = make_dataset(X_train[start:], y_train[start:])
            val_dataset = make_val_dataset(X_val, y_val, dataset)
        for trial in todo:
            t0 = time.time()
            booster = fit_booster(configs[trial], dataset, val_dataset, seed)
            rounds = int(booster.best_iteration)
            p_val = predict_booster(booster, X_val, rounds)
            threshold, val_f1 = core.best_f1_threshold(y_val, p_val)
            row = {'trial': trial, 'round': round_no, 'fraction': fraction, 'n_rows': n_rows,
                   'val_f1': val_f1, 'threshold': threshold,
                   'val_pr_auc': float(average_precision_score(y_val, p_val)), 'rounds': rounds,
                   'seconds': time.time() - t0, 'params': json.dumps(configs[trial])}
            new_row = pd.DataFrame([row])
            new_row.to_csv(log_path, mode='a', header=not os.path.exists(log_path), index=False)
            trials = pd.concat([trials, new_row], ignore_index=True) if len(trials) else new_row
            print(f"trial {trial:3d} | val F1 {val_f1:.4f} @ {threshold:.3f} | PR-AUC {row['val_pr_auc']:.4f} | "
                  f"{row['seconds']:.0f}s | leaves {configs[trial]['num_leaves']} rounds {rounds} "
                  f"lr {configs[trial]['learning_rate']:.3f}")
            del booster
        del dataset, val_dataset
        gc.collect()
        this_round = trials[(trials['round'] == round_no) & trials['trial'].isin(alive)]
        ranked = this_round.sort_values([SELECT, 'trial'], ascending=[False, True])
        alive = ranked['trial'].astype(int).tolist()
        print(f'round {round_no} best: trial {alive[0]} {SELECT} {ranked[SELECT].iloc[0]:.4f} | '
              f'median {this_round[SELECT].median():.4f} | {this_round["seconds"].sum() / 60:.1f} min')

    best_trial = alive[0]
    best_row = trials[(trials['trial'] == best_trial) & (trials['round'] == len(schedule))].iloc[0]
    best = {'trial': int(best_trial), 'select': SELECT, 'early_stopping': EARLY_STOPPING,
            'val_f1': float(best_row['val_f1']), 'val_pr_auc': float(best_row['val_pr_auc']),
            'threshold': float(best_row['threshold']), 'rounds': int(best_row['rounds']),
            'params': configs[best_trial], 'schedule': schedule, 'x0': x0, 'eta': eta, 'r0': r0, 'seed': seed}
    json.dump(best, open(f'{out_dir}/best_params.json', 'w'), indent=2)
    print(f'best configuration: trial {best_trial} ({SELECT} {best[SELECT]:.4f} on the full train rows, '
          f'{best["rounds"]} rounds)')
    print(json.dumps(configs[best_trial], indent=1))
    return configs[best_trial], trials


# ==================================================================================================
# one final run: checkpoints every CHECK_EVERY rounds, lowest-validation-loss round kept, test scored once
# ==================================================================================================

def run_name_gbt(seed, feature_set=FEATURE_SET):
    '''lightgbm_readout-<set>_seed-<seed> (CLAUDE.md §6).'''
    return f'lightgbm_readout-{feature_set}_seed-{seed}'


def gbt_result_row(name, seed, best_epoch, threshold, metrics):
    '''The flat result row with every batch_summary.csv column; params = trees kept, invariant_params = 0.'''
    row = {'run': name, 'operator': 'lightgbm', 'mp': 'none', 'readout': FEATURE_SET, 'mp_direction': 'none',
           'temporal_sampling': False, 'node_enc': 'none', 'gfp_variant': 'v0',
           'seed': seed, 'best_epoch': best_epoch, 'threshold': threshold,
           'params': best_epoch, 'invariant_params': 0}
    row.update(core.flatten_metrics(metrics))
    return row


def save_gbt_run(out_dir, row, params, paper_comparison):
    '''results.json = the flat row + the hyperparameters + F1 at the paper's 0.5 threshold; prints the split table.'''
    results = dict(row)
    results['hyperparams'] = {**{k: v for k, v in FIXED_PARAMS.items() if k not in ('verbose', 'num_threads')}, **params}
    results['paper_comparison'] = paper_comparison
    json.dump(results, open(f'{out_dir}/results.json', 'w'), indent=2)
    table = {split: {metric: row[f'{split}_{metric}'] for metric in core.METRICS} for split in core.SPLITS}
    print(pd.DataFrame(table).round(4))


def save_gbt_predictions(out_dir, data, probs, threshold):
    '''predictions.csv with the columns of gnn_core.save_predictions: every edge of every split, sorted by edge_id.'''
    frames = []
    for split in core.SPLITS:
        rows = data['rows'][split]
        prob = probs[split].astype(np.float32)
        frames.append(pd.DataFrame({
            'split':     split,
            'edge_id':   data['edge_id'][rows],
            'src':       data['src'][rows],
            'dst':       data['dst'][rows],
            'edge_time': data['edge_time'][rows],
            'y':         data['y'][rows],
            'prob':      prob,
            'pred':      (prob >= threshold).astype(int),
        }))
    table = pd.concat(frames).sort_values('edge_id')
    assert table['edge_id'].is_unique, 'edge_id must identify every scored edge once'
    table.to_csv(f'{out_dir}/predictions.csv', index=False)
    print(f'predictions: {len(table):,} rows -> {out_dir}/predictions.csv')


def train_final(params, data, seed, out_root, check_every=CHECK_EVERY):
    '''Train one seed with the chosen configuration, keep the lowest-validation-loss round, score test once.

    Early stopping on the weighted validation loss (patience EARLY_STOPPING) decides the kept round;
    the threshold is then swept on validation at that round. Writes results.json, history.csv (one
    row per checkpoint plus the kept round, epoch = boosting round), curves.png, predictions.csv and
    best.txt (the booster cut at the kept round) to <out_root>/<run>/. A run whose results.json
    exists is skipped (resume). Returns the flat result row.
    '''
    name = run_name_gbt(seed)
    out_dir = f'{out_root}/{name}'
    if os.path.exists(f'{out_dir}/results.json'):
        results = json.load(open(f'{out_dir}/results.json'))
        print(f'=== {name} === already finished (results.json found), skipping')
        return {k: v for k, v in results.items() if k not in ('hyperparams', 'paper_comparison')}
    os.makedirs(out_dir, exist_ok=True)
    print(f'=== {name} ===')
    print(f'features: {data["X"].shape[1]} | seed {seed} | rounds {params["num_round"]}'
          f' (cap; early stopping on weighted val loss, patience {EARLY_STOPPING}) | checkpoint every {check_every}')

    X_train, y_train = split_arrays(data, 'train')
    X_val, y_val = split_arrays(data, 'val')
    num_round = params['num_round']
    checkpoints = set(range(check_every, num_round, check_every)) | {num_round}
    history, last_time = [], time.time()

    def score_round(booster, k):
        '''One history row at round k (train / val loss, F1 at the val threshold, PR-AUC, ROC-AUC).'''
        nonlocal last_time
        p_train = predict_booster(booster, X_train, k)
        p_val = predict_booster(booster, X_val, k)
        threshold, val_f1 = core.best_f1_threshold(y_val, p_val)
        row = {
            'epoch':        k,
            'train_loss':   core.weighted_bce(y_train, p_train),
            'val_loss':     core.weighted_bce(y_val, p_val),
            'train_f1':     float(f1_score(y_train, (p_train >= threshold).astype(int))),
            'val_f1':       val_f1,
            'train_pr_auc': float(average_precision_score(y_train, p_train)),
            'val_pr_auc':   float(average_precision_score(y_val, p_val)),
            'val_roc_auc':  float(roc_auc_score(y_val, p_val)),
            'threshold':    threshold,
            'seconds':      time.time() - last_time,
        }
        last_time = time.time()
        return row

    def checkpoint(env):
        '''LightGBM callback: score the booster at every checkpoint round.'''
        k = env.iteration + 1
        if k not in checkpoints:
            return
        r = score_round(env.model, k)
        history.append(r)
        print(f"round {k:4d} | loss {r['train_loss']:.4f} / {r['val_loss']:.4f} | "
              f"F1 {r['train_f1']:.4f} / {r['val_f1']:.4f} @ {r['threshold']:.3f} | "
              f"PR-AUC {r['train_pr_auc']:.4f} / {r['val_pr_auc']:.4f} | {r['seconds']:.0f}s   (train / val)")

    dataset = make_dataset(X_train, y_train)
    val_dataset = make_val_dataset(X_val, y_val, dataset)
    booster = fit_booster(params, dataset, val_dataset, seed, callbacks=[checkpoint])
    del dataset, val_dataset
    # the kept round is the lowest-validation-loss round; the threshold is swept there
    best_epoch = int(booster.best_iteration)
    if best_epoch not in {r['epoch'] for r in history}:
        history.append(score_round(booster, best_epoch))
        history.sort(key=lambda r: r['epoch'])
    kept = next(r for r in history if r['epoch'] == best_epoch)
    threshold = kept['threshold']
    print(f'early stopping: lowest weighted validation loss {kept["val_loss"]:.4f} at round {best_epoch} '
          f'(validation F1 {kept["val_f1"]:.4f}, threshold {threshold:.3f})')
    pd.DataFrame(history).to_csv(f'{out_dir}/history.csv', index=False)
    booster.save_model(f'{out_dir}/best.txt', num_iteration=best_epoch)

    # final evaluation: the kept round, one validation-chosen threshold, test scored once
    metrics, probs, at_half = {}, {}, {}
    for split in core.SPLITS:
        X_split, y_split = split_arrays(data, split)
        probs[split] = predict_booster(booster, X_split, best_epoch)
        metrics[split] = core.compute_metrics(y_split, probs[split], threshold)
        at_half[f'{split}_f1_at_0.5'] = float(f1_score(y_split, (probs[split] >= PAPER_THRESHOLD).astype(int)))
    row = gbt_result_row(name, seed, best_epoch, threshold, metrics)
    save_gbt_run(out_dir, row, dict(params, early_stopping=EARLY_STOPPING), at_half)
    save_gbt_predictions(out_dir, data, probs, threshold)
    core.plot_curves(history, data['y'][data['rows']['test']], probs['test'], metrics['test']['pr_auc'],
                     best_epoch, name, f'{out_dir}/curves.png')
    del booster, probs
    gc.collect()
    return row


# ==================================================================================================
# batch summaries
# ==================================================================================================

def seed_summary(rows, out_root, filename='seed_summary_gbt.csv'):
    '''Mean and std over the seeds of every metric column -> <out_root>/<filename>; the GNN helper, GBT file name.'''
    return core.seed_summary(rows, out_root, filename)


# ==================================================================================================
# SIDE CHECK, OUTSIDE THE PROTOCOL: the test window including the tail days the pipeline removed
# ==================================================================================================
# The thesis keeps Timestamp < 2022-09-11 (Progress_Report.md section 1); the 8 later days hold 1,108
# transactions, 655 of them laundering (pattern completions only), and the paper's released split code keeps
# them in its test window (9-18 Sep). This check rebuilds the 93 features of the tail rows with the
# Data_preparation recipes, self-checks the recipes on the 10 Sep rows against the stored features, and scores
# the finished boosters. Nothing is trained or chosen here; the numbers enter no comparison.
TAIL_START, PAPER_TEST_START = '2022-09-11', '2022-09-09'
RAW_COLS = ['Timestamp', 'src_bank', 'src_account', 'dst_bank', 'dst_account', 'Amount Received',
            'Receiving Currency', 'Amount Paid', 'Payment Currency', 'Payment Format', 'label']
GFP_V0_PARAMS = {   # Data_preparation.ipynb cell 12 — the V0 sheet
    'num_threads': 8, 'time_window': 86400,
    'vertex_stats': True, 'vertex_stats_tw': 86400, 'vertex_stats_cols': [3, 4],
    'vertex_stats_feats': [0, 1, 2, 3, 4, 8, 9, 10],
    'fan': False, 'fan_bins': [2, 5, 10], 'degree': False, 'degree_bins': [2, 5, 10],
    'scatter-gather': True, 'scatter-gather_tw': 6 * 3600, 'scatter-gather_bins': [2, 3, 5],
    'temp-cycle': True, 'temp-cycle_tw': 86400, 'temp-cycle_bins': [2, 3, 5],
    'lc-cycle': True, 'lc-cycle_tw': 86400, 'lc-cycle_bins': [2, 3, 5], 'lc-cycle_len': 6,
}
GFP_BATCH = 128
TAIL_WINDOWS = ('our test (8 Sep 16:09 – 10 Sep)', 'our test + tail (– 18 Sep)',
                'paper-code test window (9 – 18 Sep)', 'tail only (11 – 18 Sep)')


def _normalise_gfp(frame, scaler):
    '''The train-fit GFP normalisation of Data_preparation (log1p where flagged, p1/p99 clip, StandardScaler).'''
    out = frame.copy()
    for group in ('log_group', 'std_group'):
        g = scaler[group]
        vals = out[g['cols']].to_numpy(dtype=np.float64)
        if g['log']:
            vals = np.log1p(np.clip(vals, 0, None))
        out[g['cols']] = g['scaler'].transform(np.clip(vals, g['p01'], g['p99'])).astype(np.float32)
    return out


def build_tail_matrix(data_dir='Data', log=print):
    '''The 93-column matrix of the tail rows (Timestamp >= TAIL_START), rebuilt from the raw file.

    Baseline features follow Data_preparation (FX rates recovered from the stored 10 Sep Amount_Log, bank risk
    frozen at the train-window rate); the GFP block is streamed causally in WSL (batch GFP_BATCH, 24 h windows)
    from the batch boundary before PAPER_TEST_START, so every tail row sees its full window; then the train-fit
    normalisation and the entity one-hots. Returns a dict with X, y, timestamps (datetime64[s]) and the
    self-checks on the 10 Sep rows (which are in both the stream and the stored files).
    '''
    import pickle
    import shutil
    import subprocess
    meta = json.load(open(f'{data_dir}/feature_meta.json'))
    base_cols, gfp_cols, node_cols = meta['BASE_EDGE_COLS'], meta['GFP_FEAT_COLS'], meta['NODE_FEAT_COLS']
    raw = pd.read_csv(f'{data_dir}/HI-Small_Trans.csv', names=RAW_COLS, header=0, low_memory=False)
    raw['Timestamp'] = pd.to_datetime(raw['Timestamp'])
    raw = raw.sort_values('Timestamp', kind='stable').reset_index(drop=True)
    is_tail = (raw['Timestamp'] >= TAIL_START).to_numpy()
    assert (~is_tail).sum() == N_EDGES and is_tail[N_EDGES:].all(), 'the kept rows must be exactly the first N_EDGES'
    n_all = len(raw)
    ts = raw['Timestamp'].to_numpy()
    sep9, sep10 = (int(np.searchsorted(ts, np.datetime64(d))) for d in (PAPER_TEST_START, '2022-09-10'))
    log(f'raw file {n_all:,} rows | kept {N_EDGES:,} | tail {is_tail.sum():,} rows, '
        f'{int(raw.label[is_tail].sum())} laundering | 9 Sep starts at row {sep9:,}, 10 Sep at {sep10:,}')
    stored = pd.read_csv(f'{data_dir}/edge_features.csv', usecols=['label'] + base_cols + gfp_cols,
                         dtype=np.float32, skiprows=range(1, sep9 + 1))
    assert len(stored) == N_EDGES - sep9 and (stored['label'].to_numpy() == raw['label'].to_numpy()[sep9:N_EDGES]).all()

    # ---- baseline features (Data_preparation cells 7-8) ----
    seconds = raw['Timestamp'].astype('int64') // 10**9
    rows10 = raw.iloc[sep10:N_EDGES]
    amount_log10 = stored['Amount_Log'].to_numpy()[sep10 - sep9:]
    ok = rows10['Amount Paid'].to_numpy() >= 1
    rate = pd.Series(np.expm1(amount_log10[ok].astype(np.float64)) / rows10['Amount Paid'].to_numpy()[ok],
                     index=rows10['Payment Currency'].to_numpy()[ok]).groupby(level=0).median()
    rate['US Dollar'] = 1.0           # the 10 Sep FX rates, carried forward to the tail days
    raw['Amount_USD'] = raw['Amount Paid'] * raw['Payment Currency'].map(rate).fillna(1.0)
    feat = pd.DataFrame(index=raw.index)
    feat['Amount_Log'] = np.log1p(raw['Amount_USD'])
    feat['Struct_Band'] = raw['Amount_USD'].between(9000, 10000).astype(np.int8)
    hour, dow = raw['Timestamp'].dt.hour, raw['Timestamp'].dt.dayofweek
    feat['Hour_Sin'], feat['Hour_Cos'] = np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24)
    feat['DayOfWeek_Sin'], feat['DayOfWeek_Cos'] = np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7)
    feat['Is_Weekend'] = (dow >= 5).astype(np.int8)
    feat['Is_Self_Loop'] = (raw['src_account'] == raw['dst_account']).astype(np.int8)
    feat['Same_Bank'] = (raw['src_bank'] == raw['dst_bank']).astype(np.int8)
    ten_days = 10 * 24 * 3600
    feat['Dt_Src_Log'] = np.log1p((seconds - seconds.groupby(raw['src_account']).shift()).fillna(ten_days))
    feat['Dt_Dst_Log'] = np.log1p((seconds - seconds.groupby(raw['dst_account']).shift()).fillna(ten_days))
    m = meta['bank_risk_smoothing_m']
    is_train = np.arange(n_all) < N_TRAIN
    prior = raw.loc[is_train, 'label'].mean()
    for col, bank_col in (('Src_Bank_Risk', 'src_bank'), ('Dst_Bank_Risk', 'dst_bank')):
        stats = raw[is_train].groupby(bank_col)['label'].agg(['sum', 'count'])
        feat[col] = raw[bank_col].map((stats['sum'] + m * prior) / (stats['count'] + m)).fillna(prior)
    for col in base_cols:
        if col.startswith('PayFmt_'):
            feat[col] = (raw['Payment Format'] == col[len('PayFmt_'):]).astype(np.int8)
    feat = feat[base_cols].astype(np.float32)
    base_diff = np.abs(feat.iloc[sep10:N_EDGES].to_numpy() - stored[base_cols].to_numpy()[sep10 - sep9:]).max()
    assert base_diff < 1e-4, f'baseline recipe does not reproduce the stored features (max diff {base_diff:.2e})'
    log(f'baseline self-check on the {N_EDGES - sep10:,} rows of 10 Sep: max |rebuilt − stored| = {base_diff:.1e}')

    # ---- GFP block, streamed causally in WSL from the batch boundary before 9 Sep ----
    account_to_idx = pickle.load(open(f'{data_dir}/account_to_idx.pkl', 'rb'))
    idx_map = dict(account_to_idx)
    for acc in pd.concat([raw['src_account'][is_tail], raw['dst_account'][is_tail]]).unique():
        if acc not in idx_map:
            idx_map[acc] = len(idx_map)
    start = (sep9 // GFP_BATCH) * GFP_BATCH
    gfp_input = np.column_stack([
        np.arange(start, n_all),
        raw['src_account'].iloc[start:].map(idx_map).to_numpy(),
        raw['dst_account'].iloc[start:].map(idx_map).to_numpy(),
        (seconds - seconds.iloc[:N_EDGES].min()).to_numpy()[start:],
        raw['Amount_USD'].to_numpy()[start:],
    ]).astype(np.float64)
    assert not np.isnan(gfp_input).any()
    tmp = os.path.abspath(f'{data_dir}/_tail_gfp_tmp')
    os.makedirs(tmp, exist_ok=True)
    np.save(f'{tmp}/input.npy', gfp_input)
    json.dump(GFP_V0_PARAMS, open(f'{tmp}/params.json', 'w'))
    to_wsl = lambda p: '/mnt/' + p[0].lower() + p[2:].replace('\\', '/')
    cmd = (f'~/gfp_env/bin/python "{to_wsl(os.path.abspath("run_gfp_wsl.py"))}" "{to_wsl(tmp)}/input.npy" '
           f'"{to_wsl(tmp)}/params.json" "{to_wsl(tmp)}/output.npy" {GFP_BATCH}')
    log(f'GFP (V0 parameters, batch {GFP_BATCH}) on {len(gfp_input):,} rows from row {start:,} in WSL ...')
    result = subprocess.run(['wsl', 'bash', '-c', cmd], capture_output=True, text=True)
    print(result.stdout.strip()[-600:])
    assert result.returncode == 0, result.stderr[-2000:]
    gfp_out = np.load(f'{tmp}/output.npy')[:, 5:]
    shutil.rmtree(tmp)
    assert gfp_out.shape == (n_all - start, len(gfp_cols))

    # self-check: the 10 Sep rows have a full 24 h of context in the stream -> compare with the stored raw GFP
    rebuilt = gfp_out[sep10 - start:N_EDGES - start]
    kept = stored[gfp_cols].to_numpy()
    kept = kept[sep10 - sep9:]
    within = np.abs(rebuilt - kept) <= 0.01 * np.maximum(np.abs(kept), 1e-6) + 1e-9
    agree = pd.Series(within.mean(axis=0), index=gfp_cols)
    groups = {'pattern bins (scatter-gather, cycles)': [c for c in gfp_cols if '_bins_' in c],
              'fan / degree / ratio': [c for c in gfp_cols if any(k in c for k in ('_fan_', '_deg_', '_ratio_'))],
              'avg / sum': [c for c in gfp_cols if any(k in c for k in ('_avg_', '_sum_'))],
              'var / skew / kurtosis': [c for c in gfp_cols if any(k in c for k in ('_var_', '_skew_', '_kurtosis_'))]}
    gfp_check = pd.DataFrame({'columns': {g: len(c) for g, c in groups.items()},
                              'rows within 1 % (min over columns)': {g: agree[c].min() for g, c in groups.items()},
                              'rows within 1 % (mean over columns)': {g: agree[c].mean() for g, c in groups.items()}})
    log('GFP self-check on the 10 Sep rows (rebuilt stream vs stored sheet):')
    print(gfp_check.round(4).to_string())

    # ---- normalise, add the entity one-hots ----
    scaler = pickle.load(open(f'{data_dir}/standard_scaler.pkl', 'rb'))
    gfp_tail = _normalise_gfp(pd.DataFrame(gfp_out[N_EDGES - start:], columns=gfp_cols), scaler)
    nodes = pd.read_csv(f'{data_dir}/node_features.csv').set_index('account_id')
    src_x = nodes.reindex(raw['src_account'][is_tail])[node_cols].fillna(0).to_numpy(np.float32)
    dst_x = nodes.reindex(raw['dst_account'][is_tail])[node_cols].fillna(0).to_numpy(np.float32)
    X_tail = np.hstack([feat[is_tail].to_numpy(np.float32), gfp_tail.to_numpy(np.float32), src_x, dst_x])
    y_tail = raw['label'].to_numpy()[is_tail]
    assert X_tail.shape == (int(is_tail.sum()), 93) and not np.isnan(X_tail).any()
    log(f'tail matrix {X_tail.shape}, {int(y_tail.sum())} laundering rows')
    return {'X': X_tail, 'y': y_tail, 'timestamps': ts[is_tail].astype('datetime64[s]'),
            'checks': {'baseline_max_diff': float(base_diff), 'gfp': gfp_check}}


def tail_check(data, data_dir, out_root, seeds=SEEDS, log=print):
    '''Score the finished boosters on the test window with and without the tail days.

    Four windows: our test rows alone (must reproduce results.json), our test + tail, the paper code's
    9-18 Sep window, the tail alone — each at the seed's own validation threshold (and at 0.5). One
    row per (seed, window) -> <out_root>/tail_check.csv; returns the four-seed summary. Outside the
    protocol: the tail features are rebuilt here, nothing is chosen on these numbers.
    '''
    X_test, y_test = split_arrays(data, 'test')
    ts_test = data['edge_time'][data['rows']['test']].astype('datetime64[s]')
    tail = build_tail_matrix(data_dir, log)
    X_both = np.vstack([X_test, tail['X']])
    y_both = np.concatenate([y_test, tail['y']])
    ts_both = np.concatenate([ts_test, tail['timestamps']])
    n_test = len(y_test)
    windows = {TAIL_WINDOWS[0]: np.arange(n_test),
               TAIL_WINDOWS[1]: np.arange(len(y_both)),
               TAIL_WINDOWS[2]: np.where(ts_both >= np.datetime64(PAPER_TEST_START))[0],
               TAIL_WINDOWS[3]: np.arange(n_test, len(y_both))}
    rows = []
    for seed in seeds:
        run_dir = f'{out_root}/{run_name_gbt(seed)}'
        res = json.load(open(f'{run_dir}/results.json'))
        booster = lgb.Booster(model_file=f'{run_dir}/best.txt')
        p = booster.predict(X_both, num_threads=THREADS).astype(np.float32)
        thr = res['threshold']
        for name, idx in windows.items():
            y, pp = y_both[idx], p[idx]
            m = core.compute_metrics(y, pp, thr)
            rows.append({'seed': seed, 'window': name, 'rows': len(idx), 'laundering': int(y.sum()),
                         'threshold': thr, 'f1': m['f1'], 'precision': m['precision'], 'recall': m['recall'],
                         'pr_auc': m['pr_auc'], 'recall_at_5pct': m['recall_at_5pct'],
                         'f1_at_0.5': float(f1_score(y, (pp >= PAPER_THRESHOLD).astype(int)))})
        assert abs(rows[-4]['f1'] - res['test_f1']) < 1e-6, (seed, rows[-4]['f1'], res['test_f1'])
        log(f"seed {seed}: our test window reproduces results.json (F1 {rows[-4]['f1']:.4f}); "
            f"+ tail {rows[-3]['f1']:.4f} | paper window {rows[-2]['f1']:.4f} | tail only {rows[-1]['f1']:.4f}")
    table = pd.DataFrame(rows)
    table.to_csv(f'{out_root}/tail_check.csv', index=False)
    summary = table.groupby('window', sort=False).agg(
        rows=('rows', 'first'), laundering=('laundering', 'first'),
        f1_mean=('f1', 'mean'), f1_std=('f1', 'std'), precision=('precision', 'mean'), recall=('recall', 'mean'),
        pr_auc=('pr_auc', 'mean'), recall_at_5pct=('recall_at_5pct', 'mean'), f1_at_0_5=('f1_at_0.5', 'mean'))
    print(f'{len(table)} rows -> {out_root}/tail_check.csv')
    return summary, table
