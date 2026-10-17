'''Shared code for the GBT cross-check: LightGBM on the same inputs as the GNNs, no message passing.

Everything that is "how a number is computed" (threshold sweep, the seven metrics, the loss
diagnostic, the curves, the batch summary) is imported from gnn_core so GBT and GNN rows are
comparable by construction. This file adds only the GBT-specific parts: the 93-column row built
from the graph files, the successive-halving search (Altman et al. 2023, Appendix C), the final
fits with boosting-round checkpoints, and the saving conventions of CLAUDE.md §6.
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

# ---------------- fixed recipe of the cross-check (EXPERIMENTS.md appendix) — never tuned per run ----------------
SEEDS        = (42, 43, 44, 45)       # final models; 42 is the run compared with the single-seed GNN runs
X0, ETA, R0  = 100, 2, 0.1            # successive halving: start configs, keep 1/ETA per round, first train fraction
CHECK_EVERY  = 25                     # boosting rounds between validation checkpoints of a final fit
THREADS      = os.cpu_count()          # all logical cores (16 on the thesis laptop)
FEATURE_SET  = 'full'                 # the only readout set of the GBT: all 81 edge + 6 + 6 node columns
N_EDGES, N_TRAIN, N_VAL_END = 5_077_237, 3_046_342, 4_061_789   # Progress_Report.md §4
POSITIVES    = {'train': 2_297, 'val': 1_082, 'test': 1_143}
PAPER_THRESHOLD = 0.5                 # the paper's decision rule, stored for comparison only

# search space: practical LightGBM ranges (name: low, high, how to sample)
SEARCH_SPACE = {
    'num_round':        (100,  1000, 'int'),
    'num_leaves':       (15,   255,  'int'),
    'learning_rate':    (0.01, 0.1,  'log'),
    'min_data_in_leaf': (20,   500,  'int'),
    'feature_fraction': (0.5,  1.0,  'float'),
    'bagging_fraction': (0.5,  1.0,  'float'),
    'lambda_l1':        (0.01, 10.0, 'log'),
    'lambda_l2':        (0.01, 10.0, 'log'),
    'scale_pos_weight': (1.0,  10.0, 'float'),   # the paper's range
}
FIXED_PARAMS = {'objective': 'binary', 'bagging_freq': 1, 'verbose': -1, 'num_threads': THREADS}
DATASET_PARAMS = {'feature_pre_filter': False}   # lets one constructed Dataset serve trials with different min_data_in_leaf
TRIAL_COLS = ['trial', 'round', 'fraction', 'n_rows', 'val_f1', 'threshold', 'val_pr_auc', 'seconds', 'params']


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

def sample_params(rng):
    '''One random configuration from SEARCH_SPACE (ints uniform, floats uniform, log ranges log-uniform).'''
    params = {}
    for name, (low, high, kind) in SEARCH_SPACE.items():
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


def fit_booster(params, dataset, seed, callbacks=None):
    '''lgb.train with the fixed recipe + one sampled configuration; num_round is taken from params.'''
    lgb_params = dict(FIXED_PARAMS)
    lgb_params.update({k: v for k, v in params.items() if k != 'num_round'})
    lgb_params.update(DATASET_PARAMS)
    lgb_params['seed'] = seed
    return lgb.train(lgb_params, dataset, num_boost_round=params['num_round'], callbacks=callbacks or [])


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

    Every round fits on the most recent `fraction` of the train rows and scores the full validation
    set with the swept-threshold F1 (test is never touched). One line per (trial, round) goes to
    <out_dir>/trials.csv as soon as it finishes, so a stopped search resumes where it was. Returns
    (best params, trials table); best = highest validation F1 of the last round, ties -> lower trial id.
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
    print(f'successive halving: x0={x0} eta={eta} r0={r0} -> rounds {schedule} '
          f'| {len(trials)} trial rows already logged')

    alive = list(range(x0))
    for round_no, (n_alive, fraction) in enumerate(schedule, 1):
        alive = alive[:n_alive]
        n_rows = int(round(fraction * n_train))
        start = n_train - n_rows                     # the most recent rows of the train window
        todo = [t for t in alive if not ((trials['trial'] == t) & (trials['round'] == round_no)).any()]
        print(f'--- round {round_no}: {n_alive} configs on the last {n_rows:,} train rows '
              f'({fraction:.0%}, {int(y_train[start:].sum()):,} laundering) | {len(todo)} to fit')
        dataset = make_dataset(X_train[start:], y_train[start:]) if todo else None
        for trial in todo:
            t0 = time.time()
            booster = fit_booster(configs[trial], dataset, seed)
            p_val = predict_booster(booster, X_val)
            threshold, val_f1 = core.best_f1_threshold(y_val, p_val)
            row = {'trial': trial, 'round': round_no, 'fraction': fraction, 'n_rows': n_rows,
                   'val_f1': val_f1, 'threshold': threshold,
                   'val_pr_auc': float(average_precision_score(y_val, p_val)),
                   'seconds': time.time() - t0, 'params': json.dumps(configs[trial])}
            new_row = pd.DataFrame([row])
            new_row.to_csv(log_path, mode='a', header=not os.path.exists(log_path), index=False)
            trials = pd.concat([trials, new_row], ignore_index=True) if len(trials) else new_row
            print(f"trial {trial:3d} | val F1 {val_f1:.4f} @ {threshold:.3f} | PR-AUC {row['val_pr_auc']:.4f} | "
                  f"{row['seconds']:.0f}s | leaves {configs[trial]['num_leaves']} rounds {configs[trial]['num_round']} "
                  f"lr {configs[trial]['learning_rate']:.3f}")
            del booster
        del dataset
        gc.collect()
        this_round = trials[(trials['round'] == round_no) & trials['trial'].isin(alive)]
        ranked = this_round.sort_values(['val_f1', 'trial'], ascending=[False, True])
        alive = ranked['trial'].astype(int).tolist()
        print(f'round {round_no} best: trial {alive[0]} val F1 {ranked["val_f1"].iloc[0]:.4f} | '
              f'median {this_round["val_f1"].median():.4f} | {this_round["seconds"].sum() / 60:.1f} min')

    best_trial = alive[0]
    best_row = trials[(trials['trial'] == best_trial) & (trials['round'] == len(schedule))].iloc[0]
    best = {'trial': int(best_trial), 'val_f1': float(best_row['val_f1']), 'threshold': float(best_row['threshold']),
            'params': configs[best_trial], 'schedule': schedule, 'x0': x0, 'eta': eta, 'r0': r0, 'seed': seed}
    json.dump(best, open(f'{out_dir}/best_params.json', 'w'), indent=2)
    print(f'best configuration: trial {best_trial} (validation F1 {best["val_f1"]:.4f} on the full train rows)')
    print(json.dumps(configs[best_trial], indent=1))
    return configs[best_trial], trials


# ==================================================================================================
# one final run: checkpoints every CHECK_EVERY rounds, best-validation-F1 round kept, test scored once
# ==================================================================================================

def run_name_gbt(seed, feature_set=FEATURE_SET):
    '''lightgbm_readout-<set>_seed-<seed> (CLAUDE.md §6).'''
    return f'lightgbm_readout-{feature_set}_seed-{seed}'


def gbt_result_row(name, seed, best_epoch, threshold, metrics):
    '''The flat result row with every batch_summary.csv column; params = trees kept, invariant_params = 0.'''
    row = {'run': name, 'operator': 'lightgbm', 'mp': 'none', 'readout': FEATURE_SET, 'mp_direction': 'none',
           'temporal_sampling': False, 'node_enc': 'none', 'gfp_variant': 'v0', 'seed': seed,
           'best_epoch': best_epoch, 'threshold': threshold,
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
    '''Train one seed with the chosen configuration, keep the best-validation-F1 round, score test once.

    Writes results.json, history.csv (one row per checkpoint, epoch = boosting round), curves.png,
    predictions.csv and best.txt (the booster cut at the kept round) to <out_root>/<run>/.
    A run whose results.json exists is skipped (resume). Returns the flat result row.
    '''
    name = run_name_gbt(seed)
    out_dir = f'{out_root}/{name}'
    if os.path.exists(f'{out_dir}/results.json'):
        results = json.load(open(f'{out_dir}/results.json'))
        print(f'=== {name} === already finished (results.json found), skipping')
        return {k: v for k, v in results.items() if k not in ('hyperparams', 'paper_comparison')}
    os.makedirs(out_dir, exist_ok=True)
    print(f'=== {name} ===')
    print(f'features: {data["X"].shape[1]} | seed {seed} | rounds {params["num_round"]} | checkpoint every {check_every}')

    X_train, y_train = split_arrays(data, 'train')
    X_val, y_val = split_arrays(data, 'val')
    num_round = params['num_round']
    checkpoints = set(range(check_every, num_round, check_every)) | {num_round}
    history, best, clock = [], {'val_f1': -1.0, 'epoch': 0, 'threshold': 0.5}, {'t0': time.time()}

    def checkpoint(env):
        k = env.iteration + 1
        if k not in checkpoints:
            return
        p_train = predict_booster(env.model, X_train, k)
        p_val = predict_booster(env.model, X_val, k)
        threshold, val_f1 = core.best_f1_threshold(y_val, p_val)
        is_best = val_f1 > best['val_f1']
        if is_best:
            best.update(val_f1=val_f1, epoch=k, threshold=threshold)
        history.append({
            'epoch':        k,
            'train_loss':   core.weighted_bce(y_train, p_train),
            'val_loss':     core.weighted_bce(y_val, p_val),
            'train_f1':     float(f1_score(y_train, (p_train >= threshold).astype(int))),
            'val_f1':       val_f1,
            'train_pr_auc': float(average_precision_score(y_train, p_train)),
            'val_pr_auc':   float(average_precision_score(y_val, p_val)),
            'val_roc_auc':  float(roc_auc_score(y_val, p_val)),
            'threshold':    threshold,
            'seconds':      time.time() - clock['t0'],
        })
        clock['t0'] = time.time()
        r = history[-1]
        print(f"round {k:4d} | loss {r['train_loss']:.4f} / {r['val_loss']:.4f} | "
              f"F1 {r['train_f1']:.4f} / {r['val_f1']:.4f} @ {threshold:.3f} | "
              f"PR-AUC {r['train_pr_auc']:.4f} / {r['val_pr_auc']:.4f} | {r['seconds']:.0f}s   (train / val)"
              + ('   * new best' if is_best else ''))

    dataset = make_dataset(X_train, y_train)
    booster = fit_booster(params, dataset, seed, callbacks=[checkpoint])
    del dataset
    pd.DataFrame(history).to_csv(f'{out_dir}/history.csv', index=False)
    best_epoch, threshold = best['epoch'], best['threshold']
    print(f'best validation F1 {best["val_f1"]:.4f} at round {best_epoch} (threshold {threshold:.3f})')
    booster.save_model(f'{out_dir}/best.txt', num_iteration=best_epoch)

    # final evaluation: the kept round, one validation-chosen threshold, test scored once
    metrics, probs, at_half = {}, {}, {}
    for split in core.SPLITS:
        X_split, y_split = split_arrays(data, split)
        probs[split] = predict_booster(booster, X_split, best_epoch)
        metrics[split] = core.compute_metrics(y_split, probs[split], threshold)
        at_half[f'{split}_f1_at_0.5'] = float(f1_score(y_split, (probs[split] >= PAPER_THRESHOLD).astype(int)))
    row = gbt_result_row(name, seed, best_epoch, threshold, metrics)
    save_gbt_run(out_dir, row, params, at_half)
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
