'''Side check, outside the protocol: the test window including the tail days the pipeline removed.

The thesis keeps Timestamp < 2022-09-11 (Progress_Report.md section 1); the 8 later days hold 1,108
transactions, 655 of them laundering (pattern completions only), and the paper's released split code keeps
them in its test window (9-18 Sep). This check rebuilds the 93 features of the tail rows with the
Data_preparation recipes, self-checks the recipes on the 10 Sep rows against the stored features, and scores
the finished boosters. Nothing is trained or chosen here; the numbers enter no comparison.

Re-exported from gnn_frauds.model.tabular.lightgbm so the notebook keeps calling gbt_core.tail_check.
'''
import json
import os
import shutil
import subprocess

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

import gnn_frauds.api as core

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
    from gnn_frauds.model.tabular.lightgbm import N_EDGES, N_TRAIN
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
    # An account's first transaction has no earlier one; 10 days is the cap used when the features were built.
    feat['Dt_Src_Log'] = np.log1p((seconds - seconds.groupby(raw['src_account']).shift()).fillna(ten_days))
    feat['Dt_Dst_Log'] = np.log1p((seconds - seconds.groupby(raw['dst_account']).shift()).fillna(ten_days))
    m = meta['bank_risk_smoothing_m']
    is_train = np.arange(n_all) < N_TRAIN
    prior = raw.loc[is_train, 'label'].mean()
    for col, bank_col in (('Src_Bank_Risk', 'src_bank'), ('Dst_Bank_Risk', 'dst_bank')):
        stats = raw[is_train].groupby(bank_col)['label'].agg(['sum', 'count'])
        # Train-window rate, smoothed as if the bank had m extra transactions at the global rate, then frozen.
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
            idx_map[acc] = len(idx_map)  # an account that appears only after 10 Sep has no stored index
    start = (sep9 // GFP_BATCH) * GFP_BATCH  # same batch edges as the stored stream; starting mid-batch would change the counts
    gfp_input = np.column_stack([
        np.arange(start, n_all),
        raw['src_account'].iloc[start:].map(idx_map).to_numpy(),
        raw['dst_account'].iloc[start:].map(idx_map).to_numpy(),
        # Seconds since the first kept transaction, the same origin as the stored sheet.
        (seconds - seconds.iloc[:N_EDGES].min()).to_numpy()[start:],
        raw['Amount_USD'].to_numpy()[start:],
    ]).astype(np.float64)
    assert not np.isnan(gfp_input).any()
    tmp = os.path.abspath(f'{data_dir}/_tail_gfp_tmp')
    os.makedirs(tmp, exist_ok=True)
    np.save(f'{tmp}/input.npy', gfp_input)
    json.dump(GFP_V0_PARAMS, open(f'{tmp}/params.json', 'w'))
    to_wsl = lambda p: '/mnt/' + p[0].lower() + p[2:].replace('\\', '/')  # Windows snapml has no GFP; the stream runs in WSL
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


def tail_check(data, data_dir, out_root, seeds=None, log=print):
    '''Score the finished boosters on the test window with and without the tail days.

    Four windows: our test rows alone (must reproduce results.json), our test + tail, the paper code's
    9-18 Sep window, the tail alone — each at the seed's own validation threshold (and at 0.5). One
    row per (seed, window) -> <out_root>/tail_check.csv; returns the four-seed summary. Outside the
    protocol: the tail features are rebuilt here, nothing is chosen on these numbers.
    seeds=None uses the four final GBT seeds.
    '''
    from gnn_frauds.model.tabular.lightgbm import (
        PAPER_THRESHOLD, SEEDS, THREADS, run_name_gbt, split_arrays,
    )
    if seeds is None:
        seeds = SEEDS
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
