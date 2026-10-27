'''Shared code for the MLP cross-check: a plain neural net on the same 93 inputs as the GBT, no message passing.

The matrix, the split views and the predictions file come from gbt_core; the threshold sweep, the seven metrics,
the loss diagnostic, the curves and the summaries from gnn_core, so MLP, GBT and GNN rows are comparable by
construction. This file adds the MLP parts: the input scaling, the three shapes (flat, funnel, resnet), one
training loop with early stopping on validation PR-AUC, the random search and the final seeds, saved with the
result files of CLAUDE.md section 6.
'''
import gc
import json
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

import gnn_frauds.api as core
import gnn_frauds.model.tabular.lightgbm as gbt_core
import yaml

from gnn_frauds.hyperparameters.gnns import conf_file
from gnn_frauds.hyperparameters.mlp import load as _load_mlp

# ---------------- fixed recipe of the cross-check (EXPERIMENTS.md appendix), never tuned per run ----------------
DEFAULT_KNOBS, SEARCH_SPACE = _load_mlp()
_random = yaml.safe_load(conf_file('optimization/random.yaml').read_text(encoding='utf-8'))
SEEDS         = tuple(int(s) for s in _random['final_seeds'])
SEARCH_SEEDS  = tuple(int(s) for s in _random['search_seeds'])
N_TRIALS      = int(_random['n_trials'])
DRAW_SEED     = int(_random['draw_seed'])
EPOCHS        = int(_random['epochs'])
PATIENCE      = int(_random['patience'])
SELECT        = _random['select']
FEATURE_SET   = 'full'             # the only input set of the MLP: all 81 edge + 6 + 6 node columns
N_EDGE_COLS   = 81                 # standardised columns; the 12 entity one-hots after them stay 0/1
STD_FLOOR     = 1e-3               # floor of the train std in the scaling
PREDICT_BATCH = 65_536             # rows per forward pass when scoring
PAPER_THRESHOLD = gbt_core.PAPER_THRESHOLD

TRIAL_COLS = ['trial', 'seed', 'shape', 'widths', 'params', 'best_epoch', 'epochs_trained',
              'val_pr_auc', 'val_f1', 'threshold', 'sec_per_epoch', 'seconds', 'knobs']


# ==================================================================================================
# data: the GBT matrix, scaled for a neural net, on the device
# ==================================================================================================

def standardise(data):
    '''Edge columns: (value - train mean) / train std (std floored at STD_FLOOR), in place.
    The 12 entity one-hot columns stay 0/1. Fit on the train rows only.'''
    X, train = data['X'], data['rows']['train']
    onehot = X[:, N_EDGE_COLS:]
    assert onehot.shape[1] == 12 and np.isin(onehot, (0.0, 1.0)).all(), 'entity columns must be 0/1'
    mean = X[train, :N_EDGE_COLS].mean(axis=0, dtype=np.float64)
    std = np.maximum(X[train, :N_EDGE_COLS].std(axis=0, dtype=np.float64), STD_FLOOR)
    edge = X[:, :N_EDGE_COLS]                      # a view: the scaling happens inside data['X']
    edge -= mean.astype(np.float32)
    edge /= std.astype(np.float32)
    assert not np.isnan(X).any()
    print(f'scaled {N_EDGE_COLS} edge columns on {train.stop - train.start:,} train rows '
          f'(std floor {STD_FLOOR}, {(std <= STD_FLOOR).sum()} columns at the floor); '
          f'largest |value| after scaling {np.abs(edge).max():.1f}; 12 one-hot columns untouched')
    return {'mean': mean, 'std': std}  # X is already scaled in place; this is only the train mean and std


def to_tensors(data):
    '''The scaled matrix and the labels as tensors on gnn_core.device (the whole matrix fits a Kaggle GPU).'''
    X = torch.from_numpy(data['X']).to(core.device)
    y = torch.from_numpy(data['y']).float().to(core.device)
    print(f'X {tuple(X.shape)} float32 ({X.numel() * 4 / 2**30:.2f} GB) and y on {core.device}')
    return X, y


# ==================================================================================================
# model: three shapes, one class
# ==================================================================================================

def layer_widths(hidden, depth, shape):
    '''Hidden widths: flat [h, h, h] · funnel [h, h/2, h/4] (floor 16) · resnet [h] * depth (the block width).'''
    assert shape in SEARCH_SPACE['shape'], f'unknown shape {shape!r}'
    if shape == 'funnel':
        return [max(16, hidden >> i) for i in range(depth)]  # halve the width at each layer, never below 16
    return [hidden] * depth


def widths_text(in_dim, widths, shape):
    '''"93 -> 512 -> 256 -> 128 -> 1", or "93 -> 256 (3 residual blocks) -> 1" for the resnet.'''
    if shape == 'resnet':
        return f'{in_dim} -> {widths[0]} ({len(widths)} residual blocks) -> 1'
    return ' -> '.join(str(w) for w in [in_dim] + widths + [1])


class ResBlock(nn.Module):
    '''z + Linear(Dropout(ReLU(Norm(Linear(z))))), both Linears width -> width (Gorishniy et al. 2021).'''

    def __init__(self, width, dropout, norm):
        super().__init__()
        self.block = nn.Sequential(nn.Linear(width, width), core.make_norm(norm, width), nn.ReLU(),
                                   nn.Dropout(dropout), nn.Linear(width, width))

    def forward(self, z):
        return z + self.block(z)


class MLP(nn.Module):
    '''flat / funnel: [Linear -> Norm -> ReLU -> Dropout] per hidden width, then Linear -> 1.
    resnet: Linear in_dim -> hidden, depth ResBlocks, then Linear -> 1. forward(x [B, in_dim]) -> logits [B].'''

    def __init__(self, in_dim, knobs):
        super().__init__()
        widths = layer_widths(knobs['hidden'], knobs['depth'], knobs['shape'])
        layers, last = [], in_dim
        if knobs['shape'] == 'resnet':
            layers.append(nn.Linear(in_dim, widths[0]))
            layers += [ResBlock(widths[0], knobs['dropout'], knobs['norm']) for _ in widths]
            last = widths[0]
        else:
            for width in widths:
                layers += [nn.Linear(last, width), core.make_norm(knobs['norm'], width), nn.ReLU(),
                           nn.Dropout(knobs['dropout'])]
                last = width
        self.net = nn.Sequential(*layers, nn.Linear(last, 1))
        self.widths = widths

    def forward(self, x):
        return self.net(x).squeeze(-1)


def draw_trials(n_trials, seed=DRAW_SEED, space=SEARCH_SPACE):
    '''{trial: knobs}: trial 0 = DEFAULT_KNOBS, the rest uniform over the space; a draw equal to an
    earlier trial is redrawn, so every setting is unique.'''
    rng = np.random.default_rng(seed)
    configs, seen, redraws = {0: dict(DEFAULT_KNOBS)}, {json.dumps(DEFAULT_KNOBS, sort_keys=True)}, 0
    while len(configs) < n_trials:
        knobs = {name: values[rng.integers(len(values))] for name, values in space.items()}
        knobs = {k: (v.item() if hasattr(v, 'item') else v) for k, v in knobs.items()}   # plain python types
        key = json.dumps(knobs, sort_keys=True)
        if key in seen:
            redraws += 1
            continue
        seen.add(key)
        configs[len(configs)] = knobs
    assert len(seen) == n_trials
    print(f'{n_trials} unique trials (trial 0 = the default knobs, {redraws} duplicate draws redrawn)')
    return configs


# ==================================================================================================
# one run: seeded shuffle, Adam at a constant LR, train / val scored every epoch, early stopping
# ==================================================================================================

@torch.no_grad()
def predict(model, X, rows):
    '''Eval-mode probabilities of the rows (a slice of X), in chunks -> float32 numpy.'''
    model.eval()
    probs = [torch.sigmoid(model(X[start:min(start + PREDICT_BATCH, rows.stop)])).cpu()
             for start in range(rows.start, rows.stop, PREDICT_BATCH)]
    return torch.cat(probs).numpy().astype(np.float32)


def train_run(knobs, X, y, rows, seed, epochs=EPOCHS, patience=PATIENCE):
    '''Train one MLP on the train rows; every epoch scores train and val (one history row, one printed
    line) and keeps the weights of the best validation PR-AUC epoch; stops after `patience` epochs
    without improvement. Returns (model with the best weights loaded, history, best) where best =
    {epoch, val_pr_auc, val_f1, threshold}.'''
    core.set_seed(seed)
    model = MLP(X.shape[1], knobs).to(X.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=knobs['lr'], weight_decay=knobs['weight_decay'])
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(float(knobs['pos_weight']), device=X.device))
    train, val = rows['train'], rows['val']
    y_train, y_val = (y[r].cpu().numpy().astype(np.int64) for r in (train, val))
    n_train, batch_size = train.stop - train.start, knobs['batch_size']
    print(f'{knobs["shape"]} {widths_text(X.shape[1], model.widths, knobs["shape"])} | params {core.count_params(model):,} | '
          f'seed {seed} | ' + ' '.join(f'{k}={v}' for k, v in knobs.items() if k not in ('shape', 'hidden', 'depth')))

    history, best, best_state = [], {'epoch': 0, 'val_pr_auc': -1.0}, None
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        order = train.start + torch.randperm(n_train, device=X.device)
        batch_losses = []
        for start in range(0, n_train, batch_size):
            idx = order[start:start + batch_size]
            optimizer.zero_grad()
            loss = criterion(model(X[idx]), y[idx])
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.detach())
        p_train, p_val = predict(model, X, train), predict(model, X, val)
        threshold, val_f1 = core.best_f1_threshold(y_val, p_val)
        row = {
            'epoch':        epoch,
            'train_loss':   core.weighted_bce(y_train, p_train),      # the thesis diagnostic, weight 8
            'val_loss':     core.weighted_bce(y_val, p_val),
            'train_f1':     float(f1_score(y_train, (p_train >= threshold).astype(int))),
            'val_f1':       val_f1,
            'train_pr_auc': float(average_precision_score(y_train, p_train)),
            'val_pr_auc':   float(average_precision_score(y_val, p_val)),
            'val_roc_auc':  float(roc_auc_score(y_val, p_val)),
            'threshold':    threshold,
            'seconds':      time.time() - t0,
            'train_loss_batches': float(torch.stack(batch_losses).mean()),   # train mode, the trial's pos_weight
            'lr':           knobs['lr'],
        }
        history.append(row)
        mark = ''
        if row['val_pr_auc'] > best['val_pr_auc']:
            best = {'epoch': epoch, 'val_pr_auc': row['val_pr_auc'], 'val_f1': val_f1, 'threshold': threshold}
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            mark = '  * new best'
        print(f"epoch {epoch:3d} | loss {row['train_loss']:.4f} / {row['val_loss']:.4f} | "
              f"F1 {row['train_f1']:.4f} / {row['val_f1']:.4f} @ {threshold:.3f} | "
              f"PR-AUC {row['train_pr_auc']:.4f} / {row['val_pr_auc']:.4f} | {row['seconds']:.1f}s   (train / val){mark}")
        if epoch - best['epoch'] >= patience:
            print(f'early stopping: no better validation PR-AUC for {patience} epochs')
            break
    model.load_state_dict(best_state)
    sec_per_epoch = float(np.mean([r['seconds'] for r in history]))
    print(f'best epoch {best["epoch"]} (val PR-AUC {best["val_pr_auc"]:.4f}, F1 {best["val_f1"]:.4f} @ {best["threshold"]:.3f}) '
          f'| {len(history)} epochs trained | {sec_per_epoch:.1f} s per epoch')
    return model, history, best


# ==================================================================================================
# random search — train rows only, scored on validation; one line per (trial, seed) in trials.csv
# ==================================================================================================

def random_search(data, X, y, out_dir, n_trials=N_TRIALS, seeds=SEARCH_SEEDS):
    '''Pick the knobs: every trial is trained with each search seed; a trial's score is the mean of its
    seeds' best validation PR-AUC. Finished (trial, seed) pairs in <out_dir>/trials.csv are skipped, so
    a stopped search resumes. Writes configs.json and best_params.json; returns (best knobs, trials).'''
    os.makedirs(out_dir, exist_ok=True)
    configs = draw_trials(n_trials)
    config_path = f'{out_dir}/configs.json'
    if os.path.exists(config_path):
        saved = {int(k): v for k, v in json.load(open(config_path)).items()}
        assert saved == configs, 'configs.json does not match the seeded draw; delete the search folder to restart'
    else:
        json.dump(configs, open(config_path, 'w'), indent=1)

    log_path = f'{out_dir}/trials.csv'
    trials = pd.read_csv(log_path) if os.path.exists(log_path) else pd.DataFrame(columns=TRIAL_COLS)
    print(f'random search: {n_trials} trials x seeds {seeds} | ranked by {SELECT} | cap {EPOCHS} epochs, '
          f'patience {PATIENCE} | {len(trials)} fits already logged')
    for trial in range(n_trials):
        for seed in seeds:
            if ((trials['trial'] == trial) & (trials['seed'] == seed)).any():
                continue
            print(f'--- trial {trial} seed {seed}')
            t0 = time.time()
            model, history, best = train_run(configs[trial], X, y, data['rows'], seed)
            row = {'trial': trial, 'seed': seed, 'shape': configs[trial]['shape'],
                   'widths': widths_text(X.shape[1], model.widths, configs[trial]['shape']),
                   'params': core.count_params(model), 'best_epoch': best['epoch'], 'epochs_trained': len(history),
                   'val_pr_auc': best['val_pr_auc'], 'val_f1': best['val_f1'], 'threshold': best['threshold'],
                   'sec_per_epoch': float(np.mean([r['seconds'] for r in history])), 'seconds': time.time() - t0,
                   'knobs': json.dumps(configs[trial])}
            new_row = pd.DataFrame([row])
            new_row.to_csv(log_path, mode='a', header=not os.path.exists(log_path), index=False)
            trials = pd.concat([trials, new_row], ignore_index=True) if len(trials) else new_row
            del model
            gc.collect()

    # winner: highest mean val PR-AUC among the trials with every seed done; tie-break val F1, then the lower id
    scores = trials.groupby('trial').agg(val_pr_auc=('val_pr_auc', 'mean'), val_f1=('val_f1', 'mean'),
                                         n_seeds=('seed', 'count')).reset_index()
    scores = scores[scores['n_seeds'] == len(seeds)]
    ranked = scores.sort_values(['val_pr_auc', 'val_f1', 'trial'], ascending=[False, False, True])
    best_trial = int(ranked['trial'].iloc[0])
    best = {'trial': best_trial, 'select': SELECT, 'val_pr_auc': float(ranked['val_pr_auc'].iloc[0]),
            'val_f1': float(ranked['val_f1'].iloc[0]), 'knobs': configs[best_trial],
            'n_trials': n_trials, 'seeds': list(seeds), 'epochs': EPOCHS, 'patience': PATIENCE, 'draw_seed': DRAW_SEED}
    json.dump(best, open(f'{out_dir}/best_params.json', 'w'), indent=2)
    print(f'best configuration: trial {best_trial} ({SELECT} {best["val_pr_auc"]:.4f}, val F1 {best["val_f1"]:.4f})')
    print(json.dumps(configs[best_trial], indent=1))
    return configs[best_trial], trials


# ==================================================================================================
# one final run: the chosen knobs, one seed, best-val-PR-AUC epoch kept, test scored once
# ==================================================================================================

def run_name_mlp(seed, feature_set=FEATURE_SET):
    '''mlp_readout-<set>_seed-<seed> (CLAUDE.md §6).'''
    return f'mlp_readout-{feature_set}_seed-{seed}'


def mlp_result_row(name, seed, best_epoch, threshold, n_params, metrics):
    '''The flat result row with every batch_summary.csv column; invariant_params = 0 as for the GBT.'''
    row = {'run': name, 'operator': 'mlp', 'mp': 'none', 'readout': FEATURE_SET, 'mp_direction': 'none',
           'temporal_sampling': False, 'node_enc': 'none', 'gfp_variant': 'v0',
           'seed': seed, 'best_epoch': best_epoch, 'threshold': threshold,
           'params': n_params, 'invariant_params': 0}
    row.update(core.flatten_metrics(metrics))
    return row


def train_final(knobs, data, X, y, seed, out_root):
    '''Train one seed with the chosen knobs, keep the best-validation-PR-AUC epoch, score test once.

    Writes results.json (the row + hyperparams + F1 at the paper's 0.5), history.csv (one row per
    epoch), curves.png, best.pt and predictions.csv to <out_root>/<run>/. A run whose results.json
    exists is skipped (resume). Returns the flat result row.
    '''
    name = run_name_mlp(seed)
    out_dir = f'{out_root}/{name}'
    if os.path.exists(f'{out_dir}/results.json'):
        results = json.load(open(f'{out_dir}/results.json'))
        print(f'=== {name} === already finished (results.json found), skipping')
        return {k: v for k, v in results.items() if k not in ('hyperparams', 'paper_comparison')}
    os.makedirs(out_dir, exist_ok=True)
    print(f'=== {name} ===')
    model, history, best = train_run(knobs, X, y, data['rows'], seed)
    best_epoch, threshold = best['epoch'], best['threshold']
    pd.DataFrame(history).to_csv(f'{out_dir}/history.csv', index=False)
    torch.save(model.state_dict(), f'{out_dir}/best.pt')

    # final evaluation: the kept epoch, one validation-chosen threshold, test scored once
    metrics, probs, at_half = {}, {}, {}
    for split in core.SPLITS:
        rows = data['rows'][split]
        probs[split] = predict(model, X, rows)
        metrics[split] = core.compute_metrics(data['y'][rows], probs[split], threshold)
        at_half[f'{split}_f1_at_0.5'] = float(f1_score(data['y'][rows], (probs[split] >= PAPER_THRESHOLD).astype(int)))
    row = mlp_result_row(name, seed, best_epoch, threshold, core.count_params(model), metrics)
    row['model_file'] = 'best.pt'
    results = dict(row)
    results['hyperparams'] = dict(knobs, widths=widths_text(X.shape[1], model.widths, knobs['shape']),
                                  epochs=EPOCHS, patience=PATIENCE, select=SELECT, optimizer='adam', lr_schedule='constant')
    results['paper_comparison'] = at_half
    json.dump(results, open(f'{out_dir}/results.json', 'w'), indent=2)
    print(pd.DataFrame({split: {m: row[f'{split}_{m}'] for m in core.METRICS} for split in core.SPLITS}).round(4))
    gbt_core.save_gbt_predictions(out_dir, data, probs, threshold)
    core.plot_curves(history, data['y'][data['rows']['test']], probs['test'], metrics['test']['pr_auc'],
                     best_epoch, name, f'{out_dir}/curves.png', select_metric='pr_auc')
    del model, probs
    gc.collect()
    return row


def seed_summary(rows, out_root, filename='seed_summary_mlp.csv'):
    '''Mean and std over the seeds of every metric column -> <out_root>/<filename>; the GNN helper, MLP file name.'''
    return core.seed_summary(rows, out_root, filename)


class MLPModel:
    """The tabular net. The nn.Module in this file is still called MLP."""

    def __init__(self, name='mlp'):
        self.name = name

    def train(self, run):
        """One fit at the fixed point, then predictions for every evaluated transaction."""
        from omegaconf import OmegaConf
        from gnn_frauds.model.gnns.run import out_root, write_resolved
        data = gbt_core.load_gbt_data(run.paths.data_dir)
        data = standardise(data)
        X, y = to_tensors(data)
        knobs = OmegaConf.to_container(run.hyperparameters.fixed, resolve=True)
        row = train_final(knobs, data, X, y, int(run.seed), out_root(run, self.name))
        write_resolved(run, f"{out_root(run, self.name)}/{row['run']}")
        return row

    def tune(self, run):
        """Unique uniform draws. Trial 0 is the fixed point."""
        global N_TRIALS, DRAW_SEED, EPOCHS, PATIENCE, SELECT, SEARCH_SEEDS
        from gnn_frauds.model.gnns.run import out_root
        N_TRIALS = run.optimization.n_trials
        DRAW_SEED = run.optimization.draw_seed
        EPOCHS = run.optimization.epochs
        PATIENCE = run.optimization.patience
        SELECT = run.optimization.select
        SEARCH_SEEDS = run.optimization.search_seeds
        data = gbt_core.load_gbt_data(run.paths.data_dir)
        data = standardise(data)
        X, y = to_tensors(data)
        return random_search(data, X, y, f'{out_root(run, self.name)}/search', n_trials=N_TRIALS, seeds=SEARCH_SEEDS)

    def retrain(self, run, seed):
        return self.train(run)
