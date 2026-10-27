"""One selecting run: train, keep the best epoch, score every split."""
import gc
import json
import os
import random
import time

import numpy as np
import pandas as pd
import psutil
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from tqdm.auto import tqdm

from gnn_frauds.dataset.hi_small import in_degree_histogram, set_seed
from gnn_frauds.experiment.metrics import (
    best_f1_threshold, compute_metrics, flatten_metrics, forward_batch, make_loader, make_loaders,
    plot_curves, predict, save_predictions, score_all_splits, top_k_metrics, weighted_bce,
)
from gnn_frauds.hyperparameters.gnns import (
    BASE_NODE_DIM, GFP_WIDTH, METRIC_COLS, METRICS, RUN_FIELDS, SEED, SPLITS, full_recipe,
)
from gnn_frauds.model.gnns.classifier import build_model, count_params, device, invariant_params
from gnn_frauds.preprocessing.node_embedding import set_node_encoding

def train_epoch(model, loader, feats, optimizer, criterion):
    '''One pass over the loader in train mode; returns the mean mini-batch loss (train mode, dropout on).'''
    model.train()
    total, count = 0.0, 0
    for batch in tqdm(loader, desc='train', leave=False):
        optimizer.zero_grad()
        logits, y = forward_batch(model, batch, feats)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total, count = total + float(loss.detach()) * len(y), count + len(y)  # weight by batch size; the last batch is shorter
    return total / max(1, count)


def smoothed(values, window=3):
    '''Centred moving average (window 3 by default, shorter at the ends) of a 1-d sequence.'''
    return pd.Series(np.asarray(values, dtype=float)).rolling(window, center=True, min_periods=1).mean().to_numpy()


def make_scheduler(optimizer, recipe):
    '''cosine (v1): CosineAnnealingLR over recipe['epochs']; constant: the LR never moves (Luo et al. 2024, tunedGNN).'''
    r = full_recipe(recipe)
    if r['lr_schedule'] == 'constant':
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 1.0)
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=r['epochs'])


def write_json_atomic(path, obj):
    '''json.dump through a temporary file + os.replace, so a killed job never leaves a half-written file.'''
    tmp = f'{path}.tmp{os.getpid()}'
    with open(tmp, 'w') as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def rng_state():
    '''Every random-number state a run consumes: python, numpy, torch CPU (neighbour sampling, shuffling), CUDA.'''
    return {'python': random.getstate(), 'numpy': np.random.get_state(), 'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def set_rng_state(state):
    '''Inverse of rng_state(); the torch states must be CPU ByteTensors.'''
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if state['cuda'] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state['cuda'])


def save_resume(path, model, optimizer, scheduler, loop_state):
    '''resume.pt: model, optimizer, scheduler, RNG and the loop state, written atomically at the end of an epoch.'''
    state = dict(loop_state, model=model.state_dict(), optimizer=optimizer.state_dict(),
                 scheduler=scheduler.state_dict(), rng=rng_state())
    tmp = f'{path}.tmp{os.getpid()}'
    torch.save(state, tmp)
    os.replace(tmp, path)


def load_resume(path, model, optimizer, scheduler):
    '''Restore model, optimizer, scheduler and RNG from resume.pt; returns the loop state (history, best, ...).'''
    state = torch.load(path, map_location='cpu', weights_only=False)   # our own file; RNG states must stay on the CPU
    model.load_state_dict(state['model'])
    optimizer.load_state_dict(state['optimizer'])
    scheduler.load_state_dict(state['scheduler'])
    set_rng_state(state['rng'])
    return state


def add_full_metrics(row, split, y, p, threshold):
    '''Every section-4 metric of one split at the epoch's validation threshold, added to a history row
    (columns the row already has, e.g. the sweep's val_f1, are kept).'''
    for name, value in compute_metrics(y, p, threshold).items():
        row.setdefault(f'{split}_{name}', value)  # keep the sweep's F1 and PR-AUC; only fill columns the row lacks


def train_one_run(model, loaders, out_dir, recipe=None, stop_epoch=None, eval_train_every=1,
                  patience=None, min_epochs=20, keep_epochs=False, resume_path=None, full_metrics=False):
    '''The training loop: per-epoch validation threshold sweep, best-validation checkpoint (best.pt), history.csv.

    recipe=None -> v1. recipe['epochs'] is the schedule length; the loop runs to stop_epoch (default: every
    epoch) or stops early when `patience` is set and the 3-epoch moving average of the validation metric has not
    improved for that many epochs (never before min_epochs). recipe['select_metric'] (val F1 = v1, or val PR-AUC)
    picks the best epoch; the returned threshold is always that epoch's validation-F1 sweep. The train split is
    scored every eval_train_every epochs (+ the last). keep_epochs saves epoch_<e>.pt every epoch; resume_path
    writes resume.pt after every epoch and continues from it when it exists; full_metrics adds every section-4
    metric of train and val to each row. Non-finite predictions (a diverged run) stop the loop.
    Returns (history rows, best_epoch, best_threshold).
    '''
    r = full_recipe(recipe)
    metric = r['select_metric']
    metric_name = {'f1': 'F1', 'pr_auc': 'PR-AUC'}[metric]
    train_loader, train_feats = loaders['train']
    val_loader,   val_feats   = loaders['val']
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([r['pos_weight']], device=device))
    optimizer = torch.optim.Adam(model.parameters(), lr=r['lr'], weight_decay=r['weight_decay'])
    scheduler = make_scheduler(optimizer, r)
    last_epoch = r['epochs'] if stop_epoch is None else int(stop_epoch)
    assert 1 <= last_epoch <= r['epochs'], f'stop_epoch {last_epoch} must lie inside the schedule (1..{r["epochs"]})'

    history = []
    best_score, best_epoch, best_threshold = -1.0, 0, 0.5
    best_smoothed, epochs_without_gain, stopped = -1.0, 0, False
    if resume_path is not None and os.path.exists(resume_path):
        state = load_resume(resume_path, model, optimizer, scheduler)
        history, stopped = state['history'], state['stopped']
        best_score, best_epoch, best_threshold = state['best']
        best_smoothed, epochs_without_gain = state['patience']
        print(f'resumed from {resume_path}: {len(history)} epochs done' + (' (the run had stopped)' if stopped else ''))
    for epoch in range(len(history) + 1, last_epoch + 1):
        if stopped:
            break
        t0 = time.time()
        lr_now = optimizer.param_groups[0]['lr']  # the LR this epoch trains with; the schedule steps after the metrics
        loss_batches = train_epoch(model, train_loader, train_feats, optimizer, criterion)

        # train and val scored in eval mode; train F1 reuses the val threshold (overfitting diagnostics)
        score_train = (epoch % eval_train_every == 0) or epoch == last_epoch
        if score_train:
            y_train, p_train, _ = predict(model, train_loader, train_feats, 'train')
        y_val, p_val, _ = predict(model, val_loader, val_feats, 'val')
        if not (np.isfinite(p_val).all() and (not score_train or np.isfinite(p_train).all())):
            print(f'non-finite predictions at epoch {epoch}: the run diverged, training stops here')
            stopped = True
        else:
            threshold, val_f1 = best_f1_threshold(y_val, p_val)
            train_f1 = f1_score(y_train, (p_train >= threshold).astype(int)) if score_train else float('nan')
            val_pr_auc = average_precision_score(y_val, p_val)

            score = val_f1 if metric == 'f1' else val_pr_auc
            is_best = score > best_score
            if is_best:
                best_score, best_epoch, best_threshold = score, epoch, threshold
                torch.save(model.state_dict(), f'{out_dir}/best.pt')
            if keep_epochs:
                torch.save(model.state_dict(), f'{out_dir}/epoch_{epoch:03d}.pt')

            history.append({
                'epoch':        epoch,
                'train_loss':   weighted_bce(y_train, p_train, r['pos_weight']) if score_train else float('nan'),
                'val_loss':     weighted_bce(y_val, p_val, r['pos_weight']),
                'train_f1':     train_f1,
                'val_f1':       val_f1,
                'train_pr_auc': average_precision_score(y_train, p_train) if score_train else float('nan'),
                'val_pr_auc':   val_pr_auc,
                'val_roc_auc':  roc_auc_score(y_val, p_val),
                'threshold':    threshold,
                'seconds':      time.time() - t0,
                'train_loss_batches': loss_batches,
                'lr':           lr_now,
            })
            row = history[-1]
            if full_metrics:
                add_full_metrics(row, 'val', y_val, p_val, threshold)
                if score_train:
                    add_full_metrics(row, 'train', y_train, p_train, threshold)
            ram_gb = psutil.Process().memory_info().rss / 2**30
            train_str = f"{row['train_loss']:.4f}" if score_train else '  -   '
            f1_str = f"{row['train_f1']:.4f}" if score_train else '  -   '
            pr_str = f"{row['train_pr_auc']:.4f}" if score_train else '  -   '
            print(f"epoch {epoch:02d} | loss {train_str} / {row['val_loss']:.4f} | "
                  f"F1 {f1_str} / {row['val_f1']:.4f} @ {threshold:.3f} | "
                  f"PR-AUC {pr_str} / {row['val_pr_auc']:.4f} | "
                  f"{row['seconds']:.0f}s | RAM {ram_gb:.1f}GB   (train / val)"
                  + ('   * new best -> best.pt' if is_best else ''))
            scheduler.step()

            if patience is not None:
                # early stopping on the smoothed validation curve of the selection metric, never before min_epochs
                current = smoothed([h[f'val_{metric}'] for h in history])[-1]
                if current > best_smoothed:
                    best_smoothed, epochs_without_gain = current, 0
                else:
                    epochs_without_gain += 1
                if epoch >= min_epochs and epochs_without_gain >= patience:
                    print(f'early stop at epoch {epoch}: smoothed val {metric_name} has not improved for {patience} epochs')
                    stopped = True
        if resume_path is not None:
            save_resume(resume_path, model, optimizer, scheduler,
                        dict(history=history, stopped=stopped, best=(best_score, best_epoch, best_threshold),
                             patience=(best_smoothed, epochs_without_gain)))
        if stopped:
            break

    if history:
        pd.DataFrame(history).to_csv(f'{out_dir}/history.csv', index=False)
    print(f'best validation {metric_name} {best_score:.4f} at epoch {best_epoch} (threshold {best_threshold:.3f}) '
          f'| {len(history)} of {r["epochs"]} epochs run')
    return history, best_epoch, best_threshold


def train_fixed_epochs(model, fit_loader, out_dir, recipe, stop_epoch, resume_path=None, eval_loaders=None):
    '''Final retrain: stop_epoch epochs on the train + val loader with the recipe's LR schedule; no model selection.

    Writes final.pt and history.csv (epoch, train_loss_batches, lr, seconds); returns the history rows.
    eval_loaders {split: (loader, feats)} adds eval-mode <split>_loss and <split>_pr_auc per epoch, for the curves
    only (val is training data here). resume_path: resume.pt after every epoch, continued from when it exists.
    '''
    r = full_recipe(recipe)
    assert 1 <= stop_epoch <= r['epochs'], f'stop_epoch {stop_epoch} must lie inside the schedule (1..{r["epochs"]})'
    loader, feats = fit_loader
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([r['pos_weight']], device=device))
    optimizer = torch.optim.Adam(model.parameters(), lr=r['lr'], weight_decay=r['weight_decay'])
    scheduler = make_scheduler(optimizer, r)
    history = []
    if resume_path is not None and os.path.exists(resume_path):
        history = load_resume(resume_path, model, optimizer, scheduler)['history']
        print(f'resumed from {resume_path}: {len(history)} of {stop_epoch} epochs done')
    for epoch in range(len(history) + 1, stop_epoch + 1):
        t0 = time.time()
        lr_now = optimizer.param_groups[0]['lr']  # the LR this epoch trains with; the schedule steps after the metrics
        loss = train_epoch(model, loader, feats, optimizer, criterion)
        assert np.isfinite(loss), f'non-finite training loss at epoch {epoch}: the retrain diverged'
        row = {'epoch': epoch, 'train_loss_batches': loss, 'lr': lr_now}
        for split, (eval_loader, eval_feats) in (eval_loaders or {}).items():
            y, p, _ = predict(model, eval_loader, eval_feats, split)
            assert np.isfinite(p).all(), f'non-finite {split} predictions at epoch {epoch}: the retrain diverged'
            row[f'{split}_loss'] = weighted_bce(y, p, r['pos_weight'])
            row[f'{split}_pr_auc'] = average_precision_score(y, p)
        row['seconds'] = time.time() - t0
        history.append(row)
        ram_gb = psutil.Process().memory_info().rss / 2**30
        evals = ''.join(f" | {s} loss {row[f'{s}_loss']:.4f} PR-AUC {row[f'{s}_pr_auc']:.4f}" for s in (eval_loaders or {}))
        print(f'epoch {epoch:02d}/{stop_epoch} | batch loss {loss:.4f} | lr {lr_now:.2e}{evals} | '
              f'{row["seconds"]:.0f}s | RAM {ram_gb:.1f}GB')
        scheduler.step()
        if resume_path is not None:
            save_resume(resume_path, model, optimizer, scheduler, dict(history=history))
    torch.save(model.state_dict(), f'{out_dir}/final.pt')
    pd.DataFrame(history).to_csv(f'{out_dir}/history.csv', index=False)
    return history


# ==================================================================================================
# plots, saving, naming
# ==================================================================================================

def run_name(operator, cfg, mp_direction='in', temporal=False, node_enc='none', gfp_variant='v0'):
    '''<model>_mp-<mp>_readout-<ro>_dir-in[_enc-<enc>][_gfp-<variant>] (CLAUDE.md section 6); the optional parts
    appear only when the knob is not at its default.

    Message passing is always over incoming edges without temporal sampling; mp_direction and temporal stay in the
    signature because the notebooks pass them, and any other value is an error.
    '''
    assert mp_direction == 'in' and not temporal, 'only mp_direction="in" without temporal sampling is implemented'
    name = f"{operator}_mp-{cfg['mp']}_readout-{cfg['readout']}_dir-in"
    if node_enc != 'none':
        name += f'_enc-{node_enc}'
    if gfp_variant != 'v0':
        name += f'_gfp-{gfp_variant}'
    return name


def result_row(name, operator, cfg, best_epoch, threshold, n_params, n_invariant, metrics,
               node_enc='none', gfp_variant='v0', seed=None):
    '''The flat result row shared by results.json and batch_summary.csv (mp_direction / temporal_sampling are
    constant columns, kept so old and new result files have the same layout).'''
    row = {'run': name, 'operator': operator, 'mp': cfg['mp'], 'readout': cfg['readout'],
           'mp_direction': 'in', 'temporal_sampling': False, 'node_enc': node_enc,
           'gfp_variant': gfp_variant, 'seed': SEED if seed is None else seed,
           'best_epoch': best_epoch, 'threshold': threshold,
           'params': n_params, 'invariant_params': n_invariant}
    row.update(flatten_metrics(metrics))
    return row


def save_run(out_dir, row, recipe=None, extra=None):
    '''Write results.json (the flat row, the architecture block, the full recipe, extras) and print the per-split table.'''
    r = full_recipe(recipe)
    results = dict(row)
    results['architecture'] = {'hidden': r['hidden'], 'layers': r['num_layers'], 'dropout': r['dropout'],
                               'neighbors': r['num_neighbors']}
    results['recipe'] = r
    results.update(extra or {})
    json.dump(results, open(f'{out_dir}/results.json', 'w'), indent=2)
    table = {split: {metric: row[f'{split}_{metric}'] for metric in METRICS} for split in SPLITS}
    print(pd.DataFrame(table).round(4))


def print_model_header(name, mp_idx, readout_idx, n_params, n_invariant, node_dim=BASE_NODE_DIM):
    '''One-line description of the run and its parameter fingerprint.'''
    print(f'=== {name} ===')
    print(f'MP edge features: {len(mp_idx)} | readout edge features: {len(readout_idx)} | node features: {node_dim}')
    print(f'total params: {n_params:,} | architecture-invariant: {n_invariant:,}')


# ==================================================================================================
# one run, one batch summary
# ==================================================================================================

def run_experiment(operator, cfg, graphs, col_sets, node_dim, out_root, mp_direction='in', temporal=False,
                   node_enc='none', enc_dir=None, gfp_variant='v0', recipe=None, eval_train_every=1, seed=None):
    '''Train and evaluate one knob configuration end-to-end (seed 42); returns the flat result row.

    node_enc = 'none' leaves graph.x and node_dim as loaded; 'rwpe<k>' appends the k RWPE columns from enc_dir,
    so only node_proj widens (6 + k -> 128). gfp_variant must be the one the graphs were loaded with
    (load_graphs); it only labels the run, the swap happened at load time. recipe=None reproduces the v1 runs.
    '''
    assert len(col_sets['gfp']) == GFP_WIDTH[gfp_variant], \
        f"graphs hold a {len(col_sets['gfp'])}-column GFP block, not the {gfp_variant!r} sheet; reload with load_graphs"
    name = run_name(operator, cfg, mp_direction, temporal, node_enc, gfp_variant)
    out_dir = f'{out_root}/{name}'
    os.makedirs(out_dir, exist_ok=True)
    mp_idx, readout_idx = col_sets[cfg['mp']], col_sets[cfg['readout']]
    node_dim = set_node_encoding(graphs, node_enc, enc_dir)   # 6 for 'none', 6 + k for 'rwpe<k>'

    run_seed = SEED if seed is None else int(seed)
    set_seed(run_seed)
    deg_hist = in_degree_histogram(graphs['train']) if operator == 'pna' else None
    model = build_model(operator, node_dim, len(mp_idx), len(readout_idx), deg_hist=deg_hist, recipe=recipe)
    n_params, n_invariant = count_params(model), invariant_params(model)
    print_model_header(name, mp_idx, readout_idx, n_params, n_invariant, node_dim)

    loaders = make_loaders(graphs, mp_idx, readout_idx, recipe)
    history, best_epoch, threshold = train_one_run(model, loaders, out_dir, recipe, eval_train_every=eval_train_every)

    # final evaluation: best checkpoint, one validation-chosen threshold, test scored once
    model.load_state_dict(torch.load(f'{out_dir}/best.pt', map_location=device))
    metrics, preds = score_all_splits(model, loaders, graphs, threshold, seed=run_seed)
    row = result_row(name, operator, cfg, best_epoch, threshold, n_params, n_invariant, metrics, node_enc, gfp_variant, seed=run_seed)
    row['model_file'] = 'best.pt'
    save_run(out_dir, row, recipe)
    save_predictions(out_dir, graphs, preds, threshold)
    plot_curves(history, preds['test']['y'], preds['test']['prob'], metrics['test']['pr_auc'],
                best_epoch, name, f'{out_dir}/curves.png')

    del model, loaders
    gc.collect()
    torch.cuda.empty_cache()
    return row


def summarize_batch(rows, out_root, filename='batch_summary.csv', check_invariant=True):
    '''Write the batch summary CSV; check every §4 column is present and (by default) invariant_params is identical.'''
    summary = pd.DataFrame(rows)
    missing = [c for c in METRIC_COLS + list(RUN_FIELDS) if c not in summary.columns]
    assert not missing, f'batch_summary.csv is missing columns: {missing}'
    if check_invariant:
        assert summary['invariant_params'].nunique() == 1, \
            f"invariant_params differ across runs: {summary['invariant_params'].tolist()}"
    summary.to_csv(f'{out_root}/{filename}', index=False)
    print(f'{len(summary)} runs -> {out_root}/{filename} | invariant_params = '
          f"{int(summary['invariant_params'].iloc[0]):,} in the first row"
          + (' (identical in every row)' if check_invariant else ''))
    return summary


def seed_summary(rows, out_root, filename='seed_summary.csv'):
    '''Mean and std over the seeds of every metric column (one configuration) -> <out_root>/<filename>; returns the one-row table.'''
    summary = pd.DataFrame(rows)
    out = {'run': summary['run'].iloc[0], 'operator': summary['operator'].iloc[0], 'mp': summary['mp'].iloc[0],
           'readout': summary['readout'].iloc[0], 'n_seeds': len(summary),
           'seeds': ' '.join(str(s) for s in summary['seed'])}
    extra = [c for c in summary.columns
             if c.startswith(('seen_', 'transfer_')) or c == 'val_seen_threshold']   # final retrain only
    for col in list(METRIC_COLS) + ['best_epoch', 'threshold'] + extra:
        out[f'{col}_mean'] = float(summary[col].mean())
        out[f'{col}_std'] = float(summary[col].std(ddof=1)) if len(summary) > 1 else 0.0
    table = pd.DataFrame([out])
    table.to_csv(f'{out_root}/{filename}', index=False)
    print(f'{len(summary)} seeds -> {out_root}/{filename}')
    return table


# ==================================================================================================
# Hydra entry: one selecting run (conf/experiment/train.yaml)
# ==================================================================================================


class TrainRun:
    """One selecting run. Hydra builds this from conf/experiment/train.yaml."""

    def __init__(self, model, features, hyperparameters, optimization, dataset, preprocessing, paths, seed):
        self.model = model
        self.features = features
        self.hyperparameters = hyperparameters
        self.optimization = optimization
        self.dataset = dataset
        self.preprocessing = preprocessing
        self.paths = paths
        self.seed = int(seed)
        self.cfg = None

    def run(self):
        """Train the selected model and score every split."""
        return self.model.train(self)

