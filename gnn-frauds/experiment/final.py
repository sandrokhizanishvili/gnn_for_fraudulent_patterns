"""Frozen-epoch retrain on the train + val graph, then one score of every split."""
import gc
import json
import os

import numpy as np
import pandas as pd
import torch

from gnn_frauds.dataset.hi_small import in_degree_histogram, make_trainval_graph, set_seed
from gnn_frauds.experiment.metrics import (
    best_f1_threshold, compute_metrics, make_loader, make_loaders, plot_final_curves,
    save_predictions, score_all_splits, threshold_sensitivity,
)
from gnn_frauds.hyperparameters.gnns import SPLITS
from gnn_frauds.experiment.train import (
    full_recipe, print_model_header, result_row, run_name, save_run, train_fixed_epochs,
)
from gnn_frauds.hyperparameters.gnns import full_recipe as _full_recipe
from gnn_frauds.model.gnns.classifier import build_model, count_params, invariant_params
from gnn_frauds.preprocessing.node_embedding import set_node_encoding

full_recipe = _full_recipe

def save_recipe(path, operator, cfg, trial, retrain_seeds=(42, 43, 44, 45), extra=None):
    '''recipe.json: the frozen knobs, best epoch, per-seed thresholds and the validation F1 that chose them.

    One threshold for every retrain seed: the mean of the tuning seeds' validation thresholds at the best epoch
    (the per-seed tuning thresholds are kept in 'tuning_thresholds' for the record). `extra` is merged in.
    '''
    tuned = {str(s): float(trial['thresholds'][str(s)]) for s in trial['seeds']}
    mean_threshold = float(np.mean(list(tuned.values())))
    thresholds = {str(s): mean_threshold for s in retrain_seeds}
    frozen = {'operator': operator, 'cfg': cfg, 'recipe': full_recipe(trial['recipe']),
              'best_epoch': int(trial['best_epoch']), 'tuning_seeds': list(trial['seeds']),
              'retrain_seeds': list(retrain_seeds), 'thresholds': thresholds, 'tuning_thresholds': tuned,
              'threshold_source': 'one threshold for every retrain seed: the mean of the tuning seeds\' '
                                  'validation thresholds at best_epoch',
              'val_f1_mean': trial['val_f1_mean'], 'val_f1_std': trial['val_f1_std'],
              'train_f1_mean': trial['train_f1_mean'], 'epochs_run': trial['epochs_run'],
              'source_stage': trial['stage'], 'source_trial': trial['trial']}
    frozen.update(extra or {})
    json.dump(frozen, open(path, 'w'), indent=2)
    print(f'frozen recipe -> {path}\n{json.dumps(frozen, indent=1)}')
    return frozen


def load_recipe(path):
    '''Read recipe.json and check it is complete; a missing file stops with a clear message.'''
    assert os.path.exists(path), (f'{path} not found: run the tuning notebook to the end (save_recipe) and '
                                  f'commit Outputs/TUNING/<FAMILY>/recipe.json before the final retrain')
    frozen = json.load(open(path))
    frozen['recipe'] = full_recipe(frozen['recipe'])
    assert 1 <= frozen['best_epoch'] <= frozen['recipe']['epochs'], 'best_epoch must lie inside the cosine schedule'
    assert set(frozen['thresholds']) == {str(s) for s in frozen['retrain_seeds']}, 'one threshold per retrain seed'
    assert all(0 < t < 1 for t in frozen['thresholds'].values())
    print(f"frozen recipe {frozen['operator']} {frozen['cfg']}: {frozen['recipe']} | best epoch {frozen['best_epoch']} "
          f"| thresholds {frozen['thresholds']} | val F1 {frozen['val_f1_mean']:.4f} ± {frozen['val_f1_std']:.4f}")
    return frozen


def final_scoring(preds, threshold, transfer_threshold):
    '''Metrics of a final run from its predictions: every split at `threshold` (the headline), the thresholded metrics
    at `transfer_threshold` (the tuned config's cut, secondary; None skips this view) and at the threshold re-swept
    on the val rows (seen in training here, secondary), and the test sensitivity table.
    Returns (metrics, extra row fields, sensitivity rows).'''
    metrics = {split: compute_metrics(preds[split]['y'], preds[split]['prob'], threshold) for split in SPLITS}
    sensitivity = threshold_sensitivity(preds['test']['y'], preds['test']['prob'], threshold)
    extra = {}
    if transfer_threshold is not None:
        extra['transfer_threshold'] = float(transfer_threshold)
        tr = {split: compute_metrics(preds[split]['y'], preds[split]['prob'], transfer_threshold) for split in SPLITS}
        extra.update({f'transfer_{split}_{m}': tr[split][m] for split in SPLITS for m in ('f1', 'precision', 'recall')})
        sensitivity.append({'point': 'tuned-config threshold', 'threshold': float(transfer_threshold),
                            'delta': float(transfer_threshold - threshold), 'f1': tr['test']['f1'],
                            'precision': tr['test']['precision'], 'recall': tr['test']['recall']})
    seen_threshold, _ = best_f1_threshold(preds['val']['y'], preds['val']['prob'])
    seen = {split: compute_metrics(preds[split]['y'], preds[split]['prob'], seen_threshold) for split in SPLITS}
    extra['val_seen_threshold'] = seen_threshold
    extra.update({f'seen_{split}_{m}': seen[split][m] for split in SPLITS for m in ('f1', 'precision', 'recall')})
    sensitivity.append({'point': 'val seen sweep', 'threshold': seen_threshold, 'delta': float(seen_threshold - threshold),
                        'f1': seen['test']['f1'], 'precision': seen['test']['precision'], 'recall': seen['test']['recall']})
    return metrics, extra, sensitivity


def run_threshold_selection(operator, cfg, graphs, col_sets, node_dim, out_root, frozen, seed, mp_direction='in',
                            temporal=False, eval_train_every=5):
    '''The validation-chosen threshold of one config under the frozen recipe: train on the train graph only for the
    frozen number of epochs (no early stopping), validate every epoch, take the validation-F1 threshold at the last
    epoch. Writes <out_root>/<run>/seed-<s>/{history.csv, curves.png, selection.json}; skipped when selection.json exists.
    Returns {'run', 'seed', 'epoch', 'threshold', 'val_f1', 'val_pr_auc', 'source'}.'''
    recipe, stop_epoch = frozen['recipe'], int(frozen['best_epoch'])
    name = run_name(operator, cfg, mp_direction, temporal)
    out_dir = f'{out_root}/{name}/seed-{seed}'
    if os.path.exists(f'{out_dir}/selection.json'):
        sel = json.load(open(f'{out_dir}/selection.json'))
        print(f"=== threshold selection {name} seed {seed} === already finished (threshold {sel['threshold']:.3f}), skipping")
        return sel
    os.makedirs(out_dir, exist_ok=True)
    mp_idx, readout_idx = col_sets[cfg['mp']], col_sets[cfg['readout']]
    set_seed(seed)
    deg_hist = in_degree_histogram(graphs['train']) if operator == 'pna' else None
    model = build_model(operator, node_dim, len(mp_idx), len(readout_idx), deg_hist=deg_hist, recipe=recipe)
    print_model_header(f'threshold selection {name} seed {seed} (train graph, {stop_epoch} epochs, frozen recipe)',
                       mp_idx, readout_idx, count_params(model), invariant_params(model), node_dim)
    loaders = make_loaders(graphs, mp_idx, readout_idx, recipe, splits=('train', 'val'))
    history, _, _ = train_one_run(model, loaders, out_dir, recipe, stop_epoch=stop_epoch, eval_train_every=eval_train_every)
    last = history[-1]
    assert last['epoch'] == stop_epoch
    plot_curves(history, None, None, None, stop_epoch, f'{name} threshold selection seed {seed}', f'{out_dir}/curves.png')
    sel = {'run': name, 'operator': operator, 'cfg': cfg, 'seed': seed, 'epoch': stop_epoch,
           'threshold': float(last['threshold']), 'val_f1': float(last['val_f1']), 'val_pr_auc': float(last['val_pr_auc']),
           'source': f'validation sweep at epoch {stop_epoch} of a train-only run with the frozen recipe, seed {seed}'}
    json.dump(sel, open(f'{out_dir}/selection.json', 'w'), indent=2)
    if os.path.exists(f'{out_dir}/best.pt'):
        os.remove(f'{out_dir}/best.pt')  # the selection keeps the threshold, not the weights
    print(f"threshold selection {name}: {sel['threshold']:.3f} (val F1 {sel['val_f1']:.4f} at epoch {stop_epoch})")
    del model, loaders
    gc.collect()
    torch.cuda.empty_cache()
    return sel


def save_config_thresholds(path, entries):
    '''config_thresholds.json: {run: {threshold, source, ...}}, one validation-chosen cut per feature config.'''
    json.dump(entries, open(path, 'w'), indent=2)
    print('config thresholds -> ' + path + ': ' + ', '.join(f"{k}={v['threshold']:.3f}" for k, v in entries.items()))


def run_final(operator, cfg, graphs, col_sets, node_dim, out_root, frozen, seed, mp_direction='in', temporal=False,
              node_enc='none', enc_dir=None, gfp_variant='v0', cfg_threshold=None, resume=False, eval_curves=False):
    '''Final retrain of one seed: train + val graph, stop at the frozen epoch, every split scored once.

    The headline threshold is cfg_threshold (this config's validation-chosen cut, see run_threshold_selection) or,
    when None, the frozen tuning threshold. Writes <out_root>/<run>/seed-<s>/: results.json (+ phase, stop_epoch,
    threshold_source, val_seen and the secondary views: transfer_* at the tuned config's threshold when cfg_threshold
    is given, seen_* at the threshold re-swept on the val rows that were trained on here), history.csv, curves.png,
    sensitivity.csv, predictions.csv, final.pt. results.json is written last, so it marks a finished run, and a
    finished run is skipped. resume=True continues a killed retrain from seed-<s>/resume.pt; eval_curves=True
    scores train and val (seen) every epoch for the curves.
    '''
    recipe, stop_epoch = frozen['recipe'], int(frozen['best_epoch'])
    frozen_threshold = float(frozen['thresholds'][str(seed)])
    threshold = frozen_threshold if cfg_threshold is None else float(cfg_threshold)
    transfer_threshold = None if cfg_threshold is None else frozen_threshold   # 2nd view only beside a config cut
    name = run_name(operator, cfg, mp_direction, temporal, node_enc, gfp_variant)
    out_dir = f'{out_root}/{name}/seed-{seed}'
    if os.path.exists(f'{out_dir}/results.json'):
        results = json.load(open(f'{out_dir}/results.json'))
        print(f'=== {name} seed {seed} === already finished (results.json found), skipping')
        return {k: v for k, v in results.items() if k not in ('architecture', 'recipe', 'threshold_sensitivity')}
    os.makedirs(out_dir, exist_ok=True)
    mp_idx, readout_idx = col_sets[cfg['mp']], col_sets[cfg['readout']]
    node_dim = set_node_encoding(graphs, node_enc, enc_dir)
    fit_graph = make_trainval_graph(graphs)

    set_seed(seed)
    deg_hist = in_degree_histogram(fit_graph) if operator == 'pna' else None   # the graph trained on
    model = build_model(operator, node_dim, len(mp_idx), len(readout_idx), deg_hist=deg_hist, recipe=recipe)
    n_params, n_invariant = count_params(model), invariant_params(model)
    print_model_header(f'{name} seed {seed} (train+val retrain, stop at epoch {stop_epoch}, threshold {threshold:.3f})',
                       mp_idx, readout_idx, n_params, n_invariant, node_dim)
    fit_loader = make_loader(fit_graph, True, mp_idx, readout_idx, recipe)
    eval_loaders = make_loaders(graphs, mp_idx, readout_idx, recipe, splits=('train', 'val')) if eval_curves else None
    history = train_fixed_epochs(model, fit_loader, out_dir, recipe, stop_epoch,
                                 resume_path=f'{out_dir}/resume.pt' if resume else None, eval_loaders=eval_loaders)
    del fit_loader, fit_graph, eval_loaders
    gc.collect()

    # scoring: the three usual loaders (val is training data here), test once; headline = this config's threshold,
    # secondary = the tuned config's threshold and the seen-val sweep (PR-AUC, ROC-AUC and top-5 % do not move)
    loaders = make_loaders(graphs, mp_idx, readout_idx, recipe)
    _, preds = score_all_splits(model, loaders, graphs, threshold, seed)
    metrics, extra, sensitivity = final_scoring(preds, threshold, transfer_threshold)
    row = result_row(name, operator, cfg, stop_epoch, threshold, n_params, n_invariant, metrics, node_enc, gfp_variant, seed)
    row['model_file'] = 'final.pt'
    row.update(phase='retrain', trained_on='train+val', stop_epoch=stop_epoch,
               threshold_source='config' if cfg_threshold is not None else 'tuning', val_seen=True, **extra)
    transfer = '' if transfer_threshold is None else \
        f"{extra['transfer_test_f1']:.4f} at the tuned config's {transfer_threshold:.3f} | "
    print(f"test F1 {metrics['test']['f1']:.4f} at this config's threshold {threshold:.3f} | {transfer}"
          f"{extra['seen_test_f1']:.4f} at the seen-val sweep {extra['val_seen_threshold']:.3f}")
    pd.DataFrame(sensitivity).to_csv(f'{out_dir}/sensitivity.csv', index=False)
    print(pd.DataFrame(sensitivity).round(4))
    save_predictions(out_dir, graphs, preds, threshold)
    plot_final_curves(history, preds['test']['y'], preds['test']['prob'], metrics['test']['pr_auc'],
                      f'{name} seed {seed}', f'{out_dir}/curves.png')
    save_run(out_dir, row, recipe, extra={'threshold_sensitivity': sensitivity})   # last: results.json = finished

    del model, loaders
    gc.collect()
    torch.cuda.empty_cache()
    return row


# ==================================================================================================
# Hydra entry: frozen-epoch retrain (conf/experiment/final.yaml)
# ==================================================================================================

GNN_NAMES = ('gin', 'pna', 'gat', 'transformer')  # LightGBM and the MLP retrain through their own classes


class FinalRun:
    """Frozen-epoch retrain. Hydra builds this from conf/experiment/final.yaml."""

    def __init__(self, model, features, hyperparameters, optimization, dataset, preprocessing, paths, seed,
                 recipe_path=None):
        self.model = model
        self.features = features
        self.hyperparameters = hyperparameters
        self.optimization = optimization
        self.dataset = dataset
        self.preprocessing = preprocessing
        self.paths = paths
        self.seed = int(seed)
        self.recipe_path = recipe_path
        self.cfg = None

    def run(self, frozen=None, seed=None, data_dir=None, out_root_dir=None, resume=False, eval_curves=False):
        """Retrain one GNN seed, or the tabular model's retrain, then score every split."""
        if self.model.name not in GNN_NAMES:
            return self.model.retrain(self, seed if seed is not None else self.seed)
        from omegaconf import OmegaConf

        from gnn_frauds.dataset.hi_small import load_graphs
        from gnn_frauds.features import feature_pair
        from gnn_frauds.model.gnns.run import out_root, write_resolved

        if frozen is None:
            assert self.recipe_path, 'experiment=final needs experiment.recipe_path=.../recipe.json'
            frozen = load_recipe(self.recipe_path)
        pre = OmegaConf.to_container(self.preprocessing, resolve=True)
        graphs, col_sets, node_dim = load_graphs(
            data_dir or self.paths.data_dir, gfp_variant=pre['gfp']['name'], gfp_dir=pre['gfp'].get('dir'))
        used = self.seed if seed is None else int(seed)
        root = out_root(self, self.model.name) if out_root_dir is None else out_root_dir
        row = run_final(
            self.model.name, feature_pair(self.features), graphs, col_sets, node_dim,
            root, frozen, used,
            node_enc=pre['node_embedding']['name'], enc_dir=pre['node_embedding'].get('dir'),
            gfp_variant=pre['gfp']['name'], resume=resume, eval_curves=eval_curves)
        write_resolved(self, f"{root}/{row['run']}/seed-{used}")
        return row

