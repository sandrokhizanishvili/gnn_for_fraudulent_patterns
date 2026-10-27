"""Knob names and the v1 point. The numbers live in conf/hyperparameters/gnns/."""
from pathlib import Path

HEADS = 4   # GATv2 / Transformer: 4 heads x 32 = 128. Not a searched knob; hidden must divide by it.

OPERATORS     = ('gin', 'pna', 'gat', 'transformer')
NODE_ENCS     = ('none', 'rwpe8', 'rwpe16')   # NODE_ENC knob: node encoding appended to the entity one-hots
BASE_NODE_DIM = 6                             # the entity one-hots every run has; encodings come after them
GFP_VARIANTS  = ('v0', 'tuned')               # GFP_VARIANT knob: which GFP sheet fills edge_attr after the 20 baseline columns
GFP_WIDTH     = {'v0': 61, 'tuned': 64}       # v0 = the sheet inside the graph files; tuned = Data/gfp_variants/tuned.npy
BASE_EDGE_DIM = 20                            # baseline edge columns, always edge_attr[:, :20]
SPLITS      = ('train', 'val', 'test')
METRICS     = ('f1', 'precision', 'recall', 'pr_auc', 'roc_auc', 'precision_at_5pct', 'recall_at_5pct')
RUN_FIELDS  = ('threshold', 'best_epoch', 'params', 'invariant_params')
METRIC_COLS = [f'{split}_{metric}' for split in SPLITS for metric in METRICS]
TRIAL_METRIC_KEYS = [f'{split}_{m}' for split in ('train', 'val') for m in METRICS + ('loss',)]   # trial.json mean / std
# compact view of batch_summary.csv for the notebook comparison table
DISPLAY_COLS = ['run', 'best_epoch', 'threshold', 'val_f1', 'test_f1', 'test_precision', 'test_recall',
                'test_pr_auc', 'test_precision_at_5pct', 'test_recall_at_5pct', 'params', 'invariant_params']

# ---------------- recipe tuning (EXPERIMENTS.md section 2b) ----------------
NORMS = ('none', 'batch', 'layer')            # normalisation after each message-passing layer
RESIDUALS = ('identity', 'linear', 'none')    # identity = v1 (h + layer(h)); linear = Luo et al. 2024 Eq. 7
LR_SCHEDULES = ('cosine', 'constant')
SELECT_METRICS = ('f1', 'pr_auc')             # validation metric that picks the best epoch (and the winning trial)
# neighbours per hop by depth: hops 1-2 are always 100, hops 3-4 50, hops 5+ 25 (fallback: 25 and 10)
NEIGHBOURS_BY_DEPTH = {1: [100], 2: [100, 100], 3: [100, 100, 50], 4: [100, 100, 50, 50],
                       5: [100, 100, 50, 50, 25], 6: [100, 100, 50, 50, 25, 25],
                       7: [100, 100, 50, 50, 25, 25, 25], 8: [100, 100, 50, 50, 25, 25, 25, 25],
                       9: [100, 100, 50, 50, 25, 25, 25, 25, 25], 10: [100, 100, 50, 50, 25, 25, 25, 25, 25, 25]}
NEIGHBOURS_FALLBACK = {3: [100, 100, 25], 4: [100, 100, 25, 25], 5: [100, 100, 25, 25, 10],
                       6: [100, 100, 25, 25, 10, 10], 7: [100, 100, 25, 25, 10, 10, 10],
                       8: [100, 100, 25, 25, 10, 10, 10, 10], 9: [100, 100, 25, 25, 10, 10, 10, 10, 10],
                       10: [100, 100, 25, 25, 10, 10, 10, 10, 10, 10]}
# edge rule: a numeric winner at the edge of its grid gets one trial one step beyond
EDGE_BEYOND = {'dropout': {0.2: 0.1, 0.4: 0.5}, 'num_layers': {2: 3, 6: 8},
               'pos_weight': {8.0: 4.0, 16.0: 32.0}, 'lr': {5e-4: 1e-4, 3e-3: 1e-2}}
SENSITIVITY_DELTAS = (-0.10, -0.05, 0.0, 0.05, 0.10)   # threshold-sensitivity table of the final retrain
VAL_PREVALENCE_NOTE = ('train - val gap: validation has 0.107 % laundering vs 0.075 % on train, '
                       'so part of the gap is prevalence, not underfitting')


def conf_file(relative):
    """A file under the repository conf/ directory."""
    # This file is gnn-frauds/hyperparameters/gnns/__init__.py; parents[3] is the repo root.
    return Path(__file__).resolve().parents[3] / 'conf' / relative


def default_recipe():
    """The v1 point, read from conf/hyperparameters/gnns/v1.yaml each call."""
    import yaml
    fixed = yaml.safe_load(conf_file('hyperparameters/gnns/v1.yaml').read_text(encoding='utf-8'))['fixed']
    fixed = dict(fixed)
    fixed['num_neighbors'] = [int(n) for n in fixed['num_neighbors']]
    fixed['ffn'] = bool(fixed['ffn'])
    fixed['self_loops'] = bool(fixed['self_loops'])
    return fixed

def full_recipe(recipe=None):
    '''The v1 recipe overlaid with the given knobs; unknown keys or a depth / neighbour mismatch are errors.'''
    out = default_recipe()
    extra = set(recipe or {}) - set(out)
    assert not extra, f'unknown recipe keys {sorted(extra)}; knobs are {sorted(out)}'
    out.update(recipe or {})
    out['num_neighbors'] = [int(n) for n in out['num_neighbors']]
    assert out['norm'] in NORMS, f"norm must be one of {NORMS}, got {out['norm']!r}"
    assert out['hidden'] % HEADS == 0, 'hidden must split into the attention heads'
    assert len(out['num_neighbors']) == out['num_layers'], 'one neighbour count per message-passing layer'
    assert out['num_layers'] >= 1 and out['epochs'] >= 1
    assert isinstance(out['self_loops'], bool), 'self_loops is a flag'
    assert out['residual'] in RESIDUALS, f"residual must be one of {RESIDUALS}, got {out['residual']!r}"
    assert out['readout_layers'] in (1, 2, 3) and out['readout_width'] > 0
    assert out['lr_schedule'] in LR_SCHEDULES, f"lr_schedule must be one of {LR_SCHEDULES}"
    assert int(out['num_workers']) == out['num_workers'] >= 0
    assert out['select_metric'] in SELECT_METRICS, f"select_metric must be one of {SELECT_METRICS}"
    return out


def recipe_tag(recipe, base=None):
    '''Trial folder name = the knobs that differ from `base` (default: the v1 recipe), e.g. d0.2_bn_l4; 'base' if none.'''
    full, ref = full_recipe(recipe), full_recipe(base)
    short = {'dropout': 'd', 'hidden': 'h', 'pos_weight': 'pw', 'lr': 'lr', 'weight_decay': 'wd',
             'batch_size': 'bs', 'epochs': 'ep'}
    parts = []
    for knob in ('dropout', 'norm', 'num_layers', 'num_neighbors', 'hidden', 'pos_weight', 'lr', 'ffn',
                 'self_loops', 'weight_decay', 'batch_size', 'epochs'):
        if full[knob] == ref[knob]:
            continue
        if knob == 'norm':
            parts.append({'batch': 'bn', 'layer': 'ln', 'none': 'nonorm'}[full['norm']])
        elif knob == 'ffn':
            parts.append('ffn' if full['ffn'] else 'noffn')
        elif knob == 'self_loops':
            parts.append('sl' if full['self_loops'] else 'nosl')
        elif knob == 'num_layers':
            parts.append(f"l{full['num_layers']}")
        elif knob == 'num_neighbors':
            # implied by the depth unless the depth is unchanged (a fallback neighbour list)
            if full['num_layers'] == ref['num_layers']:
                parts.append('nb' + '-'.join(str(n) for n in full['num_neighbors']))
        else:
            parts.append(f'{short[knob]}{full[knob]:g}')
    # num_workers and select_metric are run settings, not part of the tag
    if full['residual'] != ref['residual']:
        parts.append({'identity': 'resid', 'linear': 'reslin', 'none': 'nores'}[full['residual']])
    if full['readout_layers'] != ref['readout_layers']:
        parts.append(f"ro{full['readout_layers']}")
    if full['readout_width'] != ref['readout_width'] and full['readout_layers'] > 1:
        parts.append(f"rw{full['readout_width']:g}")
    if full['lr_schedule'] != ref['lr_schedule']:
        parts.append({'cosine': 'lrcos', 'constant': 'lrconst'}[full['lr_schedule']])
    return '_'.join(parts) or 'base'


def with_knob(recipe, knob, value, neighbours_by_depth=None):
    '''Copy of the recipe with one knob changed; num_layers also sets the neighbour list for that depth.

    Returns None when the depth has no neighbour list (dropped by the probe).
    '''
    out = dict(full_recipe(recipe))
    if knob == 'num_layers':
        table = neighbours_by_depth if neighbours_by_depth is not None else NEIGHBOURS_BY_DEPTH
        if table.get(int(value)) is None:
            return None
        out['num_layers'], out['num_neighbors'] = int(value), list(table[int(value)])
    else:
        out[knob] = value
    return full_recipe(out)


def edge_value(knob, value, grid):
    '''The edge rule: the value one step beyond the grid when `value` sits at its edge, else None.'''
    beyond = EDGE_BEYOND.get(knob, {})
    if value in beyond and value in (min(grid), max(grid)):
        return beyond[value]
    return None


# Names the notebooks import (HIDDEN, EPOCHS, ...). They are the v1 yaml, read at import.
_v1 = default_recipe()
HIDDEN = _v1['hidden']
NUM_LAYERS = _v1['num_layers']
DROPOUT = _v1['dropout']
NUM_NEIGHBORS = list(_v1['num_neighbors'])
BATCH_SIZE = _v1['batch_size']
EPOCHS = _v1['epochs']
LR = _v1['lr']
WEIGHT_DECAY = _v1['weight_decay']
POS_WEIGHT = _v1['pos_weight']
def _seed():
    import yaml  # only the default seed needs the config file; keep yaml out of the import of this module
    cfg = yaml.safe_load(conf_file('config.yaml').read_text(encoding='utf-8'))
    return int(cfg['seed'])


SEED = _seed()
