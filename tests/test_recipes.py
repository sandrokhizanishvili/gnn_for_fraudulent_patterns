"""Recipe knobs, feature configs, and the fixed-key check."""
import pytest
import yaml

from gnn_frauds.features import feature_pair
from gnn_frauds.hyperparameters import check_fixed
from gnn_frauds.hyperparameters.gnns import (
    conf_file, default_recipe, edge_value, full_recipe, recipe_tag, with_knob,
)
from gnn_frauds.hyperparameters.lightgbm import load as load_lgb
from gnn_frauds.hyperparameters.mlp import load as load_mlp

FEATURE_CONFIGS = {
    'cfg1': ('none', 'base'), 'cfg2': ('base', 'base'), 'cfg3': ('none', 'full'),
    'cfg4': ('full', 'full'), 'cfg5': ('base', 'full'), 'cfg6': ('base', 'gfp'),
    'cfg7': ('none', 'gfp'),
}


def test_default_recipe_is_v1():
    recipe = default_recipe()
    assert recipe['hidden'] == 128 and recipe['num_layers'] == 2
    assert recipe['num_neighbors'] == [100, 100]
    assert recipe['dropout'] == 0.3 and recipe['norm'] == 'none' and recipe['ffn'] is False
    assert recipe['residual'] == 'identity' and recipe['lr_schedule'] == 'cosine'
    assert recipe['select_metric'] == 'f1' and recipe['self_loops'] is True
    assert recipe['epochs'] == 20 and recipe['pos_weight'] == 8.0


def test_full_recipe_overlays_and_rejects_bad_knobs():
    recipe = full_recipe({'dropout': 0.2, 'hidden': 64})
    assert recipe['dropout'] == 0.2 and recipe['hidden'] == 64 and recipe['num_layers'] == 2
    with pytest.raises(AssertionError):
        full_recipe({'not_a_knob': 1})
    with pytest.raises(AssertionError):
        full_recipe({'hidden': 130})  # 130 does not split across 4 attention heads
    with pytest.raises(AssertionError):
        full_recipe({'num_layers': 2, 'num_neighbors': [10]})
    with pytest.raises(AssertionError):
        full_recipe({'norm': 'group'})
    with pytest.raises(AssertionError):
        full_recipe({'residual': 'skip'})
    with pytest.raises(AssertionError):
        full_recipe({'readout_layers': 4})
    with pytest.raises(AssertionError):
        full_recipe({'lr_schedule': 'step'})
    with pytest.raises(AssertionError):
        full_recipe({'select_metric': 'roc_auc'})


def test_recipe_tag_names_only_the_knobs_that_differ():
    assert recipe_tag(None) == 'base'
    assert recipe_tag({'dropout': 0.2}) == 'd0.2'
    assert recipe_tag({'norm': 'batch'}) == 'bn'
    assert recipe_tag({'norm': 'layer'}) == 'ln'
    assert recipe_tag({'ffn': True}) == 'ffn'
    assert recipe_tag({'self_loops': False}) == 'nosl'
    assert recipe_tag({'num_layers': 4, 'num_neighbors': [100, 100, 50, 50]}) == 'l4'
    assert recipe_tag({'num_neighbors': [50, 50]}) == 'nb50-50'
    assert recipe_tag({'residual': 'linear'}) == 'reslin'
    assert recipe_tag({'residual': 'none'}) == 'nores'
    assert recipe_tag({'lr_schedule': 'constant'}) == 'lrconst'
    assert recipe_tag({'readout_layers': 1, 'readout_width': 2.0}) == 'ro1'
    assert recipe_tag({'readout_layers': 3, 'readout_width': 2.0}) == 'ro3_rw2'
    assert recipe_tag({'dropout': 0.2, 'norm': 'batch'}) == 'd0.2_bn'


def test_with_knob_and_edge_value():
    base = {'dropout': 0.3}
    changed = with_knob(base, 'lr', 0.01)
    assert base == {'dropout': 0.3}
    assert changed['lr'] == 0.01 and changed['dropout'] == 0.3
    depth = with_knob(None, 'num_layers', 3)
    assert depth['num_layers'] == 3 and depth['num_neighbors'] == [100, 100, 50]
    assert with_knob(None, 'num_layers', 11) is None
    assert edge_value('dropout', 0.2, [0.2, 0.3, 0.4]) == 0.1
    assert edge_value('dropout', 0.4, [0.2, 0.3, 0.4]) == 0.5
    assert edge_value('dropout', 0.3, [0.2, 0.3, 0.4]) is None
    assert edge_value('num_layers', 2, [2, 4, 6]) == 3
    assert edge_value('num_layers', 6, [2, 4, 6]) == 8
    assert edge_value('hidden', 64, [64, 128]) is None


def test_feature_configs_match_the_seven_pairs():
    for name, (mp, readout) in FEATURE_CONFIGS.items():
        raw = yaml.safe_load(conf_file(f'features/{name}.yaml').read_text(encoding='utf-8'))
        assert feature_pair(raw) == {'mp': mp, 'readout': readout}


def test_check_fixed_rejects_unknown_keys():
    check_fixed({'hidden': 8}, {'hidden', 'dropout'})
    with pytest.raises(AssertionError):
        check_fixed({'depth': 2}, {'hidden'})


def test_tabular_hyperparameter_files_load():
    fixed, space = load_mlp()
    assert fixed['shape'] == 'funnel' and fixed['hidden'] == 512
    assert set(space['shape']) == {'flat', 'funnel', 'resnet'}
    space_lgb, fixed_lgb, point = load_lgb()
    assert point['num_leaves'] == 7 and 'verbose' in fixed_lgb
    assert space_lgb['learning_rate'][2] == 'log'
