"""Hydra wiring: the command builds the same classes the tests call directly."""
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

import gbt_core
import gnn_core
import mlp_core
from gnn_frauds.features import feature_pair
from gnn_frauds.launch import compose_experiment
from gnn_frauds.model.gnns.run import out_root, write_resolved


def test_shims_export_the_package():
    assert callable(gnn_core.run_experiment) and callable(gnn_core.load_graphs)
    assert callable(gnn_core.draw_luo_trials) and callable(gnn_core.EdgeClassifier)
    assert callable(gbt_core.train_final) and callable(gbt_core.tail_check)
    assert callable(mlp_core.train_final) and callable(mlp_core.draw_trials)


def test_compose_builds_each_experiment_without_training():
    train = compose_experiment(['model=gnns/gat', 'features=cfg5'])
    assert type(train).__name__ == 'TrainRun'
    assert train.model.name == 'gat' and train.seed == 42
    assert feature_pair(train.features) == {'mp': 'base', 'readout': 'full'}
    fixed = train.optimization.trial(train)
    assert fixed['hidden'] == 128 and list(fixed['num_neighbors']) == [100, 100]

    tune = compose_experiment(['experiment=tune', 'optimization=luo', 'model=gnns/pna', 'features=cfg4'])
    assert type(tune).__name__ == 'TuneRun' and tune.model.name == 'pna'
    assert tune.trial_index is None and tune.optimization.n_trials == 100
    assert tune.optimization.tuning_seeds == (42, 43)
    assert feature_pair(tune.features) == {'mp': 'full', 'readout': 'full'}
    assert tune.optimization.trial(tune)['epochs'] == 20  # trial 0 reads the selected hyperparameters file
    tune.trial_index = 2
    with pytest.raises(AssertionError):
        tune.optimization.trial(tune)

    final = compose_experiment(['experiment=final', 'model=gnns/transformer'])
    assert type(final).__name__ == 'FinalRun' and final.model.name == 'transformer'
    with pytest.raises(AssertionError, match='recipe_path'):
        final.run()

    net = compose_experiment(['model=tabular/mlp', 'hyperparameters=mlp/v1', 'optimization=random'])
    assert net.model.name == 'mlp' and net.optimization.trial(net) is None
    assert net.optimization.n_trials == 100 and net.optimization.patience == 10
    booster = compose_experiment([
        'model=tabular/lightgbm', 'hyperparameters=lightgbm/v1', 'optimization=halving'])
    assert booster.model.name == 'lightgbm' and booster.optimization.trial(booster) is None
    assert booster.optimization.eta == 2 and booster.optimization.n_configs == 100

    encoded = compose_experiment(['preprocessing/node_embedding=rwpe8', 'preprocessing/gfp=tuned'])
    pre = OmegaConf.to_container(encoded.preprocessing, resolve=True)
    assert pre['node_embedding']['name'] == 'rwpe8' and pre['gfp']['name'] == 'tuned'


def test_out_root_and_resolved_config(tmp_path):
    run = SimpleNamespace(paths=SimpleNamespace(output_dir='Outputs'))
    assert out_root(run, 'gin') == 'Outputs/GIN'
    assert out_root(run, 'lightgbm') == 'Outputs/GBT'
    assert out_root(run, 'mlp') == 'Outputs/MLP'
    with pytest.raises(KeyError):
        out_root(run, 'sage')
    bare = SimpleNamespace(cfg=None)
    write_resolved(bare, str(tmp_path / 'absent'))
    assert not (tmp_path / 'absent').exists()
    run.cfg = OmegaConf.create({'model': 'gin', 'seed': 42})
    write_resolved(run, str(tmp_path))
    saved = OmegaConf.load(tmp_path / 'config.yaml')
    assert saved.model == 'gin' and saved.seed == 42
