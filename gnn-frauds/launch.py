"""Compose a config and run the same experiment class the command line runs."""
from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.utils import instantiate


def compose_experiment(overrides):
    """Build the experiment class from the override list. Does not start training."""
    conf = Path(__file__).resolve().parents[1] / 'conf'
    with initialize_config_dir(config_dir=str(conf), version_base='1.3'):
        cfg = compose(config_name='config', overrides=list(overrides))
    experiment = instantiate(cfg.experiment)
    experiment.cfg = cfg  # same attachment as __main__: the run folder saves this resolved copy
    return experiment


def compose_run(overrides):
    """Build cfg from the override list and call instantiate(cfg.experiment).run()."""
    experiment = compose_experiment(overrides)
    return experiment.run()
