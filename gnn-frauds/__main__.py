"""Hydra only: load the config, build the experiment class, run it."""
from pathlib import Path

import hydra
from hydra.utils import instantiate


@hydra.main(config_path=str(Path(__file__).resolve().parents[1] / 'conf'), config_name='config', version_base='1.3')
def main(cfg):
    experiment = instantiate(cfg.experiment)
    experiment.cfg = cfg  # instantiate does not keep the resolved config; the run folder saves this copy
    experiment.run()


if __name__ == '__main__':
    main()
