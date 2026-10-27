"""Train and final use the fixed point and ignore the search space."""
from omegaconf import OmegaConf


class NoSearch:
    def trial(self, run):
        """The fixed hyperparameters of the selected model."""
        return OmegaConf.to_container(run.hyperparameters.fixed, resolve=True)
