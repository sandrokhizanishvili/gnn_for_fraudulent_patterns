"""Successive halving. The ranges it draws live in the hyperparameters file."""


class Halving:
    def __init__(self, n_configs, eta, first_fraction, early_stopping, check_every, select, final_seeds):
        self.n_configs = int(n_configs)
        self.eta = int(eta)
        self.first_fraction = float(first_fraction)
        self.early_stopping = int(early_stopping)
        self.check_every = int(check_every)
        self.select = select
        self.final_seeds = tuple(int(s) for s in final_seeds)

    def trial(self, run):
        """The search itself is successive_halving; there is no single drawn recipe."""
        return None
