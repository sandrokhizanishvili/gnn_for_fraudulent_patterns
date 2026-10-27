"""Unique uniform draws. The choice lists live in the hyperparameters file."""


class RandomSearch:
    def __init__(self, n_trials, draw_seed, epochs, patience, search_seeds, final_seeds, select):
        self.n_trials = int(n_trials)
        self.draw_seed = int(draw_seed)
        self.epochs = int(epochs)
        self.patience = int(patience)
        self.search_seeds = tuple(int(s) for s in search_seeds)
        self.final_seeds = tuple(int(s) for s in final_seeds)
        self.select = select

    def trial(self, run):
        """The search itself is random_search; trial 0 is the fixed point inside that function."""
        return None
