"""Graph transformer. The network is built in classifier.py once the feature widths are known."""


class Transformer:
    def __init__(self, name='transformer'):
        self.name = name

    def train(self, run):
        # Imported on call so loading the class does not pull in the training loop.
        from gnn_frauds.model.gnns.run import train_gnn
        return train_gnn(self.name, run)
