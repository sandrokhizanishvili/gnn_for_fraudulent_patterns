"""GATv2. The network is built in classifier.py once the feature widths are known."""


class GAT:
    def __init__(self, name='gat'):
        self.name = name

    def train(self, run):
        # Imported on call so loading the class does not pull in the training loop.
        from gnn_frauds.model.gnns.run import train_gnn
        return train_gnn(self.name, run)
