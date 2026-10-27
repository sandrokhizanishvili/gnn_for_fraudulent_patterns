"""MLP values, read from conf/hyperparameters/mlp/v1.yaml."""
import yaml

from gnn_frauds.hyperparameters.gnns import conf_file


def load():
    """(fixed point, choice lists)."""
    hp = yaml.safe_load(conf_file('hyperparameters/mlp/v1.yaml').read_text(encoding='utf-8'))
    fixed = dict(hp['fixed'])
    space = {key: list(values) for key, values in hp['space'].items()}
    return fixed, space
