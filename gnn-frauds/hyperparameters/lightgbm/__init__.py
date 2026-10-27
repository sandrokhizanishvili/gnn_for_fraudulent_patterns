"""LightGBM values, read from conf/hyperparameters/lightgbm/v1.yaml."""
import os

import yaml

from gnn_frauds.hyperparameters.gnns import conf_file


def load():
    """(search space as low/high/type rows, fixed keys that are never drawn)."""
    hp = yaml.safe_load(conf_file('hyperparameters/lightgbm/v1.yaml').read_text(encoding='utf-8'))
    space = {key: (spec['low'], spec['high'], spec['type']) for key, spec in hp['space'].items()}
    fixed = {key: value for key, value in hp['fixed'].items() if key not in hp['space']}
    fixed['num_threads'] = os.cpu_count()  # use every core; the thread count is not a searched knob
    point = {key: hp['fixed'][key] for key in hp['space']}
    return space, fixed, point
