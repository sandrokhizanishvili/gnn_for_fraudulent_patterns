'''Step "final": one array task = one seed (array index i -> seed 42 + i) of one (operator, config)'s winning recipe.

Retrains on the train + val graph for the frozen best epoch of recipe.json (written by aggregate.py) at the recipe's
constant LR, with train and val ("seen") loss and PR-AUC per epoch for the curves; then scores train, val and test
once at the frozen threshold (the mean of the two tuning seeds' validation thresholds at that epoch)
-> results.json, history.csv, curves.png, sensitivity.csv, predictions.csv, final.pt. Resumable from resume.pt;
results.json is written last and marks the seed finished.
'''
import argparse
from pathlib import Path

import common
from common import core


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--operator', choices=common.OPERATORS, required=True)
    parser.add_argument('--config', type=int, choices=sorted(common.CONFIGS), required=True)
    parser.add_argument('--index', type=int, required=True, help='array index -> seed 42 + index')
    args = parser.parse_args()
    op, cfg_id, seed = args.operator, args.config, common.final_task(args.index)
    print(f'final: array index {args.index} -> seed {seed} | {op} config {cfg_id}')
    frozen = core.load_recipe(f'{common.run_dir(op, cfg_id)}/recipe.json')
    assert frozen['operator'] == op and frozen['cfg'] == common.CONFIGS[cfg_id], 'recipe.json belongs to another pair'
    common.require_gpu()
    graphs, col_sets, node_dim = core.load_graphs(common.setting('DATA_DIR'))
    core.run_final(op, common.CONFIGS[cfg_id], graphs, col_sets, node_dim, common.final_root(op), frozen, seed,
                   resume=True, eval_curves=True)
    Path(f'{common.final_dir(op, cfg_id, seed)}/resume.pt').unlink(missing_ok=True)


if __name__ == '__main__':
    main()
