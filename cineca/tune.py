'''Step "tune": one array task = one (trial, seed) of one (operator, config); array index i -> trial i // 2, seed 42 + i % 2.

Trains the trial's recipe from trials_space.json for 100 epochs on the train graph (train and val scored every epoch,
every section-4 metric), resumable from resume.pt; the test snapshot is never loaded. Writes seed-<s>/done.json, and
the task of the seed that finishes last finalises the trial (gnn_core.finalise_trial: trial.json, one best.pt per
seed), so two processes never finalise the same trial at once.
'''
import argparse
import json
import os
import time
from pathlib import Path

import common
from common import core


def finalise(space, op, cfg_id, trial_id):
    '''trial.json once both seeds are done (CPU only; returns None while the other seed still runs).'''
    entry = space['trials'][op][trial_id]
    return core.finalise_trial(op, common.CONFIGS[cfg_id], common.trial_dir(op, cfg_id, trial_id), entry['recipe'],
                               extra={'trial_id': trial_id, 'tag': entry['tag']})


def finish_times(op, cfg_id, trial_id):
    '''{seed: time its done.json was written} of the seeds that are done.'''
    paths = {s: f'{common.trial_dir(op, cfg_id, trial_id)}/seed-{s}/done.json' for s in core.LUO_TUNING_SEEDS}
    return {s: json.load(open(p))['finished'] for s, p in paths.items() if os.path.exists(p)}


def finalise_pending(space):
    '''Every trial whose two seeds are done but which has no trial.json yet, e.g. a finalisation cut short by the time
    limit (aggregate.py calls this). A trial whose last seed finished under 10 minutes ago is left to that seed's task,
    which may be finalising it right now.'''
    for op in common.OPERATORS:
        for cfg_id in common.CONFIGS:
            for trial_id in range(space['n_trials']):
                finished = finish_times(op, cfg_id, trial_id)
                if len(finished) == 2 and time.time() - max(finished.values()) > 600:
                    finalise(space, op, cfg_id, trial_id)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--operator', choices=common.OPERATORS, required=True)
    parser.add_argument('--config', type=int, choices=sorted(common.CONFIGS), required=True)
    parser.add_argument('--index', type=int, required=True, help='array index -> trial index // 2, seed 42 + index % 2')
    args = parser.parse_args()
    space = common.load_space()
    op, cfg_id = args.operator, args.config
    trial_id, seed = common.tune_task(args.index)
    print(f'tune: array index {args.index} -> trial t{trial_id:03d}, seed {seed} | {op} config {cfg_id}')
    sdir = f'{common.trial_dir(op, cfg_id, trial_id)}/seed-{seed}'
    if not os.path.exists(f'{sdir}/done.json'):
        common.require_gpu()
        recipe = core.full_recipe(space['trials'][op][trial_id]['recipe'])
        print(f"{space['trials'][op][trial_id]['tag']} | recipe {recipe}")
        graphs, col_sets, node_dim = core.load_graphs(common.setting('DATA_DIR'), splits=('train', 'val'))
        print(f"test split loaded: {'test' in graphs}")
        t0 = time.time()
        try:
            info = core.run_tuning_seed(op, common.CONFIGS[cfg_id], graphs, col_sets, node_dim,
                                        common.trial_dir(op, cfg_id, trial_id), recipe, seed, eval_train_every=1,
                                        patience=None, resume=True, full_metrics=True)
            # no early stopping here, so fewer epochs than the schedule means the run diverged
            diverged_at = info['epochs_run'] + 1 if info['epochs_run'] < recipe['epochs'] else None
            info.update(status='diverged' if diverged_at else 'ok', diverged_at=diverged_at)
        except (core.torch.cuda.OutOfMemoryError, RuntimeError) as err:
            if not core.is_out_of_memory(err):
                raise
            # recorded, not retried: a resubmission would hit the same batch again; the trial cannot win
            print(f'OUT OF MEMORY: trial t{trial_id:03d} seed {seed} recorded as status oom')
            info = {'status': 'oom', 'diverged_at': None, 'epochs_run': None, 'error': str(err)[:200]}
        Path(f'{sdir}/best.pt').unlink(missing_ok=True)   # the trial's best.pt is chosen once both seeds are done
        done = dict(info, seed=seed, trial_id=trial_id, test_split_loaded='test' in graphs,
                    minutes=(time.time() - t0) / 60, finished=time.time(), **common.job_info())
        core.write_json_atomic(f'{sdir}/done.json', done)
        print(f"seed {seed}: {done['status']} -> {sdir}/done.json")
    finished = finish_times(op, cfg_id, trial_id)
    if len(finished) < 2:
        print('the other seed is not done yet; it will finalise the trial')
    elif max(finished, key=lambda s: (finished[s], s)) != seed:
        print('the other seed finished last; its task finalises the trial')
    else:
        finalise(space, op, cfg_id, trial_id)
        print('trial finalised')


if __name__ == '__main__':
    main()
