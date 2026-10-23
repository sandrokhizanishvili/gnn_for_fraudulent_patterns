'''Shared plumbing of the Cineca scripts: settings from config.sh, folder layout, array-index mapping, queue checks.

The tuning protocol itself (search space, draw, training, scoring) lives in gnn_core.py, none of it here. Settings
come from the environment that cineca/_env.sh exports from config.sh.
CLI (used by run_step.sh on the login node):
    python cineca/common.py plan <probe|tune|final> [--ops ...] [--cfgs ...] [--first N]
    -> one line "job name|array ids|memory|operator|config" per job array that still has unfinished, unqueued tasks
'''
import argparse
import json
import math
import os
import socket
import subprocess
import sys
import warnings

CINECA_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.dirname(CINECA_DIR)
sys.path.insert(0, CODE_DIR)
import gnn_core as core   # noqa: E402  (the package ships gnn_core.py next to cineca/)

warnings.filterwarnings('ignore', message='.*non-interactive.*')   # plt.show() under MPLBACKEND=Agg

OPERATORS = core.OPERATORS                                  # gin, pna, gat, transformer
FAMILY = {'gin': 'GIN', 'pna': 'PNA', 'gat': 'GAT', 'transformer': 'TRANSFORMER'}
CONFIGS = {2: dict(mp='base', readout='base'), 4: dict(mp='full', readout='full')}
PROBE_CONFIG = 4                                            # the heavier config; the probe verdict holds for both


def setting(name):
    '''A value from config.sh (exported by _env.sh); stops with a clear message when it is missing.'''
    value = os.environ.get(name)
    assert value not in (None, ''), f'{name} is not set: run the scripts through cineca/run_step.sh (it sources config.sh)'
    return value


def require_gpu():
    '''Jobs refuse to start on a CPU-only torch, which would burn 24 h for nothing (REQUIRE_GPU=0: setup_env.sh's CPU check).'''
    if os.environ.get('REQUIRE_GPU', '1') == '1':
        assert core.torch.cuda.is_available(), 'no GPU visible to torch: check the job request (--gres=gpu:1) and the env'
    print(f"device: {core.torch.cuda.get_device_name(0) if core.torch.cuda.is_available() else 'cpu'}")


def job_info():
    '''Where a task ran, stored in done.json.'''
    return {'host': socket.gethostname(), 'job_id': os.environ.get('SLURM_JOB_ID'),
            'gpu': core.torch.cuda.get_device_name(0) if core.torch.cuda.is_available() else 'cpu'}


# ---------------- folders ----------------

def tuning_root():
    return f"{setting('OUT_DIR')}/TUNING_LUO"


def family_dir(op):
    return f'{tuning_root()}/{FAMILY[op]}'


def probe_path(op):
    return f'{family_dir(op)}/probe.json'


def space_path():
    return f'{tuning_root()}/trials_space.json'


def run_dir(op, cfg_id):
    return f'{family_dir(op)}/{core.run_name(op, CONFIGS[cfg_id])}'


def trial_dir(op, cfg_id, trial_id):
    return f'{run_dir(op, cfg_id)}/trials/t{trial_id:03d}'


def final_root(op):
    return f"{setting('OUT_DIR')}/FINAL_LUO/{FAMILY[op]}"


def final_dir(op, cfg_id, seed):
    return f'{final_root(op)}/{core.run_name(op, CONFIGS[cfg_id])}/seed-{seed}'


# ---------------- array index -> task ----------------

def tune_task(index):
    '''Tuning array index -> (trial id, seed): both seeds of a trial are neighbours (2i, 2i + 1), so they start together.'''
    return index // 2, core.LUO_TUNING_SEEDS[index % 2]


def final_task(index):
    '''Final array index -> seed (42 + index).'''
    return core.LUO_FINAL_SEEDS[index]


def load_space():
    '''trials_space.json, written once by draw_trials.py after the probe.'''
    assert os.path.exists(space_path()), f'{space_path()} not found: run "bash cineca/run_step.sh draw" after the probe'
    return json.load(open(space_path()))


# ---------------- what is left to run ----------------

def squeue_tasks():
    '''{(job name, array index): state} of this user's pending and running array tasks.'''
    out = subprocess.run(['squeue', '-h', '-r', '-u', os.environ.get('USER', ''), '-o', '%j|%K|%T'],
                         capture_output=True, text=True, check=True).stdout
    tasks = {}
    for line in out.splitlines():
        name, index, state = line.split('|')
        if index.isdigit():
            tasks[(name, int(index))] = state
    return tasks


def job_name(step, op=None, cfg_id=None):
    return step if op is None else f'{step}_{op}_c{cfg_id}'


def finished(step, op, cfg_id, index):
    '''The finished marker of one array task: probe.json "complete", the seed's done.json, the final results.json.'''
    if step == 'probe':
        path = probe_path(OPERATORS[index])
        return os.path.exists(path) and json.load(open(path)).get('complete', False)
    if step == 'tune':
        trial_id, seed = tune_task(index)
        return os.path.exists(f'{trial_dir(op, cfg_id, trial_id)}/seed-{seed}/done.json')
    return os.path.exists(f'{final_dir(op, cfg_id, final_task(index))}/results.json')


def all_indices(step, first=None):
    if step == 'probe':
        return list(range(len(OPERATORS)))
    if step == 'tune':
        n_trials = load_space()['n_trials']
        return list(range(2 * (n_trials if first is None else min(first, n_trials))))
    return list(range(len(core.LUO_FINAL_SEEDS)))


def job_memory_gb(probe, final=False):
    '''Job memory from the probe: 1.5 x the largest peak CPU memory (main process + loader workers) + 4 GB, rounded up
    to 8 GB, at least 32 GB; +8 GB for the final retrain (it also loads the test snapshot).'''
    peaks = [a['peak_rss_gb'] for d, v in probe.items() if d.isdigit() for a in v['attempts'] if 'peak_rss_gb' in a]
    peaks += [w['peak_rss_gb'] for depth in probe['loader']['timing'].values() for w in depth.values()]
    return max(32, int(math.ceil((1.5 * max(peaks) + 4 + (8 if final else 0)) / 8)) * 8)


def memory_request(step, op):
    '''--mem of a tune / final job: config.sh MEM, or (MEM=auto) the probe-based value of that operator.'''
    if setting('MEM') != 'auto':
        return setting('MEM')
    return f"{job_memory_gb(json.load(open(probe_path(op))), final=(step == 'final'))}G"


def plan(step, ops, cfgs, first=None):
    '''What run_step.sh submits: one line "job name|array ids|memory|operator|config" per job array with work left
    (unfinished tasks that are not already pending or running).'''
    queued = squeue_tasks()
    pairs = [(None, None)] if step == 'probe' else [(op, cfg_id) for op in ops for cfg_id in cfgs]
    lines = []
    for op, cfg_id in pairs:
        name = job_name(step, op, cfg_id)
        if step == 'final' and not os.path.exists(f'{run_dir(op, cfg_id)}/recipe.json'):
            print(f'{name}: no recipe.json yet (run the aggregate step after the tuning)', file=sys.stderr)
            continue
        ids = [i for i in all_indices(step, first) if not finished(step, op, cfg_id, i) and (name, i) not in queued]
        if not ids:
            print(f'{name}: nothing to submit (finished, running or queued)', file=sys.stderr)
            continue
        mem = setting('MEM_PROBE') if step == 'probe' else memory_request(step, op)
        lines.append(f"{name}|{','.join(str(i) for i in ids)}|{mem}|{op or ''}|{cfg_id or ''}")
    return lines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('what', choices=['plan'])
    parser.add_argument('step', choices=['probe', 'tune', 'final'])
    parser.add_argument('--ops', nargs='+', default=list(OPERATORS))
    parser.add_argument('--cfgs', nargs='+', type=int, default=sorted(CONFIGS))
    parser.add_argument('--first', type=int, help='tune only trials 0..first-1 (the early check-in)')
    args = parser.parse_args()
    for line in plan(args.step, args.ops, args.cfgs, args.first):
        print(line)


if __name__ == '__main__':
    main()
