'''The Luo et al. (2024) tuning on Cineca: every step in one file (EXPERIMENTS.md section 2b).

    python cineca/luo.py <step> [--index I] [--operator OP] [--config C]      inside a SLURM job (job.sbatch)
    python cineca/luo.py plan <probe|tune|final> [--ops ...] [--cfgs ...] [--first N]   what run.sh submits
    python cineca/luo.py draw | aggregate | status                             on the login node (run.sh)

Steps: check (environment test), probe (depths that fit per operator), draw (the trial list), tune (one trial x seed),
aggregate (checks + winners), final (one retrain seed), status (progress table). The training, the search space and
the scoring live in gnn_core.py; this file maps SLURM array indices to tasks and keeps the bookkeeping. Settings come
from config.sh (exported by env.sh). Run the scripts through cineca/run.sh.
'''
import argparse
import glob
import importlib.metadata as metadata
import json
import math
import os
import platform
import shutil
import socket
import subprocess
import sys
import time
import warnings
from collections import Counter
from pathlib import Path

import pandas as pd

CINECA_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.dirname(CINECA_DIR)
sys.path.insert(0, CODE_DIR)
import gnn_core as core   # noqa: E402

warnings.filterwarnings('ignore', message='.*non-interactive.*')   # plt.show() under MPLBACKEND=Agg
torch = core.torch

OPERATORS = core.OPERATORS                                  # gin, pna, gat, transformer
FAMILY = {'gin': 'GIN', 'pna': 'PNA', 'gat': 'GAT', 'transformer': 'TRANSFORMER'}
CONFIGS = {2: dict(mp='base', readout='base'), 4: dict(mp='full', readout='full')}
PROBE_CONFIG = 4                                            # the heavier config; the probe verdict holds for both


# ==================================================================================================
# settings, folders, array index -> task
# ==================================================================================================

def setting(name):
    '''A value from config.sh; stops with a clear message when it is missing.'''
    value = os.environ.get(name)
    assert value not in (None, ''), f'{name} is not set: run the scripts through cineca/run.sh (it sources config.sh)'
    return value


def require_gpu():
    '''Jobs refuse to start on a CPU-only torch, which would burn 24 h for nothing (REQUIRE_GPU=0 for a CPU check).'''
    if os.environ.get('REQUIRE_GPU', '1') == '1':
        assert torch.cuda.is_available(), 'no GPU visible to torch: check the job request (--gres=gpu:1) and the env'
    print(f"device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")


def job_info():
    '''Where a task ran, stored in done.json.'''
    return {'host': socket.gethostname(), 'job_id': os.environ.get('SLURM_JOB_ID'),
            'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}


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


def tune_task(index):
    '''Tuning array index -> (trial id, seed): both seeds of a trial are neighbours (2i, 2i + 1), so they start together.'''
    return index // 2, core.LUO_TUNING_SEEDS[index % 2]


def final_task(index):
    '''Final array index -> seed (42 + index).'''
    return core.LUO_FINAL_SEEDS[index]


def load_space():
    '''trials_space.json, written once by the draw step after the probe.'''
    assert os.path.exists(space_path()), f'{space_path()} not found: run "bash cineca/run.sh draw" after the probe'
    return json.load(open(space_path()))


# ==================================================================================================
# what is left to run (the plan of run.sh) and the status table
# ==================================================================================================

def squeue_tasks():
    '''{(job name, array index): state} of this user's pending and running array tasks ({} without SLURM).'''
    if not shutil.which('squeue'):
        return {}
    out = subprocess.run(['squeue', '-h', '-r', '-u', os.environ.get('USER', ''), '-o', '%j|%K|%T'],
                         capture_output=True, text=True, check=True).stdout
    tasks = {}
    for line in out.splitlines():
        name, index, state = line.split('|')
        if index.isdigit():
            tasks[(name, int(index))] = state
    return tasks


def sacct_states():
    '''{(job name, array index): last SLURM state} from sacct over the last 60 days ({} without SLURM).'''
    if not shutil.which('sacct'):
        return {}
    out = subprocess.run(['sacct', '-u', os.environ.get('USER', ''), '-n', '-P', '-X', '-S', 'now-60days',
                          '-o', 'JobName,JobID,State'], capture_output=True, text=True).stdout
    states = {}
    for line in out.splitlines():   # sacct lists jobs in submission order, so the last line of a task wins
        name, job_id, state = line.split('|')
        task = job_id.split('_')[-1]
        if '_' in job_id and task.isdigit():
            states[(name, int(task))] = state.split()[0]
    return states


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
    '''One line "job name|array ids|memory|operator|config" per job array with work left (unfinished tasks that are
    not already pending or running); run.sh submits one array per line.'''
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


def status_row(step, op, cfg_id, queued, states, label):
    name = job_name(step, op, cfg_id)
    counts = Counter()
    for index in all_indices(step):
        if finished(step, op, cfg_id, index):
            counts['done'] += 1
        elif (name, index) in queued:
            counts['running' if queued[(name, index)] == 'RUNNING' else 'queued'] += 1
        elif states.get((name, index)) == 'TIMEOUT':
            counts['timeout'] += 1
        elif (name, index) in states:
            counts['failed'] += 1
        else:
            counts['missing'] += 1
    return dict(step=step, pair=label, done=f"{counts['done']}/{len(all_indices(step))}", running=counts['running'],
                queued=counts['queued'], timeout=counts['timeout'], failed=counts['failed'], missing=counts['missing'])


def step_status():
    '''One table of done / running / queued / timeout / failed / missing tasks per step and (operator, config).'''
    queued, states = squeue_tasks(), sacct_states()
    rows = [status_row('probe', None, None, queued, states, 'all operators')]
    if os.path.exists(space_path()):
        for step in ('tune', 'final'):
            for op in OPERATORS:
                for cfg_id in CONFIGS:
                    if step == 'tune' or os.path.exists(f'{run_dir(op, cfg_id)}/recipe.json'):
                        rows.append(status_row(step, op, cfg_id, queued, states, f'{op} config {cfg_id}'))
    else:
        print('trials_space.json not written yet: after the probe, run "bash cineca/run.sh draw"')
    with pd.option_context('display.width', 200, 'display.max_columns', None):
        print(pd.DataFrame(rows).to_string(index=False))
    print('tune counts seeds (2 per trial). timeout: run the step again (it resumes). '
          'failed: see logs/, run the step again once, then send the log to Sandro')


# ==================================================================================================
# step check: imports and versions, one tiny model per operator on a synthetic graph (no project data)
# ==================================================================================================

def step_check():
    print(f'python {platform.python_version()} | torch {torch.__version__} | CUDA build {torch.version.cuda}')
    for package in ('torch_geometric', 'torch_scatter', 'torch_sparse', 'torch_cluster', 'numpy', 'scipy', 'pandas',
                    'scikit-learn', 'matplotlib', 'psutil', 'tqdm'):
        print(f'  {package} {metadata.version(package)}')
    require_gpu()

    # a random graph with the project's layout: 6 node columns, 81 edge columns, ~5 % positive edges
    gen = torch.Generator().manual_seed(0)
    n_nodes, n_edges, cols = 300, 4000, list(range(81))
    graph = core.Data(x=torch.nn.functional.one_hot(torch.randint(0, 6, (n_nodes,), generator=gen), 6).float(),
                      edge_index=torch.randint(0, n_nodes, (2, n_edges), generator=gen),
                      edge_attr=torch.randn(n_edges, 81, generator=gen), edge_time=torch.arange(n_edges),
                      y=(torch.rand(n_edges, generator=gen) < 0.05).long(),
                      eval_mask=torch.ones(n_edges, dtype=torch.bool), num_nodes=n_nodes)
    recipe = core.full_recipe(dict(core.LUO_PROTOCOL, hidden=64, norm='batch', residual='linear', readout_layers=3,
                                   batch_size=512, num_neighbors=[10, 10]))
    for op in OPERATORS:
        deg_hist = core.in_degree_histogram(graph) if op == 'pna' else None
        model = core.build_model(op, 6, len(cols), len(cols), deg_hist=deg_hist, recipe=recipe)
        loader, feats = core.make_loader(graph, True, cols, cols, recipe=recipe)
        logits, y = core.forward_batch(model, next(iter(loader)), feats)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y)
        loss.backward()
        print(f'  {op:<11} params {core.count_params(model):>8,} | loss {loss.item():.4f} on {core.device}')
    print('OK')


# ==================================================================================================
# step probe (one array task per operator) -> <FAMILY>/probe.json
# ==================================================================================================

def step_probe(index):
    '''Which depths 1-10 fit on the GPU at the heaviest setting (hidden 512, BatchNorm, linear residual, readout 3 x 2,
    config 4), peak memory, epoch time, and the fastest loader worker count (0 / 4 / 6). Resumable per depth.'''
    op = OPERATORS[index]
    print(f'probe: array index {index} -> operator {op}')
    path = probe_path(op)
    if finished('probe', None, None, index):
        print(f'{path} is complete, nothing to do')
        return
    require_gpu()
    os.makedirs(family_dir(op), exist_ok=True)
    graphs, col_sets, node_dim = core.load_graphs(setting('DATA_DIR'), splits=('train', 'val'))
    print(f"test split loaded: {'test' in graphs}")
    cfg = CONFIGS[PROBE_CONFIG]
    recipe = core.full_recipe(dict(core.LUO_PROTOCOL, **core.LUO_HEAVIEST))
    probe = core.probe_memory(op, cfg, graphs, col_sets, node_dim, recipe, core.LUO_DEPTHS, path, core.LUO_MEM_FRACTION)
    fitting = sorted(int(d) for d, v in probe.items() if d.isdigit() and v['neighbors'] is not None)
    if 2 not in fitting:
        print('DEPTH 2 DOES NOT FIT AT HIDDEN 512. This needs a decision before anything else runs: '
              'stop here and send the logs folder to Sandro.')
        sys.exit(1)

    # loader workers: wall-clock s/batch at depth 2 and at the deepest depth that fits; the fastest count is used
    depth_neighbours = {d: probe[str(d)]['neighbors'] for d in sorted({2, max(fitting)})}
    timing = core.time_loader_workers(op, cfg, graphs, col_sets, node_dim, recipe, depth_neighbours,
                                      core.LOADER_WORKER_SETTINGS)
    summed = {w: sum(timing[str(d)][str(w)]['wall_s_per_batch'] for d in depth_neighbours)
              for w in core.LOADER_WORKER_SETTINGS}
    probe['loader'] = {'timing': timing, 'chosen': min(summed, key=summed.get)}
    probe['complete'] = True
    core.write_json_atomic(path, probe)
    print(f"probe {op}: depths that fit {fitting} | loader workers {probe['loader']['chosen']} "
          f"({', '.join(f'{w}: {s:.3f} s' for w, s in summed.items())}) | job memory {job_memory_gb(probe)} GB -> {path}")


# ==================================================================================================
# step draw (login node, once) -> trials_space.json, the pre-registered trial list
# ==================================================================================================

def step_draw():
    '''n_trials from conf/optimization/luo.yaml, per operator: trial 0 = the v1 knobs, trials 1..N-1 from
    gnn_core.draw_luo_trials (fixed seed, the same knobs for every operator, the depth from each operator's depths
    that fit). Refuses to run twice or once a trial folder exists.'''
    out = space_path()
    assert not os.path.exists(out), (f'{out} exists: the draw happens once. Delete it only if no trial has run yet '
                                     '(and only to change n_trials in conf/optimization/luo.yaml before the first tune step)')
    started = glob.glob(f'{tuning_root()}/*/*/trials/*')
    assert not started, f'{len(started)} trial folders exist already: the draw must come before any training'
    probes = {}
    for op in OPERATORS:
        assert finished('probe', None, None, OPERATORS.index(op)), \
            f'{probe_path(op)} missing or incomplete: run "bash cineca/run.sh probe" first'
        probes[op] = json.load(open(probe_path(op)))

    neighbours = {op: {int(d): v['neighbors'] for d, v in probe.items() if d.isdigit() and v['neighbors'] is not None}
                  for op, probe in probes.items()}
    allowed = {op: sorted(kept) for op, kept in neighbours.items()}
    workers = {op: int(probes[op]['loader']['chosen']) for op in OPERATORS}
    n_trials = int(core.LUO_N_TRIALS)
    shell_trials = os.environ.get('N_TRIALS')
    if shell_trials not in (None, '') and int(shell_trials) != n_trials:
        print(f'N_TRIALS={shell_trials} in the shell is ignored; conf/optimization/luo.yaml has {n_trials}')
    trials = core.draw_luo_trials(n_trials, allowed, neighbours, workers)
    space = {'created': time.strftime('%Y-%m-%d %H:%M:%S'), 'draw_seed': core.LUO_DRAW_SEED, 'n_trials': n_trials,
             'tuning_seeds': list(core.LUO_TUNING_SEEDS), 'final_seeds': list(core.LUO_FINAL_SEEDS),
             'protocol': core.LUO_PROTOCOL, 'space': core.LUO_SPACE, 'allowed_depths': allowed,
             'dropped_depths': {op: sorted(set(core.LUO_DEPTHS) - set(allowed[op])) for op in OPERATORS},
             'trials': trials}
    core.write_json_atomic(out, space)

    first = OPERATORS[0]
    table = pd.DataFrame([dict(id=t['id'], **{k: t['recipe'][k] for k in core.LUO_SPACE},
                               **{f'layers_{op}': trials[op][t['id']]['recipe']['num_layers'] for op in OPERATORS})
                          for t in trials[first]])
    with pd.option_context('display.max_rows', None, 'display.width', 200):
        print(table.to_string(index=False))
    print(f'\n{n_trials} trials per (operator, config) -> {out}')
    print('allowed depths:', allowed, '| dropped:', space['dropped_depths'], '| loader workers:', workers)


# ==================================================================================================
# step tune: one array task = one (trial, seed); the seed that finishes last finalises the trial
# ==================================================================================================

def finalise(space, op, cfg_id, trial_id):
    '''trial.json once both seeds are done (CPU only; None while the other seed still runs).'''
    entry = space['trials'][op][trial_id]
    return core.finalise_trial(op, CONFIGS[cfg_id], trial_dir(op, cfg_id, trial_id), entry['recipe'],
                               extra={'trial_id': trial_id, 'tag': entry['tag']})


def finish_times(op, cfg_id, trial_id):
    '''{seed: time its done.json was written} of the seeds that are done.'''
    paths = {s: f'{trial_dir(op, cfg_id, trial_id)}/seed-{s}/done.json' for s in core.LUO_TUNING_SEEDS}
    return {s: json.load(open(p))['finished'] for s, p in paths.items() if os.path.exists(p)}


def finalise_pending(space):
    '''Every trial whose two seeds are done but which has no trial.json yet (a finalisation cut short by the time
    limit). A trial whose last seed finished under 10 minutes ago is left to that seed's task.'''
    for op in OPERATORS:
        for cfg_id in CONFIGS:
            for trial_id in range(space['n_trials']):
                done = finish_times(op, cfg_id, trial_id)
                if len(done) == 2 and time.time() - max(done.values()) > 600:
                    finalise(space, op, cfg_id, trial_id)


def step_tune(op, cfg_id, index):
    '''Train the trial's recipe for 100 epochs on the train graph (train and val scored every epoch, every metric),
    resumable from resume.pt; the test snapshot is never loaded. Writes seed-<s>/done.json.'''
    space = load_space()
    trial_id, seed = tune_task(index)
    print(f'tune: array index {index} -> trial t{trial_id:03d}, seed {seed} | {op} config {cfg_id}')
    sdir = f'{trial_dir(op, cfg_id, trial_id)}/seed-{seed}'
    if not os.path.exists(f'{sdir}/done.json'):
        require_gpu()
        from gnn_frauds.launch import compose_experiment
        recipe = core.full_recipe(space['trials'][op][trial_id]['recipe'])
        print(f"{space['trials'][op][trial_id]['tag']} | recipe {recipe}")
        graphs, col_sets, node_dim = core.load_graphs(setting('DATA_DIR'), splits=('train', 'val'))
        print(f"test split loaded: {'test' in graphs}")
        run = compose_experiment([
            f'model=gnns/{op}', f'features=cfg{cfg_id}', 'hyperparameters=gnns/luo', 'optimization=luo',
            'experiment=tune', 'paths=cineca', f'experiment.trial_index={index}', f'experiment.config_id={cfg_id}',
        ])
        assert run.model.name == op
        run.run_one(graphs, col_sets, node_dim, trial_dir(op, cfg_id, trial_id), recipe, seed, trial_id,
                    job_info(), CONFIGS[cfg_id])
    done = finish_times(op, cfg_id, trial_id)
    if len(done) < 2:
        print('the other seed is not done yet; it will finalise the trial')
    elif max(done, key=lambda s: (done[s], s)) != seed:
        print('the other seed finished last; its task finalises the trial')
    else:
        finalise(space, op, cfg_id, trial_id)
        print('trial finalised')


# ==================================================================================================
# step aggregate (login node): checks with asserts, the winner of each pair, the summaries
# ==================================================================================================

def model_params(op, cfg_id, recipe):
    '''(params, invariant params) of the recipe's model, built on the meta device (no memory, no data needed).'''
    width = {'base': core.BASE_EDGE_DIM, 'full': core.BASE_EDGE_DIM + core.GFP_WIDTH['v0']}
    cfg = CONFIGS[cfg_id]
    deg_hist = torch.tensor([0, 10, 5, 2]) if op == 'pna' else None   # the degree values never change the count
    with torch.device('meta'):
        model = core.EdgeClassifier(op, core.BASE_NODE_DIM, width[cfg['mp']], width[cfg['readout']], False, deg_hist, recipe)
    return core.count_params(model), core.invariant_params(model)


def check_trial(trial, space, op, cfg_id):
    '''Every check of one finished trial; an AssertionError names the trial.'''
    tid = trial['trial_id']
    name = f'{op} config {cfg_id} t{tid:03d}'
    tdir = trial_dir(op, cfg_id, tid)
    recipe = core.full_recipe(trial['recipe'])
    assert recipe == core.full_recipe(space['trials'][op][tid]['recipe']), f'{name}: recipe differs from trials_space.json'
    for s in core.LUO_TUNING_SEEDS:
        assert json.load(open(f'{tdir}/seed-{s}/done.json'))['test_split_loaded'] is False, f'{name} seed {s}: test loaded'
    if 'val_pr_auc_mean' not in trial:        # out of memory, or diverged before its first epoch: nothing to score
        assert trial['status'] in ('oom', 'diverged'), f"{name}: no metrics but status {trial['status']!r}"
        return
    diverged = trial['diverged_at'] or {}
    histories = {}
    for s in core.LUO_TUNING_SEEDS:
        h = pd.read_csv(f'{tdir}/seed-{s}/history.csv')
        expected = diverged[str(s)] - 1 if str(s) in diverged else core.LUO_MAX_EPOCHS
        assert len(h) == expected, f'{name} seed {s}: {len(h)} epochs in history.csv, expected {expected}'
        assert (h['lr'] == recipe['lr']).all(), f"{name} seed {s}: the learning rate is not constant at {recipe['lr']}"
        assert not h[core.TRIAL_METRIC_KEYS + ['threshold']].isna().any().any(), f'{name} seed {s}: NaN in history.csv'
        histories[s] = h
    best, _, _ = core.seed_mean_best_epoch(histories, 1, 'val_pr_auc')
    assert best == trial['best_epoch'], f"{name}: best epoch {trial['best_epoch']} != {best} (argmax of the seed-mean val PR-AUC)"
    for s, h in histories.items():
        row = h.loc[h['epoch'] == best].iloc[0]
        for key in core.TRIAL_METRIC_KEYS + ['threshold']:
            assert math.isclose(trial['per_seed'][str(s)][key], float(row[key]), abs_tol=1e-12), \
                f"{name} seed {s}: trial.json {key} = {trial['per_seed'][str(s)][key]}, history.csv {row[key]} at epoch {best}"
        assert os.path.exists(f'{tdir}/seed-{s}/best.pt'), f'{name} seed {s}: best.pt missing'
        assert not glob.glob(f'{tdir}/seed-{s}/epoch_*.pt'), f'{name} seed {s}: epoch checkpoints left'
    for key in core.TRIAL_METRIC_KEYS:
        assert math.isfinite(trial[f'{key}_mean']), f'{name}: {key}_mean is not finite'
    params = model_params(op, cfg_id, recipe)
    assert (trial['params'], trial['invariant_params']) == params, \
        f"{name}: params {trial['params']} / {trial['invariant_params']}, the recipe builds {params}"


def check_final_run(results, history, frozen, name, seed):
    '''Every check of one finished final seed.'''
    assert seed in core.LUO_FINAL_SEEDS and frozen['retrain_seeds'] == list(core.LUO_FINAL_SEEDS), f'{name}: seeds'
    assert results['stop_epoch'] == frozen['best_epoch'], f'{name} seed {seed}: stop epoch differs from recipe.json'
    assert results['threshold'] == frozen['thresholds'][str(seed)], f'{name} seed {seed}: threshold differs from recipe.json'
    assert len(history) == frozen['best_epoch'], f'{name} seed {seed}: {len(history)} epochs in history.csv'
    assert not [c for c in history if c.startswith('test')], f'{name} seed {seed}: test scored during training'
    assert (history['lr'] == frozen['recipe']['lr']).all(), f'{name} seed {seed}: the learning rate is not constant'
    assert all(math.isfinite(results[c]) for c in core.METRIC_COLS), f'{name} seed {seed}: a metric is missing or not finite'


def tuning_pair(space, op, cfg_id):
    '''Checks, summary and (once all trials are in) the winner of one (operator, config).'''
    rdir = run_dir(op, cfg_id)
    trials = [json.load(open(p)) for p in sorted(glob.glob(f'{rdir}/trials/t*/trial.json'))]
    for trial in trials:
        check_trial(trial, space, op, cfg_id)
    print(f"{op} config {cfg_id}: {len(trials)}/{space['n_trials']} trials, checks passed, "
          f"statuses {pd.Series([t['status'] for t in trials], dtype=object).value_counts().to_dict()}")
    if not trials:
        return
    trial0 = [t for t in trials if t['trial_id'] == 0 and 'val_pr_auc_mean' in t]
    v1 = f'{CODE_DIR}/Outputs/{FAMILY[op]}/{core.run_name(op, CONFIGS[cfg_id])}/results.json'   # the repo's v1 run
    if trial0 and os.path.exists(v1):
        print(f"  trial 0 vs the v1 run (warning only, protocols differ): val PR-AUC {trial0[0]['val_pr_auc_mean']:.4f} "
              f"(2 seeds, 100 epochs, constant LR) vs {json.load(open(v1))['val_pr_auc']:.4f} (1 seed, 20-epoch cosine)")

    recipe_path = f'{rdir}/recipe.json'
    winner = core.choose_best_trial(trials, 'pr_auc') if len(trials) == space['n_trials'] else None
    if winner is None:
        print(f"  {len(trials)}/{space['n_trials']} trials, no winner yet")
    elif os.path.exists(recipe_path):
        assert json.load(open(recipe_path))['source_trial'] == winner['trial'], \
            f"{op} config {cfg_id}: recipe.json is not the trial with the highest mean val PR-AUC ({winner['trial']})"
    else:
        core.save_recipe(recipe_path, op, CONFIGS[cfg_id], winner, retrain_seeds=core.LUO_FINAL_SEEDS,
                         extra={'n_trials': space['n_trials'], 'select_metric': 'pr_auc', 'trial_id': winner['trial_id'],
                                'val_pr_auc_mean': winner['val_pr_auc_mean'], 'val_pr_auc_std': winner['val_pr_auc_std'],
                                'best_epoch_in_last_10': winner['best_epoch'] > core.LUO_MAX_EPOCHS - 10})
        if winner['best_epoch'] > core.LUO_MAX_EPOCHS - 10:
            print(f"  WARNING: the winner's best epoch {winner['best_epoch']} is in the last 10 of {core.LUO_MAX_EPOCHS}")
    winners = {core.LUO_STAGE: winner['trial']} if winner else {}
    core.trials_table(trials, winners).to_csv(f'{rdir}/trials_summary.csv', index=False)
    core.plot_val_curves(trials, winners, f'{rdir}/tuning_curves.png', max_labels=10, select_metric='pr_auc')


def final_pair(op, cfg_id):
    '''Checks and summaries of the final retrain seeds of one (operator, config) finished so far.'''
    recipe_path = f'{run_dir(op, cfg_id)}/recipe.json'
    if not os.path.exists(recipe_path):
        return
    frozen = core.load_recipe(recipe_path)
    pair_dir = f'{final_root(op)}/{core.run_name(op, CONFIGS[cfg_id])}'
    rows = []
    for seed in core.LUO_FINAL_SEEDS:
        if os.path.exists(f'{pair_dir}/seed-{seed}/results.json'):
            results = json.load(open(f'{pair_dir}/seed-{seed}/results.json'))
            check_final_run(results, pd.read_csv(f'{pair_dir}/seed-{seed}/history.csv'), frozen, f'{op} config {cfg_id}', seed)
            rows.append({k: v for k, v in results.items() if k not in ('architecture', 'recipe', 'threshold_sensitivity')})
    print(f'  final retrain: {len(rows)}/{len(core.LUO_FINAL_SEEDS)} seeds done, checks passed')
    if rows:
        core.summarize_batch(rows, pair_dir, 'batch_summary_final.csv')
        core.seed_summary(rows, pair_dir, 'seed_summary_final.csv')


def step_aggregate():
    space = load_space()
    finalise_pending(space)
    for op in OPERATORS:
        for cfg_id in CONFIGS:
            tuning_pair(space, op, cfg_id)
            final_pair(op, cfg_id)
    print('\nall checks passed')


# ==================================================================================================
# step final: one array task = one retrain seed of one (operator, config)'s winning recipe
# ==================================================================================================

def step_final(op, cfg_id, index):
    '''Retrain on train + val for the frozen best epoch at the frozen threshold, score train / val / test once
    -> results.json (written last), history.csv, curves.png, sensitivity.csv, predictions.csv, final.pt. Resumable.'''
    seed = final_task(index)
    print(f'final: array index {index} -> seed {seed} | {op} config {cfg_id}')
    recipe_path = f'{run_dir(op, cfg_id)}/recipe.json'
    frozen = core.load_recipe(recipe_path)
    assert frozen['operator'] == op and frozen['cfg'] == CONFIGS[cfg_id], 'recipe.json belongs to another pair'
    require_gpu()
    from gnn_frauds.launch import compose_experiment
    run = compose_experiment([
        f'model=gnns/{op}', f'features=cfg{cfg_id}', 'hyperparameters=gnns/luo', 'optimization=luo',
        'experiment=final', 'paths=cineca', f'experiment.recipe_path={recipe_path}',
    ])
    assert run.model.name == op
    run.run(frozen=frozen, seed=seed, data_dir=setting('DATA_DIR'), out_root_dir=final_root(op),
            resume=True, eval_curves=True)
    Path(f'{final_dir(op, cfg_id, seed)}/resume.pt').unlink(missing_ok=True)


# ==================================================================================================
# command line
# ==================================================================================================

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('step', choices=['plan', 'check', 'probe', 'draw', 'tune', 'aggregate', 'final', 'status'])
    parser.add_argument('plan_step', nargs='?', choices=['probe', 'tune', 'final'], help='plan: which step')
    parser.add_argument('--index', type=int, help='SLURM array index (probe, tune, final)')
    parser.add_argument('--operator', choices=OPERATORS, help='tune, final')
    parser.add_argument('--config', type=int, choices=sorted(CONFIGS), help='tune, final')
    parser.add_argument('--ops', nargs='+', default=list(OPERATORS), help='plan: operators')
    parser.add_argument('--cfgs', nargs='+', type=int, default=sorted(CONFIGS), help='plan: configs')
    parser.add_argument('--first', type=int, help='plan tune: only trials 0..first-1 (the early check-in)')
    args = parser.parse_args()
    if args.step == 'plan':
        assert args.plan_step, 'plan needs a step: probe, tune or final'
        for line in plan(args.plan_step, args.ops, args.cfgs, args.first):
            print(line)
    elif args.step == 'check':
        step_check()
    elif args.step == 'probe':
        step_probe(args.index)
    elif args.step == 'draw':
        step_draw()
    elif args.step == 'tune':
        step_tune(args.operator, args.config, args.index)
    elif args.step == 'aggregate':
        step_aggregate()
    elif args.step == 'final':
        step_final(args.operator, args.config, args.index)
    else:
        step_status()


if __name__ == '__main__':
    main()
