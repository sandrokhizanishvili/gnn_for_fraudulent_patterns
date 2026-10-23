'''status.sh: one table of done / running / queued / timeout / failed / missing tasks per step and (operator, config).

timeout -> just run the same step again (the task resumes); failed -> look at its log in logs/, run the step again
once, and send the log to Sandro if it fails again.
'''
import os
import shutil
import subprocess
from collections import Counter

import pandas as pd

import common


def last_states():
    '''{(job name, array index): last SLURM state} from sacct over the last 60 days.'''
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


def row(step, op, cfg_id, queued, states, label):
    name = common.job_name(step, op, cfg_id)
    counts = Counter()
    for index in common.all_indices(step):
        if common.finished(step, op, cfg_id, index):
            counts['done'] += 1
        elif (name, index) in queued:
            counts['running' if queued[(name, index)] == 'RUNNING' else 'queued'] += 1
        elif states.get((name, index)) == 'TIMEOUT':
            counts['timeout'] += 1
        elif (name, index) in states:
            counts['failed'] += 1
        else:
            counts['missing'] += 1
    return dict(step=step, pair=label, done=f"{counts['done']}/{len(common.all_indices(step))}", running=counts['running'],
                queued=counts['queued'], timeout=counts['timeout'], failed=counts['failed'], missing=counts['missing'])


def main():
    queued, states = common.squeue_tasks(), last_states()
    rows = [row('probe', None, None, queued, states, 'all operators')]
    if os.path.exists(common.space_path()):
        for step in ('tune', 'final'):
            for op in common.OPERATORS:
                for cfg_id in common.CONFIGS:
                    if step == 'tune' or os.path.exists(f'{common.run_dir(op, cfg_id)}/recipe.json'):
                        rows.append(row(step, op, cfg_id, queued, states, f'{op} config {cfg_id}'))
    else:
        print('trials_space.json not written yet: after the probe, run "bash cineca/run_step.sh draw"')
    with pd.option_context('display.width', 200, 'display.max_columns', None):
        print(pd.DataFrame(rows).to_string(index=False))
    print('tune counts seeds (2 per trial). timeout: run the step again (it resumes). '
          'failed: see logs/, run the step again once, then send the log to Sandro')


if __name__ == '__main__':
    main()
