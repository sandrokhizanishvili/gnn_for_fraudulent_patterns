'''Step "draw" (login node, seconds) -> Outputs/TUNING_LUO/trials_space.json, the pre-registered trial list.

Runs once, after all four probes and before any training: N_TRIALS trials per operator (config.sh); trial 0 = the v1
knobs, trials 1..N-1 from gnn_core.draw_luo_trials (fixed seed, the same knobs for every operator, the depth from
each operator's depths that fit). Refuses to overwrite trials_space.json or to run once a trial folder exists.
'''
import glob
import json
import os
import time

import pandas as pd

import common
from common import core


def main():
    out = common.space_path()
    assert not os.path.exists(out), (f'{out} exists: the draw happens once. Delete it only if no trial has run yet '
                                     '(and only to change N_TRIALS before the first tune step)')
    started = glob.glob(f'{common.tuning_root()}/*/*/trials/*')
    assert not started, f'{len(started)} trial folders exist already: the draw must come before any training'
    probes = {}
    for op in common.OPERATORS:
        assert common.finished('probe', None, None, common.OPERATORS.index(op)), \
            f'{common.probe_path(op)} missing or incomplete: run "bash cineca/run_step.sh probe" first'
        probes[op] = json.load(open(common.probe_path(op)))

    neighbours = {op: {int(d): v['neighbors'] for d, v in probe.items() if d.isdigit() and v['neighbors'] is not None}
                  for op, probe in probes.items()}
    allowed = {op: sorted(kept) for op, kept in neighbours.items()}
    workers = {op: int(probes[op]['loader']['chosen']) for op in common.OPERATORS}
    n_trials = int(common.setting('N_TRIALS'))
    trials = core.draw_luo_trials(n_trials, allowed, neighbours, workers)
    space = {'created': time.strftime('%Y-%m-%d %H:%M:%S'), 'draw_seed': core.LUO_DRAW_SEED, 'n_trials': n_trials,
             'tuning_seeds': list(core.LUO_TUNING_SEEDS), 'final_seeds': list(core.LUO_FINAL_SEEDS),
             'protocol': core.LUO_PROTOCOL, 'space': core.LUO_SPACE, 'allowed_depths': allowed,
             'dropped_depths': {op: sorted(set(core.LUO_DEPTHS) - set(allowed[op])) for op in common.OPERATORS},
             'trials': trials}
    core.write_json_atomic(out, space)

    first = common.OPERATORS[0]
    table = pd.DataFrame([dict(id=t['id'], **{k: t['recipe'][k] for k in core.LUO_SPACE},
                               **{f'layers_{op}': trials[op][t['id']]['recipe']['num_layers'] for op in common.OPERATORS})
                          for t in trials[first]])
    with pd.option_context('display.max_rows', None, 'display.width', 200):
        print(table.to_string(index=False))
    print(f'\n{n_trials} trials per (operator, config) -> {out}')
    print('allowed depths:', allowed, '| dropped:', space['dropped_depths'], '| loader workers:', workers)


if __name__ == '__main__':
    main()
