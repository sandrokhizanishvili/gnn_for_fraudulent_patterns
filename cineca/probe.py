'''Step "probe" (one array task per operator) -> Outputs/TUNING_LUO/<FAMILY>/probe.json.

Which depths 1-10 fit on the GPU at the heaviest setting (hidden 512, BatchNorm, linear residual, readout 3 layers x 2
width, config 4 = mp full / readout full; the verdict holds for config 2 too), peak GPU and CPU memory, epoch time,
and which loader worker count (0 / 4 / 6) is fastest. A depth fits when nothing ran out of memory and the peak
reserved GPU memory stays <= 85 % of the GPU (gnn_core.probe_memory). Resumable: finished depths are skipped.
This is also the first real-data test of the package.
'''
import argparse
import os
import sys

import common
from common import core


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--index', type=int, required=True, help='array index -> operator (0 gin, 1 pna, 2 gat, 3 transformer)')
    args = parser.parse_args()
    op = common.OPERATORS[args.index]
    print(f'probe: array index {args.index} -> operator {op}')
    path = common.probe_path(op)
    if common.finished('probe', None, None, args.index):
        print(f'{path} is complete, nothing to do')
        return
    common.require_gpu()
    os.makedirs(common.family_dir(op), exist_ok=True)
    graphs, col_sets, node_dim = core.load_graphs(common.setting('DATA_DIR'), splits=('train', 'val'))
    print(f"test split loaded: {'test' in graphs}")
    cfg = common.CONFIGS[common.PROBE_CONFIG]
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
          f"({', '.join(f'{w}: {s:.3f} s' for w, s in summed.items())}) | job memory {common.job_memory_gb(probe)} GB "
          f"-> {path}")


if __name__ == '__main__':
    main()
