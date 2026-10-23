'''Environment check: imports and versions, then one tiny model per operator on a synthetic graph, forward and
backward on the GPU when there is one. Ends with "OK".

Run by setup_env.sh on the login node (CPU) and by "run_step.sh check" on a GPU node (debug queue). No project data.
'''
import importlib.metadata as metadata
import platform

import common
from common import core

torch = core.torch


def main():
    print(f'python {platform.python_version()} | torch {torch.__version__} | CUDA build {torch.version.cuda}')
    for package in ('torch_geometric', 'torch_scatter', 'torch_sparse', 'torch_cluster', 'numpy', 'scipy', 'pandas',
                    'scikit-learn', 'matplotlib', 'psutil', 'tqdm'):
        print(f'  {package} {metadata.version(package)}')
    common.require_gpu()

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
    for op in common.OPERATORS:
        deg_hist = core.in_degree_histogram(graph) if op == 'pna' else None
        model = core.build_model(op, 6, len(cols), len(cols), deg_hist=deg_hist, recipe=recipe)
        loader, feats = core.make_loader(graph, True, cols, cols, recipe=recipe)
        logits, y = core.forward_batch(model, next(iter(loader)), feats)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y)
        loss.backward()
        print(f'  {op:<11} params {core.count_params(model):>8,} | loss {loss.item():.4f} on {core.device}')
    print('OK')


if __name__ == '__main__':
    main()
