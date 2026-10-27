"""Train one GNN operator from the composed config. Dims are known only after the graphs load."""
import os

from omegaconf import OmegaConf

FAMILY = {'gin': 'GIN', 'gat': 'GAT', 'pna': 'PNA', 'transformer': 'TRANSFORMER',
          'lightgbm': 'GBT', 'mlp': 'MLP'}  # folder under Outputs/; the family name, not the operator id


def out_root(run, name):
    """Outputs/<FAMILY> for this operator. The run folder under it is unchanged."""
    return f'{run.paths.output_dir}/{FAMILY[name]}'


def write_resolved(run, out_dir):
    """Copy the composed config into the run folder. Hydra does not create its own outputs tree."""
    if getattr(run, 'cfg', None) is None:
        return  # a direct call, without Hydra compose, has no resolved config to copy
    os.makedirs(out_dir, exist_ok=True)
    OmegaConf.save(run.cfg, f'{out_dir}/config.yaml')


def train_gnn(name, run):
    """One selecting run: the fixed hyperparameters, then every evaluated transaction."""
    from gnn_frauds.dataset.hi_small import load_graphs
    from gnn_frauds.experiment.train import run_experiment
    from gnn_frauds.features import feature_pair

    fixed = OmegaConf.to_container(run.hyperparameters.fixed, resolve=True)
    pre = OmegaConf.to_container(run.preprocessing, resolve=True)
    graphs, col_sets, node_dim = load_graphs(
        run.paths.data_dir, gfp_variant=pre['gfp']['name'], gfp_dir=pre['gfp'].get('dir'))
    row = run_experiment(
        name, feature_pair(run.features), graphs, col_sets, node_dim, out_root(run, name),
        node_enc=pre['node_embedding']['name'], enc_dir=pre['node_embedding'].get('dir'),
        gfp_variant=pre['gfp']['name'], recipe=fixed, seed=int(run.seed))
    write_resolved(run, f"{out_root(run, name)}/{row['run']}")
    return row
