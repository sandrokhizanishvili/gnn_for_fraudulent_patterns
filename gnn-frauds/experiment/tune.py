"""One tuning trial. A GNN trial scores train and val only. Tabular searches call their own procedure."""
import os
import time
from pathlib import Path

GNN_NAMES = ('gin', 'pna', 'gat', 'transformer')


class TuneRun:
    """One trial of cfg.optimization. Hydra builds this from conf/experiment/tune.yaml."""

    def __init__(self, model, features, hyperparameters, optimization, dataset, preprocessing, paths, seed,
                 trial_index=None, config_id=None):
        self.model = model
        self.features = features
        self.hyperparameters = hyperparameters
        self.optimization = optimization
        self.dataset = dataset
        self.preprocessing = preprocessing
        self.paths = paths
        self.seed = int(seed)
        self.trial_index = None if trial_index is None else int(trial_index)
        self.config_id = None if config_id is None else int(config_id)
        self.cfg = None

    def run(self):
        """Local path: one GNN seed, or the tabular search. Cineca calls run_one instead."""
        if self.model.name in GNN_NAMES:
            return self._local_gnn()
        return self.model.tune(self)

    def _local_gnn(self):
        from omegaconf import OmegaConf

        from gnn_frauds.dataset.hi_small import load_graphs
        from gnn_frauds.features import feature_pair
        from gnn_frauds.model.gnns.run import out_root
        from gnn_frauds.optimization.luo import run_tuning_seed

        recipe = self.optimization.trial(self)
        pre = OmegaConf.to_container(self.preprocessing, resolve=True)
        graphs, col_sets, node_dim = load_graphs(
            self.paths.data_dir, gfp_variant=pre['gfp']['name'], gfp_dir=pre['gfp'].get('dir'),
            splits=('train', 'val'))
        cfg = feature_pair(self.features)
        trial_dir = f"{out_root(self, self.model.name)}/tune/{cfg['mp']}-{cfg['readout']}/t{self.trial_index or 0:03d}"
        return run_tuning_seed(
            self.model.name, cfg, graphs, col_sets, node_dim, trial_dir, recipe, self.seed)

    def run_one(self, graphs, col_sets, node_dim, trial_dir, recipe, seed, trial_id, job, cfg):
        """The body of one Cineca array task: train this seed, record done.json, drop the seed checkpoint."""
        from gnn_frauds.experiment.train import write_json_atomic
        from gnn_frauds.optimization.luo import is_out_of_memory, run_tuning_seed

        sdir = f'{trial_dir}/seed-{seed}'
        os.makedirs(sdir, exist_ok=True)
        t0 = time.time()
        try:
            info = run_tuning_seed(
                self.model.name, cfg, graphs, col_sets, node_dim, trial_dir, recipe, seed,
                eval_train_every=1, patience=None, resume=True, full_metrics=True)
            diverged_at = info['epochs_run'] + 1 if info['epochs_run'] < recipe['epochs'] else None
            info.update(status='diverged' if diverged_at else 'ok', diverged_at=diverged_at)
        except (Exception,) as err:
            # Out of memory is a trial outcome (status oom). Every other error still fails the job.
            import torch
            if not isinstance(err, (torch.cuda.OutOfMemoryError, RuntimeError)) or not is_out_of_memory(err):
                raise
            print(f'OUT OF MEMORY: trial t{trial_id:03d} seed {seed} recorded as status oom')
            info = {'status': 'oom', 'diverged_at': None, 'epochs_run': None, 'error': str(err)[:200]}
        Path(f'{sdir}/best.pt').unlink(missing_ok=True)  # this seed's own best epoch; the trial epoch is chosen later
        done = dict(info, seed=seed, trial_id=trial_id, test_split_loaded='test' in graphs,
                    minutes=(time.time() - t0) / 60, finished=time.time(), **job)
        write_json_atomic(f'{sdir}/done.json', done)
        print(f"seed {seed}: {done['status']} -> {sdir}/done.json")
        return done
