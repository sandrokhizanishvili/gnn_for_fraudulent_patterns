# CODE_GUIDE — what the code does, file by file

Part A is the big picture; Part B lists every function. Numbers are in `Progress_Report.md`,
rules in `EXPERIMENTS.md`; those two stay the sources of truth.

The code lives in the folder `gnn-frauds/` (import name `gnn_frauds`). `conf/` holds the Hydra configs.
`gnn_core.py`, `gbt_core.py` and `mlp_core.py` re-export that package, so the old notebooks keep working.
One command runs a model: `python -m gnn_frauds model=gnns/gin features=cfg5 hyperparameters=gnns/v1 optimization=none experiment=train paths=local`.
`hyperparameters=gnns/v1` is the fixed architecture (`recipe=None`). Knob values are only in `conf/hyperparameters/`;
how a search draws them is only in `conf/optimization/`.

## Part A — the big picture

**The experiment.** Each transaction of IBM's HI-Small dataset is an edge between two accounts; the model says whether
it is money laundering (0.09 % positives). The thesis asks whether hand-made features help a GNN: the same model is
trained with different inputs, and the only thing that changes between runs is *which* edge features it sees and
*where* (inside message passing, at the final classifier, or both). Four operators (GIN, PNA, GATv2, a graph
transformer) go through the same runs, so a difference is due to the features, not the model.

**The data** (`Data_preparation.ipynb` → `Data/`). Three cumulative PyTorch Geometric graphs: train (first 60 % of
transactions in time), val (train + next 20 %, only the val part labelled), test (everything, only the last 20 %
labelled). Earlier transactions are visible as context but never counted in loss or metrics. Nodes = accounts with 6
entity one-hot columns. Edges = transactions with 81 columns: 20 baseline (amount, timing, bank, payment format) and
61 GFP, IBM's Graph Feature Preprocessor counts of fan-in, fan-out, scatter-gather and cycle patterns around the two
accounts within a time window. Normalisation is fitted on train rows only; `Data_checks.ipynb` verifies all of it.

**One transaction through the model** (`gnn_core.EdgeClassifier`). The full graph does not fit on a GPU, so a batch
is 8,192 labelled transactions plus a sampled neighbourhood (up to 100 incoming transactions per hop, two hops). The
6 node columns are projected to a 128-dimensional state. Two message-passing layers update each account's state from
the accounts that paid it; *how* the messages are combined is the operator, the only place the four models differ.
The readout concatenates the sender's state, the receiver's state and the transaction's own features, and an MLP
outputs one logit. Loss: binary cross-entropy with the laundering class weighted 8×.

**The seven configurations.** `mp` = edge columns the messages carry, `readout` = edge columns the classifier sees;
`base` = 20 baseline, `gfp` = 61 GFP, `full` = all 81, `none` = no edge features.

| Config | mp | readout | Tests |
|---|---|---|---|
| 1 | none | base | topology + message passing alone (reference) |
| 2 | base | base | baseline features inside the messages |
| 3 | none | full | GFP only at the classifier |
| 4 | full | full | GFP everywhere |
| 5 | base | full | GFP at the classifier only |
| 6 | base | gfp | are raw features redundant after message passing used them? |
| 7 | none | gfp | GFP alone vs baseline alone (compare with 1) |

Within one operator only the conv's edge projection and the first readout layer change width between configs;
`invariant_params()` counts the rest and the notebooks assert it is identical.

**One run** (`gnn_core.run_experiment`). Seed 42; 20 epochs of Adam with a cosine schedule; after every epoch
predict on train and val, pick the threshold that maximises val F1, log the metrics, keep the best weights; reload the
best epoch and score train, val and **test once** at the validation threshold; write `results.json`,
`predictions.csv`, `curves.png`. Test is never used to choose an epoch, a threshold or a hyperparameter. Folder names
say what the run was (`Outputs/PNA/pna_mp-base_readout-full_dir-in/` = PNA, config 5); `batch_summary.csv` per family
is the table the report copies.

**The files.**

| Group | Files | Status |
|---|---|---|
| Shared library | `gnn-frauds/` (import `gnn_frauds`); `gnn_core.py` re-exports it | live |
| Main grid | `GIN_`, `PNA_`, `GAT_`, `TRANSFORMER_fixed_architecture.ipynb` (differ only in `OPERATOR`) | live |
| Variants | `RWPE_fixed_architecture.ipynb` (+ `RWPE_encoding.ipynb`, `rwpe_compute.py`); `GFP_fixed_architecture.ipynb` (+ `GFP_experiments.ipynb`) | live |
| Cross-checks without message passing | `GBT_fixed_features.ipynb` (`gbt_core.py`), `MLP_fixed_features.ipynb` (`mlp_core.py`); `Typology_analysis.ipynb` breaks detections down by laundering pattern | live |
| One command | `Run_benchmark.ipynb`, `python -m gnn_frauds`, `conf/` | live |
| Tuning, current | `cineca/` — Luo et al. (2024) random search, 100 trials × 2 seeds per operator and config 2 / 4, then a 5-seed retrain (`EXPERIMENTS.md` §2b, `cineca/README_LORIS.md`) | live |
| Tuning, old | `GIN_tuning*`, `GAT_tuning*`, `Outputs/TUNING/`, `Outputs/FINAL/` | **superseded**, kept for the record |
| Other | `Outputs/` (every finished run), `Data/` (not versioned), `thesis/`, `Presentation/`, `Papers/` | — |

**Reading order.** This page; `Progress_Report.md` §1–4 and §7; `GIN_fixed_architecture.ipynb` (eight code cells);
then in `gnn_core.py`: `load_graphs`, `EdgeClassifier`, `MPLayer`, `make_conv`, `make_loader`, `train_one_run`,
`run_experiment`; then `EXPERIMENTS.md` §2b and `cineca/README_LORIS.md`.

## Part B — every function, file by file

### gnn_core.py

The root file re-exports `gnn_frauds`. The functions below live in `gnn-frauds/` (`model/gnns/classifier.py`,
`experiment/train.py`, `experiment/metrics.py`, `experiment/final.py`, `optimization/luo.py`, `dataset/hi_small.py`).
Entry points and what they call:

```
notebook         → run_experiment → build_model, make_loaders, train_one_run, score_all_splits, save_run
luo.py tune      → run_tuning_seed → build_model, make_loaders, train_one_run
                   finalise_trial  → seed_mean_best_epoch, score_trial
luo.py aggregate → choose_best_trial, save_recipe, trials_table, plot_val_curves
luo.py final     → run_final → make_trainval_graph, train_fixed_epochs, score_all_splits, final_scoring
```

Four ideas explain the structure: the recipe is a dict and `None` means v1, so one code path reproduces the grid and
runs any tuned setting; only `make_conv` knows the operator; test is touched in one place per run and never loaded
by the tuning; every long job can die and resume (`resume.pt`, atomic writes, skip-if-result-exists).
**(old)** marks functions used only by the superseded one-pass tuning.

| Function | What it does |
|---|---|
| Constants | v1 recipe (hidden 128, 2 layers, dropout 0.3, [100, 100] neighbours, batch 8,192, 20 epochs, Adam 1e-3, pos_weight 8, seed 42); data widths; metric names; tuning grids; `NEIGHBOURS_BY_DEPTH` / `_FALLBACK` (neighbours per hop for 1–10 layers); `device`. |
| `default_recipe`, `full_recipe` | The v1 knobs; overlay given knobs on them and validate. Every function calls `full_recipe` first. |
| `recipe_tag` | Folder name from the knobs that differ from v1 (`d0.2_bn_l4`). |
| `with_knob`, `edge_value` **(old)** | One-knob copies and the "one step beyond the grid" rule. |
| `set_seed` | Seeds Python, NumPy, torch CPU and GPU. |
| `normalise_gfp_sheet` | The Data_preparation GFP normalisation (log1p, p1/p99 clip, standardise, train-fit) for the tuned sheet. |
| `load_graphs` | Reads the three snapshots and `feature_meta.json`; returns graphs, `col_sets` (`none/base/gfp/full` → column indices), node width. `splits=('train','val')` never reads test. Optionally swaps in the tuned GFP sheet. |
| `set_node_encoding` | Appends the RWPE columns to the 6 entity columns (`none` = unchanged). |
| `make_trainval_graph` | The val snapshot with every edge labelled, for the final retrain. |
| `in_degree_histogram` | Degree histogram PNA needs for its scalers. |
| `gin_mlp`, `make_conv`, `make_norm` | GIN's inner MLP; **the one operator-specific line** (GINE/GIN, PNA, GATv2, TransformerConv, with or without edge features); none / BatchNorm / LayerNorm. |
| `MPLayer` | conv → optional linear residual → norm → ReLU → dropout → identity residual (v1) → optional FFN. |
| `EdgeClassifier` | `node_proj` → `num_layers` MPLayers → readout on [h_src ‖ h_dst ‖ e_seed] → logit. |
| `build_model`, `count_params`, `invariant_params` | Build on the device; total parameters; parameters whose shape ignores the input widths (must match across configs). |
| `make_loader`, `make_loaders` | PyG `LinkNeighborLoader` over a split's evaluated edges; the graph carries only the `mp` columns, the seeds' `readout` columns travel separately. |
| `forward_batch`, `predict` | One forward pass on a batch; eval-mode probabilities over a loader. |
| `best_f1_threshold` | The validation cut with the highest F1; chosen this way in every run. |
| `weighted_bce`, `top_k_metrics`, `compute_metrics`, `threshold_sensitivity` | Loss from eval-mode probabilities; precision/recall at the top 5 %; the seven metrics of a split; F1 at threshold ± 0.05 / 0.10 and 0.5 (robustness only). |
| `score_all_splits`, `save_predictions`, `flatten_metrics` | Score train / val / test once at one threshold (re-seeded, so sampling is identical); `predictions.csv`; `{split: {metric}}` → flat keys. |
| `train_epoch`, `smoothed`, `make_scheduler` | One pass over the loader; 3-point moving average; cosine (v1) or constant LR. |
| `write_json_atomic`, `rng_state`, `set_rng_state`, `save_resume`, `load_resume`, `add_full_metrics` | Crash-safe JSON; capture and restore every RNG; `resume.pt` per epoch; all seven metrics into a history row. |
| `train_one_run` | **The loop of every run that selects an epoch**: train, predict train and val, sweep the threshold, log, save `best.pt` when the selection metric (val F1 in v1, val PR-AUC in Luo) improves; optional early stopping, per-epoch checkpoints, resume; a non-finite prediction stops the run as diverged. |
| `train_fixed_epochs` | **The loop of the final retrain**: a fixed number of epochs on train + val, no selection; saves `final.pt`. |
| `plot_curves`, `plot_final_curves` | `curves.png` for a selecting run / a final run. |
| `run_name`, `result_row`, `save_run`, `print_model_header` | Folder name; the flat result row; `results.json`; the run header. |
| `run_experiment` | **One grid run end to end** (Part A). |
| `summarize_batch`, `seed_summary` | `batch_summary.csv` with the invariant-params assert; mean ± std over seeds. |
| `run_tuning_seed` | One (recipe, seed) run on train with validation every epoch, no test loader, checkpoint per epoch. |
| `seed_mean_best_epoch`, `score_trial` | Argmax of the seeds' mean validation curve; per-seed metrics at that epoch, checkpoints deleted, `trial.json`. |
| `run_tuning_trial`, `choose_winner`, `summarize_stage`, `summarize_tuning`, `neighbours_from_probe`, `probe_depths`, `run_threshold_selection`, `save_config_thresholds` **(old)** | The one-knob-at-a-time search, its winner rule (beat the incumbent by more than seed noise), its summaries, its time-based probe and its per-config threshold run. |
| `trials_table`, `load_trials`, `plot_val_curves` | One row per trial (`trials_summary.csv`); read every `trial.json`; `tuning_curves.png`. |
| `is_out_of_memory`, `process_rss_gb`, `probe_attempt`, `neighbour_candidates`, `probe_one` | Recognise CUDA OOM; CPU memory incl. workers; timed batches + one val pass with peak memory; primary then fallback neighbour list; one attempt with OOM recorded, not raised. |
| `probe_memory`, `time_loader_workers` | **Luo probe**: a depth fits if peak reserved GPU memory ≤ 85 %; stops at the first depth that fits with neither list; resumable. Time 0 / 4 / 6 sampling workers. |
| `save_recipe`, `load_recipe` | `recipe.json`: winning knobs, best epoch, one threshold per retrain seed (mean of the tuning seeds' thresholds); read and check it. |
| `final_scoring` | All metrics at the frozen threshold, plus two secondary views (another config's cut; the cut re-swept on the now-seen val rows) and the sensitivity table. |
| `run_final` | **One seed of the final retrain**: train + val graph, `train_fixed_epochs` to the frozen epoch, test once at the frozen threshold, all result files; skipped if `results.json` exists. |
| `LUO_*` constants, `luo_recipe` | The grid (64,800 cells before depth), depths 1–10, 100 epochs, draw seed, seeds 42–43 / 42–46, constant LR + PR-AUC selection, the heaviest probe setting, the 85 % rule; one trial's full recipe. |
| `draw_luo_trials` | **The pre-registered draw**: fixed-seed partial Fisher–Yates over the cells (trial *i* never changes when trials are added), a depth per operator from those that fit. |
| `choose_best_trial`, `delete_trial_checkpoints` | Highest mean val PR-AUC, tie-break F1, then id; trials without metrics cannot win. Remove per-epoch weights, keep `best.pt`. |
| `finalise_trial` | Once both seeds are done: best epoch from the seed-mean curve, that epoch's checkpoint → `best.pt`, rest deleted, curves redrawn, `trial.json`; OOM or instant divergence → a `trial.json` without metrics. |

### gbt_core.py — LightGBM cross-check

The root file re-exports `gnn_frauds.model.tabular.lightgbm`. The ranges are `conf/hyperparameters/lightgbm/v1.yaml`;
the halving schedule is `conf/optimization/halving.yaml`.
Same 93 inputs as the GNN readout (81 edge columns + the entity one-hots of sender and receiver), no message passing.
Metrics, threshold sweep, curves and summaries are imported from `gnn_core`. Called by `GBT_fixed_features.ipynb`:
`load_gbt_data → successive_halving → train_final × 4 seeds → seed_summary`. The tail appendix is
`gnn-frauds/model/tabular/tail.py`, re-exported here so the notebook still calls `gbt_core.tail_check`.

| Function | What it does |
|---|---|
| Constants | seeds 42–45; halving 100 configs, keep half per round, start on 10 % of train rows; checkpoint every 25 rounds; early stopping 100 rounds; selection by val PR-AUC; the search space (small trees, strong L2, slow learning); expected row and laundering counts. |
| `load_gbt_data`, `split_arrays` | The matrix from the graph files, labels from each split's snapshot, row ranges; (X, y) of one split. |
| `sample_params`, `make_dataset`, `make_val_dataset`, `fit_booster`, `predict_booster` | Random config; binned LightGBM datasets (val weights laundering by 8 so its loss matches the thesis loss); one fit with early stopping on that loss; probabilities from the first n trees. |
| `halving_schedule`, `successive_halving` | The rounds; **the search**: each round keeps the best half on more train rows, one line per (trial, round) to `trials.csv` (resumable), `best_params.json`. Test never touched. |
| `run_name_gbt`, `gbt_result_row`, `save_gbt_run`, `save_gbt_predictions` | Name; flat row (`params` = trees, `invariant_params` = 0); `results.json` incl. F1 at the paper's 0.5; `predictions.csv` in the GNN layout. |
| `train_final` | **One final seed**: fit, score train / val every 25 rounds, keep the lowest-val-loss round, sweep the threshold there, test once; `results.json`, `history.csv`, `curves.png`, `best.txt`. |
| `seed_summary` | `gnn_core.seed_summary` under the GBT file name. |

### `gnn-frauds/model/tabular/tail.py` — tail window, outside the protocol

Re-exported by `lightgbm.py` (and so by `gbt_core`). Rebuilds the 93 features of the eight tail days the dataset
cut removed and scores the finished boosters. Nothing is chosen on these numbers.

| Function | What it does |
|---|---|
| `_normalise_gfp`, `build_tail_matrix`, `tail_check` | Train-fit GFP normalisation; the tail matrix (baseline recipe, causal GFP stream in WSL, entity one-hots), self-check against stored Sep 10 rows; four windows at each seed's validation threshold → `tail_check.csv`. |

### mlp_core.py — MLP cross-check

The root file re-exports `gnn_frauds.model.tabular.mlp`. The fixed point and the choice lists are
`conf/hyperparameters/mlp/v1.yaml`; the draw settings are `conf/optimization/random.yaml`.
A plain neural net on the GBT matrix: separates "it is a neural net" from "it uses the graph". Called by
`MLP_fixed_features.ipynb`: `gbt_core.load_gbt_data → standardise → to_tensors → random_search → train_final × 4 → seed_summary`.

| Function | What it does |
|---|---|
| Constants | seeds 42–45; 100 trials, one seed each; cap 60 epochs, patience 10 on val PR-AUC; trial 0 = a 512 → 256 → 128 funnel; the search space (width, depth, shape, norm, dropout, lr, weight decay, pos_weight, batch size). |
| `standardise`, `to_tensors` | Scale the 81 edge columns with train mean / std (one-hots untouched); whole matrix to the GPU. |
| `layer_widths`, `widths_text`, `ResBlock`, `MLP` | Widths per shape (flat, funnel, resnet); printable shape; `z + Linear(Dropout(ReLU(Norm(Linear(z)))))`; one class for the three shapes ending in Linear → 1. |
| `draw_trials`, `predict` | Trial 0 = defaults, rest uniform, duplicates redrawn; chunked eval-mode probabilities. |
| `train_run` | **The loop**: seeded shuffle, Adam at constant LR, train and val scored every epoch with the GNN history columns, best val PR-AUC weights kept, early stopping. |
| `random_search` | Every trial to `trials.csv` (resumable), ranked by val PR-AUC, `best_params.json`. |
| `run_name_mlp`, `mlp_result_row`, `train_final`, `seed_summary` | Name; flat row; **one final seed** (test once at the kept epoch's val threshold, all result files, skipped if done); mean ± std over seeds. |

### rwpe_compute.py — random-walk positional encoding

For every account, the probability that a k-step walk along the money flow returns to it (k = 1 … K); high at step k
for an account on a k-cycle. Exact scipy sparse powers on CPU, one snapshot per call, self-loops dropped. Run by
`RWPE_encoding.ipynb`; output consumed by `gnn_core.set_node_encoding`.

| Function | What it does |
|---|---|
| `peak_ram_gb`, `load_edge_index` | Peak memory; sender / receiver index of every transaction, cached. |
| `compute_rwpe` | P = D_out⁻¹ A, then the diagonal of P, P², …, P^K → [nodes × K]. |
| `main` | `<split> <K> <data_dir> <out_dir>`: first n edges of the snapshot, compute, check [0, 1], save `rwpe_k<K>_<split>.pt` + log. |

### run_gfp_wsl.py

IBM's Graph Feature Preprocessor is missing from the Windows snapml build, so the notebooks and `gbt_core` call this
script through WSL. It streams the transactions in batches of 128 so an edge only sees earlier ones (a whole-file
`fit_transform` would leak the future). A script, no functions: load input array and parameter JSON, transform batch
by batch, save.

### cineca/ — the Luo tuning on Leonardo

SLURM packaging of `EXPERIMENTS.md` §2b; the training is `gnn_core`. Instructions: `cineca/README_LORIS.md`.

Seven files. Loris's own values (account, module) live in `config.local.sh`, not in git, read after `config.sh`.

| File | What it does |
|---|---|
| `config.sh` | Every setting with its default: account, partition, queues, modules, folders, trials, parallelism, memory, time. |
| `env.sh` | Sourced by `run.sh` and `job.sbatch`: `config.sh`, then `config.local.sh`, modules, the Python env on `PATH`, headless plots. |
| `environment.yml` | The conda environment, pinned to the Kaggle stack of the finished runs. |
| `run.sh <step>` | The one command: `check`, `probe`, `draw`, `tune`, `aggregate`, `final`, `status`, `pack`. For GPU steps it asks `luo.py plan` what is missing and submits one job array per (operator, config); re-running submits only unfinished, unqueued tasks. `pack` runs the aggregate checks and writes the three result archives. |
| `job.sbatch` | The one job script: `luo.py $STEP --index $SLURM_ARRAY_TASK_ID` (+ operator and config). |
| `luo.py` | Every step, one function each (below). |
| `README_LORIS.md` | The operator manual: upload, environment, limits, the commands in order, progress, what to send back. |

| `luo.py` function | What it does |
|---|---|
| `setting`, `require_gpu`, `job_info` | Read a `config.sh` value; refuse a CPU-only torch; host / job / GPU for `done.json`. |
| `tuning_root` … `final_dir` | The folder layout under `Outputs/TUNING_LUO` and `Outputs/FINAL_LUO`. |
| `tune_task`, `final_task`, `load_space` | Array index → (trial, seed) (indices 2i, 2i + 1 = the two seeds of trial i) / → retrain seed; read `trials_space.json`. |
| `squeue_tasks`, `sacct_states`, `job_name`, `finished`, `all_indices`, `job_memory_gb`, `memory_request`, `plan` | Queued tasks; past task states; `tune_gin_c2`; does a task's result file exist; all indices of a step; job memory from the probe; **what `run.sh` submits**. |
| `status_row`, `step_status` | done / running / queued / timeout / failed / missing per step and pair. |
| `step_check` | Versions, then one tiny model per operator on a random graph, forward + backward on the GPU → `OK`. |
| `step_probe` | Per operator: `probe_memory` at the heaviest setting on config 4 (stops if depth 2 does not fit), `time_loader_workers`; writes `probe.json`. |
| `step_draw` | Once, after the probes: `draw_luo_trials` → `trials_space.json`; refuses to run twice or after a trial started. |
| `finalise`, `finish_times`, `finalise_pending`, `step_tune` | Finalise one trial; when each seed finished; finalise trials cut short; **one (trial, seed) task**: train + val graphs only, `run_tuning_seed` for 100 epochs with resume, OOM / divergence recorded as a status, `done.json`, and the last seed finalises the trial. |
| `model_params`, `check_trial`, `check_final_run`, `tuning_pair`, `final_pair`, `step_aggregate` | Parameter counts on the meta device; **every assertion on a trial** (recipe = draw, test never loaded, 100 epochs, constant LR, no NaN, best epoch recomputed, `trial.json` = `history.csv`, one `best.pt`, params match); the final-seed assertions; per pair: checks, trial 0 vs the repo's v1 run, winner → `recipe.json`, summaries; final summaries; all eight pairs. |
| `step_final` | One retrain seed: load and check `recipe.json`, `run_final` with resume, remove `resume.pt`. |
| `main` | The command line: `luo.py <step>` with `--index`, `--operator`, `--config`; `luo.py plan <step>` for `run.sh`. |

### Presentation/make_figures.py

The slide charts (main F1 chart, RWPE and tuned-GFP paired charts, daily counts) from the `batch_summary*.csv` files;
nothing typed in.
