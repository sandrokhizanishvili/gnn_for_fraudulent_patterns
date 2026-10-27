# GNN for Fraudulent Pattern Detection (Master Thesis)

Detecting money-laundering transactions with Graph Neural Networks on the IBM AML **HI-Small**
synthetic dataset (Altman et al., NeurIPS 2023). Extends my Graph Mining course project
(`../AML_GNN_GMA/`, LI-Small) with a re-validated, leakage-checked pipeline.

**New to the repository? Start with [CODE_GUIDE.md](CODE_GUIDE.md)**: part A is the big picture (what the code
does, one transaction through the model, the seven feature configurations, which files are live and which are
superseded, a reading order); part B describes every function of every `.py` file.

Full write-up of the work so far: **[Progress_Report.md](Progress_Report.md)**.

## Notebooks (run in order)

| Notebook | Kernel | What it does |
|---|---|---|
| `EDA.ipynb` | any Python 3 with pandas/matplotlib/yfinance | Exploratory analysis of HI-Small; USD conversion of amounts; findings that drive the feature design |
| `Data_preparation.ipynb` | `graph_feature_preprocessor` | Truncation at Sep 10, 20 baseline edge features, 6 node features, 61 GFP structural features, 60/20/20 temporal split, train-fit normalization, PyG graph snapshots |
| `GFP_experiments.ipynb` | `graph_feature_preprocessor` | One data-tuned GFP sheet (`tuned`: 48 h windows, cycles ≤ 12, bins [2, 4, 6, 8]) next to the paper baseline V0, with the measurements on the 370 annotated attempts that motivate it, causality and alignment checks |
| `Data_checks.ipynb` | `graph_feature_preprocessor` | Shows what every artifact in `Data/` is and looks like, and verifies it (row counts, alignment, leakage properties, graph consistency) with a summary table |
| `GIN_fixed_architecture.ipynb` | Kaggle (GPU) | GIN edge classifier with a **fixed architecture**; knobs select which edge features enter message passing and which the classifier sees |
| `GAT_fixed_architecture.ipynb` | Kaggle (GPU) | Same notebook for the GATv2 operator (4 heads × 32 = 128, edge features via `edge_dim`); only the config cell differs — everything shared is imported from `gnn_core.py` |
| `PNA_fixed_architecture.ipynb` | Kaggle (GPU) | Same notebook for the PNA operator (mean / max / min / std aggregators × degree scalers calibrated on the training-graph in-degree histogram, edge features via `edge_dim`); only the config cell differs |
| `TRANSFORMER_fixed_architecture.ipynb` | Kaggle (GPU) | Same notebook for the graph transformer operator (PyG `TransformerConv`, 4 heads × 32 = 128, attention with edge features via `edge_dim` over the sampled neighbourhood — local, not full-graph); only the config cell differs |
| `RWPE_encoding.ipynb` | `graph_feature_preprocessor` (CPU; also runs on Kaggle) | Random-walk positional encoding (RWPE) per snapshot, k = 8 and 16: toy sanity test against PyG `AddRandomWalkPE`, edge-list check against the graph files, self-loop tables (why self-loops are dropped from P), compute via `rwpe_compute.py` with a per-step time / RAM / fill-in log, checks, diagnostics, value scale → `Data/rwpe/` |
| `RWPE_fixed_architecture.ipynb` | Kaggle (GPU) | Same notebook for the node-encoding runs (`NODE_ENC = rwpe16` on configs 5 and 4 of every operator, all eight in one session); RWPE files from the Kaggle dataset `hi-small-rwpe`; results in `Outputs/RWPE/<FAMILY>/`, summary `batch_summary_rwpe.csv` per family |
| `GFP_fixed_architecture.ipynb` | Kaggle (GPU) | Same notebook for the data-tuned GFP runs (`GFP_VARIANT = tuned` on configs 5 and 4 of every operator, all eight in one session); the tuned sheet from the Kaggle dataset `hi-small-gfp-tuned`, swapped into the GFP block at load time (graph files untouched); results in `Outputs/GFP_TUNED/<FAMILY>/`, summary `batch_summary_gfp_tuned.csv` per family; verdict in `Progress_Report.md` §7.6 |
| `GIN_tuning.ipynb` | Kaggle (GPU) | *Superseded by the Luo tuning (`cineca/`).* Recipe tuning of the GIN family on config 2 (mp base / readout base), one knob at a time in a pre-registered order (normalisation → layers → FFN → pos_weight → learning rate → dropout: architecture first, regularisation last), 2 seeds per trial, selection on validation only, early stopping on cosine T_max 100; never builds a test loader; resumable across Kaggle sessions; committed without outputs (every number is in `Outputs/TUNING/GIN/`); froze `Outputs/TUNING/GIN/recipe.json` = v1 + BatchNorm after each layer, best epoch 34 (rules in `EXPERIMENTS.md` §2b) |
| `GIN_tuning_final.ipynb` | Kaggle (GPU) | *Superseded by the Luo tuning (13 of 16 runs done).* Retrains the frozen GIN recipe on the train + val graph (built in memory) for the frozen number of epochs on configs 2, 4, 5 and 6 (edge features in message passing), seeds 42–45 each, one validation-chosen threshold per config (selection run on the train graph; `Outputs/TUNING/GIN/config_thresholds.json`), test scored once per run, mean ± std over the four seeds per config and threshold-sensitivity tables → `Outputs/FINAL/GIN/retrain/`; resumable |
| `GAT_tuning.ipynb` / `GAT_tuning_final.ipynb` | Kaggle (GPU) | *Superseded by the Luo tuning (the retrain never ran).* The same two notebooks for the GATv2 family (only `OPERATOR` and the paths differ): tuning on GAT config 2 (one seed per trial, an extra self-loop stage) froze `Outputs/TUNING/GAT/recipe.json` = the v1 recipe unchanged, best epoch 44; then the final retrain of configs 2, 4, 5 and 6 → `Outputs/FINAL/GAT/retrain/` |
| `GBT_fixed_features.ipynb` | `graph_feature_preprocessor` (CPU, local) | GBT cross-check: LightGBM on the same 93 inputs the GNNs see (81 edge features + the 6 + 6 entity one-hots of both endpoints), no message passing — hand-crafted graph features vs learned ones. One successive-halving search (x0 = 100, train rows only, scored on validation, ranked by validation PR-AUC) over small, strongly regularised trees, every fit stopped on the weighted validation loss (patience 100), then four seeds under the thesis protocol; results in `Outputs/GBT/`, search log in `Outputs/GBT/search/`, summaries `batch_summary_gbt.csv` and `seed_summary_gbt.csv`; verdict in `Progress_Report.md` §7.7. Ends with a clearly labelled appendix outside the protocol: the boosters scored on the test window including the removed tail days (`tail_check.csv`) |
| `MLP_fixed_features.ipynb` | Kaggle (GPU) | MLP cross-check: a plain neural net on the same 93 inputs as the GBT (edge columns standardised on the train rows, one-hots untouched), no message passing — the GNN readout's input without the learned node states, so it tells message passing from "a neural net" in the GBT comparison. One random search (100 unique trials, one seed each, over three shapes flat / funnel / resnet, widths, depth, normalisation, dropout, learning rate, weight decay, `pos_weight`, batch size; every fit stopped on validation PR-AUC, trials ranked by validation PR-AUC), then four seeds under the thesis protocol; results in `Outputs/MLP/`, search log in `Outputs/MLP/search/`, summaries `batch_summary_mlp.csv` and `seed_summary_mlp.csv`; resumable across Kaggle sessions when the search log is committed in between; verdict in `Progress_Report.md` §7.13 |

`gnn-frauds/` — the importable package (`gnn_frauds`). `conf/` chooses the model, the features, the hyperparameters
and the search. `python -m gnn_frauds` and `Run_benchmark.ipynb` call the same experiment class.
`gnn_core.py` re-exports the package for the operator notebooks: the fixed model template,
`build_model(operator, …)`, loaders, the training loop with validation threshold sweep, metrics on
train / val / test, curves, saving and `invariant_params()`. Notebooks only set the operator, the
config batch and paths. The `NODE_ENC` knob (`none | rwpe8 | rwpe16`) appends a pre-computed node
encoding to the 6 entity columns; only `node_proj` widens. The `GFP_VARIANT` knob (`v0 | tuned`)
replaces the 61 V0 GFP columns of `edge_attr` by the 64-column tuned sheet at load time, normalised
with the same train-fit recipe; only the edge projections and the readout input widen. The training recipe
is a dict (`recipe=None` = the v1 constants, reproducing every finished run exactly); the tuning helpers
(`run_tuning_trial`, `choose_winner`, `probe_depths`, `save_recipe` / `load_recipe`), the train + val graph
(`make_trainval_graph`) and the final retrain (`run_final`) live in the same file, as does the Luo et al. (2024)
tuning: the knobs `residual`, `readout_layers`, `readout_width`, `lr_schedule`, `num_workers`, `select_metric`
(defaults = v1), resume from `resume.pt`, the memory probe, the pre-registered draw (`draw_luo_trials`) and the
two-seed finalisation (`finalise_trial`).

`cineca/` — the Luo et al. (2024) re-tuning of every operator on configs 2 and 4 (100 random trials × 2 seeds per
pair, then a 5-seed retrain of each winner; rules in `EXPERIMENTS.md` §2b), packaged for Cineca (Leonardo Booster,
SLURM): `config.sh` (settings; personal values in an untracked `config.local.sh`), `env.sh`,
`run.sh <check|probe|draw|tune|aggregate|final|status|pack>` (the one command), `job.sbatch`
(the one job script), `luo.py` (every step; tune and final use the same experiment class as the command above),
`README_LORIS.md`. The conda file is `env/environment.yml`. `n_trials` is `conf/optimization/luo.yaml`.
Results land in `Outputs/TUNING_LUO/` and `Outputs/FINAL_LUO/`.

`gbt_core.py` — the GBT cross-check next to the GNNs: builds the 93-column row from the graph files,
runs the successive-halving search (resumable `trials.csv`; every fit stops on the weighted validation
loss), trains the final seeds with validation checkpoints every 25 boosting rounds and writes the same result files as the GNN runs. Threshold sweep,
metrics, loss diagnostic, curves and batch summary are imported from `gnn_core.py`, never copied.

`mlp_core.py` — the MLP cross-check next to the GBT: scales the GBT matrix (train-row mean / std on the 81
edge columns), defines the three shapes (flat, funnel, resnet) in one `MLP` class, trains
with Adam at a constant learning rate and early stopping on validation PR-AUC, runs the resumable random
search (`trials.csv`) and the final seeds, and writes the same result files as the GNN runs. The matrix and
the predictions file come from `gbt_core.py`; threshold sweep, metrics, loss diagnostic, curves and summaries
from `gnn_core.py` — never copied.

`rwpe_compute.py` — exact RWPE of one snapshot on CPU (scipy; P = D_out⁻¹ A without self-loops,
step k = diag(Pᵏ), same maths as PyG `AddRandomWalkPE`), logging non-zeros, seconds and RAM per step.

`run_gfp_wsl.py` — helper that runs IBM SnapML's Graph Feature Preprocessor **inside WSL**
(the Windows snapml build lacks it), streaming edges in batches of 128 so features stay causal.
Requires a WSL venv: `python3 -m venv ~/gfp_env && ~/gfp_env/bin/pip install 'numpy<2' snapml`.

## Graph

- **Nodes** = accounts (515,070 after truncation), **edges** = transactions (5,077,237 after truncation), directed temporal multigraph. The 590,819 self-loops (11.6 % of edges, 8 laundering) are kept as edges and flagged by `Is_Self_Loop`.
- **Task** = edge classification (`Is Laundering`), 4,522 positives (0.089%).
- **Node features (6):** entity-type one-hot.
- **Edge features (81):** 20 baseline (USD log-amount, structuring band, time cyclicals, same-bank, time since previous txn of sender/receiver, causal leakage-guarded bank target encoding, payment-format OHE) + 61 GFP (scatter-gather, temporal & simple cycles, vertex statistics).
- **Snapshots:** `train_graph.pt` (train edges), `val_graph.pt` (train+val, val evaluated), `test_graph.pt` (all, test evaluated); context edges carry label −1.

## Outputs (`Data/`, not versioned — regenerate with the notebooks)

`edge_features.csv` · `node_features.csv` · `feature_meta.json` (feature groups, dims, ablation grid) ·
`standard_scaler.pkl` · `train/val/test_graph.pt` · `account_to_idx.pkl` · `gfp_variants/tuned.npy` (+ `tuned_cols.json`; Kaggle dataset `hi-small-gfp-tuned`) ·
`rwpe/rwpe_k{8,16}_{train,val,test}.pt` (RWPE node encodings, float32 [515,070, k])

## Model comparison

One fixed architecture (2 message-passing layers, hidden 128, dropout 0.3, same readout and
training recipe) trained per operator (GIN/GINE, PNA, GATv2, graph transformer) over
seven feature configurations: which edge features enter message passing (none / 20 baseline /
81 baseline+GFP) × which the classifier sees (baseline / baseline+GFP / GFP only). Headline
metric minority-class F1 at a validation-chosen threshold, with PR-AUC and top-5 % recall;
every metric on train, val and test. Results per family in `Outputs/<FAMILY>/` and in
[Progress_Report.md](Progress_Report.md) §7.1–7.4. Winners are chosen on validation F1 (gaps
under 0.02 are ties; test is shown, never used to choose): GIN-5; GAT-4 ≈ GAT-5; PNA-2 ≈ PNA-4 ≈
PNA-5; TR-4 ≈ TR-5 — in every family, baseline features in message passing with all 81 features
at the readout is among the best (test F1 0.49–0.53 for GIN and GATv2, 0.59–0.62 for PNA and the
graph transformer). The RWPE node encoding (§7.5) changes none of this: ties for GIN, PNA and the
transformer, a borderline gain for GATv2 only. The data-tuned GFP sheet (§7.6: 48 h windows,
cycles up to 12 hops, four bins) gives the same picture: ties for GIN, PNA and the transformer, a
gain for GATv2 config 5 only; the paper's V0 sheet stays the default. The GBT cross-check (§7.7:
LightGBM on the same 93 inputs, no message passing, hyperparameters from one successive-halving
search, every fit stopped on the validation loss, four seeds) lands at test F1 0.520 ± 0.008 and PR-AUC 0.483 — level with the best
GIN run and above GATv2, but 0.08–0.09 below PNA and the graph transformer: on identical inputs,
message passing with those two operators adds something the hand-crafted graph features alone do not.
The MLP cross-check (§7.13: a plain neural net on the same 93 inputs, recipe from one random search of
100 settings, four seeds) lands at test F1 0.489 ± 0.006 and PR-AUC 0.419 — below the trees and level
with the best GATv2 run — so the PNA / transformer lead over LightGBM is message passing, not "a neural
net beats trees".
