# CLAUDE.md — gnn_for_fraudulent_patterns (Master Thesis)

Read this fully at the start of every session. Its purpose is **consistency**: the thesis's
whole argument rests on every run being comparable, and every document saying the same thing.

## 1. What this project is

Master thesis, Sapienza (laurea magistrale).

**Thesis question (top level): how are Graph Neural Networks able to identify fraudulent
(money-laundering) patterns in a transaction network?**

The thesis answers it through one controlled study — **does manual feature engineering help
GNNs detect money laundering?** Engineered features (20 baseline transaction features; 61 GFP
structural features; later, node positional/structural encodings) are added to otherwise
*identical* models across GNN operators (GIN/GINE, PNA, GATv2, a graph transformer), so
any difference is attributable to the features, not the model. The results speak back to the
top-level question: *which* patterns a GNN catches on its own (topology + message passing),
*which* it needs help with (pre-computed structure at the readout), and *which* typologies
(cycles, fan-in/out, scatter-gather, stack…) each setup actually detects (analysis A2).

When writing or summarising, always keep these two levels visible: the broad "how do GNNs
find laundering patterns" question is the thesis; feature engineering is the method used to
probe it. Do not describe the thesis as only a feature-engineering study.

Extends my Graph Mining course project (`../AML_GNN_GMA/`, LI-Small) — but that pipeline is
**not ground truth**: several parts were re-tested and revised (see the two snapml pitfalls in `Progress_Report.md` §3).

**Dataset:** IBM AML **HI-Small** (Altman et al., NeurIPS 2023), truncated at Sep 10 2022.
5,077,237 transactions · 4,522 laundering (0.089%) · 515,070 accounts · 370 annotated attempts,
8 typologies. Files in `Data/` (never versioned).

**Task:** edge classification — nodes = accounts, edges = transactions, directed temporal multigraph.

**The experiment grid (7 feature configs × 4 operators, plus data variants and cross-checks)** is
defined in `EXPERIMENTS.md`. That file is the single source of truth for what is planned, running,
and done.

## 2. Source-of-truth documents — read before acting, keep in sync

| Document | Role |
|---|---|
| `EXPERIMENTS.md` | Experiment plan + run tracker: **status and pointers only, no numbers**. Read first for any modelling task. |
| `Progress_Report.md` | Markdown mirror of the Notion documentation page (**Notion is the main copy**): data, features, GFP, models, protocol, results tables with curves and interpretation, repository map, plus the reference list. Same section numbers as Notion. |
| `README.md` | Public repo summary: notebooks, graph stats, outputs. |
| GitHub | https://github.com/sandrokhizanishvili/gnn_for_fraudulent_patterns |
| Notion "experiments" page | Same content as `EXPERIMENTS.md`, kept in sync — https://app.notion.com/p/gnn_for_fraudulent_patterns-experiments-3e011ae27ef280889f9cd78f4cfb1a78 |
| Notion "documentation" page | **Main copy** of the write-up; `Progress_Report.md` mirrors it — https://app.notion.com/p/gnn_for_fraudulent_patterns-documentation-3e011ae27ef2807cb4dfc9582cf70298 |

**Consistency rule:** a fact (a metric, a dimension, a count, a hyperparameter, a run status)
must never differ between these documents. When you change one, update the others in the same
task — or, for Notion, give me the exact text to paste (or update it via the Notion tools if
they are available). Never leave a number in one place that contradicts another.

## 3. The fixed architecture — never change it

**Status (19 Oct 2026):** the table below is the **v1 recipe**, fixed a priori for every finished run.
After the 13 Oct meeting the recipe is tuned **once per operator** on config 2 (base / base) and then
frozen; it is never tuned per feature config (`EXPERIMENTS.md` §2b holds the pre-registered rules and
the stage tracker). **GIN is tuned and frozen (19 Oct): `Outputs/TUNING/GIN/recipe.json` = v1 with
BatchNorm after each message-passing layer** (`h ← h + Dropout(ReLU(BatchNorm(conv(h))))`), everything
else unchanged — 2 layers, hidden 128, dropout 0.3, no FFN, [100, 100], batch 8,192, Adam 1e-3 with
cosine T_max 100, pos_weight 8, weight decay 1e-5 — trained for the frozen 34 epochs; the family's invariant
parameter count under this recipe is **68,099** (67,587 + the two BatchNorm affines), identical across configs; the stages for
layers, FFN, pos_weight, LR and dropout all kept the incumbent. Every GIN v2 run (`GIN_tuning_final.ipynb`,
later the seven feature configs) loads that file; the finished v1 GIN runs stay as they are. PNA, GATv2
**GATv2 is tuned and frozen (21 Oct): `Outputs/TUNING/GAT/recipe.json` = the v1 recipe unchanged** (no
normalisation, 2 layers, no FFN, self-loops on, pos_weight 8, LR 1e-3, dropout 0.3 — every stage kept the v1 value,
one seed per trial) **with best epoch 44** on the 100-epoch cosine; its invariant parameter count stays 67,585. PNA
and the Transformer keep v1 until they are tuned the same way. In `gnn_core.py`, `recipe=None`
reproduces v1 exactly; a tuned run passes the recipe dict loaded from `recipe.json` (the only copy of the
frozen numbers). This note is extended per operator when its recipe is frozen.

Fixed **a priori for ALL runs** (chosen so the 81-dim feature set is never compressed):

| Fixed | Value |
|---|---|
| Message-passing layers | 2 |
| Hidden dim | **128** |
| Dropout | 0.3 |
| Node MLP (GIN family) | Linear → BatchNorm → ReLU → Linear, learnable ε, residual |
| GATv2 | 4 heads × 32 = 128; `add_self_loops=True` in v1 (recipe knob `self_loops`, tuned for GATv2 only) |
| PNA | PyG `PNAConv`, aggregators mean/max/min/std × scalers identity/amplification/attenuation, towers 1, in-degree histogram of the training graph; same dims/depth |
| Transformer | PyG `TransformerConv` with edge features, 4 heads × 32 = 128, same 2-layer template (local attention over sampled neighbours — full-graph attention is infeasible at 5M edges) |
| Readout | concat `[h_src ‖ h_dst ‖ e_seed]` → Linear 128 → ReLU → Dropout → Linear 1 |
| Sampling | `LinkNeighborLoader`, [100, 100], batch 8,192 seed edges |
| Loss / optimiser | `BCEWithLogitsLoss(pos_weight=8)` / Adam 1e-3, cosine, 20 epochs |
| Seed | 42 (seed sweep on the winning runs later; see "Cross-checks" in `EXPERIMENTS.md`) |
| Invariant params | GIN family **67,587** · GATv2 family **67,585** · PNA family **427,265** · Transformer family **133,121** — must be identical across feature configs within a family; verify every run |

**Comparability across operators:** the rule is the same number of layers (2) and the same
embedding dimension (128) for every operator; the parameter count may differ by operator, because
each operator's internals differ (PNA's 12 aggregator × scaler views feed a 1,664 → 128 MLP per
layer; the Transformer has four 128 × 128 projections per layer — query, key, value, skip — where
GATv2 has two). Total parameters over the 7 configs: GIN 103,043–131,843 · GATv2 103,041–131,585 ·
PNA 561,537–623,105 · Transformer 168,577–197,121. Within a family only the edge projections and
the readout input widen.

**Everything outside the message-passing operator is identical for every operator** — the
operator is the only box that may differ. Concretely, shared by GIN / GATv2 / PNA / Transformer
and enforced by `gnn_core.py` (`EdgeClassifier`, `MPLayer`), never re-implemented in a notebook:
- **Before MP:** node features → `node_proj` Linear 6 → 128 → ReLU; the `mp` knob selects the
  same edge columns for every operator and hands them to the conv as `edge_attr`. How the conv
  consumes them (GINE adds, GATv2 attends, PNA concatenates) is part of the operator.
- **Each of the 2 layers:** the same wrapper `h ← h + Dropout(0.3)(ReLU(conv(h, edges)))`; conv
  input 128, conv output 128, residual. Only `conv` differs.
- **After MP:** the same readout on the seed edge, concat `[h_src ‖ h_dst ‖ e_seed]` → Linear 128
  → ReLU → Dropout 0.3 → Linear 1; the same loader, loss, optimiser, schedule, threshold selection,
  checkpointing and scoring.
- **Normalisation:** none in the shared template (no LayerNorm anywhere). The only normalisation
  is the BatchNorm inside GIN's node MLP, which is part of the GIN operator. Adding a shared
  normalisation layer would change the fixed architecture and invalidate the finished runs.
Anything that would make a projection, dropout, readout or normalisation differ between
operators is a change to the fixed architecture — stop and say so.

**The only things that may change between runs are the knobs:**
`OPERATOR` (gin | pna | gat | transformer) · `MP_EDGE_FEATS` (none | base | full) ·
`READOUT_EDGE_FEATS` (base | full | gfp) · `NODE_ENC` (none | rwpe8 | rwpe16 | node2vec) ·
`GFP_VARIANT` (v0 | tuned) · `MP_DIRECTION` (in | bidirectional) ·
`TEMPORAL_SAMPLING` (on | off). Adding features changes **only input-projection widths**.

If I ask for something that would change dims, depth, hyperparameters, or protocol — even
"just to see" — **stop and tell me it breaks the fixed-architecture rule** before writing code.
If a comparison needs a different setting, it is a new, separately labelled experiment, never a
silent edit to an existing run.

## 3b. Planned extensions — pointer

RWPE / Node2Vec node encodings, the transformer operator and the GFP variants are specified in
**`EXPERIMENTS.md`** (sections 1–2 and the appendix "Fixed sizes for the planned extensions")
and on the Notion experiments page, including the reasoning behind every size. Do not restate
them here. Rules that apply when implementing:
- Their sizes (RWPE k = 16 / 8, Node2Vec dim = the RWPE k kept, 4 heads × 32) are fixed a
  priori — never tune them per run.
- Plan in `EXPERIMENTS.md` + Notion first, get my OK, then the notebook.
- **RWPE (decided 4 Oct):** a node feature, computed once per snapshot on that snapshot's own
  edges (train: train edges; val: train + val; test: all) — the same maths as PyG
  `AddRandomWalkPE` (`add_rwpe()` in `../AML_GNN_GMA/gine_rwpe.ipynb`): P = D_out⁻¹ A, step k
  = diag(Pᵏ). Computed by `rwpe_compute.py` (scipy, exact, checked against the PyG transform
  in `RWPE_encoding.ipynb`), saved as `Data/rwpe/rwpe_k{8,16}_{split}.pt`, loaded by the
  `NODE_ENC` knob (`none | rwpe8 | rwpe16`) → only `node_proj` widens to (6 + k) → 128;
  `invariant_params` counts `node_proj` at the base width 6, so it stays at the family value.
  The walk is **directed** (follows the money; decided 4 Oct after seeing that only ~1 % of
  accounts get a non-zero vector): non-zero = the account sits on a directed money cycle of
  length ≤ 16; an undirected walk would mostly re-encode degree (already in GFP) and is not
  computable exactly (P² through hubs of degree ~168k).
- **Self-loops are dropped when building P** (graph files and model inputs unchanged). Why:
  self-loops are 18 % of train edges but 2 % of val/test, and 81 % of them are Reinvestment,
  so with them step 1 would mostly repeat `Is_Self_Loop` / `PayFmt_Reinvestment` and shift
  between splits. Step 1 is therefore 0 for every account (asserted).
- **k rule:** k = 16 first, decided on validation F1 only (gap < 0.02 = tie). k = 8 runs only
  if k = 16 shows an uplift (> 0.02 validation F1 over the matching run without RWPE); then
  keep k = 8 if it ties with k = 16, otherwise keep k = 16. No uplift at k = 16 → stop, RWPE
  is a null result. Test is never used to choose.
- Stage 1 = 8 runs: config 5 (mp base / readout full) and config 4 (mp full / readout full)
  with `rwpe16` for GIN, GATv2, PNA and Transformer, from `RWPE_fixed_architecture.ipynb`
  (all eight in one Kaggle session). Always comment on overfitting from the curves (config 4
  overfit for GIN, mildly for PNA and the Transformer, without RWPE).
- Known caveat, written with the results: val/test RWPE also reflects edges later than the
  seed edge — the same caveat as neighbour sampling.
- **GFP tuned (decided 4 Oct):** the 64-column sheet replaces the V0 block at load time
  (`GFP_VARIANT`, `NODE_ENC` stays `none`); only the edge projections and the readout's first
  Linear widen (337 → 340 inputs), `invariant_params` unchanged. 8 runs = configs 5 and 4 of
  every operator from `GFP_fixed_architecture.ipynb`, each judged against the family's own V0
  run on validation F1 (gap < 0.02 = tie; test never used to choose). A win stays a separate
  finding; V0 remains the default sheet for every other experiment until I decide otherwise.
  Always comment on overfitting from the curves, especially config 4.
- **GBT cross-check (decided 17 Oct, recipe fixed 18 Oct):** LightGBM on the same 93 inputs (81
  `edge_attr` columns + the 6 + 6 entity one-hots of both endpoints), no message passing — hand-crafted
  graph features vs learned ones. Hyperparameters are chosen **once** by successive halving (x0 = 100,
  η = 2, r0 = 0.1; train rows only, scored on validation, trials ranked by validation PR-AUC) and then
  frozen for the 4 seeds (42–45); nothing is tuned per seed. Every fit stops on the weighted validation
  loss (patience 100; the kept round = the lowest-loss round, threshold swept there). Code in
  `gbt_core.py` (imports the metric / threshold / plot helpers from `gnn_core.py`, never copies them);
  one notebook `GBT_fixed_features.ipynb` runs **locally on CPU** (`graph_feature_preprocessor` env +
  lightgbm), the one exception to the Kaggle loop of §7. The search space and row conventions are in
  the `EXPERIMENTS.md` appendix. Only test F1 is a fair score for these runs (validation picks
  hyperparameters, kept round and threshold); say so wherever the GBT is compared. A first, wider
  search without early stopping memorised the train rows and was removed on 20 Oct together with its
  `GBT_VARIANT` knob; a new variant needs a new decision, not a knob.

## 4. Evaluation protocol — never deviate

- Threshold swept on **validation** each epoch (best-F1 point); keep the **best-val-F1 checkpoint**
  and **save that threshold** — it is part of the result, not a detail.
- **Test is scored exactly once** per run, with the validation-chosen checkpoint and threshold.
  No decision of any kind is ever made on test.
- **Every metric is computed on all three splits — train, val, test** — using the *same*
  validation-chosen threshold for the thresholded ones. A result with only test numbers is
  incomplete; a result without train numbers hides overfitting.
- **Metrics saved per split** (exactly these, in `results.json` and `batch_summary.csv`):

  | Metric | Notes |
  |---|---|
  | `f1` | minority-class F1 at the val-best-F1 threshold — **headline metric** |
  | `precision` | at the same val-best-F1 threshold, saved as its own column |
  | `recall` | at the same val-best-F1 threshold, saved as its own column |
  | `pr_auc` | threshold-free robustness check |
  | `roc_auc` | reported, not used for decisions (≈0.97 everywhere; uninformative at 1:1,120) |
  | `precision_at_5pct` | precision when flagging the top 5% highest-scored transactions |
  | `recall_at_5pct` | share of laundering caught in that top 5% |

  Plus once per run: `threshold`, `best_epoch`, `params`, `invariant_params`.
  Column naming: `<split>_<metric>` → e.g. `train_f1`, `val_pr_auc`, `test_recall_at_5pct`.
  `batch_summary.csv` must contain all of them; if a column is missing the run is not "done".
- Each model gets its own validation-chosen threshold (same procedure, not same number).
- **Curves (`curves.png`):** train and val loss, F1, PR-AUC per epoch, test precision–recall
  curve — and a **vertical dashed line at `best_epoch`** on every per-epoch panel, labelled,
  so the saved checkpoint is visible at a glance. Always comment on overfitting.
- Differences within seed spread are **ties**. A single-seed result is preliminary.
- **Predictions are saved for every run** (`predictions.csv`, see §6): every evaluated edge
  of train, val and test with its probability and an edge identifier. They feed the analyses
  A2–A4 (per-typology recall, error analysis, operating points) the way
  `../AML_GNN_GMA/models_results_analysis.ipynb` joined prediction files to
  `HI-Small_Patterns.txt`. A run without this file is not "done".

## 5. Data pipeline & leakage — treat as settled, protect it

The pipeline is complete and verified (59 checks in `Data_checks.ipynb`). Do not "improve" it
without an explicit request. Known invariants:

- **Temporal 60/20/20 positional split**, boundaries fixed *before* feature engineering.
  Train 3,046,342 · val 1,015,447 · test 1,015,448 edges. Cumulative PyG snapshots
  (`train_graph.pt`, `val_graph.pt`, `test_graph.pt`); context edges carry label −1.
- **EDGE_DIM 81** = 20 baseline (`[:, :20]`) + 61 GFP (`[:, 20:81]`). **NODE_DIM 6** (+ k
  when `NODE_ENC = rwpe<k>`; the RWPE files in `Data/rwpe/` are appended at run time, the
  graph files never change).
- **GFP is causal:** computed by streaming time-sorted edges through `transform` in batches
  of 128 (`run_gfp_wsl.py`, runs in WSL because Windows snapml lacks GFP). Single-batch
  `fit_transform` leaks the future — never use it.
- **Bank target encoding:** train-window rates frozen for val/test; strictly-earlier-only for
  train rows; smoothing m = 200.
- **Normalisation** fit on train only (`standard_scaler.pkl`). The data-tuned GFP sheet
  (`Data/gfp_variants/tuned.npy`, 64 columns: 48 h windows, cycles ≤ 12, bins [2, 4, 6, 8])
  replaces the GFP block of `edge_attr` at load time (`GFP_VARIANT = tuned` in
  `gnn_core.load_graphs`: same train-fit recipe via `normalise_gfp_sheet()`, width 20 + 64 = 84;
  the graph files never change; the "GFP tuned to the data" item in `EXPERIMENTS.md`).
- Any change that could let future information reach training is a bug. Flag it explicitly.
  If results look too good (big jumps, val ≫ train, near-perfect scores), suspect leakage first.

## 6. Run artifacts & naming

Every completed run writes to `Outputs/<FAMILY>/<run_name>/` (node-encoding runs:
`Outputs/RWPE/<FAMILY>/<run_name>/`; GFP-variant runs: `Outputs/GFP_TUNED/<FAMILY>/<run_name>/`):
`results.json`, `history.csv`, `curves.png`, `best.pt`, `predictions.csv`; plus
`batch_summary.csv` per batch (all columns of section 4, every split).

`predictions.csv` (plain CSV, ~230 MB per run; written once, at final scoring with the best
checkpoint; one row per evaluated edge of every split, sorted by `edge_id`):

| Column | Meaning |
|---|---|
| `split` | train / val / test |
| `edge_id` | position in the graph edge list = row index of `Data/edge_features.csv` (which holds `src_account`, `dst_account`, `Timestamp`, `label` for the typology join) |
| `src`, `dst` | node indices (`account_to_idx`) |
| `edge_time` | unix seconds |
| `y` | label |
| `prob` | model probability (sigmoid of the logit), full float32 precision |
| `pred` | predicted class 0/1 = `prob >= threshold` with the run's validation-chosen threshold, computed at scoring time (use this column rather than re-thresholding the CSV: re-comparing parsed floats can flip the edge that defines the threshold) |

It is written by `gnn_core.save_predictions()`; operator notebooks never re-implement it.
Run name: `<model>_mp-<mp>_readout-<ro>_dir-<dir>[_enc-<enc>][_gfp-<variant>][_temporal]`
(optional parts only when the knob is not at its default).
GBT cross-check runs: `Outputs/GBT/lightgbm_readout-full_seed-<seed>/` with the same files, except
`best.txt` (the LightGBM booster, gitignored) instead of `best.pt`; `history.csv` has one row per
scored boosting round (`epoch` = round); the result row keeps every `batch_summary.csv` column with
`operator = lightgbm`, `mp = none`, `mp_direction = none`, `params` = trees kept,
`invariant_params = 0`; `results.json` adds `hyperparams` and `paper_comparison` (F1 at 0.5).
Search log: `Outputs/GBT/search/`; summaries `Outputs/GBT/batch_summary_gbt.csv` and
`seed_summary_gbt.csv`; `Outputs/GBT/tail_check.csv` = the side check of the notebook appendix (boosters scored
on the test window including the removed tail days) — outside the protocol, never used to choose or compare.
Recipe-tuning runs (`EXPERIMENTS.md` §2b): `Outputs/TUNING/<FAMILY>/s<k>_<knob>/<trial>/` holds `trial.json`
(the trial's scoring: knobs, seed-mean best epoch, per-seed thresholds, train / val metrics mean ± std, the
gap) and `seed-<s>/history.csv`, `curves.png`; per stage `stage_summary.csv`; per family
`trials_summary.csv`, `tuning_curves.png`, `probe.json` and **`recipe.json`** (the frozen recipe — the only
copy of its numbers, loaded by the final notebook; never a dict in code). Tuning runs build no test loader
and write no `predictions.csv`; per-epoch checkpoints are deleted after the post-trial scoring; `best.pt`
stays on Kaggle. Trial tag = the knobs that differ from the v1 recipe + T_max (`d0.2`, `bn`, `ln`, `l4`,
`pw16`, `lr0.003`, `ffn`, joined by `_`; `base` = none). Hyperparameter rows are **trials**; **config** keeps
meaning the feature configurations 1–7.
Final retrain (train + val graph, frozen epoch and threshold): `Outputs/FINAL/<FAMILY>/retrain/<run>/seed-<s>/`
with `results.json` (plus `phase`, `stop_epoch`, `frozen_threshold`, `val_seen`, the recipe block),
`history.csv` (train-mode mini-batch loss and LR per epoch), `curves.png`, `sensitivity.csv` (test F1 /
precision / recall at the frozen threshold ± 0.05 / ± 0.10 and at 0.5), `predictions.csv` and `final.pt`
(the last two not versioned); summaries `batch_summary_retrain.csv`, `seed_summary_retrain.csv`. Run names
keep the convention above; the seed is the sub-folder.
`results/` holds only the archived hidden-64 reference of Run 1 (superseded; keep it).

After a run finishes: mark its row ✅ in `EXPERIMENTS.md` with the `Outputs/` pointer (no
metrics there); put the numbers, curves and RQ interpretation in `Progress_Report.md` §7 (and the Notion documentation page);
mirror both to their Notion pages. Mark superseded rows `SUPERSEDED`, never delete them.

## 7. Repository conventions

- **One notebook per operator, one shared code file.** `GIN_fixed_architecture.ipynb`,
  `PNA_fixed_architecture.ipynb`, `GAT_fixed_architecture.ipynb` and
  `TRANSFORMER_fixed_architecture.ipynb` each run one operator's batch and keep that family's
  results readable in one place. Everything
  that must be identical across operators lives in **`gnn_core.py`** and is imported, never
  copied: the model template (layers, readout, dropout), `build_model(operator, …)`, the
  training loop, threshold selection, `score_all_splits()`, `plot_curves()`, `save_run()`,
  `invariant_params()`. A notebook may set `OPERATOR`, the batch of knob configs, and paths —
  nothing else. If a notebook needs to redefine a shared function, that is a bug in
  `gnn_core.py`; fix it there.
- All operator notebooks have the **same section order and the same cells**; only the config
  cell differs. On Kaggle, cell 1 clones/pulls the repo and `sys.path.append`s it so the
  committed `gnn_core.py` is used.
  Exception: PNA's model-preview cell (cell 9) also passes
  `core.in_degree_histogram(graphs['train'])`, which `PNAConv` needs; all other notebooks omit
  it. Why: the other operators ignore the histogram, and `run_experiment()` computes it itself
  for PNA, so the argument would be dead code there and would mean editing executed notebooks.
  Second exception: `RWPE_fixed_architecture.ipynb` runs node-encoding configs for every
  operator in one session (so one executed notebook holds all their training logs); its
  config entries carry `operator` and `node_enc`, its runs land in `Outputs/RWPE/<FAMILY>/<run>/` with `batch_summary_rwpe.csv`
  per family, separate from the operator batches in `Outputs/<FAMILY>/`.
  Third exception: `GFP_fixed_architecture.ipynb` does the same for the GFP-variant runs: config
  entries carry `operator` and `gfp_variant`, the graphs are loaded once with
  `GFP_VARIANT = tuned` (one sheet per batch), runs land in `Outputs/GFP_TUNED/<FAMILY>/<run>/`
  with `batch_summary_gfp_tuned.csv` per family.
  Fourth exception: `GBT_fixed_features.ipynb` is not an operator notebook — LightGBM, no message
  passing (§3b). It keeps the section skeleton (environment, configuration, load, search, runs,
  comparison) but imports `gbt_core.py`, has no Kaggle cells and runs locally on CPU in the
  `graph_feature_preprocessor` env (Claude Code may run it in the background via nbconvert; it is
  resumable: logged search trials and finished runs are skipped on rerun).
  Fifth exception: the recipe-tuning pair `<OP>_tuning.ipynb` / `<OP>_tuning_final.ipynb` (GIN first;
  one pair per operator as it is tuned, same cells, only the knobs cell differs). The tuning notebook
  sets `OPERATOR`, `CFG` (config 2), `STAGES`, `SEEDS`, `TUNE` and runs the pre-registered stages through
  `gnn_core.run_tuning_trial` / `choose_winner` / `probe_depths` / `save_recipe`; it never builds a test
  loader, scores the train split every 5th epoch (`eval_train_every = 5`), and its `OUT_ROOT` sits inside
  the Kaggle repo clone so committed trials are skipped on the next session. The final notebook loads
  `recipe.json` (and stops with a clear message if it is missing), builds the train + val graph in
  memory and runs `run_final` for configs 2, 4, 5 and 6 × seeds 42–45 (the epoch from the config-2 tuning; one validation-chosen
  threshold per config, every config by the same selection run via `run_threshold_selection` →
  `Outputs/TUNING/<FAMILY>/config_thresholds.json`; the frozen tuning threshold of `recipe.json` is kept as the
  secondary `transfer_*` view; test metrics reported as mean ± std over the four; secondary view per seed: the threshold re-swept
  on the seen val rows and the thresholded metrics at it, `val_seen_threshold` / `seen_<split>_<metric>`, never the
  headline — PR-AUC is threshold-free and read first). **This pair is committed without
  outputs** (decided 18 Oct 2026): its executed form is several MB of inline curve images and logs,
  which GitHub's 1 MB content limit blocks from being imported into Kaggle, and every number the
  outputs show is already in `Outputs/TUNING/<FAMILY>/` (`history.csv`, `trial.json`, `curves.png`,
  `stage_summary.csv`, `probe.json`) and `Outputs/FINAL/<FAMILY>/`; the operator notebooks keep their
  outputs as before.
- **`recipe=None` reproduces v1.** Every `gnn_core.py` entry point takes an optional recipe dict
  (`full_recipe`, `DEFAULT = the module constants`). After any change to `gnn_core.py`, the regression
  check is part of the local verification: `run_experiment(recipe=None)` on the smoke slice with the
  previous file and the new one in the same process must give the same `history.csv` values and
  `best.pt` tensors, and a v1 `best.pt` must still load with `strict=True`.
- **Kaggle environment cell — keep verbatim in every operator notebook.** The pinned
  PyTorch / PyG stack below is known to work on Kaggle GPU; do not "simplify", reorder, or
  change versions without my OK (a mismatched torch / torch-scatter build is the usual cause of
  a wasted Kaggle session):
  ```
  !pip uninstall -y torch torchvision torchaudio torch-scatter torch-sparse pyg_lib
  !pip install torch==2.8.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
  !pip install torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-2.8.0+cu126.html
  !pip install torch_geometric
  ```
  Then the repo clone/pull + `sys.path.append`. If a new operator needs an extra package
  (e.g. `pyg-lib` for temporal sampling), add it as a separate line after these, not inside.
- Data-prep notebooks use the `graph_feature_preprocessor` kernel; EDA uses plain Python.
- **Claude Code cannot run training.** All notebooks execute on **Kaggle GPU**, by me. The loop is:
  1. Claude Code edits the notebook locally (plan → my OK → edit).
  2. I commit + push; I upload/run the notebook on Kaggle.
  3. I commit + push the Kaggle results (`Outputs/<FAMILY>/<run>/results.json`, `history.csv`,
     `curves.png`, `batch_summary.csv`) and pull them locally.
  4. Claude Code reads the results and updates `EXPERIMENTS.md`, `Progress_Report.md`, Notion.
  So: never claim a run "is done" from an edit. Keep every notebook
  runnable top-to-bottom on a fresh Kaggle kernel (paths, installs).
- **Verify locally before hand-off (CPU, no GPU here).** Before giving me a notebook to run on
  Kaggle, make sure it actually works: run it end-to-end yourself on a tiny slice of the data
  (a few tens of thousands of edges with some positives, few neighbours, 1 epoch). Do this in
  a scratch copy or a temporary script — **do not add a smoke-test flag or any test-only
  code to the notebook itself.** Every cell must execute, every output file must appear with
  all §4 columns, `invariant_params` must print. Delete the scratch outputs afterwards; never
  report those numbers. In the hand-off message, say the check passed and what it covered.
- **What is committed:** the small result files above (they are how results reach the local
  repo) and `results/`. **Not committed:** `Data/`, `best.pt` checkpoints, `predictions.csv`
  (`.gitignore`: `*.pt`, `*predictions*`); keep them locally under `Outputs/` for the analyses.
- Don't add dependencies without saying why. Don't rewrite working cells for style.
- `gnn_core.py` follows the same code-style rules below; each function has a 1–3 line docstring.
- **Commits:** never commit or push on your own. After each completed run or document sync,
  *propose* a commit (message = what changed, in one line) and wait for my OK. Never push
  without an explicit yes. `Data/`, `Outputs/`, checkpoints stay out of git.

### Code style — simple, readable, debuggable

- **Simplest solution that works.** Plain functions and loops over clever abstractions; no
  class hierarchies, decorators, or metaprogramming unless there is no simpler way.
- **One idea per cell, one job per function.** A cell should do something I can name in
  five words. A function should fit on one screen.
- **Easy to debug:** print shapes, counts and a few example rows at each step; assert what
  must be true (`assert edge_attr.shape[1] == 81`); no silent `try/except`.
- **Readable names:** `readout_edge_dim`, not `red`; no single-letter variables outside
  short loops and maths.
- **Comments:** short, and only where the *why* isn't obvious — not what the line does.
  One line above a block beats a paragraph. Keep the existing comment style of the notebook.
- **Markdown cells:** every section starts with a short markdown cell — what this section
  does, what it needs, what it produces. **Bullet points, not paragraphs**: 2–5 bullets,
  each one line. Plain English, no filler, no repetition of the code. If a markdown cell
  runs longer than ~6 lines, cut it or turn it into a table.
- **Consistency across notebooks:** same section order, same cells, same output format as
  `GIN_fixed_architecture.ipynb`. Shared logic goes in `gnn_core.py`, not into cells.
- If a piece of code is hard to explain in one markdown line, it is too complicated —
  simplify it before moving on.
- Long training: print per-epoch progress; keep runs resumable from `best.pt` where feasible.

## 8. `Progress_Report.md` and the Notion pages — writing style

These are **working documents**, not the thesis. The LaTeX thesis (academic tone) comes at the
end, in Overleaf. Until then, write them so that a colleague who has never seen the project
understands each section in one read.

### How to write

- **Plain English.** Short sentences, everyday words. Say "we removed the last week of data
  because it has almost no normal transactions", not "temporal truncation was applied to
  mitigate distributional artefacts". If a sentence needs a second read, rewrite it.
- **No filler and no hedging language.** Cut "it is worth noting that", "in order to",
  "leverage", "utilise", "robust", "comprehensive", "novel", "significantly" (unless there is
  a test), "state-of-the-art". Just state the thing.
- **One idea per bullet, one line per bullet.** If a bullet wraps to three lines, split it or
  turn it into a table row.
- **Numbers live in tables, never in prose.** Text says what a table shows ("GFP at the readout
  gives the biggest gain"); the table holds the values. A number appears in exactly one place.
- **Say what was done and what it means, in that order.** "What: X. Why: Y. Result: Z." is a
  fine skeleton for any subsection. Do not narrate the process ("first we tried… then we…").
- **Honest verbs.** "shows", "suggests", "we could not tell" — not "proves", "demonstrates
  clearly", "confirms". Ties within seed noise are called ties.
- **Explain jargon once, at first use, in brackets:** "GFP (pre-computed graph-pattern
  features)", "readout (the final classifier)". After that, use the short name.
- **Before / after examples** of the target style:
  - ✗ "The single-batch fit_transform paradigm was found to induce temporal leakage."
    ✓ "Running GFP on the whole dataset at once leaks the future: an account's first
    transaction already sees its later ones. We fixed it by streaming edges in batches of 128."
  - ✗ "GATv2 leverages an attention mechanism to adaptively weight neighbour contributions."
    ✓ "GATv2 learns which neighbours matter and weights them accordingly."
  - ✗ "Results demonstrate a significant uplift attributable to structural features."
    ✓ "Adding GFP features at the readout raises test F1 from 0.47 to 0.53 (single seed)."

### Structure

- **Notion is the main copy.** Edit the Notion page first (or at the same time), then mirror
  the change into `Progress_Report.md` with the same section numbers.
- **Fixed section order** (keep it, so readers always know where to look):
  0. **Overview** — thesis question, one short paragraph, links to the repo and the task list
  1. **Dataset** — one property → value table (counts, time span, imbalance, typologies)
  2. **Features** — node features table, baseline edge features table (feature → what →
     how computed → evidence), dropped features, one leakage rule
  3. **GFP configuration & variants** — one table: feature group × (V0, variants), why the
     variants, the two snapml pitfalls
  4. **Graph & splits** — one table with edges, evaluated edges, windows, laundering per split
  5. **Models — four operators** — one table of fixed values + knobs, then one subsection per
     operator (rule, edge features, invariant params, status)
  6. **Evaluation protocol** — bullets + one metrics table (metric → what → role)
  7. **Results — message passing × data variants** — per family: one subsection per run
     (metrics table on all splits + curves), then "Best of the family" with the ranked table
     and ≤ 3 verdict bullets tied to the thesis question
  8. **Repository map** — file → content
- **Each section opens with one line** saying what it contains. No section longer than a
  screen; split into a table or move detail to the notebook.
- **Mark status honestly:** ✅ done · 🔄 running · ⬜ todo · *SUPERSEDED* — never delete history.

## 9. How to work with me

- **Plain English, simple and concise.** If you introduce a tool or concept, one or two
  sentences on what it is and why is enough.
- Before a large change (new operator, new experiment block, restructuring a notebook):
  a short plan (3–6 bullets), then wait for my yes. Small fixes: do them and say what you did.
- Ambiguity: ask **one** focused question rather than guessing.
- Be honest about what a result shows and doesn't. Overclaiming is worse than a null result.
  Ties within seed noise are ties.
- Thesis text is written in **Overleaf (LaTeX)**. When I ask for writing, produce LaTeX
  snippets I can paste, reuse `Progress_Report.md` wording where sensible, and cite by
  author + year (Altman et al. 2023; Xu et al. 2019; Corso et al. 2020; Bouritsas et al.;
  Liu et al. 2021). The PDFs are in `../AML_GNN_GMA/Papers/`.
- I work on this in the evenings alongside a full-time job: prefer small, finishable steps and
  tell me clearly where we stopped and what is next.
