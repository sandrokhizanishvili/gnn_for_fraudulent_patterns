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
*identical* models across GNN operators (GIN/GINE, PNA, GATv2, later a graph transformer), so
any difference is attributable to the features, not the model. The results speak back to the
top-level question: *which* patterns a GNN catches on its own (topology + message passing),
*which* it needs help with (pre-computed structure at the readout), and *which* typologies
(cycles, fan-in/out, scatter-gather, stack…) each setup actually detects (analysis A2).

When writing or summarising, always keep these two levels visible: the broad "how do GNNs
find laundering patterns" question is the thesis; feature engineering is the method used to
probe it. Do not describe the thesis as only a feature-engineering study.

Extends my Graph Mining course project (`../AML_GNN_GMA/`, LI-Small) — but that pipeline is
**not ground truth**: several parts were re-tested and revised (see `Progress_Report.md` §4.3, §5.2).

**Dataset:** IBM AML **HI-Small** (Altman et al., NeurIPS 2023), truncated at Sep 10 2022.
5,077,237 transactions · 4,522 laundering (0.089%) · 515,070 accounts · 370 annotated attempts,
8 typologies. Files in `Data/` (never versioned).

**Task:** edge classification — nodes = accounts, edges = transactions, directed temporal multigraph.

**Research questions (RQ1–RQ4) and the 18-run grid** are defined in `EXPERIMENTS.md`. That file
is the single source of truth for what is planned, running, and done.

## 2. Source-of-truth documents — read before acting, keep in sync

| Document | Role |
|---|---|
| `EXPERIMENTS.md` | Experiment plan + run tracker: **status and pointers only, no numbers**. Read first for any modelling task. |
| `Progress_Report.md` | Detailed write-up of everything done (data, features, leakage findings) and **the only home of results tables, curves and interpretation**. |
| `README.md` | Public repo summary: notebooks, graph stats, outputs. |
| GitHub | https://github.com/sandrokhizanishvili/gnn_for_fraudulent_patterns |
| Notion "experiments" page | Mirror of `EXPERIMENTS.md` as a task list — https://app.notion.com/p/gnn_for_fraudulent_patterns-experiments-3e011ae27ef280889f9cd78f4cfb1a78 |
| Notion "documentation" page | Mirror of `Progress_Report.md`, condensed — https://app.notion.com/p/gnn_for_fraudulent_patterns-documentation-3e011ae27ef2807cb4dfc9582cf70298 |

**Consistency rule:** a fact (a metric, a dimension, a count, a hyperparameter, a run status)
must never differ between these documents. When you change one, update the others in the same
task — or, for Notion, give me the exact text to paste (or update it via the Notion tools if
they are available). Never leave a number in one place that contradicts another.

## 3. The fixed architecture — never change it

Fixed **a priori for ALL runs** (chosen so the 81-dim feature set is never compressed):

| Fixed | Value |
|---|---|
| Message-passing layers | 2 |
| Hidden dim | **128** |
| Dropout | 0.3 |
| Node MLP (GIN family) | Linear → BatchNorm → ReLU → Linear, learnable ε, residual |
| GATv2 | 4 heads × 32 = 128 |
| PNA | needs training-graph in-degree histogram; same dims/depth |
| Transformer | PyG `TransformerConv` with edge features, 4 heads × 32 = 128, same 2-layer template (local attention over sampled neighbours — full-graph attention is infeasible at 5M edges) |
| Readout | concat `[h_src ‖ h_dst ‖ e_seed]` → Linear 128 → ReLU → Dropout → Linear 1 |
| Sampling | `LinkNeighborLoader`, [100, 100], batch 8,192 seed edges |
| Loss / optimiser | `BCEWithLogitsLoss(pos_weight=8)` / Adam 1e-3, cosine, 20 epochs |
| Seed | 42 (S4 adds more seeds) |
| Invariant params (GIN family) | **67,587** — must be identical across feature configs; verify every run |

**The only things that may change between runs are the knobs:**
`OPERATOR` (gin | pna | gat | transformer) · `MP_EDGE_FEATS` (none | base | full) ·
`READOUT_EDGE_FEATS` (base | full | gfp) · `NODE_ENC` (none | rwpe | node2vec) ·
`GFP_VARIANT` (v0 | win48 | win120 | lc10 | rich) · `MP_DIRECTION` (in | bidirectional) ·
`TEMPORAL_SAMPLING` (on | off). Adding features changes **only input-projection widths**.

If I ask for something that would change dims, depth, hyperparameters, or protocol — even
"just to see" — **stop and tell me it breaks the fixed-architecture rule** before writing code.
If a comparison needs a different setting, it is a new, separately labelled experiment, never a
silent edit to an existing run.

## 3b. Planned extensions — pointer

Block 3 (RWPE / Node2Vec), the transformer operator and the GFP variants (S2) are specified in
**`EXPERIMENTS.md` → "Planned extensions"** and the Notion experiments page, including the
reasoning behind every size. Do not restate them here. Rules that apply when implementing:
- Their sizes (k = 8, dim 8, 4 heads × 32) are fixed a priori — never tune them per run.
- Plan in `EXPERIMENTS.md` + Notion first, get my OK, then the notebook.
- RWPE: reuse `add_rwpe()` from `../AML_GNN_GMA/gine_rwpe.ipynb`.

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

The pipeline is complete and verified (61 checks in `Data_checks.ipynb`). Do not "improve" it
without an explicit request. Known invariants:

- **Temporal 60/20/20 positional split**, boundaries fixed *before* feature engineering.
  Train 3,046,342 · val 1,015,447 · test 1,015,448 edges. Cumulative PyG snapshots
  (`train_graph.pt`, `val_graph.pt`, `test_graph.pt`); context edges carry label −1.
- **EDGE_DIM 81** = 20 baseline (`[:, :20]`) + 61 GFP (`[:, 20:81]`). **NODE_DIM 6.**
- **GFP is causal:** computed by streaming time-sorted edges through `transform` in batches
  of 128 (`run_gfp_wsl.py`, runs in WSL because Windows snapml lacks GFP). Single-batch
  `fit_transform` leaks the future — never use it.
- **Bank target encoding:** train-window rates frozen for val/test; strictly-earlier-only for
  train rows; smoothing m = 200.
- **Normalisation** fit on train only (`standard_scaler.pkl`). GFP variants (win48 / win120 /
  lc10 / rich) in `Data/gfp_variants/` are swappable into `edge_attr[:, 20:81]` after the same
  train-fit recipe (experiment S2).
- Any change that could let future information reach training is a bug. Flag it explicitly.
  If results look too good (big jumps, val ≫ train, near-perfect scores), suspect leakage first.

## 6. Run artifacts & naming

Every completed run writes to `Outputs/<FAMILY>/<run_name>/`:
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
`results/` holds only the archived hidden-64 reference of Run 1 (superseded; keep it).

After a run finishes: mark its row ✅ in `EXPERIMENTS.md` with the `Outputs/` pointer (no
metrics there); put the numbers, curves and RQ interpretation in `Progress_Report.md` §7–8;
mirror both to their Notion pages. Mark superseded rows `SUPERSEDED`, never delete them.

## 7. Repository conventions

- **One notebook per operator, one shared code file.** `GIN_fixed_architecture.ipynb`,
  `PNA_fixed_architecture.ipynb`, `GAT_fixed_architecture.ipynb` (later `TRANSFORMER_…`) each
  run one operator's batch and keep that family's results readable in one place. Everything
  that must be identical across operators lives in **`gnn_core.py`** and is imported, never
  copied: the model template (layers, readout, dropout), `build_model(operator, …)`, the
  training loop, threshold selection, `score_all_splits()`, `plot_curves()`, `save_run()`,
  `invariant_params()`. A notebook may set `OPERATOR`, the batch of knob configs, and paths —
  nothing else. If a notebook needs to redefine a shared function, that is a bug in
  `gnn_core.py`; fix it there.
- All operator notebooks have the **same section order and the same cells**; only the config
  cell differs. On Kaggle, cell 1 clones/pulls the repo and `sys.path.append`s it so the
  committed `gnn_core.py` is used.
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
end, in Overleaf. Until then:

- **Plain English, short.** Bullet points over paragraphs. One idea per bullet. Tables for
  anything with numbers. No sentence that just restates a table.
- **Fixed section order** (keep it, so readers always know where to look):
  1. **What & why** — thesis question and RQ1–RQ4 (5 lines max)
  2. **Data** — one table (counts, split windows, imbalance) + the truncation decision
  3. **Features** — baseline 20 / GFP 61 / node 6, each block one table: feature → evidence
  4. **Leakage & verification** — the findings and the fix, as bullets
  5. **Fixed architecture & protocol** — one diagram, one table of fixed values, the knobs
  6. **Results** — one ranked table per family, all splits; then "what it means" as
     ≤ 5 bullets tied to the RQs; curves of the winner
  7. **Open items** — pointer to `EXPERIMENTS.md`, nothing duplicated
  8. **Artifact inventory** — file → content
- **Every number has a home.** A metric appears in exactly one table; text refers to it,
  never repeats it with different rounding.
- **Mark status honestly:** ✅ done · 🔄 running · ⬜ todo · *SUPERSEDED* — never delete history.
- When a section grows beyond a screen, split it into a table or move detail to the notebook.

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
