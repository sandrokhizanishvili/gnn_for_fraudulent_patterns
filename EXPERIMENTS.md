# Experiment Plan — GNN for Fraudulent Pattern Detection

**Thesis idea:** does manual feature engineering help Graph Neural Networks detect money
laundering? The study adds engineered features (baseline transaction features, and the GFP
structural features in particular) to otherwise identical models, across several GNN variants,
to identify the best-working model + feature combination for AML.

**Research questions:**
- **RQ1 — Do edge features help GNNs?** Within each block, the with/without-edge-features
  pairs (runs 1↔2, 3↔4, 5↔6 and 7↔8, 9↔10, 11↔12) differ only in whether transaction
  features enter message passing.
- **RQ2 — Do GFP structural features improve detection further?** Every Block 1 run vs its
  Block 2 twin — same model, only the feature set gains the 61 GFP columns.
- **RQ3 — Operator choice:** which GNN variant (GIN/GINE, PNA, GATv2) works best under the
  same capacity budget, and does the answer change once engineered features are present?
- **RQ4 — Best AML variant:** which model × feature combination wins overall — including
  whether a GBT on the same features matches the best GNN (S3), and whether tuning the GFP
  windows to the data beats the paper's config (S2)?

Master thesis experiment tracker. Dataset: IBM AML **HI-Small**, truncated to Sep 1–10 2022
(5,077,237 transactions, 4,522 laundering, 60/20/20 temporal split).

**Protocol for every run:** threshold chosen on validation, best-val-F1 checkpoint, test scored
once; headline metric minority-class F1, with PR-AUC / ROC-AUC / precision / recall alongside.

**Architecture rule:** 2 message-passing layers, hidden 64, dropout 0.3, same readout MLP and
training recipe for every model — no dimension or layer changes, ever. Adding features changes
only input-projection widths. Within a family the invariant-parameter count must match exactly;
across families dims/depth are equal but operators differ (both counts recorded per run).

**Status:** ✅ done · 🔄 running · ⬜ todo

---

## Phase 0 — Data pipeline (complete)

- ✅ EDA with day-specific USD conversion (`EDA.ipynb`)
- ✅ Data preparation: truncation, 20 baseline edge features, 6 node features, causal bank
  target encoding, 61 causal GFP features, normalization, PyG snapshots
- ✅ GFP parameter variant blocks win48 / win120 / lc10 / rich
- ✅ Artifact verification, 61 checks (`Data_checks.ipynb`)
- ✅ Fixed-architecture GIN notebook (feature / direction / temporal knobs, overfitting diagnostics)

---

## Block 1 — Baseline features (readout sees the 20 baseline features)

| # | Model | Message passing | Question | Status |
|---|---|---|---|---|
| 1 | GIN | node features only | topology alone, sum aggregation | ✅ test F1 0.385 · PR-AUC 0.314 (seed 42, best epoch 13, no overfitting) |
| 2 | GINE | + 20 base edge feats | edge features inside sum MP | ⬜ |
| 3 | PNA | node features only | multi-aggregator MP | ⬜ (needs PNA/GAT added to the notebook) |
| 4 | PNA | + 20 base edge feats | edge features inside PNA | ⬜ |
| 5 | GATv2 | node features only | attention aggregation | ⬜ |
| 6 | GATv2 | + 20 base edge feats | edge features inside attention | ⬜ |

## Block 2 — GFP variants of the same six (feature set = base + 61 GFP)

| # | Model | Message passing | Status |
|---|---|---|---|
| 7 | GIN | node features only, readout base+GFP | ⬜ |
| 8 | GINE | + 81 feats in MP and readout | ⬜ |
| 9 | PNA | node features only, readout base+GFP | ⬜ |
| 10 | PNA | + 81 feats in MP and readout | ⬜ |
| 11 | GATv2 | node features only, readout base+GFP | ⬜ |
| 12 | GATv2 | + 81 feats in MP and readout | ⬜ |
| 13 | GINE | + 20 base feats in MP, readout = embeddings + GFP only (61) | ⬜ |
| 14 | PNA | + 20 base feats in MP, readout = embeddings + GFP only (61) | ⬜ |
| 15 | GATv2 | + 20 base feats in MP, readout = embeddings + GFP only (61) | ⬜ |

Runs 13–15 are the **split-roles** variant: message passing digests the raw transaction
attributes into the embeddings, the classifier sees only `[h_src, h_dst, GFP]` — do raw
features become redundant at the decision layer once MP has consumed them?

**Together: a 3 × 2 × 2 factorial (operator × edge-feats-in-MP × GFP) plus the split-roles
variant — the uplift of each manually engineered feature block per operator family, under a
fixed architecture.**

Infrastructure: ⬜ extend `GIN_fixed_architecture.ipynb` with an `OPERATOR = gin | pna | gat`
knob (same layer structure; only the conv swaps).

---

## Secondary experiments

### S1 — Direction ablation
- ⬜ best model from Blocks 1–2 re-run with bidirectional MP (second conv over reversed edges)
- *Does seeing outgoing money help? Fan-out is invisible to incoming-only aggregation.*

### S2 — GFP parameter variants
- ⬜ prep: normalize win48 / win120 / lc10 / rich with the train-fit recipe (swap `edge_attr[:, 20:81]`)
- ⬜ best model re-run with each variant (4 runs)
- *Do data-driven windows beat the paper's config? Correlation analysis says yes (+0.064 → +0.097).*

### S3 — GBT baselines (LightGBM / XGBoost)
- ⬜ `GBT_baselines.ipynb`: rows = edge + src-node + dst-node features; feature sets base vs base+GFP (vs gfp-only)
- ⬜ GFP variant comparison in GBT (cheap cross-check of S2)
- *Does message passing add anything beyond engineered features? (paper: GFP+GBT ≥ GNN)*

### S4 — Seeds & robustness
- ⬜ 3–4 seeds on headline runs (block winners, S1–S3 winners); report mean ± std
- *Differences within seed spread are ties.*

### S5 — Temporal causal sampling (optional)
- ⬜ pyg-lib on Kaggle, then best model with `TEMPORAL_SAMPLING` on vs off
- *Features are causal; default neighbour sampling is not. How much does honesty cost?*

---

## Analysis

- ⬜ A1 — aggregation: one table + plots across all runs (`results/` → comparison notebook)
- ⬜ A2 — per-typology recall: join test predictions to `HI-Small_Patterns.txt` — which of the
  8 laundering patterns does each model catch (cycles vs fan-in vs stack …)?
- ⬜ A3 — error analysis of the best model: FP/FN by amount band, payment format, bank risk, degree
- ⬜ A4 — operating points: precision@K, recall at fixed precision (deployment view)

## Writing

- ⬜ W1 — data & methodology chapters (reuse `Progress_Report.md`)
- ⬜ W2 — experiments & results chapter (A1 tables, architecture diagrams done)
- ⬜ W3 — discussion: leakage findings (single-batch GFP, `time_window` cap), fixed-architecture
  methodology, limitations (synthetic data, single dataset, fixed capacity)

---

## Standing rules

1. Only the operator / feature / direction / temporal knobs change between runs — never dims,
   depth, hyperparameters or protocol; verify `invariant_params` within each family.
2. Every completed run: `results.json`, `history.csv`, `curves.png` (ideally the executed
   notebook) into `results/`, named `<model>_mp-<mp>_readout-<ro>_dir-<dir>[_temporal]_seed<k>`.
3. Test is scored once per run with the validation-chosen checkpoint and threshold; no
   decisions are ever made on test.
