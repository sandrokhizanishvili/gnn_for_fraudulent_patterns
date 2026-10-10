# Experiments — GNN for Fraudulent Pattern Detection

**Thesis question:** how are Graph Neural Networks able to identify fraudulent (money-laundering)
patterns in a transaction network?

**How we test it:** the same fixed-size GNN is trained many times; only two things change —
**what data goes in** (section 1) and **how the network aggregates neighbours** (section 2).
Each combination is one **run** (section 3). If a run scores better, the credit goes to the data
or the operator, never to a bigger model.

Results and numbers → `Progress_Report.md` and the Notion documentation page.
Mirrored on the Notion experiments page (kept in sync).

*A box is ticked when the results are pushed to GitHub and the documentation is updated.*

---

## Now

- [x] GIN batch (GIN-1 … GIN-7) trained under the full evaluation protocol — Kaggle, 26 Sep;
      results in `Progress_Report.md` §7.1 (best by validation F1: GIN-5)
- [x] PNA batch (PNA-1 … PNA-7) trained under the same protocol — Kaggle, 27 Sep;
      results in `Progress_Report.md` §7.3 (best by validation F1: PNA-2 ≈ PNA-4 ≈ PNA-5, a tie)
- [x] GATv2 batch (GAT-1 … GAT-7) trained under the same protocol — Kaggle, 26 Sep;
      results in `Progress_Report.md` §7.2 (best by validation F1: GAT-4 ≈ GAT-5, a tie)
- [x] Graph Transformer batch (TR-1 … TR-7) trained under the same protocol — Kaggle, 28 Sep;
      results in `Progress_Report.md` §7.4 (best by validation F1: TR-4 ≈ TR-5, a tie; TR-2 at the edge)
- [x] RWPE stage 1 (8 runs, `rwpe16`) trained on Kaggle, 5 Oct; results in `Progress_Report.md`
      §7.5 (ties for GIN, PNA and the Transformer; GAT-5 + RWPE just above the tie band → one
      k = 8 run on GAT config 5 next, section 3 "RWPE runs")
- [x] GFP `tuned` batch (8 runs, `GFP_VARIANT = tuned`) trained on Kaggle, 5–6 Oct; results in
      `Progress_Report.md` §7.6 (ties for GIN, PNA, the Transformer and GAT-4; GAT-5 + tuned the
      only run above the tie band; V0 stays the default sheet, section 3 "GFP tuned runs")

---

## 1 · Data — what goes into the model

Dataset: IBM AML **HI-Small** (synthetic bank transactions, 5.08 M transactions, 0.09 %
laundering). Each transaction is an edge, each account a node.

- [x] **Baseline edge features** (20 per transaction) — amount, timing, bank, payment format
- [x] **GFP structural edge features** (61 per transaction) — pre-computed graph patterns around
      the transaction (cycles, fan-in/out, scatter-gather, degree statistics), IBM's Graph Feature
      Preprocessor with the paper's windows; simple cycles capped at length 6 (paper: 10)
- [x] **Node features** (6 per account) — account entity type
- [x] **GFP tuned to the data** (64 per transaction) — one alternative sheet, `tuned`: 48 h windows
      (scatter-gather 12 h), simple cycles up to 12 hops, four pattern bins [2, 4, 6, 8]; chosen from
      the durations and sizes of HI-Small's 370 annotated attempts (`GFP_experiments.ipynb`,
      `Progress_Report.md` §3). Runs: configs 5 and 4 of every operator with
      `GFP_VARIANT = tuned` (section 3, "GFP tuned runs"), each compared with the family's own V0
      run on validation F1 (gap < 0.02 = tie). Done (§7.6): seven ties, one gain (GAT-5); V0 stays
      the default sheet
- [ ] **RWPE node encoding** (16 per account, k = 16 first) — directed random-walk return
      probabilities (walks follow the money: non-zero = the account sits on a directed money
      cycle of length ≤ 16); one vector per snapshot from
      that snapshot's own edges, self-loops dropped (they are 18 % of train edges but 2 % of
      val/test, and `Is_Self_Loop` / `PayFmt_Reinvestment` already carry them);
      `RWPE_encoding.ipynb` → `Data/rwpe/`. Runs: configs 5 and 4 of every operator with RWPE
      (section 3, "RWPE runs"). Rule, on validation F1 only (gap < 0.02 = tie): k = 8 runs only
      if k = 16 shows an uplift (> 0.02 validation F1 over the matching run without RWPE); then
      keep k = 8 if it ties with k = 16, otherwise keep k = 16. No uplift at k = 16 → stop,
      RWPE is a null result. Test is never used to choose.
- [ ] **Node2Vec node encoding** (dim = the RWPE k kept) — learned alternative to RWPE, if time allows

---

## 2 · Message passing — how the model processes it

All operators share one code file (`gnn_core.py`) and one fixed template — 2 layers, width 128,
dropout 0.3, identical training recipe — so only the aggregation rule differs.

- [x] **GIN / GINE** — sums neighbour messages · `GIN_fixed_architecture.ipynb` · 7 runs
- [x] **PNA** — several aggregators at once (mean, max, min, std) · `PNA_fixed_architecture.ipynb` · 7 runs
- [x] **GATv2** — attention decides which neighbours matter · `GAT_fixed_architecture.ipynb` · 7 runs
- [x] **Graph Transformer** — attention with edge features over the sampled neighbourhood ·
      `TRANSFORMER_fixed_architecture.ipynb` · 7 runs

**Two extra questions about message passing itself (optional)**

- [ ] **Direction** — does the network also need to see outgoing money? Best model re-run with
      bidirectional aggregation
- [ ] **Causal sampling** (optional) — sample only earlier neighbours; how much does strict
      honesty cost?

---

## 3 · Runs — Data × Message passing

*Each family runs the same seven feature configurations: which edge features enter message
passing / which the final classifier sees. "base" = 20 baseline, "base+GFP" = all 81,
"GFP only" = 61, "none" = topology only. Pointer = result folder once done.*

### GIN family

- [x] **GIN-1** · none / base — topology alone, the reference point → `Outputs/GIN/gin_mp-none_readout-base_dir-in/`
- [x] **GIN-2** · base / base — edge features inside message passing → `Outputs/GIN/gin_mp-base_readout-base_dir-in/`
- [x] **GIN-3** · none / base+GFP — GFP only at the decision layer → `Outputs/GIN/gin_mp-none_readout-full_dir-in/`
- [x] **GIN-4** · base+GFP / base+GFP — GFP everywhere → `Outputs/GIN/gin_mp-full_readout-full_dir-in/`
- [x] **GIN-5** · base / base+GFP — GFP at the decision layer only → `Outputs/GIN/gin_mp-base_readout-full_dir-in/`
- [x] **GIN-6** · base / GFP only — are raw features redundant once message passing has used them? → `Outputs/GIN/gin_mp-base_readout-gfp_dir-in/`
- [x] **GIN-7** · none / GFP only — GFP alone vs baseline alone (compare with GIN-1) → `Outputs/GIN/gin_mp-none_readout-gfp_dir-in/`

### PNA family

- [x] **PNA-1** · none / base → `Outputs/PNA/pna_mp-none_readout-base_dir-in/`
- [x] **PNA-2** · base / base → `Outputs/PNA/pna_mp-base_readout-base_dir-in/`
- [x] **PNA-3** · none / base+GFP → `Outputs/PNA/pna_mp-none_readout-full_dir-in/`
- [x] **PNA-4** · base+GFP / base+GFP → `Outputs/PNA/pna_mp-full_readout-full_dir-in/`
- [x] **PNA-5** · base / base+GFP → `Outputs/PNA/pna_mp-base_readout-full_dir-in/`
- [x] **PNA-6** · base / GFP only → `Outputs/PNA/pna_mp-base_readout-gfp_dir-in/`
- [x] **PNA-7** · none / GFP only → `Outputs/PNA/pna_mp-none_readout-gfp_dir-in/`

### GATv2 family

- [x] **GAT-1** · none / base → `Outputs/GAT/gat_mp-none_readout-base_dir-in/`
- [x] **GAT-2** · base / base → `Outputs/GAT/gat_mp-base_readout-base_dir-in/`
- [x] **GAT-3** · none / base+GFP → `Outputs/GAT/gat_mp-none_readout-full_dir-in/`
- [x] **GAT-4** · base+GFP / base+GFP → `Outputs/GAT/gat_mp-full_readout-full_dir-in/`
- [x] **GAT-5** · base / base+GFP → `Outputs/GAT/gat_mp-base_readout-full_dir-in/`
- [x] **GAT-6** · base / GFP only → `Outputs/GAT/gat_mp-base_readout-gfp_dir-in/`
- [x] **GAT-7** · none / GFP only → `Outputs/GAT/gat_mp-none_readout-gfp_dir-in/`

### Transformer family

- [x] **TR-1** · none / base → `Outputs/TRANSFORMER/transformer_mp-none_readout-base_dir-in/`
- [x] **TR-2** · base / base → `Outputs/TRANSFORMER/transformer_mp-base_readout-base_dir-in/`
- [x] **TR-3** · none / base+GFP → `Outputs/TRANSFORMER/transformer_mp-none_readout-full_dir-in/`
- [x] **TR-4** · base+GFP / base+GFP → `Outputs/TRANSFORMER/transformer_mp-full_readout-full_dir-in/`
- [x] **TR-5** · base / base+GFP → `Outputs/TRANSFORMER/transformer_mp-base_readout-full_dir-in/`
- [x] **TR-6** · base / GFP only → `Outputs/TRANSFORMER/transformer_mp-base_readout-gfp_dir-in/`
- [x] **TR-7** · none / GFP only → `Outputs/TRANSFORMER/transformer_mp-none_readout-gfp_dir-in/`

### RWPE runs — node encoding (`RWPE_fixed_architecture.ipynb`, all eight in one Kaggle session)

*Config 5 (base / base+GFP) and config 4 (base+GFP / base+GFP) of each family with the k = 16
RWPE node encoding (`NODE_ENC = rwpe16`); each compared with the family's own config 5 / 4 above
on validation F1. Summary file: `Outputs/RWPE/<FAMILY>/batch_summary_rwpe.csv`; the operator batches stay in `Outputs/<FAMILY>/`.*

- [x] **GIN-5 + RWPE** · base / base+GFP · rwpe16 → `Outputs/RWPE/GIN/gin_mp-base_readout-full_dir-in_enc-rwpe16/`
- [x] **GIN-4 + RWPE** · base+GFP / base+GFP · rwpe16 → `Outputs/RWPE/GIN/gin_mp-full_readout-full_dir-in_enc-rwpe16/`
- [x] **GAT-5 + RWPE** · base / base+GFP · rwpe16 → `Outputs/RWPE/GAT/gat_mp-base_readout-full_dir-in_enc-rwpe16/`
- [x] **GAT-4 + RWPE** · base+GFP / base+GFP · rwpe16 → `Outputs/RWPE/GAT/gat_mp-full_readout-full_dir-in_enc-rwpe16/`
- [x] **PNA-5 + RWPE** · base / base+GFP · rwpe16 → `Outputs/RWPE/PNA/pna_mp-base_readout-full_dir-in_enc-rwpe16/`
- [x] **PNA-4 + RWPE** · base+GFP / base+GFP · rwpe16 → `Outputs/RWPE/PNA/pna_mp-full_readout-full_dir-in_enc-rwpe16/`
- [x] **TR-5 + RWPE** · base / base+GFP · rwpe16 → `Outputs/RWPE/TRANSFORMER/transformer_mp-base_readout-full_dir-in_enc-rwpe16/`
- [x] **TR-4 + RWPE** · base+GFP / base+GFP · rwpe16 → `Outputs/RWPE/TRANSFORMER/transformer_mp-full_readout-full_dir-in_enc-rwpe16/`
- [ ] **GAT-5 + rwpe8** · base / base+GFP · rwpe8 → `Outputs/RWPE/GAT/gat_mp-base_readout-full_dir-in_enc-rwpe8/` — the only configuration above the tie band at k = 16 (rule in section 1); k = 8 is kept only if it ties with k = 16. No k = 8 runs for GIN, PNA, Transformer (null result at k = 16)

### GFP tuned runs — data-tuned GFP sheet (`GFP_fixed_architecture.ipynb`, all eight in one Kaggle session)

*Config 5 (base / base+GFP) and config 4 (base+GFP / base+GFP) of each family with the `tuned`
GFP sheet in place of V0 (`GFP_VARIANT = tuned`: the 64 columns are normalised like V0 and swapped
into the GFP block at load time, `edge_attr` 84 wide, graph files untouched; `NODE_ENC = none`,
one factor at a time). Summary file: `Outputs/GFP_TUNED/<FAMILY>/batch_summary_gfp_tuned.csv`.*

- [x] **GIN-5 + tuned** · base / base+GFP · tuned → `Outputs/GFP_TUNED/GIN/gin_mp-base_readout-full_dir-in_gfp-tuned/`
- [x] **GIN-4 + tuned** · base+GFP / base+GFP · tuned → `Outputs/GFP_TUNED/GIN/gin_mp-full_readout-full_dir-in_gfp-tuned/`
- [x] **GAT-5 + tuned** · base / base+GFP · tuned → `Outputs/GFP_TUNED/GAT/gat_mp-base_readout-full_dir-in_gfp-tuned/`
- [x] **GAT-4 + tuned** · base+GFP / base+GFP · tuned → `Outputs/GFP_TUNED/GAT/gat_mp-full_readout-full_dir-in_gfp-tuned/`
- [x] **PNA-5 + tuned** · base / base+GFP · tuned → `Outputs/GFP_TUNED/PNA/pna_mp-base_readout-full_dir-in_gfp-tuned/`
- [x] **PNA-4 + tuned** · base+GFP / base+GFP · tuned → `Outputs/GFP_TUNED/PNA/pna_mp-full_readout-full_dir-in_gfp-tuned/`
- [x] **TR-5 + tuned** · base / base+GFP · tuned → `Outputs/GFP_TUNED/TRANSFORMER/transformer_mp-base_readout-full_dir-in_gfp-tuned/`
- [x] **TR-4 + tuned** · base+GFP / base+GFP · tuned → `Outputs/GFP_TUNED/TRANSFORMER/transformer_mp-full_readout-full_dir-in_gfp-tuned/`

**Decision rule, fixed in advance:** each run is compared with the family's own V0 config 5 / 4
above on validation F1 only (gap < 0.02 = tie); test is never used to choose. If tuned wins on
most configs, that is reported as a separate finding ("the data-tuned sheet helps"); V0 stays the
default sheet for every other experiment (RWPE, direction, sampling, seeds) until I decide
otherwise. Every result comments on overfitting from the curves, especially config 4 (overfit
for GIN, mildly for PNA and the Transformer, on V0).

### Cross-checks (optional)

- [ ] **Gradient-boosted trees** on the same features, no graph — does message passing add
      anything beyond engineered features?
- [ ] **Seeds** — 3–4 random seeds on the winning runs, mean ± std; differences inside the seed
      spread count as ties

---

## 4 · Analysis & writing

- [ ] One comparison table + plots across all runs
- [ ] **Per-typology recall** — which of the 8 laundering patterns (cycle, fan-in, stack …) each
      model actually catches; this answers the thesis question directly
- [ ] Error analysis of the best model — what it gets wrong and why
- [ ] Operating points — precision at a fixed alert budget, recall at fixed precision
- [ ] Thesis chapter: data & methodology
- [ ] Thesis chapter: experiments & results
- [ ] Thesis chapter: discussion — leakage findings, fixed-architecture method, limitations

---

## Log

*Newest first. Unticked = in progress · ticked = finished and synced.*

- [x] **5–6 Oct** — GFP tuned batch: all eight runs (configs 5 and 4 × four operators,
      `GFP_VARIANT = tuned`) in one Kaggle session of `GFP_fixed_architecture.ipynb`; results,
      curves and predictions in `Outputs/GFP_TUNED/`, sections and comparison tables in
      `Progress_Report.md` §7.6: seven ties, GAT-5 + tuned the only run above the tie band
      (+0.042 validation F1); V0 stays the default sheet
- [x] **4 Oct** — `GFP_VARIANT` knob in `gnn_core.py` (v0 | tuned: the tuned sheet is normalised
      like V0 and swapped into the GFP block at load time, `edge_attr` 81 → 84, only the edge
      projections and the readout input widen, `invariant_params` unchanged);
      `GFP_fixed_architecture.ipynb` ready with the 8 configs (5 and 4 of every operator);
      decision rule in section 3 "GFP tuned runs"
- [x] **4 Oct** — GFP `tuned` sheet (64 columns) computed in a rewritten `GFP_experiments.ipynb`;
      the four single-factor sheets of August (win48 / win120 / lc10 / rich) removed as
      uninformative; the "52 % / 84 % of hop gaps" claim did not reproduce and was replaced by the
      recomputed per-typology durations (`Progress_Report.md` §3); V0 relabelled "paper, cycles ≤ 6";
      `Data_checks.ipynb` §6 now checks the tuned sheet (59 checks, all pass)
- [x] **5 Oct** — RWPE stage 1: all eight runs (configs 5 and 4 × four operators, `rwpe16`) in one
      Kaggle session of `RWPE_fixed_architecture.ipynb`; results, curves and predictions in
      `Outputs/RWPE/`, sections and comparison tables in `Progress_Report.md` §7.5
- [x] **4 Oct** — RWPE node encoding: `rwpe_compute.py` + `RWPE_encoding.ipynb` (k = 8 and
      k = 16 per snapshot, self-loops dropped, checked against PyG `AddRandomWalkPE`,
      `Data/rwpe/`); `NODE_ENC` knob in `gnn_core.py` (only `node_proj` widens);
      `RWPE_fixed_architecture.ipynb` ready with the 8 stage-1 configs, batch not yet run
- [x] **28 Sep** — Graph Transformer batch (TR-1 … TR-7) trained on Kaggle under the full
      evaluation protocol; results, curves and predictions in `Outputs/TRANSFORMER/`, table in
      `Progress_Report.md` §7.4
- [x] **27 Sep** — PNA batch (PNA-1 … PNA-7) trained on Kaggle under the full evaluation
      protocol; results, curves and predictions in `Outputs/PNA/`, table in `Progress_Report.md` §7.3
- [x] **27 Sep** — Graph Transformer operator in `gnn_core.py` (`TransformerConv`, 4 heads × 32,
      edge features via `edge_dim`); `TRANSFORMER_fixed_architecture.ipynb` ready with the seven
      configs, batch not yet run
- [x] **27 Sep** — PNA operator in `gnn_core.py` (`PNAConv`, four aggregators × three scalers,
      training-graph in-degree histogram); `PNA_fixed_architecture.ipynb` ready with the seven
      configs, batch not yet run
- [x] **27 Sep** — optional gradient accumulation removed from `gnn_core.py` again: never used
      (the GAT batch fitted at 8,192), training loop back to the one every run used
- [x] **26 Sep** — GAT batch (GAT-1 … GAT-7) trained on Kaggle under the full evaluation
      protocol; results, curves and predictions in `Outputs/GAT/`, table in `Progress_Report.md` §7.2
- [x] **26 Sep** — GATv2 operator in `gnn_core.py` (+ optional gradient accumulation, default off);
      `GAT_fixed_architecture.ipynb` ready with the seven configs, batch not yet run
- [x] **26 Sep** — GIN batch (GIN-1 … GIN-7) trained on Kaggle under the full evaluation
      protocol; results, curves and predictions in `Outputs/GIN/`, table in `Progress_Report.md` §7.1
- [x] **26 Sep** — shared code moved to `gnn_core.py`; GIN notebook on the full evaluation
      protocol (all splits, top-5 % metrics, saved threshold, best-epoch line, saved predictions)
- [x] **23 Sep** — first GIN-family batch (5 runs) at width 128 — superseded by the 26 Sep retrain
- [x] **Sep** — data pipeline complete: EDA, 20 + 61 + 6 features, causal GFP computation, four
      GFP variants, 61 verification checks

---

## Appendix — technical details for the code (not on the Notion page)

**Knobs** (the only things that change between runs): `OPERATOR` (gin | pna | gat | transformer) ·
`MP_EDGE_FEATS` (none | base | full) · `READOUT_EDGE_FEATS` (base | full | gfp) ·
`NODE_ENC` (none | rwpe8 | rwpe16 | node2vec) · `GFP_VARIANT` (v0 | tuned) ·
`MP_DIRECTION` (in | bidirectional) · `TEMPORAL_SAMPLING` (on | off).

**Config order per family** (`CONFIGS` in every operator notebook, in this order):

| # | `mp` | `readout` |
|---|---|---|
| 1 | none | base |
| 2 | base | base |
| 3 | none | full |
| 4 | full | full |
| 5 | base | full |
| 6 | base | gfp |
| 7 | none | gfp |

RWPE runs use the same `mp` / `readout` dicts plus `operator` and `node_enc = 'rwpe16'`
(configs 5 and 4); the run name adds `_enc-rwpe16`; the summary is `batch_summary_rwpe.csv`.
GFP tuned runs use the same dicts plus `operator` and `gfp_variant = 'tuned'` (configs 5 and 4);
the run name adds `_gfp-tuned`; the summary is `batch_summary_gfp_tuned.csv`; `results.json` and
every batch summary carry a `gfp_variant` field (`v0` for all other runs).

**Fixed sizes for the planned extensions** (chosen a priori, never tuned): RWPE k = 16 first
(steps 1–16; covers the GFP cycle limit of 6 and the paper's 10 with margin), k = 8 (the course
project's value, = steps 1–8 of k = 16) only by the rule in section 1; P is built without
self-loops, so step 1 is 0 for every account; Node2Vec dim = the RWPE k kept (matched so the
comparison is about the kind of encoding, trained on the train graph only, zero vector for
unseen accounts); Transformer = PyG `TransformerConv`, 4 heads × 32 = 128 (mirrors GATv2 so only
the attention mechanism differs; local attention over the sampled `[100, 100]` neighbourhood).
GFP variants replace the GFP block of `edge_attr` (columns 20 onwards; 20 + 64 = 84 wide for
tuned) at load time in `gnn_core.load_graphs` (`GFP_VARIANT`), after the same train-fit
normalisation as V0 (vertex statistics: log1p → clip at train p1 / p99 → StandardScaler, fit on
the train rows; pattern bins untouched); the graph files never change, and only the edge
projections and the readout's first Linear widen (337 → 340 inputs). On Kaggle the tuned sheet
comes from the separate dataset `hi-small-gfp-tuned` (`tuned.npy` + `tuned_cols.json`).
RWPE on the val/test snapshots sees later edges than a seed edge — same caveat as neighbour
sampling; reported as a limitation. RWPE files: `Data/rwpe/rwpe_k{8,16}_{train,val,test}.pt`,
float32 [515,070, k], rows in `account_to_idx` order, computed by `rwpe_compute.py` (scipy,
exact, same maths as PyG `AddRandomWalkPE`: P = D_out⁻¹ A, multi-edges counted, never-senders
stay 0); on Kaggle they come from the separate dataset `hi-small-rwpe`.

**Protocol, architecture, artifacts and standing rules:** see `CLAUDE.md` §3–§6. Nothing is
"done" until results are pushed and this file, `Progress_Report.md` and both Notion pages agree.
