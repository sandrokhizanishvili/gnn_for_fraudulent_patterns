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
- [x] GBT cross-check — LightGBM on the same 93 inputs, no message passing: one successive-halving
      search (x0 = 100) on the train rows, then 4 seeds; run locally on CPU, 17–18 Oct
      (`GBT_fixed_features.ipynb`, section 3 "GBT cross-check runs"); results in `Progress_Report.md` §7.7
      (test F1 level with the best GIN run, above GATv2, well below PNA and the Transformer)
- [ ] GBT regularised batch (`GBT_VARIANT = reg`) — same 93 inputs; smaller trees, early stopping on the
      weighted validation loss (patience 100), trials ranked by validation PR-AUC, 4 seeds; runs locally
      (`GBT_reg_fixed_features.ipynb`, section 3 "GBT regularised runs"); the first batch stays as the
      unregularised reference
- [ ] **GIN recipe tuning** (decided 17 Oct, after the 13 Oct meeting) — the training recipe is tuned once
      per operator on config 2 and then frozen; GIN first, PNA / GATv2 / Transformer later with the same code
      and notebook template. Rules and stage tracker in section 2b; `GIN_tuning.ipynb` (Kaggle, 3–5 sessions)
      → `Outputs/TUNING/GIN/recipe.json`; then `GIN_tuning_final.ipynb` retrains on train + val with 4 seeds
      (section 3b); results → `Progress_Report.md` §7.8–7.9

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
dropout 0.3, identical training recipe — so only the aggregation rule differs. This is the **v1 recipe**
(fixed a priori, 20 epochs, seed 42); section 2b tunes it once per operator, GIN first.

- [x] **GIN / GINE** — sums neighbour messages · `GIN_fixed_architecture.ipynb` · 7 runs
- [x] **PNA** — several aggregators at once (mean, max, min, std) · `PNA_fixed_architecture.ipynb` · 7 runs
- [x] **GATv2** — attention decides which neighbours matter · `GAT_fixed_architecture.ipynb` · 7 runs
- [x] **Graph Transformer** — attention with edge features over the sampled neighbourhood ·
      `TRANSFORMER_fixed_architecture.ipynb` · 7 runs
- [x] **No message passing — LightGBM** on the same features (hand-crafted graph features vs
      message passing) · `GBT_fixed_features.ipynb` · 1 search + 4 seeds (section 3, "GBT cross-check runs")

**Two extra questions about message passing itself (optional)**

- [ ] **Direction** — does the network also need to see outgoing money? Best model re-run with
      bidirectional aggregation
- [ ] **Causal sampling** (optional) — sample only earlier neighbours; how much does strict
      honesty cost?

---

## 2b · Recipe tuning — one pass per operator, then frozen (GIN first)

Hyperparameters were tuned sequentially, one at a time, in a pre-registered order, selected on validation
only; Luo et al. (2024) and GNN+ (2025) also used non-exhaustive searches. The recipe is tuned on
**config 2 (mp base / readout base)** and never per feature config: every feature config of an operator
reuses its frozen recipe unchanged, so the feature comparison stays a fixed-architecture comparison.
Hyperparameter rows are called **trials**; **config** keeps meaning the feature configurations 1–7.
Why: in the v1 runs the best epoch sits at 17–20 of 20 in 16 of 44 runs, train F1 is below validation F1
in every run, and one seed cannot separate gains inside the 0.02 tie band.
Why this order (architecture first, then optimizer, regularisation last): dropout depends on the final
capacity (depth, FFN), and the best learning rate depends on the normalisation, the depth and the loss
scale (pos_weight), so the architecture is fixed first and the regularisation last. The starting
incumbent is the v1 recipe (dropout 0.3, no norm, 2 layers, no FFN, pos_weight 8, LR 1e-3).

**Rules, fixed before the first run** (`GIN_tuning.ipynb`; the code is generic, `gnn_core.py`):

| Item | Rule |
|---|---|
| Config / seeds | config 2; tuning seeds 42, 43; final retrain seeds 42, 43, 44, 45 |
| Stages, in order | 1 normalisation after each MP layer {none, BatchNorm, LayerNorm} · 2 layers {2, 4, 6} · 3 FFN after each MP layer {off, on} · 4 pos_weight {8, 16} · 5 learning rate {5e-4, 1e-3, 3e-3} · 6 dropout {0.2, 0.3, 0.4}. If a deeper model wins stage 2, stages 3–6 run at that depth |
| Edge rule | a numeric winner at the edge of its grid gets one extra trial one step beyond (dropout 0.1 / 0.5; layers 3 / 8; pos_weight 4 / 32; LR 1e-4 / 1e-2) before the next stage |
| Fixed, not tuned | hidden 128, batch 8,192 seed edges, weight decay 1e-5, residuals on. Optimiser and scheduler: Adam + CosineAnnealingLR(T_max = 100), no warm-up, identical for every trial and for the final retrain; only the starting LR varies (stage 5). The final retrain uses the identical scheduler and stops at the frozen best epoch |
| Dropout scope | one value for the MP layers, the readout MLP and the FFN; no separate readout dropout |
| Layer | `h ← h + Dropout(ReLU(Norm(conv(h))))`; Norm is the identity for `none`; GIN's inner-MLP BatchNorm stays (it is the operator) |
| FFN (stage 3) | `h ← Norm(h + Linear(2h→h)(Dropout(ReLU(Linear(h→2h)(h)))))`, Norm = the stage-1 choice (LayerNorm if stage 1 kept `none`); expansion ×2 and the inner dropout are assumptions (GNN+ does not fix them) |
| Neighbours per hop | hops 1–2 always [100, 100]; 4 layers [100, 100, 50, 50]; 6 layers [100, 100, 50, 50, 25, 25]; edge depths 3 = [100, 100, 50], 8 = [100, 100, 50, 50, 25, 25, 25, 25]. Fallback 1 (far hops only): 4 → [100, 100, 25, 25], 6 → [100, 100, 25, 25, 10, 10]; if that fails too, the depth is dropped and documented |
| Probe (first session) | per depth 2 / 4 / 6 at the heaviest setting (BatchNorm + FFN): 2–3 warm-up batches, ~20 timed training batches with peak GPU memory, plus one full prediction pass over the val graph (train + val context) with its peak memory and time. Pass = both fit and the epoch estimate is ≤ ~3× the 2-layer one; the budget is recomputed from these numbers → `probe.json` |
| Epochs | cosine T_max = 100, at most 100 epochs; early stop when the 3-epoch moving average of validation F1 has not improved for 10 epochs, never before epoch 20; a run that reaches 100 is flagged |
| Best epoch of a trial | argmax of the 3-epoch smoothed **seed-mean** validation F1 curve over the epochs both seeds reached; per-seed threshold = that seed's validation threshold at that epoch |
| Winner of a stage | a trial replaces the incumbent only if its seed-mean validation F1 is higher by more than max(0.01, the larger of the two seed stds); otherwise the incumbent stays; ties go to the simpler option (fewer layers, no norm, no FFN) |
| Train vs validation at the best epoch | each (trial, seed) keeps a temporary checkpoint per epoch; both seeds run back to back; after the second seed the checkpoints at the seed-mean best epoch are scored on train and validation (that seed's threshold) → `trial.json`, then the checkpoints are deleted. A trial counts as done only once this is written; a session that stops between the two seeds re-runs the first |
| Train metrics during training | train split scored every 5th epoch and at the last one; validation every epoch |
| Summary tables | `stage_summary.csv` / `trials_summary.csv`: one row per trial with every hyperparameter, best epoch, stopped epoch per seed, train and validation F1 / PR-AUC (mean ± std), the train − validation gap, winner flag. Caveat next to the gap: validation has 0.107 % laundering vs 0.075 % on train, so part of the gap is prevalence, not underfitting |
| Freeze | `Outputs/TUNING/GIN/recipe.json` — knobs, neighbour lists, T_max, best epoch, thresholds for all four retrain seeds (42 and 43 their own at the best epoch; 44 and 45 the mean of the two), validation F1 mean ± std, source stage. The single source of the frozen numbers; no copy in code |
| Final retrain | `GIN_tuning_final.ipynb`: train + val graph (built in memory, appendix), T_max = 100 unchanged, stop at the frozen best epoch, frozen per-seed threshold, seeds 42–45, test scored once per seed; reported: test metrics as mean ± std over the four seeds (F1, PR-AUC), threshold sensitivity (frozen −0.10 … +0.10 and 0.5), train / val metrics with val marked as seen in training |
| Known limitation | the train + val graph has 496 batches per epoch vs 372 in tuning, so stopping at the frozen epoch means ~33 % more update steps with the same per-epoch LR profile; reported, not corrected |
| Not refit | GFP scalers and the bank target encoding stay fit on the train window (a consistent refit would change the val rows' features between phases) |
| Afterwards | every GIN feature config reuses the frozen recipe (validation-chosen epoch and threshold per config, 3 seeds); PNA, GATv2 and the Transformer are tuned the same way, each in its own session block |

**GIN stage tracker** (pointer = `Outputs/TUNING/GIN/<stage>/`, no numbers here):

- [x] Probe — depths 2 / 4 / 6 at BatchNorm + FFN → `Outputs/TUNING/GIN/probe.json` (Kaggle, 17–18 Oct:
      2 and 4 layers pass with their primary lists, 6 layers dropped by the 3× rule with both lists; depth 3
      probed later for the edge rule → `probe_3L.json`)
- [x] Stage 1 · norm {none, BatchNorm, LayerNorm} → `Outputs/TUNING/GIN/s1_norm/` — winner BatchNorm
- [x] Stage 2 · layers {2, 4, 6} → `Outputs/TUNING/GIN/s2_layers/` — incumbent 2 layers kept (6 skipped by
      the probe; edge trial 3 layers ran, no win)
- [x] Stage 3 · FFN {off, on} → `Outputs/TUNING/GIN/s3_ffn/` — incumbent FFN off kept
- [x] Stage 4 · pos_weight {8, 16} → `Outputs/TUNING/GIN/s4_pos_weight/` — incumbent 8 kept (edge trial 4
      ran, no win)
- [ ] Stage 5 · LR {5e-4, 1e-3, 3e-3} → `Outputs/TUNING/GIN/s5_lr/` (session 1 was cut inside the 5e-4 trial;
      it is redone from scratch next session)
- [ ] Stage 6 · dropout {0.2, 0.3, 0.4} → `Outputs/TUNING/GIN/s6_dropout/`
- [ ] Edge-rule trials (only if triggered): stage 2 → `s2_layers/bn_l3/` (done), stage 4 →
      `s4_pos_weight/bn_pw4/` (done); stages 5–6 pending
- [ ] Freeze → `Outputs/TUNING/GIN/recipe.json`, `trials_summary.csv`, `tuning_curves.png`

---

## 3 · Runs — Data × Message passing (v1 recipe — fixed a priori, 20 epochs, seed 42)

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

### GBT cross-check runs — hand-crafted graph features vs message passing (`GBT_fixed_features.ipynb`, local CPU)

*LightGBM on the same 93 inputs the GNNs see — the 81 edge features (20 baseline + 61 GFP V0) plus
the 6 + 6 entity one-hots of the two endpoints — on the same positional 60/20/20 split and the same
protocol (threshold swept on validation, best-validation-F1 boosting round kept, test scored once,
every metric on every split). No message passing: the GFP block is the only graph information, so the
contrast is pre-computed graph features vs learned ones. Hyperparameters are chosen once, by the
paper's successive halving (Appendix C): x0 = 100 random configs → 100 @ 10 % → 50 @ 20 % → 25 @ 40 %
→ 12 @ 80 % → 6 @ 100 % of the train rows (the most recent rows), each scored on the full validation
set; then frozen for the four seeds. Test is never used until the four final models are scored.
Runs land in `Outputs/GBT/<run>/`, the search log in `Outputs/GBT/search/`, the summaries in
`Outputs/GBT/batch_summary_gbt.csv` and `seed_summary_gbt.csv`.*

- [x] **GBT search** · LightGBM, successive halving on the train rows, scored on validation → `Outputs/GBT/search/best_params.json` + `trials.csv`
- [x] **GBT-42** · all 93 features · seed 42 → `Outputs/GBT/lightgbm_readout-full_seed-42/` — the run compared with the single-seed GNN runs
- [x] **GBT-43** · all 93 features · seed 43 → `Outputs/GBT/lightgbm_readout-full_seed-43/`
- [x] **GBT-44** · all 93 features · seed 44 → `Outputs/GBT/lightgbm_readout-full_seed-44/`
- [x] **GBT-45** · all 93 features · seed 45 → `Outputs/GBT/lightgbm_readout-full_seed-45/`

**Decision rule, fixed in advance:** the four seeds give mean ± std; the GBT is set against the best
run of every family and against configs 3 / 7 (GFP at the readout, nothing in message passing). Only
test F1 is a fair score for the GBT: validation is used three times (hyperparameters, kept round,
threshold), so its F1 is optimistic; validation is shown for the overfitting check only. A gap inside
the 4-seed spread is a tie. Bank target encoding is re-verified as train-label-only before the run
(trees exploit a leaky encoding immediately). The paper's GFP + LightGBM (62.86 ± 0.25) is a reference
point, not a target: different raw features, simple cycles ≤ 6 vs 10, truncated test window,
validation-chosen threshold vs 0.5 (F1 at 0.5 is stored alongside), x0 = 100 vs 1000 with a narrower
search space, LightGBM 4.6 vs 3.1.1.

### GBT regularised runs — `GBT_VARIANT = reg` (`GBT_reg_fixed_features.ipynb`, local CPU)

*Why: the first GBT batch memorises the train rows (train PR-AUC 1.00 against 0.55 on validation);
its validation loss rises from round 200 on while validation F1 and PR-AUC keep creeping up, so any
ranking metric keeps the last round. Same 93 inputs, split and protocol; what changes is the capacity
and the stopping rule. Search space: leaves 7–63, `min_data_in_leaf` 200–2,000, L2 1–100, learning rate
0.01–0.05, feature fraction 0.3–0.8 (bagging fraction 0.5–1, L1 0.01–10, `scale_pos_weight` 1–10 as
before); up to 2,000 rounds with **early stopping on the weighted validation loss** (the loss the curves
show, laundering rows weighted 8), patience 100 rounds: every trial and every final seed keeps its
lowest-validation-loss round, and the threshold is swept on validation at that round. Search trials
are ranked by **validation PR-AUC** (threshold-free) instead of F1. Same halving budget (x0 = 100,
η = 2, r0 = 0.1), same four seeds. Runs land in `Outputs/GBT_REG/<run>/` (names end in `_gbt-reg`),
the search log in `Outputs/GBT_REG/search/`, the summaries in `batch_summary_gbt_reg.csv` and
`seed_summary_gbt_reg.csv`.*

- [ ] **GBT-reg search** · LightGBM, successive halving with early stopping, ranked by validation PR-AUC → `Outputs/GBT_REG/search/best_params.json` + `trials.csv`
- [ ] **GBT-reg-42** · all 93 features · seed 42 → `Outputs/GBT_REG/lightgbm_readout-full_seed-42_gbt-reg/` — the run compared with the single-seed GNN runs
- [ ] **GBT-reg-43** · all 93 features · seed 43 → `Outputs/GBT_REG/lightgbm_readout-full_seed-43_gbt-reg/`
- [ ] **GBT-reg-44** · all 93 features · seed 44 → `Outputs/GBT_REG/lightgbm_readout-full_seed-44_gbt-reg/`
- [ ] **GBT-reg-45** · all 93 features · seed 45 → `Outputs/GBT_REG/lightgbm_readout-full_seed-45_gbt-reg/`

**Decision rule, fixed in advance:** the regularised batch is the headline GBT in the comparison with
the GNNs (the model whose train curve is informative); the first batch stays reported as the
unregularised reference. Both are judged on test F1 and PR-AUC (the fair scores, four seeds); a gap
inside the seed spread is a tie. The train / validation gap of the new batch is reported from the
curves; if it does not close, that is stated, not hidden by picking a different round.

### Cross-checks (optional)

- [x] **Gradient-boosted trees** on the same features, no message passing — does message passing add
      anything beyond engineered features? → planned as the "GBT cross-check runs" block above
- [ ] **Seeds** — 3–4 random seeds on the winning runs, mean ± std; differences inside the seed
      spread count as ties → covered by the tuned-recipe runs of section 3b (3 seeds)

---

## 3b · Runs v2 — tuned recipes (GIN first)

*Runs with a frozen recipe from section 2b. The final retrain uses the train + val graph, the frozen epoch
and threshold, and scores test once per seed; summaries give mean ± std over the seeds. Run names keep the
v1 convention; the seed is a sub-folder.*

### GIN final retrain (`GIN_tuning_final.ipynb`, one Kaggle session)

- [ ] **GIN-2 v2 · seed 42** · base / base · frozen recipe → `Outputs/FINAL/GIN/retrain/gin_mp-base_readout-base_dir-in/seed-42/`
- [ ] **GIN-2 v2 · seed 43** → `Outputs/FINAL/GIN/retrain/gin_mp-base_readout-base_dir-in/seed-43/`
- [ ] **GIN-2 v2 · seed 44** (threshold = mean of seeds 42 and 43) → `Outputs/FINAL/GIN/retrain/gin_mp-base_readout-base_dir-in/seed-44/`
- [ ] **GIN-2 v2 · seed 45** (threshold = mean of seeds 42 and 43) → `Outputs/FINAL/GIN/retrain/gin_mp-base_readout-base_dir-in/seed-45/`
- [ ] Summaries → `Outputs/FINAL/GIN/retrain/batch_summary_retrain.csv`, `seed_summary_retrain.csv` (mean ± std over the four seeds); per-seed `sensitivity.csv`

**Decision rule, fixed in advance:** nothing is chosen here. Test F1 mean ± std and PR-AUC over the four seeds are reported next
to the v1 GIN-2 run (single seed, 20 epochs) with the protocol difference stated; the sensitivity table shows
how much the frozen threshold matters. The GIN feature configs under the frozen recipe follow in a later block.

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

- [ ] **17–18 Oct** — GIN tuning session 1 (one 12 h Kaggle session): probe + stages 1–4 finished, stage 5
      cut by the session limit (its unfinished trial is redone next session); results in
      `Outputs/TUNING/GIN/`, decisions re-checked against the rule; incumbent after stage 4 = BatchNorm,
      2 layers, no FFN, pos_weight 8; Notion synced
- [ ] **17 Oct** — recipe tuning planned (after the 13 Oct meeting with Aris and Loris): one pass per
      operator on config 2, GIN first; rules, stages and tracker in section 2b, final-retrain rows in
      section 3b, train + val graph and `recipe.json` schema in the appendix; `gnn_core.py` changes and
      `GIN_tuning.ipynb` / `GIN_tuning_final.ipynb` to follow; Notion experiments page synced
- [ ] **18 Oct** — GBT regularised batch planned: `GBT_VARIANT` knob in `gbt_core.py` (v1 reproduced
      unchanged; `reg` = smaller search space, early stopping on the weighted validation loss with
      patience 100, trials ranked by validation PR-AUC); `GBT_reg_fixed_features.ipynb`; block "GBT
      regularised runs" in section 3; Notion synced
- [x] **17–18 Oct** — GBT cross-check run: successive-halving search (193 fits, 5.8 h, winner 856 rounds ×
      216 leaves) and the four seeds (22 min each) locally on 16 CPU threads; results, curves and
      predictions in `Outputs/GBT/`, tables and verdict in `Progress_Report.md` §5.5 / §7.7 / §8; Notion
      synced
- [x] **17 Oct** — GBT cross-check planned: `gbt_core.py` + `GBT_fixed_features.ipynb` (LightGBM on
      the 93-column row, successive halving x0 = 100 on the train rows scored on validation, 4 seeds,
      thesis protocol, runs locally on CPU); block "GBT cross-check runs" in section 3, decision rule
      fixed in advance; Notion experiments page synced
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

**GBT cross-check** (`gbt_core.py`, `GBT_fixed_features.ipynb`; runs locally in the
`graph_feature_preprocessor` env on CPU, not on Kaggle): one row per edge = `edge_attr` (81 columns
exactly as in the graph files: baseline raw, GFP vertex statistics normalised on train) ‖ `x[src]` (6)
‖ `x[dst]` (6) = 93 columns, built from `test_graph.pt` alone (rows `[:3,046,342]` train,
`[3,046,342:4,061,789]` val, the rest test; the counts and the 2,297 / 1,082 / 1,143 laundering rows
are asserted). Search space — practical LightGBM ranges, not the paper's Table 10 (which goes to
16,384 leaves): `num_round` 100–1000 · `num_leaves` 15–255 · `learning_rate` 0.01–0.1 (log-uniform) ·
`min_data_in_leaf` 20–500 · `feature_fraction` 0.5–1 · `bagging_fraction` 0.5–1 (`bagging_freq` 1) ·
`lambda_l1`, `lambda_l2` 0.01–10 (log-uniform) · `scale_pos_weight` 1–10 (the paper's range);
`objective = binary`, library defaults otherwise. Halving: each (trial, round) is appended to
`Outputs/GBT/search/trials.csv`, so a stopped search resumes; the best config of the last round (full
train) goes to `best_params.json`. Final fits: train and validation scored every 25 boosting rounds and
at the last round (`history.csv`, `epoch` = round), the round with the best validation F1 is kept
(`best_epoch`), its threshold saved, test scored once at that round. Row columns as in every
`batch_summary.csv`: `operator = lightgbm`, `mp = none`, `readout = full`, `mp_direction = none`,
`node_enc = none`, `gfp_variant = v0`, `seed`, `params` = trees kept, `invariant_params = 0` (no
architecture-invariant block). Run name `lightgbm_readout-full_seed-<seed>`; `results.json` also holds
`hyperparams` and `paper_comparison` (F1 at threshold 0.5 on val / test, the paper's decision rule).
`best.txt` (the booster) and `predictions.csv` stay out of git.
**GBT variants** (`gbt_core.VARIANTS`, knob `GBT_VARIANT`): `v1` = the first batch exactly as run (no early
stopping, best-validation-F1 round, trials ranked by validation F1, `SEARCH_SPACE`); `reg` = `SEARCH_SPACE_REG`
(leaves 7–63, `min_data_in_leaf` 200–2,000, L2 1–100, learning rate 0.01–0.05, feature fraction 0.3–0.8,
`num_round` = 2,000 as a cap), early stopping with patience 100 on LightGBM's `binary_logloss` over a
validation Dataset whose laundering rows carry weight 8 — the same minimum as `gnn_core.weighted_bce`, the
loss of the curves — so `booster.best_iteration` is the kept round (`best_epoch`); the threshold is swept on
validation at that round; trials ranked by `val_pr_auc`. `trials.csv` gains a `rounds` column (kept
rounds); `results.json` rows carry `gbt_variant` (absent = v1) and the `hyperparams` block adds
`early_stopping`; `history.csv` holds the 25-round checkpoints plus the kept round. Run
name suffix `_gbt-reg`; outputs under `Outputs/GBT_REG/`.

**Recipe tuning** (section 2b; `gnn_core.py`, `GIN_tuning.ipynb`, `GIN_tuning_final.ipynb`):
- *Recipe dict.* `DEFAULT_RECIPE` = the v1 constants (hidden 128, 2 layers, dropout 0.3, norm none, FFN off,
  neighbours [100, 100], batch 8,192, epochs 20 = cosine T_max, LR 1e-3, weight decay 1e-5, pos_weight 8);
  `recipe=None` anywhere reproduces v1 exactly (identity norm registers no parameters, the FFN is not
  registered, so old `best.pt` files load unchanged). `epochs` is the schedule length; how many epochs run is
  a separate stop argument.
- *Trial folders.* `Outputs/TUNING/GIN/s<k>_<knob>/<trial>/` with `trial.json` and `seed-<s>/history.csv`,
  `curves.png`; trial tag = the knob value (`d0.2`, `bn`, `ln`, `l4`, `pw16`, `lr3e-3`, `ffn`). Tuning runs
  build no test loader and write no `predictions.csv`; per-epoch checkpoints are deleted after the post-trial
  scoring.
- *Train + val graph* (final retrain only, built in memory, no new file): the val snapshot's edges, features
  and times; labels = the train snapshot's labels on the first 3,046,342 rows and the val labels after;
  `eval_mask` all true. Asserted: the first 3,046,342 edges of the val snapshot equal the train snapshot,
  3,379 laundering edges, 4,061,789 seeds (496 batches per epoch). RWPE, when used, comes from the val
  snapshot file. The PNA degree histogram would come from this graph (the graph trained on).
- *`recipe.json`.* `operator`, `cfg`, `recipe` (all knobs incl. `num_neighbors`, `epochs` = T_max),
  `best_epoch`, `thresholds` {42, 43: own; 44, 45: mean of the two}, `threshold_source`, `val_f1_mean`,
  `val_f1_std`, `stopped_epochs`, `source_stage`. Loaded and re-validated by the final notebook, which stops
  with a clear message if the file is missing.
- *Final-retrain outputs.* `Outputs/FINAL/GIN/retrain/<run>/seed-<s>/`: `results.json` (plus `phase`,
  `stop_epoch`, `frozen_threshold`, `val_seen`, the recipe block), `history.csv` (train-mode mini-batch loss
  per epoch), `curves.png`, `sensitivity.csv`, `predictions.csv` (not versioned).

**Protocol, architecture, artifacts and standing rules:** see `CLAUDE.md` §3–§6. Nothing is
"done" until results are pushed and this file, `Progress_Report.md` and both Notion pages agree.
