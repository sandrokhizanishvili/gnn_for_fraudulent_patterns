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
      search (x0 = 100) on the train rows with early stopping on the weighted validation loss, trials
      ranked by validation PR-AUC, then 4 seeds; run locally on CPU, 18 Oct (`GBT_fixed_features.ipynb`,
      section 3 "GBT cross-check runs"); results in `Progress_Report.md` §7.7 (test F1 level with the
      best GIN run, above GATv2, well below PNA and the Transformer; a first, wider search without early
      stopping memorised the train rows and was dropped on 20 Oct)
- [ ] *SUPERSEDED (22 Oct)* **GIN recipe tuning** (decided 17 Oct, after the 13 Oct meeting) — the training recipe is tuned once
      per operator on config 2 and then frozen; GIN first, PNA / GATv2 / Transformer later with the same code
      and notebook template. Rules and stage tracker in section 2b. **Tuning done (two Kaggle sessions, 17–19
      Oct): `Outputs/TUNING/GIN/recipe.json` frozen = v1 + BatchNorm after each MP layer, best epoch 34.**
      Next: `GIN_tuning_final.ipynb` retrains on train + val with 4 seeds (section 3b); results →
      `Progress_Report.md` §7.8–7.9
- [ ] *SUPERSEDED (22 Oct)* **GATv2 recipe tuning** (started 19 Oct on the second Kaggle account, in parallel with the GIN retrain) — the same
      pre-registered protocol on GAT config 2 (`GAT_tuning.ipynb` → `Outputs/TUNING/GAT/recipe.json`, tracker in
      section 2b). **Tuning done (two sessions, 19–21 Oct): no knob beat the v1 recipe, frozen = v1 unchanged with
      best epoch 44 on the 100-epoch cosine** (results in `Progress_Report.md` §7.11). Next: `GAT_tuning_final.ipynb` on
      configs 2 / 4 / 5 / 6 × 4 seeds (section 3b) → §7.12; heads stay 4 × 32;
      the outer residual is the only residual in the GATv2 layer (PyG's `residual` option stays off, as in v1); one
      extra stage, self-loops {on, off} (`add_self_loops`), sits between FFN and pos_weight; one seed per trial
      (budget), noted as the one protocol difference from GIN
- [ ] **Luo et al. (2024) re-tuning — every operator × configs 2 and 4** (decided 22 Oct; replaces the two tuning items
      above and their final retrains, whose files stay as they are) — random search over the paper's space plus our
      additions, 100 trials × 2 seeds per (operator, config), then a 5-seed retrain on train + val of each winner; runs on
      Cineca (Leonardo Booster) by Loris from `cineca/`. Rules and tracker in section 2b, retrain rows in section 3b.
      Package checked locally on the slice (22 Oct) and sent to Loris (23 Oct); next: check → probe → draw → early
      check-in on trials 0–9
- [x] **MLP cross-check** — a plain neural net on the same 93 inputs as the GBT (20 baseline + 61 GFP V0
      + 6 + 6 entity one-hots), no message passing: one random search (100 unique trials, one seed each; shapes flat /
      funnel / resnet; every fit stopped on validation PR-AUC) on the train rows, scored on validation, then 4 seeds;
      one Kaggle GPU session, 23–24 Oct (`MLP_fixed_features.ipynb`, `mlp_core.py`, section 3 "MLP cross-check runs");
      results in `Progress_Report.md` §7.13 (below the trees and the best GIN run, level with GATv2, far below PNA and
      the Transformer: the PNA / Transformer lead over LightGBM is message passing, not "a neural net")

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
- [x] **No message passing — MLP** on the same features (is the GNN gain over the trees message passing,
      or just a neural net?) · `MLP_fixed_features.ipynb` · 1 search + 4 seeds (section 3, "MLP cross-check runs")

**Two extra questions about message passing itself (optional)**

- [ ] **Direction** — does the network also need to see outgoing money? Best model re-run with
      bidirectional aggregation
- [ ] **Causal sampling** (optional) — sample only earlier neighbours; how much does strict
      honesty cost?

---

## 2b · Recipe tuning — Luo et al. (2024), per operator and per config (decided 22 Oct 2026)

**What:** the training recipe is re-tuned following Luo et al. (2024), "Classic GNNs are Strong Baselines", as closely
as possible, separately for every operator (GIN, PNA, GATv2, Transformer) **and** for config 2 (mp base / readout base)
and config 4 (mp full / readout full). That makes 8 independent tunings, each with its own frozen recipe, best epoch and
threshold, followed by a 5-seed retrain of each winner (section 3b).
**Why:** the one-pass tuning below kept hidden 128 and 2 layers fixed and tuned on config 2 only. The paper shows that
classic GNNs need their own width, depth, normalisation, residual and dropout to be compared fairly.
**Consequence:** hidden size, depth and readout now differ between configs 2 and 4. **Each config is compared at its
own tuned recipe, so a difference between configs 2 and 4 mixes the feature effect with the recipe.** The
fixed-architecture comparison of the v1 grid (section 3) is unchanged.
**Where:** on Cineca (Leonardo Booster, A100 64 GB). Loris runs it from `cineca/` (`cineca/README_LORIS.md`); Sandro has
no Cineca access. Results come back as archives and are committed as usual.

**Protocol, fixed before any trial runs:**

| Item | Rule |
|---|---|
| Search | Random search: **100 trials per (operator, config)**, so 800 trials. The supervisor asked for at least 100. The full grid (64,800 non-depth cells × 10 depths) is far too large; GNN+ (Luo et al. 2025) is non-exhaustive too ("we do not perform an exhaustive search") |
| The draw | Trial 0 = the v1 knobs. Trials 1–99 are drawn **once**, after the probe and before any training, with a fixed seed (20261022), by `gnn_core.draw_luo_trials`. It is a partial Fisher–Yates shuffle of the 64,800 cells, the same cells for every operator and both configs, plus a depth drawn from each operator's depths that fit. The draw is prefix-stable and uses Python's version-stable `random.random()`. Output: `Outputs/TUNING_LUO/trials_space.json`, the pre-registration |
| Seeds | **2 per trial (42, 43)**, run as separate GPU jobs → 200 runs per pair, 1,600 in all |
| Training | Adam at a **constant LR**, **100 epochs, no early stopping** (tunedGNN `medium_graph/main.py`). Train and validation are scored every epoch with every section-4 metric. The test split is never loaded (asserted and logged by every task) |
| Best epoch | argmax of the **seed-mean validation PR-AUC** curve (raw, first epoch on ties), not the mean of the per-seed best epochs |
| Metrics of a trial | Every metric on train and validation at that epoch, per seed and as mean ± std. They are read from `history.csv`, i.e. the same predictions the epoch was chosen on |
| Threshold | Mean of the two seeds' validation-F1-maximising thresholds at the best epoch (per-seed values kept) |
| Winner | Highest mean val PR-AUC, tie-break mean val F1. **F1 stays the headline metric in every table.** A winner whose best epoch is in 91–100 is flagged |
| Failed trials | A seed with non-finite predictions stops there; the trial is scored over the epochs both seeds reached and flagged `diverged_at`. A seed out of GPU memory makes the trial `oom` (no metrics, cannot win) |
| Depths | 1–10 minus those that do not fit in GPU memory; there is no time rule. The probe runs at the heaviest setting (hidden 512, BatchNorm, linear residual, readout 3 × 2, config 4). A depth fits when nothing runs out of memory and the peak reserved memory is ≤ 85 % of the GPU; the fallback neighbour list is tried when the primary one does not fit, and the probe stops at the first depth that fits with neither list |
| Neighbours per hop | Hops 1–2 100, hops 3–4 50, hops 5+ 25. Fallback: hops 3–4 25, hops 5+ 10 |
| Loader workers | The probe times 0 / 4 / 6 sampling worker processes and the fastest is used. Workers are non-persistent, so resume stays exact (checked locally) |
| Final | The winner is retrained on train + val for its best epoch, seeds 42–46, at the frozen threshold. Test is scored once per seed; mean ± std. The separate threshold-selection run is not needed: the winning trial already is two train-only runs of the recipe, validated at the frozen epoch |
| Checks | Asserts inside `cineca/luo.py` (`step_aggregate`), run every time (also by `run.sh pack`). Every trial: 100 epochs × 2 seeds, constant LR, recipe = `trials_space.json`, test never loaded, best epoch recomputed from the two histories, `trial.json` values = `history.csv`, parameter count = the recipe's model, no NaN. Winner: max mean val PR-AUC. Final runs: seeds 42–46, stop epoch and threshold = `recipe.json`, test never scored during training (no test column in `history.csv`). Trial 0 vs the v1 run of the same config is printed as a warning only |

**Search space:**

| Knob | Values | Source |
|---|---|---|
| Learning rate | {0.0005, 0.001, 0.003, 0.005, 0.01} | Luo §4 {0.001, 0.005, 0.01} + 0.0005 / 0.003 from its large-graph rows (Tables 8–10) |
| Hidden dim | {64, 128, 256, 512}: one value for the node embedding, the MP edge projections, every MP layer and the readout base; GATv2 / Transformer 4 heads × hidden/4 | Luo §4 {64, 256, 512}; **128 is our addition** (the size of every earlier run) |
| Normalisation after each MP layer | {none, BatchNorm, LayerNorm} | Luo §4 |
| Residual | {linear, none}: linear `h ← Dropout(ReLU(Norm(conv(h) + W_r h)))`, none `h ← Dropout(ReLU(Norm(conv(h))))` | Luo Eq. 7, tunedGNN `medium_graph/model.py` |
| Dropout (after the activation, also in the readout) | {0.2, 0.3, 0.5, 0.7} | Luo §4 |
| Weight decay (Adam) | {0, 5e-5, 5e-4, 1e-3, 1e-2} | tunedGNN run scripts; the paper's text names none |
| MP layers | {1, …, 10} minus the memory-dropped depths | Luo §4 |
| pos_weight | {4, 8, 16} | **our addition** (supervisor) |
| Readout layers | {1, 2, 3}: 1 = a single `Linear(2h + e → 1)` (the paper's head), 2 = the v1 readout, 3 = one more hidden layer; each hidden layer is Linear → ReLU → Dropout | **our addition** (the paper does not tune the head) |
| Readout width | {0.5, 1, 2} × hidden per readout hidden layer; drawn but unused when readout layers = 1 (recorded as unused) | **our addition** |
| Fixed | batch 8,192 seed edges, FFN off, GATv2 self-loops on, PNA aggregators / scalers / towers | — |

**Deviations from the paper:**

| Ours | Paper / tunedGNN | Reason |
|---|---|---|
| Train + val retrain at the frozen epoch, 5 seeds | test at the best-validation epoch of the same run, 5 runs | Supervisor's protocol |
| 100-epoch cap | per-dataset epochs 100–2,500, mostly full-batch | Our epochs are mini-batch (~372 steps each); winners with a best epoch in 91–100 are flagged |
| Random search, 100 trials | not stated (a subset of the grid) | 648,000 configurations per pair |
| 2 seeds per tuning trial | tuning seeds not stated | Noise control at an affordable cost |
| Best epoch and winner chosen on val PR-AUC (tie-break val F1) | the dataset's own metric | With only 2 seeds per trial the best-threshold F1 is noisy and optimistic; PR-AUC needs no threshold (the same choice as our GBT search) |
| pos_weight, hidden 128, readout layers / width tuned | not in the space | Our additions (above) |
| Edge classification on a temporal multigraph, neighbour sampling [100, 100, …] with 8,192 seed edges | node classification; ogbn-proteins code uses fanout 32 and 10 batches per epoch | Our task and loader |
| A depth is kept only if it fits with a 15 % memory margin | not specified | A mid-trial OOM would waste GPU time |
| Fixed operator internals (4 heads, GATv2 self-loops, PNA aggregators / scalers / towers) | — | Kept as in v1 |
| Trial 0 = v1 knobs (identity residual, wd 1e-5, outside the grid) | — | Shows whether tuning beats v1 |

Matches the paper: Adam, constant LR (tunedGNN), full epochs with best-validation-epoch selection, linear residual, FFN
off, no jumping knowledge, 5 final seeds. The epoch counts of ogbn-proteins in the paper are 100 (GCN) and 1,000
(GraphSAGE, GAT).

**Tracker** (pointer = the folder; no numbers here):

- [x] Package `cineca/` + the `gnn_core.py` extensions; regression check (`recipe=None` unchanged) and local smoke test
      on the slice (22 Oct)
- [x] **23 Oct** — Hand-off to Loris: `handoff_loris.zip` on Google Drive (`README_LORIS.md` + `handoff_code.tar.gz` + `handoff_data.tar.gz`)
- [ ] Environment check on a GPU node (`run.sh check`)
- [ ] Probe ×4 → `Outputs/TUNING_LUO/<FAMILY>/probe.json` (depths that fit, loader workers, job memory)
- [ ] Draw → `Outputs/TUNING_LUO/trials_space.json`
- [ ] Early check-in: trials 0–9 of every pair → `results_small.tar.gz` to Sandro before the rest
- [ ] Tuning GIN config 2 / config 4 → `Outputs/TUNING_LUO/GIN/<run>/` (`trials/`, `trials_summary.csv`, `tuning_curves.png`, `recipe.json`)
- [ ] Tuning PNA config 2 / config 4 → `Outputs/TUNING_LUO/PNA/<run>/`
- [ ] Tuning GATv2 config 2 / config 4 → `Outputs/TUNING_LUO/GAT/<run>/`
- [ ] Tuning Transformer config 2 / config 4 → `Outputs/TUNING_LUO/TRANSFORMER/<run>/`
- [ ] Aggregate (checks pass, 8 winners frozen)

### 2b (old) · One-pass recipe tuning on config 2 — *SUPERSEDED (22 Oct 2026)*

*Kept for the record, not continued. This tuning kept hidden 128 and 2 layers fixed, ran on config 2 only and reused
that recipe for every config. It is replaced by the Luo tuning above. Its files stay where they are
(`Outputs/TUNING/GIN`, `Outputs/TUNING/GAT`, `Outputs/FINAL/GIN`).*

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
| Freeze | `Outputs/TUNING/GIN/recipe.json` — knobs, neighbour lists, T_max, best epoch, one threshold for all four retrain seeds (the mean of the two tuning seeds' validation thresholds at the best epoch; the per-seed tuning values are kept in `tuning_thresholds`; decided 19 Oct), validation F1 mean ± std, source stage. The single source of the frozen numbers; no copy in code |
| Final retrain | `GIN_tuning_final.ipynb`: train + val graph (built in memory, appendix), T_max = 100 unchanged, stop at the frozen best epoch, **one validation-chosen threshold per config** (decided 20 Oct: every config, the tuned one included, gets one selection run — train graph only, frozen recipe, frozen number of epochs, seed 42 — and takes the validation-F1 threshold at the last epoch → `Outputs/TUNING/<FAMILY>/config_thresholds.json`; the test F1 at the frozen tuning threshold of `recipe.json` is kept as a secondary `transfer_*` view, because a cut chosen for one config's score scale penalises configs whose scores sit higher or lower, which the GIN retrain showed for config 4), seeds 42–45, test scored once per seed; reported: test metrics as mean ± std over the four seeds (F1, PR-AUC), threshold sensitivity (frozen −0.10 … +0.10 and 0.5), train / val metrics with val marked as seen in training. Secondary view, stored next to it and never the headline (decided 19 Oct): per seed the threshold re-swept on the validation rows (trained on in this phase) and the thresholded metrics at it (`val_seen_threshold`, `seen_<split>_<f1|precision|recall>`, one extra row in `sensitivity.csv`), so the drift of a seen-data sweep is visible; PR-AUC, ROC-AUC and the top-5 % metrics are threshold-free, unaffected, and read first |
| Known limitation | the train + val graph has 496 batches per epoch vs 372 in tuning, so stopping at the frozen epoch means ~33 % more update steps with the same per-epoch LR profile; reported, not corrected |
| Not refit | GFP scalers and the bank target encoding stay fit on the train window (a consistent refit would change the val rows' features between phases) |
| Afterwards | the GIN feature configs with edge features in message passing — 2 (base / base), 4 (base+GFP / base+GFP), 5 (base / base+GFP), 6 (base / GFP only) — are retrained in `GIN_tuning_final.ipynb` exactly like config 2: train + val graph, the frozen epoch transferred unchanged from the config-2 tuning, the threshold chosen per config on validation (selection run, see the Final retrain row), 4 seeds 42–45 (mean ± std); configs 1, 3 and 7 (`mp = none`) are not rerun; PNA, GATv2 and the Transformer are tuned the same way, each in its own session block |

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
- [x] Stage 5 · LR {5e-4, 1e-3, 3e-3} → `Outputs/TUNING/GIN/s5_lr/` — incumbent 1e-3 kept (the 5e-4 trial cut
      in session 1 was redone from scratch in session 2)
- [x] Stage 6 · dropout {0.2, 0.3, 0.4} → `Outputs/TUNING/GIN/s6_dropout/` — incumbent 0.3 kept
- [x] Edge-rule trials: stage 2 → `s2_layers/bn_l3/`, stage 4 → `s4_pos_weight/bn_pw4/`; not triggered in
      stages 5–6 (the incumbent sat in the middle of both grids)
- [x] Freeze (19 Oct) → `Outputs/TUNING/GIN/recipe.json` (= v1 + BatchNorm, best epoch 34, thresholds per
      retrain seed), `trials_summary.csv`, `tuning_curves.png`

**GATv2 stage tracker** (`GAT_tuning.ipynb`, second Kaggle account; pointer = `Outputs/TUNING/GAT/<stage>/`, same rules and grids plus one operator-specific stage, heads fixed at 4 × 32; **one tuning seed (42) per trial** instead of two — a budget decision taken on 19 Oct, so the seed std is 0 and the winner margin is the fixed 0.01, the best epoch comes from the single smoothed validation curve, and the frozen threshold is that seed's; the final retrain keeps four seeds):

- [x] Probe — depths 2 / 4 / 6 at BatchNorm + FFN → `Outputs/TUNING/GAT/probe.json` (Kaggle, 19–20 Oct: 2 and 4 layers pass, 6 layers dropped by the 3× rule with both lists; depth 3 probed for the edge rule → `probe_3L.json`)
- [x] Stage 1 · norm {none, BatchNorm, LayerNorm} → `Outputs/TUNING/GAT/s1_norm/` — incumbent none kept (BatchNorm +0.006, below the 0.01 margin)
- [x] Stage 2 · layers {2, 4, 6} → `Outputs/TUNING/GAT/s2_layers/` — incumbent 2 layers kept (6 skipped by the probe; edge trial 3 layers ran, no win)
- [x] Stage 3 · FFN {off, on} → `Outputs/TUNING/GAT/s3_ffn/` — incumbent FFN off kept
- [x] Stage 4 · self-loops {on, off} (GATv2's `add_self_loops`, v1 = on; the node's attention to itself is a weighted message inside the operator, not a residual) → `Outputs/TUNING/GAT/s4_self_loops/` — incumbent on kept
- [x] Stage 5 · pos_weight {8, 16} → `Outputs/TUNING/GAT/s5_pos_weight/` — incumbent 8 kept (edge trial 4 ran, no win)
- [x] Stage 6 · LR {5e-4, 1e-3, 3e-3} → `Outputs/TUNING/GAT/s6_lr/` — incumbent 1e-3 kept
- [x] Stage 7 · dropout {0.2, 0.3, 0.4} → `Outputs/TUNING/GAT/s7_dropout/` — incumbent 0.3 kept
- [x] Edge-rule trials: stage 2 → `s2_layers/l3/`, stage 5 → `s5_pos_weight/pw4/`; not triggered elsewhere
- [x] Freeze (21 Oct) → `Outputs/TUNING/GAT/recipe.json` (= the v1 recipe unchanged, best epoch 44 on the 100-epoch cosine, threshold 0.483 for every retrain seed — the transfer view; the headline thresholds come from the per-config selection runs), `trials_summary.csv`, `tuning_curves.png`

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
protocol (threshold swept on validation, test scored once, every metric on every split). No message
passing: the GFP block is the only graph information, so the contrast is pre-computed graph features
vs learned ones. Every fit stops on the **weighted validation loss** (the loss the curves show,
laundering rows weighted 8), patience 100 rounds, and keeps its lowest-loss round; the threshold is
swept on validation at that round. Hyperparameters are chosen once, by the paper's successive halving
(Appendix C): x0 = 100 random configs → 100 @ 10 % → 50 @ 20 % → 25 @ 40 % → 12 @ 80 % → 6 @ 100 % of
the train rows (the most recent rows), each scored on the full validation set and ranked by
**validation PR-AUC** (threshold-free); then frozen for the four seeds. Search space: leaves 7–63,
`min_data_in_leaf` 200–2,000, L2 1–100, learning rate 0.01–0.05, feature fraction 0.3–0.8, bagging
fraction 0.5–1, L1 0.01–10, `scale_pos_weight` 1–10, up to 2,000 rounds as a cap. Test is never used
until the four final models are scored. Runs land in `Outputs/GBT/<run>/`, the search log in
`Outputs/GBT/search/`, the summaries in `Outputs/GBT/batch_summary_gbt.csv` and `seed_summary_gbt.csv`.
(A first batch with a wider space — leaves up to 255, 1,000 rounds, no early stopping, the
best-validation-F1 round kept — memorised the train rows, train PR-AUC 1.00, and was removed on
20 Oct; log entries of 17–18 Oct.)*

- [x] **GBT search** · LightGBM, successive halving on the train rows with early stopping, ranked by validation PR-AUC → `Outputs/GBT/search/best_params.json` + `trials.csv`
- [x] **GBT-42** · all 93 features · seed 42 → `Outputs/GBT/lightgbm_readout-full_seed-42/` — the run compared with the single-seed GNN runs
- [x] **GBT-43** · all 93 features · seed 43 → `Outputs/GBT/lightgbm_readout-full_seed-43/`
- [x] **GBT-44** · all 93 features · seed 44 → `Outputs/GBT/lightgbm_readout-full_seed-44/`
- [x] **GBT-45** · all 93 features · seed 45 → `Outputs/GBT/lightgbm_readout-full_seed-45/`

**Decision rule, fixed in advance:** the four seeds give mean ± std; the GBT is set against the best
run of every family and against configs 3 / 7 (GFP at the readout, nothing in message passing). Only
test F1 and PR-AUC are fair scores for the GBT: validation is used three times (hyperparameters, kept
round, threshold), so its scores are optimistic; validation is shown for the overfitting check only. A
gap inside the 4-seed spread is a tie. The train / validation gap is reported from the curves; if it
does not close, that is stated, not hidden by picking a different round. Bank target encoding is
re-verified as train-label-only before the run (trees exploit a leaky encoding immediately). The
paper's GFP + LightGBM (62.86 ± 0.25) is a reference point, not a target: different raw features,
simple cycles ≤ 6 vs 10, truncated test window, validation-chosen threshold vs 0.5 (F1 at 0.5 is
stored alongside), x0 = 100 vs 1000 with a narrower search space, LightGBM 4.6 vs 3.1.1.

### MLP cross-check runs — a plain neural net on the GBT inputs (`MLP_fixed_features.ipynb`, Kaggle GPU)

*An MLP on the same 93 inputs as the GBT — the 81 edge features (20 baseline + 61 GFP V0) plus the 6 + 6
entity one-hots of the two endpoints — on the same positional 60/20/20 split and the same protocol. It is
the GNN readout's own input `[e_seed ‖ x_src ‖ x_dst]` without the learned node states, so against the GBT
block it separates "message passing" from "a neural net": an MLP level with LightGBM means the PNA /
Transformer gain is message passing; an MLP level with them means the gain was the net. The 81 edge
columns are standardised with the train-row mean / std (std floor 1e-3); the one-hots stay 0/1 — the
only input treatment the trees did not need. Every fit (search trial or final seed) runs
Adam at a constant learning rate for at most 60 epochs and stops after 10 epochs without a better
**validation PR-AUC**, keeping that epoch; the threshold is swept on validation there. The recipe is chosen
once by random search: 100 unique trials, trial 0 = the a-priori pick (funnel 93 → 512 → 256 → 128 → 1,
BatchNorm, dropout 0.2, learning rate 1e-3, weight decay 1e-4, pos_weight 8, batch 8,192), each trained once
(seed 42) on the train rows, scored on validation, ranked by validation PR-AUC (tie-break validation
F1); then frozen for the four seeds. Search space: shape {flat, funnel, resnet} ×
first width {128, 256, 512, 1024} × depth {2, 3, 4, 5} × normalisation {none, batch} × dropout {0.1, 0.2,
0.3, 0.5} × learning rate {3e-4, 1e-3, 3e-3} × weight decay {0, 1e-5, 1e-4, 1e-3} × pos_weight {4, 8, 16} ×
batch {2,048, 8,192}. Test is never used until the four final models are scored. Runs land in
`Outputs/MLP/<run>/`, the search log in `Outputs/MLP/search/`, the summaries in
`Outputs/MLP/batch_summary_mlp.csv` and `seed_summary_mlp.csv`.*

- [x] **MLP search** · random search, 100 trials, one seed each, on the train rows, ranked by validation PR-AUC → `Outputs/MLP/search/best_params.json` + `trials.csv` (winner: trial 28, flat 93 → 512 × 4 → 1, BatchNorm)
- [x] **MLP-42** · all 93 features · seed 42 → `Outputs/MLP/mlp_readout-full_seed-42/` — the run compared with the single-seed GNN runs
- [x] **MLP-43** · all 93 features · seed 43 → `Outputs/MLP/mlp_readout-full_seed-43/`
- [x] **MLP-44** · all 93 features · seed 44 → `Outputs/MLP/mlp_readout-full_seed-44/`
- [x] **MLP-45** · all 93 features · seed 45 → `Outputs/MLP/mlp_readout-full_seed-45/`

**Decision rule, fixed in advance:** the four seeds give mean ± std; the MLP is set against the GBT (four
seeds), the best run of every family and configs 3 / 7 (GFP at the readout, nothing in message passing).
Only test F1 and PR-AUC are fair scores for the MLP: validation is used three times (hyperparameters, kept
epoch, threshold), so its scores are optimistic; validation is shown for the overfitting check only. A gap
inside the 4-seed spread is a tie. The train / validation gap is reported from the curves; if it does not
close, that is stated, not hidden by picking a different epoch. The search may span Kaggle sessions:
`Outputs/MLP/search/trials.csv` is committed in between and the logged fits are skipped on the next run.

### Cross-checks (optional)

- [x] **Gradient-boosted trees** on the same features, no message passing — does message passing add
      anything beyond engineered features? → planned as the "GBT cross-check runs" block above
- [x] **A plain neural net** on the same features, no message passing — is the GNN gain over the trees the
      net or the message passing? → the "MLP cross-check runs" block above (done: the net loses to the trees)
- [ ] **Seeds** — 3–4 random seeds on the winning runs, mean ± std; differences inside the seed
      spread count as ties → covered by the tuned-recipe runs of section 3b (4 seeds)

---

## 3b · Runs v2 — tuned recipes

*Runs with a frozen recipe from section 2b. The final retrain uses the train + val graph, the frozen epoch
and threshold, and scores test once per seed; summaries give mean ± std over the seeds. Run names keep the
v1 convention; the seed is a sub-folder.*

### Luo-tuned final retrain (Cineca, `cineca/luo.py final`; 8 pairs × seeds 42–46, after `aggregate` froze each `recipe.json`)

- [ ] **GIN-2 Luo** · base / base → `Outputs/FINAL_LUO/GIN/gin_mp-base_readout-base_dir-in/seed-<s>/`
- [ ] **GIN-4 Luo** · base+GFP / base+GFP → `Outputs/FINAL_LUO/GIN/gin_mp-full_readout-full_dir-in/seed-<s>/`
- [ ] **PNA-2 Luo** · base / base → `Outputs/FINAL_LUO/PNA/pna_mp-base_readout-base_dir-in/seed-<s>/`
- [ ] **PNA-4 Luo** · base+GFP / base+GFP → `Outputs/FINAL_LUO/PNA/pna_mp-full_readout-full_dir-in/seed-<s>/`
- [ ] **GAT-2 Luo** · base / base → `Outputs/FINAL_LUO/GAT/gat_mp-base_readout-base_dir-in/seed-<s>/`
- [ ] **GAT-4 Luo** · base+GFP / base+GFP → `Outputs/FINAL_LUO/GAT/gat_mp-full_readout-full_dir-in/seed-<s>/`
- [ ] **TR-2 Luo** · base / base → `Outputs/FINAL_LUO/TRANSFORMER/transformer_mp-base_readout-base_dir-in/seed-<s>/`
- [ ] **TR-4 Luo** · base+GFP / base+GFP → `Outputs/FINAL_LUO/TRANSFORMER/transformer_mp-full_readout-full_dir-in/seed-<s>/`
- [ ] Summaries per pair: `batch_summary_final.csv`, `seed_summary_final.csv`; per seed `sensitivity.csv`, `predictions.csv`

**Decision rule, fixed in advance:** nothing is chosen on test. For each pair, report test F1 and PR-AUC as mean ± std
over the 5 seeds, with the threshold-sensitivity table and validation marked as seen. Configs 2 and 4 of an operator are
each at their own tuned recipe, so their difference mixes features and recipe; say so wherever they are compared.
Across operators, read test PR-AUC (threshold-free) first, then F1 at the frozen threshold.

### GIN final retrain (`GIN_tuning_final.ipynb`, configs 2 / 4 / 5 / 6 × seeds 42–45, resumable across Kaggle sessions) — *SUPERSEDED (22 Oct)*

*Kept as it is; GIN-6 seeds 43–45 and the summaries will not be run. Replaced by the Luo-tuned retrain above.*

- [ ] Threshold selection runs (train graph, frozen recipe, 34 epochs, seed 42) for configs 2 / 4 / 5 / 6 → `Outputs/TUNING/GIN/config_thresholds/<run>/seed-42/` and `config_thresholds.json` (the frozen tuning threshold 0.654 stays as the transfer view)

- [x] **GIN-2 v2** · base / base · frozen recipe, 4 seeds → `Outputs/FINAL/GIN/retrain/gin_mp-base_readout-base_dir-in/seed-<s>/`
- [x] **GIN-4 v2** · base+GFP / base+GFP · same recipe and epoch, own threshold, 4 seeds → `Outputs/FINAL/GIN/retrain/gin_mp-full_readout-full_dir-in/seed-<s>/`
- [x] **GIN-5 v2** · base / base+GFP · 4 seeds → `Outputs/FINAL/GIN/retrain/gin_mp-base_readout-full_dir-in/seed-<s>/`
- [ ] **GIN-6 v2** · base / GFP only · 4 seeds → `Outputs/FINAL/GIN/retrain/gin_mp-base_readout-gfp_dir-in/seed-<s>/` (seed 42 done; 43–45 next session)
- [ ] Summaries → `Outputs/FINAL/GIN/retrain/batch_summary_retrain.csv` (one row per run), `seed_summary_retrain.csv` (one row per config, mean ± std over the four seeds; also one file per config); per-run `sensitivity.csv`

**Decision rule, fixed in advance:** nothing is chosen here. Test F1 mean ± std and PR-AUC over the four seeds are reported next
to the v1 runs of the same configs (single seed, 20 epochs) with the protocol difference stated; the sensitivity table shows
how much the threshold matters. Configs 4, 5 and 6 reuse the epoch tuned on config 2 (stated caveat) but get their own
validation-chosen threshold from a selection run (config 2 as well, so all four follow one procedure); test F1 at the
frozen tuning threshold stays in the files as the `transfer_*` view. Across configs, test PR-AUC (threshold-free) is read first, then F1 at the config's own threshold, then recall at
the 5 % alert budget. Configs 1, 3 and 7 (no edge features in message passing) are not rerun.

### GATv2 final retrain (`GAT_tuning_final.ipynb`, configs 2 / 4 / 5 / 6 × seeds 42–45, after `Outputs/TUNING/GAT/recipe.json` exists) — *SUPERSEDED (22 Oct), never run*

- [ ] Threshold selection runs for GAT configs 2 / 4 / 5 / 6 → `Outputs/TUNING/GAT/config_thresholds/`
- [ ] **GAT-2 v2** · base / base · frozen GATv2 recipe, 4 seeds → `Outputs/FINAL/GAT/retrain/gat_mp-base_readout-base_dir-in/seed-<s>/`
- [ ] **GAT-4 v2** · base+GFP / base+GFP · 4 seeds → `Outputs/FINAL/GAT/retrain/gat_mp-full_readout-full_dir-in/seed-<s>/`
- [ ] **GAT-5 v2** · base / base+GFP · 4 seeds → `Outputs/FINAL/GAT/retrain/gat_mp-base_readout-full_dir-in/seed-<s>/`
- [ ] **GAT-6 v2** · base / GFP only · 4 seeds → `Outputs/FINAL/GAT/retrain/gat_mp-base_readout-gfp_dir-in/seed-<s>/`
- [ ] Summaries → `Outputs/FINAL/GAT/retrain/batch_summary_retrain.csv`, `seed_summary_retrain.csv`; per-run `sensitivity.csv`

Same decision rule as for GIN: nothing is chosen on test; configs 4, 5 and 6 reuse the epoch tuned on GAT config 2 and get their own validation-chosen threshold.

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

- [x] **24 Oct** — MLP cross-check finished in one Kaggle T4 session (notebook `sandrokh/notebook12baf08d19`, 9 h search +
      4 final seeds); output pulled with the Kaggle CLI into `Outputs/MLP/` (small files committed; `best.pt`,
      `predictions.csv` local); `Progress_Report.md` §7.13 written, MLP row added to the §7.7 comparison, README summary
      updated. Notion not synced (page not shared with the connector)
- [x] **23 Oct** — MLP cross-check planned and coded: `mlp_core.py` (train-row scaling of the edge columns, the
      three shapes flat / funnel / resnet in one `MLP` class, one training loop with early stopping on validation
      PR-AUC, the resumable random search, the final seeds; matrix and predictions file from `gbt_core`, metrics /
      curves / summaries from `gnn_core`) and `MLP_fixed_features.ipynb` (Kaggle GPU, committed without outputs
      until run); block in section 3, space and row conventions in the appendix, §5.6 in `Progress_Report.md`.
      Smoke test on a slice of the real matrix (150 k train / 60 k val / 60 k test rows, CPU): 3 search fits with
      resume, one run per shape, 2 final seeds, both summaries, finished seed skipped on rerun — passed; the
      scratch outputs were deleted. Next: Sandro runs the notebook on Kaggle (the search may need more than one
      session: commit `Outputs/MLP/search/trials.csv` in between) → results → `Progress_Report.md` §7.13
- [ ] **22 Oct** — Luo et al. (2024) re-tuning decided and packaged for Cineca (Loris runs it; Sandro has no access):
      every operator × configs 2 and 4, random search of 100 trials × 2 seeds over the paper's space plus our additions,
      constant LR, 100 epochs without early stopping, best epoch and winner on the seed-mean val PR-AUC, 5-seed retrain on
      train + val (rules section 2b). New recipe knobs in `gnn_core.py`: `residual` (identity = v1 | linear | none),
      `readout_layers`, `readout_width`, `lr_schedule` (cosine = v1 | constant), `num_workers`, `select_metric` (f1 =
      v1 | pr_auc). Also new: resume from `resume.pt`, the memory probe (which fixes a
      tensor leak after an out-of-memory error), `load_graphs(splits=...)`, the draw and two-seed finalisation.
      `recipe=None` regression on the slice: every v1 path is identical (histories, checkpoints, predictions, results)
      and old checkpoints load strictly. `cineca/` package and end-to-end smoke test on the slice from a scratch copy.
      The GIN / GATv2 one-pass tuning and the GIN retrain are SUPERSEDED (files kept). Estimate before the probe (T4
      timings, ±50 %): ≈ 15,000 GPU-h if the expected depths fit, ≈ 27,000 if every depth fits. Next: hand-off,
      probe, draw, early check-in

- [x] **20 Oct** — tail-window side check moved from a scratch script into the repo: appendix of
      `GBT_fixed_features.ipynb` (clearly marked as outside the protocol) + `gbt_core.tail_check`, result
      `Outputs/GBT/tail_check.csv`; same numbers as the 18–19 Oct scratch run (0.638 ± 0.005 on test + tail)
- [x] **20 Oct** — GBT cleaned up to one batch and one notebook: the unregularised first batch (17–18 Oct)
      memorised the train rows and was removed from the repo; the regularised batch is now *the* GBT
      (`Outputs/GBT/`, run names `lightgbm_readout-full_seed-<s>`, summaries `*_gbt.csv`, result files
      re-labelled, curves re-plotted from the same histories and predictions); `gbt_core.py` lost the
      `GBT_VARIANT` knob and the first search space (early stopping + PR-AUC ranking are the fixed recipe);
      `GBT_reg_fixed_features.ipynb` folded into `GBT_fixed_features.ipynb`; `Progress_Report.md` §7.7
      rewritten (old §7.7 / §7.8 merged, §7.9–7.10 renumbered to §7.8–7.9); Notion left as is (skipped on request)
- [ ] **20–21 Oct** — GATv2 tuning session 2 (second Kaggle account, 6.7 h, ran to the end): stages 1–4 skipped from the
      committed files, the pos_weight 4 edge trial redone, stages 6 (LR) and 7 (dropout) run, no further edge trial; every
      challenger landed 0.01–0.02 below the v1 incumbent (one seed per trial, margin 0.01), recipe frozen →
      `Outputs/TUNING/GAT/recipe.json` = v1 unchanged, best epoch 44, val F1 0.549 (v1 GAT-2 at 20 epochs: 0.498);
      GATv2 tuning total ≈ 18.7 GPU hours; the Kaggle GitHub link committed the executed `GAT_tuning.ipynb` (stripped
      again). Next: `GAT_tuning_final.ipynb` (selection runs + 16 retrain runs, ≈ 28 h, resumable over sessions)
- [ ] **20 Oct** — per-config threshold for the final retrain (decided after the GIN session-1 numbers: at config 2's
      0.654 config 4 lost 0.04 F1 to precision while its PR-AUC was the best; its F1 was still rising at +0.10): one
      selection run per config, config 2 included (train graph, frozen recipe, frozen epoch, seed 42, validation sweep at
      the last epoch) → `config_thresholds.json`; `run_final` scores at the config's threshold (headline) and keeps the
      frozen tuning threshold as the `transfer_*` view; finished runs are re-scored from their saved predictions
      (`rescore_final_run`), nothing is retrained; `GIN_tuning_final.ipynb` / `GAT_tuning_final.ipynb` gain section 4b;
      smoke-tested on the slice
- [ ] **19–20 Oct** — GATv2 tuning session 1 (second Kaggle account, 12 h, cut by the session cap): probe (6 layers dropped,
      4.9× the 2-layer epoch), stages 1–4 finished, all four kept the v1 value — none / 2 layers / no FFN / self-loops on
      (BatchNorm +0.006 and the rest below the 0.01 margin; one seed per trial) — stage 5 ran pos_weight 8 and 16 and was
      cut inside the pos_weight 4 edge trial, which is redone next session; files in `Outputs/TUNING/GAT/`; ≈ 68–77 s per
      2-layer epoch, 135 s for 3 layers, 181 s for 4
- [ ] **19–20 Oct** — GIN final retrain session 1 (Kaggle, 12 h, cut by the session cap): 13 of the 16 runs finished —
      configs 2, 4, 5 with all four seeds, config 6 seed 42; seeds 43–45 of config 6 next session; result files in
      `Outputs/FINAL/GIN/retrain/` (predictions and `final.pt` kept locally, not versioned); per epoch 72–78 s for
      configs 2 / 5 / 6 (496 batches), 127 s for config 4; the seen-val threshold sweep landed 0.59–0.82 against the
      frozen 0.654 and never beat it on test by more than the seed spread
- [ ] **18–19 Oct** — GIN tuning session 2 (Kaggle, 5.4 h): probe re-run with the same verdict, stages 1–4
      skipped from the committed `trial.json` files, stage 5 (2.6 h) and stage 6 (2.7 h) run, no edge trial
      triggered, recipe frozen → `Outputs/TUNING/GIN/recipe.json` = v1 + BatchNorm (2 layers, no FFN,
      pos_weight 8, LR 1e-3, dropout 0.3), best epoch 34, val F1 0.609 ± 0.019 over the two seeds; every
      challenger of stages 2–6 landed 0.01–0.05 below the incumbent, none passed the winner margin; decisions
      re-checked locally with `choose_winner`; GIN tuning total ≈ 17.4 GPU hours. `GIN_tuning.ipynb` is now
      committed without outputs (GitHub's 1 MB limit blocked the Kaggle import; CLAUDE.md §7). Next:
      `GIN_tuning_final.ipynb` (4 seeds, ~2 h); Notion synced
- [x] **17–18 Oct** — GIN tuning session 1 (one 12 h Kaggle session): probe + stages 1–4 finished, stage 5
      cut by the session limit (its unfinished trial was redone in session 2); results in
      `Outputs/TUNING/GIN/`, decisions re-checked against the rule; incumbent after stage 4 = BatchNorm,
      2 layers, no FFN, pos_weight 8; Notion synced
- [ ] **17 Oct** — recipe tuning planned (after the 13 Oct meeting with Aris and Loris): one pass per
      operator on config 2, GIN first; rules, stages and tracker in section 2b, final-retrain rows in
      section 3b, train + val graph and `recipe.json` schema in the appendix; `gnn_core.py` changes and
      `GIN_tuning.ipynb` / `GIN_tuning_final.ipynb` to follow; Notion experiments page synced
- [x] **18 Oct** — GBT regularised batch run: search (193 fits, 1.35 h, winner 25 leaves, learning rate 0.010,
      stopped at round 1,380) and the four seeds (27–54 min each) locally; results, curves and predictions in
      `Outputs/GBT_REG/`, tables and verdict in `Progress_Report.md` §7.8 (§5.5 / §7.7 / §8 adjusted); Notion synced
      — since 20 Oct this is the only GBT batch: `Outputs/GBT/`, §7.7
- [x] **18 Oct** — GBT regularised batch planned: `GBT_VARIANT` knob in `gbt_core.py` (v1 reproduced
      unchanged; `reg` = smaller search space, early stopping on the weighted validation loss with
      patience 100, trials ranked by validation PR-AUC); its own notebook `GBT_reg_fixed_features.ipynb`
      (merged into `GBT_fixed_features.ipynb` on 20 Oct); block "GBT regularised runs" in section 3; Notion synced
- [x] **17–18 Oct** — GBT cross-check run: successive-halving search (193 fits, 5.8 h, winner 856 rounds ×
      216 leaves) and the four seeds (22 min each) locally on 16 CPU threads; results, curves and
      predictions in `Outputs/GBT/`, tables and verdict in `Progress_Report.md` §5.5 / §7.7 / §8; Notion
      synced — removed on 20 Oct (memorised the train rows; one paragraph of §7.7 keeps the account)
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
are asserted). Search space (`gbt_core.SEARCH_SPACE`) — small trees and strong regularisation, not the
paper's Table 10 (which goes to 16,384 leaves): `num_leaves` 7–63 · `learning_rate` 0.01–0.05
(log-uniform) · `min_data_in_leaf` 200–2,000 · `feature_fraction` 0.3–0.8 · `bagging_fraction` 0.5–1
(`bagging_freq` 1) · `lambda_l1` 0.01–10, `lambda_l2` 1–100 (log-uniform) · `scale_pos_weight` 1–10 (the
paper's range) · `num_round` 2,000 as a cap; `objective = binary`, library defaults otherwise. Every fit
(search trial or final seed) uses early stopping with patience 100 (`gbt_core.EARLY_STOPPING`) on
LightGBM's `binary_logloss` over a validation Dataset whose laundering rows carry weight 8 — the same
minimum as `gnn_core.weighted_bce`, the loss of the curves — so `booster.best_iteration` is the kept
round (`best_epoch`); the threshold is swept on validation at that round. Halving: each (trial, round) is
appended to `Outputs/GBT/search/trials.csv` (with the kept `rounds`), so a stopped search resumes; trials
are ranked by `val_pr_auc` (`gbt_core.SELECT`); the best config of the last round (full train) goes to
`best_params.json`. Final fits: train and validation scored every 25 boosting rounds plus the kept round
(`history.csv`, `epoch` = round), test scored once at the kept round. Row columns as in every
`batch_summary.csv`: `operator = lightgbm`, `mp = none`, `readout = full`, `mp_direction = none`,
`node_enc = none`, `gfp_variant = v0`, `seed`, `params` = trees kept, `invariant_params = 0` (no
architecture-invariant block). Run name `lightgbm_readout-full_seed-<seed>`; `results.json` also holds
`hyperparams` (with `early_stopping`) and `paper_comparison` (F1 at threshold 0.5 on train / val / test,
the paper's decision rule). `best.txt` (the booster) and `predictions.csv` stay out of git. The first,
unregularised batch and the `GBT_VARIANT` knob that labelled the two batches were removed on 20 Oct (log).
*Side check, outside the protocol* (notebook appendix, `gbt_core.tail_check`, `build_tail_matrix`): the 1,108
tail rows (Timestamp ≥ 11 Sep, 655 laundering) get their 93 features rebuilt from `Data/HI-Small_Trans.csv`
— baseline recipe of `Data_preparation` (10 Sep FX rates carried forward, bank risk frozen at the train
rate), GFP streamed causally in WSL with the V0 parameters from the batch boundary before 9 Sep, train-fit
normalisation, entity one-hots; self-checked on the 10 Sep rows (baseline exact, GFP per column group
within 1 %) — and the four `best.txt` boosters are scored at their validation thresholds and at 0.5 on four
windows (our test = must reproduce `results.json`; our test + tail; the paper code's 9–18 Sep window; the
tail alone) → `Outputs/GBT/tail_check.csv`. Nothing is chosen on these numbers; they size the window effect.

**MLP cross-check** (`mlp_core.py`, `MLP_fixed_features.ipynb`; Kaggle GPU): the GBT matrix of
`gbt_core.load_gbt_data` (93 columns, same rows, same asserted counts), then `mlp_core.standardise`: columns
0–80 (the edge features) get `(value − train mean) / train std` (std floored at 1e-3, no clipping: the
check on the full train rows showed a ±10 clip would merge the highest bank-risk values and the pattern-bin
counts, the informative rows); columns 81–92 (the entity one-hots) stay 0/1; the whole matrix is moved to the GPU once (1.76 GB). Model
(`mlp_core.MLP`): flat = `[Linear → Norm → ReLU → Dropout]` per hidden layer, every layer `hidden` wide;
funnel = the same blocks with the width halving per layer (`hidden`, `hidden/2`, …, floor 16); resnet =
`Linear 93 → hidden`, then `depth` blocks `z + Linear(Dropout(ReLU(Norm(Linear(z)))))` (the tabular ResNet of
Gorishniy et al. 2021), both Linears `hidden → hidden`; every shape ends in `Linear → 1`; `norm` = batch
(`BatchNorm1d`) or none. Trial 0 = funnel, first width 512, depth 3, batch norm, dropout 0.2, learning rate
1e-3, weight decay 1e-4, pos_weight 8, batch 8,192 (≈ 214 k parameters). Draw (`draw_trials`): trials 1–99
uniform over `SEARCH_SPACE` with `numpy.random.default_rng(42)`, a draw equal to an earlier trial is redrawn,
all 100 settings unique → `Outputs/MLP/search/configs.json`. Training (`train_run`): `set_seed(seed)`, seeded
shuffle of the train rows every epoch, Adam (`lr`, `weight_decay`), `BCEWithLogitsLoss(pos_weight)`, no
schedule; after every epoch train and val are scored (`history.csv` columns as the GNN runs: `train_loss` /
`val_loss` = `gnn_core.weighted_bce` at weight 8 whatever the trial's `pos_weight`, `train_f1` at the val
threshold, PR-AUC, ROC-AUC, `threshold`, `seconds`, plus `train_loss_batches` = the mini-batch loss at the
trial's weight and `lr`); the state of the best `val_pr_auc` epoch is kept and reloaded at the end; a run
stops after 10 epochs without improvement (cap 60). Search log `trials.csv`, one row per (trial, seed):
`trial, seed, shape, widths, params, best_epoch, epochs_trained, val_pr_auc, val_f1, threshold,
sec_per_epoch, seconds, knobs`; winner = highest `val_pr_auc` (tie-break `val_f1`, then the lower id; `SEARCH_SEEDS = (42,)`, one
fit per trial) → `best_params.json`. Finals: one `train_run` per
seed on the train rows, test scored once at the kept epoch. Row columns as in every `batch_summary.csv`:
`operator = mlp`, `mp = none`, `readout = full`, `mp_direction = none`, `node_enc = none`, `gfp_variant = v0`,
`seed`, `params` = `count_params`, `invariant_params = 0`. Run name `mlp_readout-full_seed-<seed>`;
`results.json` also holds `hyperparams` (the knobs, the widths, epochs, patience, select) and
`paper_comparison` (F1 at 0.5). `best.pt` and `predictions.csv` stay out of git. The notebook keeps
`OUT_ROOT` inside the Kaggle repo clone, so committed search fits and finished seeds are skipped on the next
session.

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
- *`recipe.json`.* `operator`, `cfg`, `recipe` (all knobs incl. `num_neighbors`, `epochs` = T_max, `self_loops` = GATv2's `add_self_loops`, ignored by the other operators),
  `best_epoch`, `thresholds` (one value for every retrain seed = the mean of the tuning seeds' thresholds; used by the tuned config and kept as the `transfer_*` view for the others), `tuning_thresholds`, `threshold_source`, `val_f1_mean`,
  `val_f1_std`, `stopped_epochs`, `source_stage`. Loaded and re-validated by the final notebook, which stops
  with a clear message if the file is missing.
- *Final-retrain outputs.* `Outputs/FINAL/GIN/retrain/<run>/seed-<s>/`: `results.json` (plus `phase`,
  `stop_epoch`, `frozen_threshold`, `val_seen`, the recipe block), `history.csv` (train-mode mini-batch loss
  per epoch), `curves.png`, `sensitivity.csv`, `predictions.csv` (not versioned).

**Luo tuning** (section 2b; `gnn_core.py` last section, `cineca/`):
- *Knobs* (defaults = v1, so `recipe=None` is unchanged):
  - `residual` (identity | linear | none): linear registers `layers.i.residual_proj` (counted as invariant).
  - `readout_layers` (1 | 2 | 3) / `readout_width` (× hidden): the classifier stays `classifier.0 …`, and
    `classifier.0.` is always the width-dependent first Linear.
  - `lr_schedule` (cosine | constant = `LambdaLR(1.0)`).
  - `num_workers` (loader workers, non-persistent).
  - `select_metric` (f1 | pr_auc): picks `best.pt`, the seed-mean best epoch and the winner.
  - `recipe_tag` adds `reslin`, `nores`, `ro<k>`, `rw<w>`, `lrconst`; `num_workers` and `select_metric` are not in
    the tag.
- *Folders:* `Outputs/TUNING_LUO/trials_space.json`, `logs/`.
  - Per family: `probe.json`.
  - Per run (`<model>_mp-<mp>_readout-<ro>_dir-in`): `trials/t<id>/trial.json`, `curves.png` (seed means), and
    `seed-<s>/{history.csv, curves.png, done.json, best.pt}`.
  - Per run, also `trials_summary.csv`, `tuning_curves.png`, `recipe.json`.
  - Final: `Outputs/FINAL_LUO/<FAMILY>/<run>/seed-<s>/{results.json, history.csv, curves.png, sensitivity.csv,
    predictions.csv, final.pt}` (results.json written last), plus `batch_summary_final.csv` and `seed_summary_final.csv`.
- *`trials_space.json`:*
  - draw seed, N, the space and the protocol;
  - allowed / dropped depths per operator (the reasons are in each `probe.json`);
  - every trial's full recipe (with its neighbour list and loader workers) and tag (`readout_width_used`).
  - The job memory is computed from `probe.json` at submit time (`MEM=auto`).
  - Array index i of a tuning job = trial i // 2, seed 42 + i % 2.
- *Resume:*
  - `train_one_run` / `train_fixed_epochs` write `resume.pt` last in every epoch, atomically. It holds the model,
    optimizer, scheduler, history, best / patience state and every RNG state.
  - A resubmitted job continues at the next epoch: bit-exact on CPU with 1 thread (tested), up to GPU scatter (or
    multi-thread CPU) summation-order nondeterminism otherwise.
  - `done.json` (written last) marks a finished seed; `trial.json` a finished trial; `results.json` (written last) a
    finished final seed.
- *Checkpoints:* every epoch's checkpoint is kept until both seeds of the trial are done; then `finalise_trial` keeps
  the best epoch's as `seed-<s>/best.pt`, deletes the rest and `resume.pt`, and writes `trial.json` last (CPU only,
  from the two histories). Only the task of the seed that finished last runs it, so two processes never finalise the
  same trial at once; the aggregate step finalises a trial left over (e.g. cut by the time limit) once its last seed
  finished over 10 minutes ago.
- *Job memory:* 1.5 × the probe's peak CPU memory (main + workers) + 4 GB, rounded up to 8 GB (final + 8 GB);
  `MEM=auto` in `config.sh`.

**Protocol, architecture, artifacts and standing rules:** see `CLAUDE.md` §3–§6. Nothing is
"done" until results are pushed and this file, `Progress_Report.md` and both Notion pages agree.
