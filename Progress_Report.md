# GNN for Fraudulent Pattern Detection — Progress Report

**Master Thesis, Sapienza University of Rome**
**Stage covered:** exploratory data analysis, data preparation, feature engineering, and GFP parameter experiments on the IBM AML **HI-Small** dataset.
**Builds on:** my Graph Mining course project ([`AML_GNN_GMA/`](../AML_GNN_GMA/)), which used the LI-Small variant. This thesis work does **not** treat the course pipeline as ground truth — several of its components were re-tested and revised (see §4.3, §5.2).

---

## 1. Dataset

The project uses the **HI-Small** variant of *IBM Transactions for Anti-Money Laundering* [[2]](#references), a synthetic dataset generated with the AMLworld simulator [[1]](#references):

| Property | Value |
|---|---|
| Transactions | 5,078,345 (raw) |
| Accounts | 515,080 in transactions (515,070 after the Sep-10 truncation); 518,581 in the accounts file |
| Time span | 2022-09-01 → 2022-09-18 (real activity only through 09-10, see §3) |
| Laundering rate | 0.102% (1 : 980) — ~2× the LI-Small rate, hence "Higher Illicit" |
| Currencies | 15 (native-unit amounts; USD conversion required, §2.2) |
| Payment formats | 7 (Cheque, Credit Card, ACH, Cash, Reinvestment, Wire, Bitcoin) |
| Laundering pattern attempts | 370, annotated in `HI-Small_Patterns.txt`, 8 typologies |

Files: `Data/HI-Small_Trans.csv`, `Data/HI-Small_accounts.csv`, `Data/HI-Small_Patterns.txt`.

## 2. Exploratory Data Analysis (`EDA.ipynb`)

### 2.1 Key findings

- **Extreme class imbalance:** 5,177 laundering among 5.08M transactions (0.102%, 1:980). Standard accuracy is meaningless; training needs class weighting and evaluation needs minority-class F1 / PR-AUC (the paper's protocol [[1]](#references)).
- **ACH dominates format risk:** 11.8% of volume but a 0.75% laundering rate (~7× average). Reinvestment and Wire contain zero laundering. Bitcoin is *below* average here (0.038%) — opposite of LI-Small, showing findings do not transfer between variants.
- **Temporal signal:** laundering follows business hours (midday peak 0.17% vs midnight 0.03%) and is strongly elevated on weekends (Sunday 0.31% ≈ 3× weekday rates).
- **Currency mismatch carries no signal:** 1.42% of transactions have payment ≠ receiving currency, with **zero** laundering among them.
- **Graph structure:** out-degree is scale-free (median 2, max 168,672); in-degree is tight (median 16, max 1,084). Top-1% in-degree hubs are *cleaner* than average (0.095% vs 0.116%) — laundering lives in mid-degree accounts, which motivates GNN neighbourhood aggregation over degree rules.
- **Entity types (6):** Individual (0.139%) and Corporation (0.127%) are above-average risk senders; Country (0.034%) and Direct (0%) below.
- **Pattern typologies:** 370 attempts spread nearly evenly over all 8 types (Cycle 54, Gather-Scatter 51, … Fan-In 40) — unlike LI-Small's 117 skewed attempts, giving supervision on every structural pattern.

### 2.2 Currency normalisation

Raw `Amount Paid` is denominated per-currency (the raw maximum ≈ 1.05 **trillion** is a Yen row). All amounts are converted with **day-specific close rates** from Yahoo Finance (via `yfinance`; weekends forward-filled; USD = 1.0). After conversion: median $863, p99 ≈ $3.0M; laundering amounts concentrate at $2K–$13K (median $5.7K).

## 3. Data Preparation (`Data_preparation.ipynb`)

### 3.1 Truncation at September 10

Per the dataset creator's explanation on Kaggle [[3]](#references), the *Small* datasets contain only **10 days of real data (Sep 1–10)**; later rows exist solely to let in-flight laundering patterns complete and contain **no legitimate transactions**. Measured on HI-Small: the post-Sep-10 tail is 1,108 rows at a **59% laundering rate** (≈600× the true rate), sitting exactly where a temporal test split ends. All rows with `Timestamp ≥ 2022-09-11` are therefore dropped: 5,077,237 transactions and 4,522 positives (0.089%) remain (655 positives lost).

### 3.2 Temporal split

Following the paper's protocol [[1]](#references): a positional **60/20/20 split** over the time-sorted stream.

| Split | Edges | Window | Laundering |
|---|---|---|---|
| Train | 3,046,342 | 09-01 00:00 → 09-06 13:34 | 0.086% |
| Val | 1,015,447 | → 09-08 16:09 | ~0.09% |
| Test | 1,015,448 | → 09-10 23:59 | ~0.09% |

The split boundaries are defined **before** feature engineering because one feature (bank risk, §4.2) must be fit on the training window only.

## 4. Feature Engineering

### 4.1 Baseline edge features (20) — each justified by measured signal

| Feature | Evidence on HI-Small (base rate 0.089%) |
|---|---|
| `Amount_Log` = log1p(USD amount) | heavy-tailed → log-normal after transform |
| `Struct_Band` (USD $9–10k) | 0.275% laundering — **3.1× lift**; classic structuring band under the $10k reporting threshold |
| `Hour_Sin/Cos`, `DayOfWeek_Sin/Cos`, `Is_Weekend` | business-hours peak; Sunday ≈ 3× |
| `Is_Self_Loop` | 11.6% of rows, near-zero risk (mostly Reinvestment) |
| `Same_Bank` | same-bank 0.012% vs cross-bank 0.101% — **8× separation** |
| `Dt_Src_Log` (time since sender's previous txn) | laundering senders fire in bursts: median gap **0.0h** vs 0.3h |
| `Dt_Dst_Log` (time since receiver's previous txn) | laundering receivers are dormant mules: median gap **8.2h** vs 0.4h |
| `Src_Bank_Risk`, `Dst_Bank_Risk` | 30,470 banks; among banks with ≥5k txns, rates range 0% → 0.68% (sender and receiver roles differ) |
| Payment-format OHE (7) | ACH 7× average; Reinvestment/Wire zero |

### 4.2 Bank risk target encoding (edge-level) — three-layer leakage protection

Banks are the one categorical where target encoding beats OHE (30,470 categories). The encoding is an **edge** feature so that it can be made causal per transaction; it is protected against target leakage by:

1. **Temporal isolation** — rates are computed on train rows only and *frozen*; val/test rows look them up (unseen banks → global train prior), mirroring a production system.
2. **Causal expanding window for train rows** — a train edge at time *t* receives the smoothed rate of its bank's transactions with timestamp **< t** only (cumulative count/sum excluding the current row), so neither its own label nor any future label can enter its feature — the same causality principle as GFP.
3. **Smoothing** — `(positives + m·prior)/(count + m)` with `m = 200` shrinks small banks toward the prior, addressing the noisy-rate instability that degraded the LI-Small course project's target-encoding experiment.

A node-level variant (a static per-account bank rate) was considered and rejected: a static attribute cannot exclude an account's own training labels from its bank's rate, so it would reintroduce within-train target leakage. In GNN configurations whose message passing uses no edge features, the bank signal therefore enters only at the edge readout `[h_src, edge, h_dst]`.

### 4.3 Dropped from the course-project pipeline (evidence-based)

| Dropped | Reason |
|---|---|
| `Is_ACH` | exact duplicate of the `PayFmt_ACH` one-hot column |
| `Currency_Mismatch` | zero positives in its 1.4% share |
| `Bank_ID_Norm` (node) | min-max-scaled arbitrary ID = meaningless ordinal; superseded by the edge-level bank risk encoding (§4.2) |
| Rejected candidates: round-amount flags (share ≈ 0.00%), paid≠received flags (never occurs) | measured, no support in data |

### 4.4 Node features (6)

Entity-type one-hot (Partnership, Corporation, Sole, Country, Individual, Direct) extracted from the entity name. Node features are deliberately minimal, matching the paper's GNN setup: an account's behaviour is learned from its transactions and neighbourhood rather than a static profile.

## 5. GFP Structural Features

### 5.1 Configuration

IBM SnapML's **Graph Feature Preprocessor** [[4]](#references), [[5]](#references) computes AML subgraph patterns per transaction, configured per the paper's Appendix D [[1]](#references): scatter-gather (6h window), temporal cycles (24h), length-constrained simple cycles (24h, length ≤ 6 as a cost compromise vs the paper's 10), and vertex statistics (fan/degree/ratio + avg/sum/var/skew/kurtosis of timestamps and USD amounts, 24h) — **61 features**. Fan/degree histograms are disabled in the reference config since vertex stats carry continuous versions of the same quantities.

**Windows note:** the Windows build of `snapml` lacks the GFP native code, so GFP runs inside WSL/Ubuntu through `run_gfp_wsl.py` — the same Linux build the paper's Kaggle-era experiments used.

### 5.2 Leakage discovery: single-batch `fit_transform` sees the future

A causality check (an account's first-ever transaction must have no history) exposed that calling `fit_transform` on the entire dataset at once produces **future leakage**: first transactions received out-degrees up to **176,127** — the account's entire future — plus a double-insertion artifact (`fit` and `transform` each insert the batch; a 3-edge toy example returned deg = 6). This affects the LI-Small course-project features, which were computed single-batch.

**Fix:** stream the time-sorted edges through `transform` in **batches of 128**, exactly the paper's protocol [[1]](#references) (Appendix D). Verified on toy data (first edge: fan = deg = 1) and on the full run: **99.02%** of first transactions are perfectly causal, worst case deg = 9 — bounded by a single batch (~22 seconds of traffic), the exposure the paper accepts.

### 5.3 Normalisation and graph construction

- GFP vertex stats: `log1p → clip [train p1, p99] → StandardScaler`, fit on train only (`ratio`/`skew` skip the log). Pattern-histogram bins and baseline features are left untouched (bounded/binary/log-scaled already).
- **Cumulative PyG snapshots** per the paper's dynamic-graph protocol [[1]](#references): `train_graph.pt` (train edges), `val_graph.pt` (train+val, val evaluated), `test_graph.pt` (all, test evaluated); context edges carry label −1 and are excluded from loss/metrics.
- Final dimensions: **EDGE_DIM 81** (20 baseline + 61 GFP), **NODE_DIM 6**.

## 6. GFP Parameter Experiments (`GFP_experiments.ipynb`)

### 6.1 Why the paper's defaults are questionable for HI-Small

The paper fixes one GFP configuration for all datasets. Measuring HI-Small's own 370 annotated attempts:

| Measurement | Result | Implication |
|---|---|---|
| Attempt duration | only **21% complete within 24h**; 90% within 120h | 24h windows see a fraction of most patterns |
| Consecutive-hop gap within attempts | 52% ≤ 24h; **84% ≤ 48h** | a 24h window often cannot connect adjacent hops |
| Cycle sizes | 2–12 accounts; **37% longer than 6**; only 5/54 exceed 10 | `lc-cycle_len = 6` misses a third of cycles |
| Fan degrees | median 7.5, p75 = 12, max 16 | bins [2,3,5] dump ~75% of fan patterns into one bin |

(This *refuted* an initial "shorter windows" hypothesis — HI-Small patterns play out over days.)

### 6.2 A second library pitfall: the global `time_window` silently caps pattern windows

The first variant run produced cycle features **bit-identical** to the 24h reference despite
`temp-cycle_tw = 48h` being accepted by `set_params`. Controlled experiments showed that snapml
**silently caps every per-pattern window by the global `time_window` parameter** (and retains
graph history only up to roughly twice it): raising `time_window` together with the pattern
windows makes cycle detections increase as expected (83→93 temp-cycle rows on a 1.2M-edge
slice). The long-window variants therefore set `time_window` to their largest pattern window.
A related default: enabling fan/degree histograms uses a **12h** window unless `fan_tw`/
`degree_tw` are set explicitly — the `rich` variant sets them to 24h.

### 6.3 Variants computed (one factor changed per variant)

| Variant | Change vs paper config V0 | Cols | Runtime |
|---|---|---|---|
| V0 `paper` | — (inside `edge_features.csv`) | 61 | — |
| V1 `win48` | scatter-gather 12h; cycles/vertex-stats 48h (+ global window 48h) | 61 | 445s |
| V2 `win120` | scatter-gather 24h; cycles/vertex-stats 120h (+ global window 120h) | 61 | 601s |
| V3 `lc10` | lc-cycle length 10 (the paper's true setting) | 61 | 283s |
| V4 `rich` | + fan/degree histograms (24h, data-driven bins [2,4,8,13]); + min/max/median vertex stats | 101 | 889s |

All computed with the same causal batch-128 streaming and saved as row-aligned float32 blocks in `Data/gfp_variants/` with column-name JSONs.

### 6.4 First-look signal (univariate Pearson correlation with the label)

The long-window variants **strengthen the cycle features substantially**, exactly as the
pattern-duration measurements of §6.1 predicted:

| Variant | Strongest feature | Corr | Features with abs(corr) > 0.01 |
|---|---|---|---|
| V0 / lc10 / rich | `temp-cycle_bins_2-3` | +0.064 | 11 / 11 / 20 |
| win48 | `lc-cycle_bins_3-5` | **+0.082** | 12 |
| win120 | `temp-cycle_bins_3-5` | **+0.097** | 13 |

With 120h windows the larger cycle bins (`3-5`, `5-inf`) — nearly silent at 24h — become the
top features, i.e. the multi-day, multi-hop cycles the annotations describe are only visible
to GFP once the window covers them. Univariate correlation cannot capture interactions, so the
final variant ranking is deferred to the GBT/GNN comparison; this summary confirms every
variant carries signal and that longer windows surface more of it.

### 6.5 Verification audit

All artifacts passed an integrity audit: row/label counts, NaN checks, feature-dimension
consistency, graph snapshot shapes/eval masks/label conservation (2,297 + 1,082 + 1,143 =
4,522), and a cross-run determinism check — `lc10`'s unchanged feature blocks are bit-identical
to V0 from an independent run, proving row alignment; its changed block (and the corrected
long-window blocks) differ as intended.

## 7. Model-Comparison Plan (next stage)

- **Feature ablations** (encoded as `ABLATIONS` in `feature_meta.json`), for every model: (A) node features + structure only, (B) + baseline edge features, (C) + GFP features, (D) full — isolating the uplift of each block. For the GNNs this is refined into a 2-factor grid: which edge features enter **message passing** (none / baseline / baseline+GFP) × which enter the **final classifier** `[h_src, edge, h_dst]` (baseline / baseline+GFP / GFP only). GBT rows are built as edge features + source-node features + destination-node features (entity types).
- **Fixed-architecture GNN protocol** (`gnn_core.py`, imported by every operator notebook): one model class — 2 GIN layers, hidden 128, node MLP `Linear → BatchNorm → ReLU → Linear`, learnable ε, residual, dropout 0.3, `[100,100]` neighbour sampling, `pos_weight` 8, Adam 1e-3 (weight decay 1e-5), 20 epochs — for every configuration; `GINConv` when no edge features enter message passing, `GINEConv` otherwise (identical node MLPs, only the edge term differs). Each run records an *architecture-invariant parameter count* that must match across feature sets, making comparability verifiable. Additional axes held constant within a comparison: message-passing direction (incoming-only vs incoming + outgoing with separate convs) and optional temporal neighbour sampling.
- **Models:** gradient-boosted trees (LightGBM/XGBoost per the paper's GBT baselines [[1]](#references)) and GNNs (GIN with edge features, GIN+EU, PNA [[6]](#references)); minority-class F1 as the headline metric, PR-AUC alongside.
- **GFP variant comparison:** V0–V4 swapped into the same models to rank the parameter choices of §6.2; winning factors may be combined.

### 7.1 Results — GIN family (hidden 128, incoming MP, seed 42; batch of 26 Sep 2026)

Seven feature configurations through the identical architecture (`invariant_params` = 67,587 in
every run; 20 epochs; threshold chosen on validation; test scored once with the best-val-F1
checkpoint). Artifacts in `Outputs/GIN/<run>/`, training logs in the executed
`GIN_fixed_architecture.ipynb`. Ranked by test F1. Precision, recall, PR-AUC, ROC-AUC and
Recall@5 % are **test** values at the validation-chosen threshold.

| Run | mp / readout | best ep | F1 train | F1 val | **F1 test** | Precision | Recall | PR-AUC | ROC-AUC | Recall@5 % |
|---|---|---|---|---|---|---|---|---|---|---|
| GIN-5 | base / base+GFP | 8 | 0.534 | 0.609 | **0.525** | 0.678 | 0.429 | 0.484 | 0.980 | 0.867 |
| GIN-4 | base+GFP / base+GFP | 10 | 0.563 | 0.598 | 0.512 | 0.743 | 0.390 | 0.479 | 0.974 | 0.827 |
| GIN-3 | none / base+GFP | 11 | 0.453 | 0.548 | 0.471 | 0.635 | 0.375 | 0.435 | 0.980 | 0.861 |
| GIN-2 | base / base | 15 | 0.548 | 0.554 | 0.456 | 0.659 | 0.348 | 0.392 | 0.961 | 0.771 |
| GIN-6 | base / GFP only | 19 | 0.526 | 0.533 | 0.440 | 0.507 | 0.388 | 0.395 | 0.961 | 0.770 |
| GIN-1 | none / base | 13 | 0.432 | 0.439 | 0.371 | 0.481 | 0.302 | 0.328 | 0.972 | 0.814 |
| GIN-7 | none / GFP only | 20 | 0.273 | 0.319 | 0.256 | 0.265 | 0.248 | 0.198 | 0.941 | 0.731 |

Precision@5 % is 0.016–0.020 for every run and is not shown: at 0.11 % prevalence its ceiling
is 0.023, so it cannot separate models. Recall@5 % (share of laundering inside the top-5 %
alerts) is the informative operating-point number. All 21 per-split metrics are in
`Outputs/GIN/batch_summary.csv`.

**What it means** (single seed, preliminary; the 23 Sep batch of the same five configs landed
within ±0.02 F1 of these numbers, so differences below ~0.02 are ties):

- **RQ1 — base edge features in message passing help:** +0.085 (GIN-1→2) and +0.054 (GIN-3→5).
- **RQ2 — GFP features help further, at the readout:** +0.100 on GIN (1→3), +0.069 on GINE
  (2→5); the best configuration is **+0.154 over the plain baseline** at identical capacity.
- **Where the GFP uplift lives — the decision layer.** GFP inside message passing too (GIN-4)
  does not beat GIN-5 (−0.013, a tie) and overfits: validation loss rises from epoch 10 while
  train PR-AUC keeps climbing (0.62 vs 0.49 on validation at epoch 20).
- **GFP complements, it does not replace.** GFP alone at the readout (GIN-7, 0.256) is far below
  the 20 baseline features alone (GIN-1, 0.371), and dropping the baseline features from the
  readout (GIN-6, 0.440) costs 0.085 against GIN-5. Both GFP-only runs were still improving at
  epochs 19–20, with high thresholds (0.65–0.69).
- **Overfitting:** mild for the winner (validation loss flat after epoch 8, train F1 0.534 vs
  test 0.525); strongest for GIN-4 as above. ROC-AUC is 0.94–0.98 everywhere and uninformative
  at this imbalance.

Curves of the winner: `Outputs/GIN/gin_mp-base_readout-full_dir-in/curves.png` (dashed line =
saved checkpoint, epoch 8).

## 8. Artifact Inventory

| File | Content |
|---|---|
| `EDA.ipynb` | executed EDA with all findings of §2 |
| `Data_preparation.ipynb` | executed pipeline of §3–§5 |
| `GFP_experiments.ipynb` | executed variant generation of §6 |
| `Data_checks.ipynb` | executed inspection + verification of every artifact (§6.5) |
| `gnn_core.py` | shared code for every operator: fixed model template, `build_model(operator, …)`, loaders, training loop with validation threshold sweep, metrics on all splits, curves, saving, `invariant_params()` |
| `GIN_fixed_architecture.ipynb` | executed Kaggle notebook of the GIN family: config cell (7 runs) + loop over `gnn_core` (§7.1) |
| `GAT_fixed_architecture.ipynb` | same notebook for the GATv2 operator; only the config cell differs |
| `run_gfp_wsl.py` | causal batched GFP bridge (Windows → WSL) |
| `Data/edge_features.csv` | 5,077,237 × (meta + 81 features), pre-normalisation |
| `Data/node_features.csv`, `Data/account_to_idx.pkl` | node features (6) and account indexing |
| `Data/feature_meta.json` | feature groups, dims, ablation grid, provenance notes |
| `Data/standard_scaler.pkl` | train-fit normalisation parameters |
| `Data/train_graph.pt`, `val_graph.pt`, `test_graph.pt` | cumulative PyG snapshots |
| `Data/gfp_variants/{win48,win120,lc10,rich}.npy` + `_cols.json` | GFP variant feature blocks |
| `Outputs/GIN/<run>/` | per-run `results.json` (all splits, all metrics), `history.csv`, `curves.png`, `best.pt`, `predictions.csv` (every scored edge with `edge_id`, `prob`, `pred`; not versioned) + `batch_summary.csv` per batch (§7.1) |

---

## References

1. **E. Altman, J. Blanuša, L. von Niederhäusern, B. Egressy, A. Anghel, K. Atasu.** *Realistic Synthetic Financial Transactions for Anti-Money Laundering Models.* NeurIPS 2023 Datasets and Benchmarks. arXiv:2306.16424. (Local copy: `AML_GNN_GMA/Papers/2306.16424v3.pdf`.)
2. **IBM Transactions for Anti-Money Laundering (AML)** — Kaggle dataset by E. Altman. https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml
3. **E. Altman**, Kaggle discussion #427517 — on the 10-day real-data span of the Small datasets and the laundering-only tail days. https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml/discussion/427517
4. **Snap ML — Graph Feature Preprocessor documentation.** IBM. https://snapml.readthedocs.io/en/latest/graph_preprocessor.html
5. **J. Blanuša et al.** Graph feature preprocessing / subgraph-pattern extraction underlying GFP (cycle enumeration and graph pattern mining papers cited as [12, 13, 48, 49] in [1]).
6. **IBM Multi-GNN reference implementation** (GIN+EU, PNA baselines for this dataset). https://github.com/IBM/Multi-GNN
7. **yfinance** — Yahoo Finance market data downloader used for day-specific FX rates. https://github.com/ranaroussi/yfinance
