# GNN for Fraudulent Pattern Detection (Master Thesis)

Detecting money-laundering transactions with Graph Neural Networks on the IBM AML **HI-Small**
synthetic dataset (Altman et al., NeurIPS 2023). Extends my Graph Mining course project
(`../AML_GNN_GMA/`, LI-Small) with a re-validated, leakage-checked pipeline.

Full write-up of the work so far: **[Progress_Report.md](Progress_Report.md)**.

## Notebooks (run in order)

| Notebook | Kernel | What it does |
|---|---|---|
| `EDA.ipynb` | any Python 3 with pandas/matplotlib/yfinance | Exploratory analysis of HI-Small; USD conversion of amounts; findings that drive the feature design |
| `Data_preparation.ipynb` | `graph_feature_preprocessor` | Truncation at Sep 10, 20 baseline edge features, 6 node features, 61 GFP structural features, 60/20/20 temporal split, train-fit normalization, PyG graph snapshots |
| `GFP_experiments.ipynb` | `graph_feature_preprocessor` | Four GFP parameter variants (win48, win120, lc10, rich) with data-driven rationale |
| `Data_checks.ipynb` | `graph_feature_preprocessor` | Shows what every artifact in `Data/` is and looks like, and verifies it (row counts, alignment, leakage properties, graph consistency) with a summary table |
| `GIN_fixed_architecture.ipynb` | Kaggle (GPU) | GIN edge classifier with a **fixed architecture**; knobs select which edge features enter message passing / the classifier, incoming-only vs bidirectional aggregation, and optional temporal sampling |

`run_gfp_wsl.py` — helper that runs IBM SnapML's Graph Feature Preprocessor **inside WSL**
(the Windows snapml build lacks it), streaming edges in batches of 128 so features stay causal.
Requires a WSL venv: `python3 -m venv ~/gfp_env && ~/gfp_env/bin/pip install 'numpy<2' snapml`.

## Graph

- **Nodes** = accounts (515,070 after truncation), **edges** = transactions (5,077,237 after truncation), directed temporal multigraph, self-loops kept and flagged.
- **Task** = edge classification (`Is Laundering`), 4,522 positives (0.089%).
- **Node features (6):** entity-type one-hot.
- **Edge features (81):** 20 baseline (USD log-amount, structuring band, time cyclicals, same-bank, time since previous txn of sender/receiver, causal leakage-guarded bank target encoding, payment-format OHE) + 61 GFP (scatter-gather, temporal & simple cycles, vertex statistics).
- **Snapshots:** `train_graph.pt` (train edges), `val_graph.pt` (train+val, val evaluated), `test_graph.pt` (all, test evaluated); context edges carry label −1.

## Outputs (`Data/`, not versioned — regenerate with the notebooks)

`edge_features.csv` · `node_features.csv` · `feature_meta.json` (feature groups, dims, ablation grid) ·
`standard_scaler.pkl` · `train/val/test_graph.pt` · `account_to_idx.pkl` · `gfp_variants/*.npy`

## Planned model comparison

GBT (LightGBM / XGBoost) and GNNs (GIN+EU, PNA) over the ablation grid
A (nodes + structure) → B (+ baseline edges) → C (+ GFP) → D (full), and across the five GFP
variants; minority-class F1 and PR-AUC as metrics.
