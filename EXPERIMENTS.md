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

- [ ] 🔄 GIN batch (GIN-1 … GIN-5) retrained under the full evaluation protocol — Kaggle, started 26 Sep
- [ ] GIN-6 and GIN-7 (second small batch)
- [ ] PNA notebook + batch (7 runs)
- [ ] GATv2 notebook + batch (7 runs)

---

## 1 · Data — what goes into the model

Dataset: IBM AML **HI-Small** (synthetic bank transactions, 5.08 M transactions, 0.09 %
laundering). Each transaction is an edge, each account a node.

- [x] **Baseline edge features** (20 per transaction) — amount, timing, bank, payment format
- [x] **GFP structural edge features** (61 per transaction) — pre-computed graph patterns around
      the transaction (cycles, fan-in/out, scatter-gather, degree statistics), IBM's Graph Feature
      Preprocessor with the paper's default windows
- [x] **Node features** (6 per account) — account entity type
- [ ] **GFP with data-tuned windows** — four alternative settings (longer windows, longer cycles,
      extra statistics) are computed; retrain the best model with each to see if they beat the
      paper's defaults
- [ ] **RWPE node encoding** (8 per account) — random-walk return probabilities that tell the
      network where an account sits in the graph; retrain the best model per operator with it
- [ ] **Node2Vec node encoding** (8 per account) — learned alternative to RWPE, if time allows

---

## 2 · Message passing — how the model processes it

All operators share one code file (`gnn_core.py`) and one fixed template — 2 layers, width 128,
dropout 0.3, identical training recipe — so only the aggregation rule differs.

- [x] **GIN / GINE** — sums neighbour messages · `GIN_fixed_architecture.ipynb` · 7 runs
- [ ] **PNA** — several aggregators at once (mean, max, min, std) · `PNA_fixed_architecture.ipynb` · 7 runs
- [ ] **GATv2** — attention decides which neighbours matter · `GAT_fixed_architecture.ipynb` · 7 runs
- [ ] **Graph Transformer** — attention with edge features over the sampled neighbourhood ·
      `TRANSFORMER_fixed_architecture.ipynb` · 7 runs

**Two extra questions about message passing itself**

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

- [ ] 🔄 **GIN-1** · none / base — topology alone, the reference point → `Outputs/GIN/gin_mp-none_readout-base_dir-in/`
- [ ] 🔄 **GIN-2** · base / base — edge features inside message passing → `Outputs/GIN/gin_mp-base_readout-base_dir-in/`
- [ ] 🔄 **GIN-3** · none / base+GFP — GFP only at the decision layer → `Outputs/GIN/gin_mp-none_readout-full_dir-in/`
- [ ] 🔄 **GIN-4** · base+GFP / base+GFP — GFP everywhere → `Outputs/GIN/gin_mp-full_readout-full_dir-in/`
- [ ] 🔄 **GIN-5** · base / base+GFP — GFP at the decision layer only → `Outputs/GIN/gin_mp-base_readout-full_dir-in/`
- [ ] **GIN-6** · base / GFP only — are raw features redundant once message passing has used them?
- [ ] **GIN-7** · none / GFP only — GFP alone vs baseline alone (compare with GIN-1)

### PNA family

- [ ] **PNA-1** · none / base
- [ ] **PNA-2** · base / base
- [ ] **PNA-3** · none / base+GFP
- [ ] **PNA-4** · base+GFP / base+GFP
- [ ] **PNA-5** · base / base+GFP
- [ ] **PNA-6** · base / GFP only
- [ ] **PNA-7** · none / GFP only

### GATv2 family

- [ ] **GAT-1** · none / base
- [ ] **GAT-2** · base / base
- [ ] **GAT-3** · none / base+GFP
- [ ] **GAT-4** · base+GFP / base+GFP
- [ ] **GAT-5** · base / base+GFP
- [ ] **GAT-6** · base / GFP only
- [ ] **GAT-7** · none / GFP only

### Transformer family

- [ ] **TR-1** · none / base
- [ ] **TR-2** · base / base
- [ ] **TR-3** · none / base+GFP
- [ ] **TR-4** · base / base+GFP
- [ ] **TR-5** · base+GFP / base+GFP
- [ ] **TR-6** · base / GFP only
- [ ] **TR-7** · none / GFP only

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

- [ ] **26 Sep** — GIN batch (GIN-1 … GIN-5) retraining on Kaggle under the full evaluation
      protocol; results, curves and predictions expected today
- [x] **26 Sep** — shared code moved to `gnn_core.py`; GIN notebook on the full evaluation
      protocol (all splits, top-5 % metrics, saved threshold, best-epoch line, saved predictions)
- [x] **23 Sep** — first GIN-family batch (5 runs) at width 128 — superseded by the 26 Sep retrain
- [x] **Sep** — data pipeline complete: EDA, 20 + 61 + 6 features, causal GFP computation, four
      GFP variants, 61 verification checks

---

## Appendix — technical details for the code (not on the Notion page)

**Knobs** (the only things that change between runs): `OPERATOR` (gin | pna | gat | transformer) ·
`MP_EDGE_FEATS` (none | base | full) · `READOUT_EDGE_FEATS` (base | full | gfp) ·
`NODE_ENC` (none | rwpe | node2vec) · `GFP_VARIANT` (v0 | win48 | win120 | lc10 | rich) ·
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

**Fixed sizes for the planned extensions** (chosen a priori, never tuned): RWPE k = 8 (covers
the GFP cycle limit of 6 with margin; matches the course project); Node2Vec dim = 8 (matched to
RWPE so the comparison is about the kind of encoding, trained on the train graph only, zero
vector for unseen accounts); Transformer = PyG `TransformerConv`, 4 heads × 32 = 128 (mirrors
GATv2 so only the attention mechanism differs; local attention over the sampled `[100, 100]`
neighbourhood). GFP variants are swapped into `edge_attr[:, 20:81]` after the same train-fit
normalisation. RWPE on the val/test snapshots sees later edges than a seed edge — same caveat
as neighbour sampling; reported as a limitation.

**Protocol, architecture, artifacts and standing rules:** see `CLAUDE.md` §3–§6. Nothing is
"done" until results are pushed and this file, `Progress_Report.md` and both Notion pages agree.
