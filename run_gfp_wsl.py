"""Run IBM SnapML's GraphFeaturePreprocessor inside WSL, streaming in batches.

The Windows build of snapml lacks the Graph Feature Preprocessor, so Data_preparation.ipynb and
GFP_experiments.ipynb shell out to this script through WSL (venv ~/gfp_env, created with
python3 -m venv ~/gfp_env && ~/gfp_env/bin/pip install 'numpy<2' snapml).

Why batches: fit_transform on the whole dataset at once lets every edge see all other edges, future ones
included (an account's first transaction gets its full future degree). Streaming `transform` in small batches
is causal: an edge only sees earlier batches and its own. Altman et al. use batch size 128; so do we.

Usage (from WSL):
    ~/gfp_env/bin/python run_gfp_wsl.py <input.npy> <params.json> <output.npy> [batch_size]

Input : float64 array [txn_id, src_idx, dst_idx, ts_sec, amount_usd], time-sorted
Output: float32 array, the GFP output (input columns passed through + features)
"""
import json
import sys
import time

import numpy as np
from snapml import GraphFeaturePreprocessor

inp_path, params_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
batch_size = int(sys.argv[4]) if len(sys.argv) > 4 else 128

with open(params_path) as f:
    params = json.load(f)

X = np.load(inp_path)
print(f'[wsl] input {X.shape}, streaming in batches of {batch_size} '
      f'({params["num_threads"]} threads) ...', flush=True)

gfp = GraphFeaturePreprocessor()
gfp.set_params(params)

t0 = time.time()
out = None  # allocated once the first batch shows the output width
for start in range(0, len(X), batch_size):
    batch_out = gfp.transform(X[start:start + batch_size])
    if out is None:
        out = np.empty((len(X), batch_out.shape[1]), dtype=np.float32)
    out[start:start + len(batch_out)] = batch_out
    if start % 1_000_000 < batch_size:
        print(f'[wsl]   {start:,}/{len(X):,} edges  ({time.time()-t0:.0f}s)', flush=True)

print(f'[wsl] done in {time.time()-t0:.1f}s, output {out.shape}', flush=True)

np.save(out_path, out)
print('[wsl] saved', out_path, flush=True)
