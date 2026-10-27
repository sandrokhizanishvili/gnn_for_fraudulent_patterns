"""GFP sheet normalisation. The variant (v0 or tuned) is chosen in conf/preprocessing/gfp/."""
import numpy as np
from sklearn.preprocessing import StandardScaler

def normalise_gfp_sheet(values, cols, n_train):
    '''The GFP normalisation of Data_preparation.ipynb section 6, fit on the first n_train rows only.

    Pattern bins (*_bins_*) untouched; fan / deg / sum / avg / var / kurtosis: log1p -> clip at train p1 / p99
    -> StandardScaler; ratio / skew: clip -> StandardScaler. values [E, len(cols)] float32 -> float32.
    '''
    assert values.shape == (values.shape[0], len(cols)) and values.dtype == np.float32
    assert not np.isnan(values).any(), 'GFP sheet contains NaN'
    pattern_cols = [c for c in cols if '_bins_' in c]  # small integer bin counts; scaling them would leave the V0 recipe
    vertex_cols  = [c for c in cols if c not in pattern_cols]
    log_cols = [c for c in vertex_cols if any(s in c for s in ['fan', 'deg', 'sum', 'avg', 'var', 'kurtosis'])]
    std_cols = [c for c in vertex_cols if c not in log_cols]
    print(f'GFP normalisation: untouched pattern bins {len(pattern_cols)} | log+scale {len(log_cols)} '
          f'| scale-only {len(std_cols)} | fit on the first {n_train:,} rows')

    out = values.copy()
    for group, use_log in [(log_cols, True), (std_cols, False)]:
        if not group:
            continue
        idx = [cols.index(c) for c in group]
        vals = values[:, idx]
        if use_log:
            vals = np.log1p(np.clip(vals, 0, None))  # those columns are heavy-tailed counts; log1p is the stored V0 recipe
        p01 = np.percentile(vals[:n_train], 1, axis=0)
        p99 = np.percentile(vals[:n_train], 99, axis=0)
        vals = np.clip(vals, p01, p99)
        scaler = StandardScaler().fit(vals[:n_train])
        out[:, idx] = scaler.transform(vals).astype(np.float32)
        del vals
    if vertex_cols:
        vertex = out[:, [cols.index(c) for c in vertex_cols]]
        print(f'vertex stats after normalisation: min={vertex.min():.2f} max={vertex.max():.2f} '
              f'mean={vertex.mean():.3f} std={vertex.std():.3f}')
    assert not np.isnan(out).any()
    return out


