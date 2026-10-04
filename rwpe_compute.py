'''Random-walk positional encoding (RWPE) of one graph snapshot, exact, on CPU.

RWPE(v)[k - 1] = probability that a k-step random walk starting at v is back at v, k = 1..K,
with P = D_out^-1 A built from the snapshot's own edges WITHOUT self-loops (src == dst dropped;
decision in CLAUDE.md 3b). Same maths as PyG's AddRandomWalkPE (checked in RWPE_encoding.ipynb),
written with scipy so every step's time, memory and fill-in can be logged.

usage: python rwpe_compute.py <split> <K> <data_dir> <out_dir> [--keep-self-loops]
  split    train | val | test  (cumulative snapshots: first SPLIT_EDGES[split] rows of edge_features.csv)
  writes   <out_dir>/rwpe_k<K>_<split>.pt   (float32 [num_nodes, K], rows ordered by account_to_idx)
           <out_dir>/log_<split>_k<K>.txt   (one line per step: nnz of P^k, seconds, RAM)
  --keep-self-loops  diagnostic only: keeps src == dst edges, writes rwpe_k<K>_<split>_withloops.pt
'''
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd
import psutil
import scipy.sparse as sp
import torch

# edges of each cumulative snapshot = the first n rows of Data/edge_features.csv (time-sorted)
SPLIT_EDGES = {'train': 3_046_342, 'val': 4_061_789, 'test': 5_077_237}
NUM_NODES   = 515_070


def peak_ram_gb():
    '''Peak resident memory of this process in GB (Windows: peak_wset; Linux: ru_maxrss).'''
    info = psutil.Process().memory_info()
    if hasattr(info, 'peak_wset'):
        return info.peak_wset / 2**30
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20


def load_edge_index(data_dir, out_dir):
    '''[2, E] int32 node indices of every transaction (account_to_idx order), cached as edge_index_all.npy.'''
    cache = f'{out_dir}/edge_index_all.npy'
    if os.path.exists(cache):
        return np.load(cache)
    edges = pd.read_csv(f'{data_dir}/edge_features.csv', usecols=['src_account', 'dst_account'])
    with open(f'{data_dir}/account_to_idx.pkl', 'rb') as f:
        account_to_idx = pickle.load(f)
    assert len(edges) == SPLIT_EDGES['test'], f'{len(edges):,} rows, expected {SPLIT_EDGES["test"]:,}'
    assert len(account_to_idx) == NUM_NODES
    edge_index = np.stack([edges['src_account'].map(account_to_idx).to_numpy(),
                           edges['dst_account'].map(account_to_idx).to_numpy()]).astype(np.int32)
    assert not np.isnan(edge_index).any() and edge_index.min() >= 0 and edge_index.max() < NUM_NODES
    np.save(cache, edge_index)
    return edge_index


def compute_rwpe(src, dst, num_nodes, walk_length, keep_self_loops=False, log=print):
    '''RWPE for all nodes from a directed edge list -> float32 [num_nodes, walk_length].

    Multi-edges count with their multiplicity (as in PyG); rows of nodes that never send stay 0.
    log() receives one line per step with nnz(P^k), seconds and RAM.
    '''
    src, dst = np.asarray(src), np.asarray(dst)
    if not keep_self_loops:
        keep = src != dst
        src, dst = src[keep], dst[keep]
    out_degree = np.bincount(src, minlength=num_nodes)
    adjacency = sp.csr_matrix((np.ones(len(src), dtype=np.float32), (src, dst)), shape=(num_nodes, num_nodes))
    adjacency.sum_duplicates()
    inv_degree = (1.0 / np.maximum(out_degree, 1)).astype(np.float32)
    transition = (sp.diags(inv_degree) @ adjacency).tocsr().astype(np.float32)   # P = D_out^-1 A

    rwpe = np.zeros((num_nodes, walk_length), dtype=np.float32)
    power = transition.copy()                                                    # P^1
    rwpe[:, 0] = power.diagonal()
    log(f'step  1 | nnz(P^1)={power.nnz:>13,} | peak RAM {peak_ram_gb():.2f} GB')
    for k in range(2, walk_length + 1):
        t0 = time.time()
        power = (power @ transition).tocsr()                                     # P^k
        rwpe[:, k - 1] = power.diagonal()
        log(f'step {k:>2} | nnz(P^{k})={power.nnz:>13,} | {time.time() - t0:6.1f}s | '
            f'RAM now {psutil.Process().memory_info().rss / 2**30:.2f} GB | peak {peak_ram_gb():.2f} GB')
    return rwpe


def main():
    split, walk_length, data_dir, out_dir = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
    keep_self_loops = '--keep-self-loops' in sys.argv[5:]
    assert split in SPLIT_EDGES, f'split must be one of {list(SPLIT_EDGES)}'
    os.makedirs(out_dir, exist_ok=True)
    suffix = '_withloops' if keep_self_loops else ''
    out_path = f'{out_dir}/rwpe_k{walk_length}_{split}{suffix}.pt'
    log_path = f'{out_dir}/log_{split}_k{walk_length}{suffix}.txt'
    log_file = open(log_path, 'w')

    def log(line):
        print(line, flush=True)
        log_file.write(line + '\n')
        log_file.flush()

    t_start = time.time()
    edge_index = load_edge_index(data_dir, out_dir)[:, :SPLIT_EDGES[split]]
    n_loops = int((edge_index[0] == edge_index[1]).sum())
    log(f'{split}: {edge_index.shape[1]:,} edges, {n_loops:,} self-loops '
        f'({"kept" if keep_self_loops else "dropped"}), K={walk_length}, nodes={NUM_NODES:,}')
    rwpe = compute_rwpe(edge_index[0], edge_index[1], NUM_NODES, walk_length, keep_self_loops, log)
    assert rwpe.shape == (NUM_NODES, walk_length) and not np.isnan(rwpe).any()
    assert rwpe.min() >= 0 and rwpe.max() <= 1
    torch.save(torch.from_numpy(rwpe), out_path)
    log(f'saved {out_path} {tuple(rwpe.shape)} | total {time.time() - t_start:.0f}s | peak RAM {peak_ram_gb():.2f} GB')
    log_file.close()


if __name__ == '__main__':
    main()
