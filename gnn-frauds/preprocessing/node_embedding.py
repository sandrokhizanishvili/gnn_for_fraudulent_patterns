"""Node encoding appended to the 6 entity columns. The choice is conf/preprocessing/node_embedding/."""
import torch

from gnn_frauds.hyperparameters.gnns import NODE_ENCS

def set_node_encoding(graphs, node_enc, enc_dir=None):
    '''Set graph.x of every snapshot to [entity one-hots || node encoding] for the NODE_ENC knob.

    none -> the 6 entity columns only (as every run so far); rwpe<k> -> that snapshot's k RWPE
    columns from <enc_dir>/rwpe_k<k>_<split>.pt appended. Returns the node dim (6 or 6 + k).
    '''
    assert node_enc in NODE_ENCS, f'unknown node encoding {node_enc!r}; choose from {NODE_ENCS}'
    for split, graph in graphs.items():
        if 'x_base' not in graph:
            graph.x_base = graph.x        # remembered once, so the knob can change between runs of a batch
        if node_enc == 'none':
            graph.x = graph.x_base
            continue
        k = int(node_enc[len('rwpe'):])
        enc = torch.load(f'{enc_dir}/rwpe_k{k}_{split}.pt', weights_only=True).float()
        assert enc.shape == (graph.num_nodes, k), f'{split}: encoding {tuple(enc.shape)} != ({graph.num_nodes}, {k})'
        assert not torch.isnan(enc).any() and enc.min() >= 0 and enc.max() <= 1  # return probabilities of a directed walk
        graph.x = torch.cat([graph.x_base, enc], dim=1)
    node_dim = graphs['train'].x.shape[1]
    assert all(g.x.shape[1] == node_dim for g in graphs.values())
    return node_dim


