"""The shared edge classifier. gin.py, gat.py, pna.py and transformer.py select the operator."""
import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv, GINConv, GINEConv, PNAConv, TransformerConv

from gnn_frauds.hyperparameters.gnns import BASE_NODE_DIM, HEADS, NORMS, OPERATORS, full_recipe

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # one device for the model, the loss and the batches

def gin_mlp(hidden):
    '''The GIN node MLP: Linear -> BatchNorm -> ReLU -> Linear, identical in every configuration.'''
    return nn.Sequential(nn.Linear(hidden, hidden), nn.BatchNorm1d(hidden), nn.ReLU(),
                         nn.Linear(hidden, hidden))


def make_conv(operator, hidden, mp_edge_dim, deg_hist=None, self_loops=True):
    '''One graph convolution for the operator; the only place operators differ.

    mp_edge_dim = 0 -> edge features stay out of message passing. self_loops is GATv2's add_self_loops
    (v1 = True); the other operators ignore it. PyG defaults are written out where they matter.
    '''
    if operator == 'gin':
        # Edge features go through GINE; without them the layer is GIN. ε is learned, not fixed at 0.
        if mp_edge_dim > 0:
            return GINEConv(gin_mlp(hidden), train_eps=True, edge_dim=mp_edge_dim)
        return GINConv(gin_mlp(hidden), train_eps=True)
    if operator == 'pna':
        # 4 aggregators x 3 scalers, one tower (one MLP on the full width); scalers use the train in-degree histogram
        assert deg_hist is not None, 'pna needs the training-graph in-degree histogram (in_degree_histogram)'
        pna_kwargs = dict(aggregators=['mean', 'max', 'min', 'std'],
                          scalers=['identity', 'amplification', 'attenuation'], deg=deg_hist, towers=1)
        if mp_edge_dim > 0:
            return PNAConv(hidden, hidden, edge_dim=mp_edge_dim, **pna_kwargs)
        return PNAConv(hidden, hidden, **pna_kwargs)
    if operator == 'gat':
        # HEADS heads concatenated -> hidden; a self-loop gets the mean of the node's incoming edge features
        if mp_edge_dim > 0:
            return GATv2Conv(hidden, hidden // HEADS, heads=HEADS, edge_dim=mp_edge_dim,
                             add_self_loops=self_loops, fill_value='mean', bias=True)
        return GATv2Conv(hidden, hidden // HEADS, heads=HEADS,
                         add_self_loops=self_loops, fill_value='mean', bias=True)
    if operator == 'transformer':
        # HEADS heads concatenated -> hidden; root_weight = the node's own state through lin_skip.
        # Dropout stays in MPLayer, so every operator drops the same activation. The conv's own dropout stays off.
        if mp_edge_dim > 0:
            return TransformerConv(hidden, hidden // HEADS, heads=HEADS, concat=True,
                                   edge_dim=mp_edge_dim, beta=False, dropout=0.0, root_weight=True)
        return TransformerConv(hidden, hidden // HEADS, heads=HEADS, concat=True,
                               beta=False, dropout=0.0, root_weight=True)
    raise ValueError(f'unknown operator {operator!r}; choose from {OPERATORS}')


def make_norm(norm, hidden):
    '''none -> Identity (no parameters, so v1 checkpoints still load); batch -> BatchNorm1d; layer -> LayerNorm.'''
    if norm == 'none':
        return nn.Identity()
    if norm == 'batch':
        return nn.BatchNorm1d(hidden)
    if norm == 'layer':
        return nn.LayerNorm(hidden)
    raise ValueError(f'unknown norm {norm!r}; choose from {NORMS}')


class MPLayer(nn.Module):
    '''One message-passing step with a residual: h <- h + Dropout(ReLU(Norm(conv(h, edges)))).

    conv_in aggregates the accounts that paid a node (the module name is fixed by the saved checkpoints).
    norm='none' (v1) is the identity. ffn adds the GNN+ block h <- Norm(h + W2(Dropout(ReLU(W1 h)))) after the
    residual; its Norm is the layer's norm choice, or LayerNorm when that is 'none'. residual: 'identity' (v1, the
    formula above), 'linear' (Luo et al. 2024 Eq. 7: h <- Dropout(ReLU(Norm(conv(h) + W_r h)))) or 'none' (no skip).
    Only ffn and the linear residual add parameters, so v1 checkpoints load unchanged.
    '''

    def __init__(self, operator, hidden, mp_edge_dim, dropout, deg_hist=None, norm='none', ffn=False,
                 self_loops=True, residual='identity'):
        super().__init__()
        self.use_edges = mp_edge_dim > 0
        self.conv_in = make_conv(operator, hidden, mp_edge_dim, deg_hist, self_loops)
        self.dropout = nn.Dropout(dropout)
        self.norm = make_norm(norm, hidden)
        self.ffn = None
        if ffn:
            self.ffn = nn.Sequential(nn.Linear(hidden, 2 * hidden), nn.ReLU(), nn.Dropout(dropout),
                                     nn.Linear(2 * hidden, hidden))
            self.ffn_norm = make_norm(norm if norm != 'none' else 'layer', hidden)
        self.residual = residual
        if residual == 'linear':
            self.residual_proj = nn.Linear(hidden, hidden)   # registered last, so v1 layers initialise unchanged

    def forward(self, h, edge_index, edge_attr):
        out = self.conv_in(h, edge_index, edge_attr) if self.use_edges else self.conv_in(h, edge_index)
        if self.residual == 'linear':
            out = out + self.residual_proj(h)                 # the paper's residual enters before the norm
        out = self.dropout(torch.relu(self.norm(out)))
        h = h + out if self.residual == 'identity' else out  # linear already added W_r h inside out; 'none' has no skip
        if self.ffn is not None:
            h = self.ffn_norm(h + self.ffn(h))
        return h


class EdgeClassifier(nn.Module):
    '''Fixed-architecture edge classifier: node_proj -> num_layers message-passing layers -> readout MLP.

    x [N, node_dim] -> node_proj Linear -> hidden, ReLU -> MPLayers -> readout on the seed edge:
    concat [h_src || h_dst || e_seed] -> (readout_layers - 1) x (Linear -> ReLU -> Dropout) -> Linear 1 (logit).
    Between feature configs only the conv's edge projection and the readout's first Linear change width.
    Module names (node_proj, layers.i.conv_in, classifier) are fixed by the saved checkpoints. recipe=None -> v1.
    bidirectional (message passing over reversed edges too) is not implemented; the parameter stays because the
    notebooks pass it (always False).
    '''

    def __init__(self, operator, node_dim, mp_edge_dim, readout_edge_dim, bidirectional=False, deg_hist=None, recipe=None):
        super().__init__()
        assert not bidirectional, 'only message passing over incoming edges is implemented'
        r = full_recipe(recipe)
        hidden = r['hidden']
        self.node_proj = nn.Linear(node_dim, hidden)
        self.layers = nn.ModuleList(
            [MPLayer(operator, hidden, mp_edge_dim, r['dropout'], deg_hist, r['norm'], r['ffn'], r['self_loops'], r['residual'])
             for _ in range(r['num_layers'])])
        width = max(1, int(round(r['readout_width'] * hidden)))   # neurons per readout hidden layer
        in_dim, readout = 2 * hidden + readout_edge_dim, []
        for _ in range(r['readout_layers'] - 1):  # the last Linear is the logit, so it is not one of these blocks
            readout += [nn.Linear(in_dim, width), nn.ReLU(), nn.Dropout(r['dropout'])]
            in_dim = width
        self.classifier = nn.Sequential(*readout, nn.Linear(in_dim, 1))

    def forward(self, x, edge_index, edge_attr, seed_index, seed_edge_feats):
        '''x [N, node_dim], edge_index [2, E], edge_attr [E, mp_dim] or None,
        seed_index [2, B], seed_edge_feats [B, ro_dim] -> logits [B].'''
        h = torch.relu(self.node_proj(x))
        for layer in self.layers:
            h = layer(h, edge_index, edge_attr)
        z = torch.cat([h[seed_index[0]], h[seed_index[1]], seed_edge_feats], dim=1)
        return self.classifier(z).squeeze(-1)


def build_model(operator, node_dim, mp_edge_dim, readout_edge_dim, bidirectional=False, deg_hist=None, recipe=None):
    '''Build the template for one operator, feature configuration and recipe (None = v1), on the device.'''
    assert operator in OPERATORS, f'unknown operator {operator!r}'
    model = EdgeClassifier(operator, node_dim, mp_edge_dim, readout_edge_dim, bidirectional, deg_hist, recipe)
    return model.to(device)


def count_params(model):
    '''Total number of trainable parameters.'''
    return sum(p.numel() for p in model.parameters())


def invariant_params(model):
    '''Parameters whose shape does not depend on the feature widths; identical across the feature configs of a family.

    Left out: the edge projection inside the conv (.lin. / lin_edge / edge_encoder, and PNA's message MLP input
    pre_nns.0.0., which widens when edge features enter) and the readout's first Linear (classifier.0.).
    node_proj.weight is counted at the base node width 6, so a node encoding widens it without changing the count.
    v1 values: GIN 67,587 · GATv2 67,585 · PNA 427,265 · Transformer 133,121; a tuned recipe has its own constant.
    '''
    width_dependent = ('.lin.', 'lin_edge', 'edge_encoder', 'pre_nns.0.0.', 'classifier.0.')
    total = 0
    for name, p in model.named_parameters():
        if name == 'node_proj.weight':
            total += model.node_proj.out_features * BASE_NODE_DIM
        elif not any(tag in name for tag in width_dependent):
            total += p.numel()
    return total

