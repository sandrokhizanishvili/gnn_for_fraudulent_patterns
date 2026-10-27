"""The four operators share one classifier. Only the convolution may differ."""
import pytest
import torch

from gnn_frauds.dataset.hi_small import in_degree_histogram, set_seed
from gnn_frauds.model.gnns.classifier import (
    EdgeClassifier, MPLayer, build_model, count_params, device, invariant_params, make_conv, make_norm,
)
from gnn_frauds.model.gnns.gat import GAT
from gnn_frauds.model.gnns.gin import GIN
from gnn_frauds.model.gnns.pna import PNA
from gnn_frauds.model.gnns.transformer import Transformer
from tests.support import tiny_recipe

OPERATORS = ('gin', 'gat', 'pna', 'transformer')
# v1 family constants: hidden 128, 2 layers, identity residual, no shared norm (CLAUDE.md section 3).
V1_INVARIANT = {'gin': 67_587, 'gat': 67_585, 'pna': 427_265, 'transformer': 133_121}


def _deg():
    edge_index = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]], dtype=torch.long)
    from torch_geometric.data import Data
    return in_degree_histogram(Data(edge_index=edge_index, num_nodes=6))


@pytest.mark.parametrize('operator', OPERATORS)
@pytest.mark.parametrize('residual', ['identity', 'linear', 'none'])
@pytest.mark.parametrize('edges', [False, True])
def test_each_operator_returns_finite_logits(operator, residual, edges):
    recipe = tiny_recipe(residual=residual, ffn=False, norm='none')
    mp_dim = 3 if edges else 0
    model = build_model(operator, 6, mp_dim, 5, deg_hist=_deg() if operator == 'pna' else None, recipe=recipe)
    x = torch.randn(6, 6, device=device)
    edge_index = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]], device=device)
    edge_attr = torch.randn(5, mp_dim, device=device) if edges else None
    seed_index = torch.tensor([[0, 1], [1, 2]], device=device)
    seed_feats = torch.randn(2, 5, device=device)
    model.train()
    train_out = model(x, edge_index, edge_attr, seed_index, seed_feats)
    model.eval()
    eval_out = model(x, edge_index, edge_attr, seed_index, seed_feats)
    assert train_out.shape == eval_out.shape == (2,)
    assert torch.isfinite(train_out).all() and torch.isfinite(eval_out).all()


def test_v1_invariant_params_match_the_family_constants():
    degree = _deg()
    for operator, expected in V1_INVARIANT.items():
        narrow = build_model(operator, 6, 0, 20, deg_hist=degree if operator == 'pna' else None)
        wide = build_model(operator, 6, 81, 81, deg_hist=degree if operator == 'pna' else None)
        encoded = build_model(operator, 22, 20, 20, deg_hist=degree if operator == 'pna' else None)
        assert invariant_params(narrow) == invariant_params(wide) == invariant_params(encoded) == expected
        assert count_params(wide) > count_params(narrow)


def test_recipe_none_matches_an_explicit_v1_recipe_and_reloads():
    set_seed(0)
    first = build_model('gin', 6, 20, 20, recipe=None)
    set_seed(0)
    second = build_model('gin', 6, 20, 20, recipe={})
    for left, right in zip(first.parameters(), second.parameters()):
        assert torch.equal(left, right)
    second.load_state_dict(first.state_dict(), strict=True)


def test_linear_residual_and_ffn_add_parameters():
    plain = build_model('gin', 6, 4, 4, recipe=tiny_recipe())
    linear = build_model('gin', 6, 4, 4, recipe=tiny_recipe(residual='linear'))
    with_ffn = build_model('gin', 6, 4, 4, recipe=tiny_recipe(ffn=True))
    batch = build_model('gin', 6, 4, 4, recipe=tiny_recipe(norm='batch'))
    assert count_params(linear) > count_params(plain)
    assert count_params(with_ffn) > count_params(plain)
    assert count_params(batch) > count_params(plain)


def test_unknown_operator_norm_and_bidirectional_are_rejected():
    with pytest.raises(ValueError):
        make_conv('sage', 8, 0)
    with pytest.raises(AssertionError):
        make_conv('pna', 8, 0, deg_hist=None)
    with pytest.raises(ValueError):
        make_norm('group', 8)
    with pytest.raises(AssertionError):
        EdgeClassifier('gin', 6, 0, 4, bidirectional=True, recipe=tiny_recipe())
    with pytest.raises(AssertionError):
        build_model('sage', 6, 0, 4)
    layer = MPLayer('gin', 8, 0, 0.0, ffn=True, norm='none', residual='linear')
    assert layer.ffn is not None and hasattr(layer, 'residual_proj')
    assert isinstance(make_norm('none', 8), torch.nn.Identity)


def test_operator_classes_only_store_the_name():
    assert GIN().name == 'gin'
    assert GAT().name == 'gat'
    assert PNA().name == 'pna'
    assert Transformer().name == 'transformer'
    assert callable(GIN().train)
