"""Tests for MCDropoutWrapper: uncertainty-aware inference with MatGL models."""

from __future__ import annotations

import pytest
import torch
from pymatgen.core import Lattice, Structure

import matgl  # noqa: F401
from matgl.models._chgnet import CHGNet
from matgl.models._m3gnet import M3GNet
from matgl.utils.uncertainty import MCDropoutWrapper

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def mos_structure():
    return Structure(Lattice.cubic(4.0), ["Mo", "S"], [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])


@pytest.fixture(scope="module")
def fe_structure():
    return Structure(Lattice.cubic(2.87), ["Fe", "Fe"], [[0, 0, 0], [0.5, 0.5, 0.5]])


# ---------------------------------------------------------------------------
# MCDropoutWrapper — construction
# ---------------------------------------------------------------------------


class TestMCDropoutWrapperConstruction:
    def test_chgnet_with_explicit_dropout(self, mos_structure):
        """CHGNet initialised with final_dropout > 0 should wrap without errors."""
        model = CHGNet(element_types=("Mo", "S"), final_dropout=0.1)
        wrapper = MCDropoutWrapper(model, dropout_p=0.1)
        assert len(wrapper._stochastic_modules) > 0

    def test_chgnet_identity_replaced(self, mos_structure):
        """CHGNet with default final_dropout=0 (nn.Identity) should have Identity replaced."""
        model = CHGNet(element_types=("Mo", "S"), final_dropout=0.0)
        wrapper = MCDropoutWrapper(model, dropout_p=0.1)
        # The Identity should have been replaced with a real Dropout.
        assert isinstance(model.final_dropout, torch.nn.Identity)
        assert isinstance(wrapper.model.final_dropout, torch.nn.Dropout)
        assert wrapper.model.final_dropout.p == pytest.approx(0.1)

    def test_m3gnet_wraps(self):
        """M3GNet with is_intensive=True uses an MLP final_layer that Dropout can be injected into."""
        model = M3GNet(element_types=("Mo", "S"), is_intensive=True)
        wrapper = MCDropoutWrapper(model, dropout_p=0.1)
        assert len(wrapper._stochastic_modules) > 0

    def test_m3gnet_extensive_raises(self):
        """M3GNet with is_intensive=False uses WeightedReadOut (GatedMLP), which has no injectable
        dropout site compatible with MC Dropout. Verify this raises rather than silently misbehaves."""
        model = M3GNet(element_types=("Mo", "S"), is_intensive=False)
        with pytest.raises(ValueError, match="No dropout layers found"):
            MCDropoutWrapper(model, dropout_p=0.1)

    def test_invalid_dropout_p_zero(self):
        model = CHGNet(element_types=("Mo", "S"))
        with pytest.raises(ValueError, match="dropout_p must be in"):
            MCDropoutWrapper(model, dropout_p=0.0)

    def test_invalid_dropout_p_one(self):
        model = CHGNet(element_types=("Mo", "S"))
        with pytest.raises(ValueError, match="dropout_p must be in"):
            MCDropoutWrapper(model, dropout_p=1.0)

    def test_unknown_readout_attrs_raises(self):
        model = CHGNet(element_types=("Mo", "S"))
        with pytest.raises(ValueError, match="No dropout layers found"):
            MCDropoutWrapper(model, dropout_p=0.1, readout_attrs=("nonexistent_attr",))


# ---------------------------------------------------------------------------
# MCDropoutWrapper — predict_uncertainty
# ---------------------------------------------------------------------------


class TestPredictUncertainty:
    def test_output_shapes_single_structure(self, mos_structure):
        model = CHGNet(element_types=("Mo", "S"), final_dropout=0.1)
        wrapper = MCDropoutWrapper(model, dropout_p=0.1)
        mean, std = wrapper.predict_uncertainty(mos_structure, n_passes=5)
        assert mean.shape == torch.Size([])
        assert std.shape == torch.Size([])

    def test_output_shapes_multiple_structures(self, mos_structure, fe_structure):
        model = CHGNet(element_types=("Mo", "S", "Fe"), final_dropout=0.1)
        wrapper = MCDropoutWrapper(model, dropout_p=0.1)
        mean, std = wrapper.predict_uncertainty([mos_structure, fe_structure], n_passes=5)
        assert mean.shape == torch.Size([2])
        assert std.shape == torch.Size([2])

    def test_std_positive_with_dropout(self, mos_structure):
        """With dropout active, predictions across passes should differ (std > 0)."""
        model = CHGNet(element_types=("Mo", "S"), final_dropout=0.3)
        wrapper = MCDropoutWrapper(model, dropout_p=0.3)
        _, std = wrapper.predict_uncertainty(mos_structure, n_passes=20)
        assert std.item() > 0.0

    def test_std_zero_deterministic_baseline(self, mos_structure):
        """With dropout p=~0, all passes should agree (std ≈ 0)."""
        model = CHGNet(element_types=("Mo", "S"), final_dropout=0.0)
        # Use a tiny p so we can still wrap, but effectively no dropout signal
        wrapper = MCDropoutWrapper(model, dropout_p=1e-6)
        _, std = wrapper.predict_uncertainty(mos_structure, n_passes=10)
        assert std.item() < 1e-3

    def test_model_returns_to_eval_after_predict(self, mos_structure):
        """Model must be fully in eval() mode after predict_uncertainty returns."""
        model = CHGNet(element_types=("Mo", "S"), final_dropout=0.1)
        wrapper = MCDropoutWrapper(model, dropout_p=0.1)
        wrapper.predict_uncertainty(mos_structure, n_passes=3)
        for m in wrapper.model.modules():
            assert not m.training, f"{m} still in training mode after predict_uncertainty"

    def test_m3gnet_uncertainty(self, mos_structure):
        model = M3GNet(element_types=("Mo", "S"), is_intensive=True)
        wrapper = MCDropoutWrapper(model, dropout_p=0.1)
        mean, std = wrapper.predict_uncertainty(mos_structure, n_passes=5)
        assert torch.isfinite(mean)
        assert std >= 0.0

    def test_mean_finite(self, mos_structure):
        model = CHGNet(element_types=("Mo", "S"), final_dropout=0.1)
        wrapper = MCDropoutWrapper(model, dropout_p=0.1)
        mean, _ = wrapper.predict_uncertainty(mos_structure, n_passes=5)
        assert torch.isfinite(mean)

    def test_n_passes_one_raises(self, mos_structure):
        model = CHGNet(element_types=("Mo", "S"), final_dropout=0.1)
        wrapper = MCDropoutWrapper(model, dropout_p=0.1)
        with pytest.raises(ValueError, match="n_passes must be"):
            wrapper.predict_uncertainty(mos_structure, n_passes=1)


# ---------------------------------------------------------------------------
# MCDropoutWrapper — backbone-once fast path (cache_backbone)
# ---------------------------------------------------------------------------


def _batch_for(wrapper, structures):
    from torch_geometric.data import Batch

    graphs = [wrapper._to_graph(s, None) for s in structures]
    batch = Batch.from_data_list([g for g, _ in graphs])
    sfs = [sf for _, sf in graphs]
    return batch, (torch.stack(sfs) if sfs[0].numel() > 0 else None)


class TestCacheBackbone:
    def test_m3gnet_fast_path_is_used(self, mos_structure, fe_structure):
        """M3GNet (intensive) readout is terminal, so the cached path must engage."""
        model = M3GNet(element_types=("Mo", "S", "Fe"), is_intensive=True)
        wrapper = MCDropoutWrapper(model, dropout_p=0.1)
        batch, bsf = _batch_for(wrapper, [mos_structure, fe_structure])
        assert wrapper._predict_cached(batch, bsf, 4) is not None

    def test_chgnet_falls_back(self, mos_structure):
        """CHGNet pools *after* dropout, so the head is not terminal -> fallback (None)."""
        model = CHGNet(element_types=("Mo", "S"), final_dropout=0.0)
        wrapper = MCDropoutWrapper(model, dropout_p=0.1)
        batch, bsf = _batch_for(wrapper, [mos_structure])
        assert wrapper._predict_cached(batch, bsf, 4) is None
        # public API still works via the naive fallback
        mean, std = wrapper.predict_uncertainty(mos_structure, n_passes=5)
        assert torch.isfinite(mean)
        assert std >= 0.0

    def test_cached_equals_naive(self, mos_structure, fe_structure):
        """With near-zero dropout both paths collapse to the deterministic prediction,
        so the fast path must match the naive loop tightly (exactness of head replay)."""
        model = M3GNet(element_types=("Mo", "S", "Fe"), is_intensive=True)
        wrapper = MCDropoutWrapper(model, dropout_p=1e-6)
        structs = [mos_structure, fe_structure]
        m_fast, _ = wrapper.predict_uncertainty(structs, n_passes=8, cache_backbone=True)
        m_slow, _ = wrapper.predict_uncertainty(structs, n_passes=8, cache_backbone=False)
        assert torch.allclose(m_fast, m_slow, atol=1e-4)

    def test_cached_shapes_match_naive(self, mos_structure, fe_structure):
        model = M3GNet(element_types=("Mo", "S", "Fe"), is_intensive=True)
        wrapper = MCDropoutWrapper(model, dropout_p=0.1)
        structs = [mos_structure, fe_structure]
        m_fast, s_fast = wrapper.predict_uncertainty(structs, n_passes=5, cache_backbone=True)
        m_slow, s_slow = wrapper.predict_uncertainty(structs, n_passes=5, cache_backbone=False)
        assert m_fast.shape == m_slow.shape == torch.Size([2])
        assert s_fast.shape == s_slow.shape == torch.Size([2])
        assert torch.isfinite(m_fast).all()
        assert (s_fast >= 0).all()


@pytest.mark.parametrize("model_type", ["M3GNet", "TensorNet"])
@pytest.mark.parametrize("cache_backbone", [False, True])
def test_hidden_dropout_is_not_output_scaling(model_type, cache_backbone):
    """Equal deterministic outputs can have different uncertainty from hidden features."""
    from matgl.layers import MLP
    from matgl.models import TensorNet
    from matgl.utils.uncertainty import _inject_dropout

    torch.manual_seed(0)
    factory = M3GNet if model_type == "M3GNet" else TensorNet
    kwargs = {} if model_type == "M3GNet" else {"use_warp": False}
    model = factory(element_types=("Mo", "S"), is_intensive=True, **kwargs)
    wrapper = MCDropoutWrapper(model, dropout_p=0.5)
    layers = wrapper.model.final_layer.layers
    last_linear = max(i for i, layer in enumerate(layers) if isinstance(layer, torch.nn.Linear))
    assert isinstance(layers[last_linear - 1], torch.nn.Dropout)
    assert not any(isinstance(layer, torch.nn.Dropout) for layer in model.final_layer.layers)

    # The same sum (and thus deterministic prediction) with different hidden-feature norms.
    head = MLP([2, 1], bias_last=False)
    drop = _inject_dropout(head, 0.5)
    assert drop is not None
    with torch.no_grad():
        head.layers[-1].weight.fill_(1.0)
    x = torch.tensor([[1.0, 0.0], [0.5, 0.5]])
    torch.manual_seed(0)
    samples = torch.stack([head(x).squeeze(-1) for _ in range(2000)])
    assert torch.allclose(samples.mean(0), torch.ones(2), atol=0.05)
    ratios = samples.std(0) / samples.mean(0).abs()
    assert ratios[0] > ratios[1] + 0.15

    # Exercise the full model and both replay paths with identical random draws.
    structure = Structure(Lattice.cubic(4.0), ["Mo", "S"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    torch.manual_seed(42)
    mean, std = wrapper.predict_uncertainty(structure, n_passes=20, cache_backbone=cache_backbone)
    torch.manual_seed(42)
    other_mean, other_std = wrapper.predict_uncertainty(structure, n_passes=20, cache_backbone=not cache_backbone)
    assert torch.allclose(mean, other_mean, atol=1e-6)
    assert torch.allclose(std, other_std, atol=1e-6)

    structures = [Structure(Lattice.cubic(a), ["Mo", "S"], [[0, 0, 0], [0.5, 0.5, 0.5]]) for a in (3.0, 4.0, 5.0)]
    mean, std = wrapper.predict_uncertainty(structures, n_passes=1000, cache_backbone=True)
    ratios = std / mean.abs().clamp_min(1e-8)
    assert (ratios.max() - ratios.min()).item() > 0.001
    assert not torch.allclose(ratios, torch.ones_like(ratios), atol=0.1)


def test_wrapper_preserves_used_model_and_training_mode(mos_structure):
    model = M3GNet(element_types=("Mo", "S"), is_intensive=True).train()
    # A grad-connected feature_dict from an earlier call must not prevent copying.
    model.predict_structure(mos_structure)
    original_features = model.feature_dict
    original_state = {key: value.clone() for key, value in model.state_dict().items()}
    original_modules = list(model.modules())
    wrapper = MCDropoutWrapper(model)
    wrapper.predict_uncertainty(mos_structure, n_passes=3)
    assert model.training
    assert list(model.modules()) == original_modules
    assert model.feature_dict is original_features
    assert all(torch.equal(model.state_dict()[key], value) for key, value in original_state.items())
    assert wrapper.model is not model
