from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

import matgl
from matgl.models._tensornet import TensorNet, _warp_available


def _set_pos_and_pbc(graph, lat):
    """Attach `pos` and `pbc_offshift` to `graph`."""
    graph.pbc_offshift = torch.matmul(graph.pbc_offset, lat[0])
    graph.pos = graph.frac_coords @ lat[0]


def _device_of(graph):
    return graph.pos.device


def _make_tensornet(**kwargs):
    """Construct TensorNet with `use_warp=False` so non-warp paths are exercised."""
    kwargs.setdefault("use_warp", False)
    return TensorNet(**kwargs)


def _check_scalar_output(output):
    assert torch.numel(output) == 1
    assert torch.isfinite(output).all()


def test_model(graph_MoS, tmp_path):
    torch.manual_seed(0)

    _, graph, _ = graph_MoS

    activations = ["swish", "tanh", "sigmoid", "softplus2", "softexp"]
    for act in activations:
        model = _make_tensornet(is_intensive=False, activation_type=act)
        model.to(_device_of(graph))
        output = model(g=graph)
        _check_scalar_output(output)

    model.save(tmp_path)
    TensorNet.load(tmp_path)

    model = _make_tensornet(is_intensive=False, equivariance_invariance_group="SO(3)")
    model.to(_device_of(graph))
    output = model(g=graph)
    _check_scalar_output(output)


@pytest.mark.parametrize("use_smooth", [True, False])
def test_model_spherical_bessel(graph_MoS, use_smooth):
    """SphericalBessel RBF must size the embedding/interaction layers correctly.

    The non-smooth basis emits ``max_l * max_n`` features (the smooth one emits
    ``max_n``); a regression test that the input Linear layers are sized to match.
    """
    torch.manual_seed(0)
    _, graph, _ = graph_MoS
    model = _make_tensornet(is_intensive=False, rbf_type="SphericalBessel", use_smooth=use_smooth, max_n=4, max_l=3)
    model.to(_device_of(graph))
    output = model(g=graph)
    _check_scalar_output(output)


def test_exceptions():
    with pytest.raises(ValueError, match="Invalid activation type"):
        _ = _make_tensornet(element_types=None, is_intensive=False, activation_type="whatever")
    with pytest.raises(ValueError, match=r"Classification task cannot be extensive."):
        _ = _make_tensornet(element_types=["Mo", "S"], is_intensive=False, task_type="classification")


def test_model_intensive(graph_MoS):
    torch.manual_seed(0)
    structure, graph, _ = graph_MoS
    lat = torch.tensor(np.array([structure.lattice.matrix]), dtype=matgl.float_th, device=_device_of(graph))
    _set_pos_and_pbc(graph, lat)
    model = _make_tensornet(element_types=["Mo", "S"], is_intensive=True)
    model.to(_device_of(graph))
    output = model(g=graph)
    _check_scalar_output(output)


def test_model_intensive_with_weighted_atom(graph_MoS):
    torch.manual_seed(0)
    structure, graph, _ = graph_MoS
    lat = torch.tensor(np.array([structure.lattice.matrix]), dtype=matgl.float_th, device=_device_of(graph))
    _set_pos_and_pbc(graph, lat)
    model = _make_tensornet(element_types=["Mo", "S"], is_intensive=True, readout_type="weighted_atom")
    model.to(_device_of(graph))
    output = model(g=graph)
    _check_scalar_output(output)


def test_model_intensive_with_ReduceReadOut(graph_MoS):
    torch.manual_seed(0)
    structure, graph, _ = graph_MoS
    lat = torch.tensor(np.array([structure.lattice.matrix]), dtype=matgl.float_th, device=_device_of(graph))
    _set_pos_and_pbc(graph, lat)
    model = _make_tensornet(is_intensive=True, readout_type="reduce_atom")
    model.to(_device_of(graph))
    output = model(g=graph)
    _check_scalar_output(output)


def test_model_intensive_with_classification(graph_MoS):
    torch.manual_seed(0)
    structure, graph, _ = graph_MoS
    lat = torch.tensor(np.array([structure.lattice.matrix]), dtype=matgl.float_th, device=_device_of(graph))
    _set_pos_and_pbc(graph, lat)
    model = _make_tensornet(element_types=["Mo", "S"], is_intensive=True, task_type="classification")
    model.to(_device_of(graph))
    output = model(g=graph)
    _check_scalar_output(output)
    assert 0.0 <= output.item() <= 1.0


def test_return_features(graph_MoS):
    structure, graph, _ = graph_MoS
    lat = torch.tensor(np.array([structure.lattice.matrix]), dtype=matgl.float_th, device=_device_of(graph))
    _set_pos_and_pbc(graph, lat)

    model = _make_tensornet(element_types=["Mo", "S"], is_intensive=True)
    model.to(_device_of(graph))

    out = model.predict_structure(structure, return_features=False)
    assert isinstance(out, torch.Tensor)

    out_feats = model.predict_structure(structure, return_features=True)
    assert isinstance(out_feats, dict)
    assert "final" in out_feats
    assert "readout" in out_feats
    assert "edge_attr" in out_feats
    assert "embedding" in out_feats
    assert "gc_1" in out_feats

    assert out_feats["final"].shape == torch.Size([])
    assert out_feats["readout"].shape[0] == structure.num_sites

    out_feats_subset = model.predict_structure(structure, return_features=True, output_layers=["final", "gc_1"])
    assert set(out_feats_subset.keys()) == {"final", "gc_1"}


# ---------------------------------------------------------------------------
# Warp-only regression and parity tests (nvalchemiops installed)
# ---------------------------------------------------------------------------


def _skip_if_no_warp():
    if not _warp_available:
        pytest.skip("Warp tests require nvalchemiops to be installed.")


def test_warp_model_regression(graph_MoS):
    _skip_if_no_warp()
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)

    expected = {
        "swish": torch.tensor(0.0827),
        "tanh": torch.tensor(-0.0258),
        "sigmoid": torch.tensor(0.0360),
        "softplus2": torch.tensor(0.1165),
        "softexp": torch.tensor(0.1100),
    }

    _, graph, _ = graph_MoS
    for act, ref in expected.items():
        model = TensorNet(is_intensive=False, activation_type=act, use_warp=True)
        output = model(g=graph)
        assert torch.allclose(output, ref, atol=1e-4)


def test_warp_model_intensive(graph_MoS):
    _skip_if_no_warp()
    structure, graph, _ = graph_MoS
    lat = torch.tensor(np.array([structure.lattice.matrix]), dtype=matgl.float_th)
    graph.pbc_offshift = torch.matmul(graph.pbc_offset, lat[0])
    graph.pos = graph.frac_coords @ lat[0]
    model = TensorNet(element_types=["Mo", "S"], is_intensive=True, use_warp=True)
    output = model(g=graph)
    _check_scalar_output(output)


def test_warp_model_intensive_with_weighted_atom(graph_MoS):
    _skip_if_no_warp()
    structure, graph, _ = graph_MoS
    lat = torch.tensor(np.array([structure.lattice.matrix]), dtype=matgl.float_th)
    graph.pbc_offshift = torch.matmul(graph.pbc_offset, lat[0])
    graph.pos = graph.frac_coords @ lat[0]
    model = TensorNet(element_types=["Mo", "S"], is_intensive=True, readout_type="weighted_atom", use_warp=True)
    output = model(g=graph)
    _check_scalar_output(output)


def test_warp_model_intensive_with_ReduceReadOut(graph_MoS):
    _skip_if_no_warp()
    structure, graph, _ = graph_MoS
    lat = torch.tensor(np.array([structure.lattice.matrix]), dtype=matgl.float_th)
    graph.pbc_offshift = torch.matmul(graph.pbc_offset, lat[0])
    graph.pos = graph.frac_coords @ lat[0]
    model = TensorNet(is_intensive=True, readout_type="reduce_atom", use_warp=True)
    output = model(g=graph)
    _check_scalar_output(output)


def test_warp_backward(graph_MoS):
    _skip_if_no_warp()
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)

    expected_cell_grad = torch.tensor(
        [
            [-0.000909, 0.000000, 0.000000],
            [0.000000, -0.000909, 0.000000],
            [0.000000, 0.000000, -0.000909],
        ]
    )

    structure, graph, _ = graph_MoS
    cell = torch.tensor(structure.lattice.matrix, dtype=matgl.float_th).requires_grad_(True)

    graph.pbc_offshift = torch.matmul(graph.pbc_offset, cell)
    graph.pos = graph.frac_coords @ cell

    model = TensorNet(is_intensive=False, activation_type="swish")
    model.train()

    energy = model(g=graph)
    cell_grad = torch.autograd.grad(energy, cell, create_graph=True)[0]

    assert torch.allclose(cell_grad, expected_cell_grad, atol=1e-6)


def test_warp_double_backward(graph_MoS):
    _skip_if_no_warp()
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)

    expected_cell_grad2 = torch.tensor(
        [
            [-0.0000037, -0.000000, -0.000000],
            [-0.000000, -0.0000037, -0.000000],
            [-0.000000, -0.000000, -0.0000037],
        ]
    )

    structure, graph, _ = graph_MoS
    cell = torch.tensor(structure.lattice.matrix, dtype=matgl.float_th).requires_grad_(True)
    cell.retain_grad()

    graph.pbc_offshift = torch.matmul(graph.pbc_offset, cell)
    graph.pos = graph.frac_coords @ cell

    model = TensorNet(is_intensive=False, activation_type="swish")
    model.train()

    energy = model(g=graph)
    cell_grad = torch.autograd.grad(energy, cell, create_graph=True)[0]
    loss = (cell_grad * cell_grad).sum()
    loss.backward()

    assert torch.allclose(cell.grad, expected_cell_grad2, atol=1e-6)


def _build_pair_from_pretrained(repo_id: str) -> tuple[TensorNet, TensorNet]:
    from matgl.utils.io import _get_file_paths

    fpaths = _get_file_paths(Path(repo_id))
    map_location = "cpu" if not torch.cuda.is_available() else None
    state = torch.load(fpaths["state.pt"], map_location=map_location, weights_only=False)
    init_blob = torch.load(fpaths["model.pt"], map_location=map_location, weights_only=False)
    inner_init_args = dict(init_blob["model"]["init_args"])

    inner_state = {k[len("model.") :]: v for k, v in state.items() if k.startswith("model.")}

    model_warp = TensorNet(**{**inner_init_args, "use_warp": True})
    model_pyg = TensorNet(**{**inner_init_args, "use_warp": False})
    model_warp.load_state_dict(inner_state, strict=False)
    model_pyg.load_state_dict(inner_state, strict=False)
    model_warp.eval()
    model_pyg.eval()
    return model_warp, model_pyg


@pytest.mark.parametrize("structure_fixture", ["MoS", "LiFePO4", "Li3InCl6"])
def test_warp_pyg_parity_pretrained(structure_fixture, request):
    _skip_if_no_warp()
    structure = request.getfixturevalue(structure_fixture)
    model_warp, model_pyg = _build_pair_from_pretrained("materialyze/TensorNet-PES-MatPES-PBE-2025.2")

    from matgl.ext.pymatgen import Structure2Graph

    converter = Structure2Graph(element_types=model_pyg.element_types, cutoff=model_pyg.cutoff)
    g, lat, _ = converter.get_graph(structure)
    g.pbc_offshift = torch.matmul(g.pbc_offset, lat[0])
    g.pos = g.frac_coords @ lat[0]

    with torch.no_grad():
        out_warp = model_warp(g=g)
        out_pyg = model_pyg(g=g)

    assert torch.allclose(out_warp, out_pyg, atol=1e-5, rtol=1e-5), (
        f"{structure_fixture}: warp={out_warp.detach().cpu()} vs pyg={out_pyg.detach().cpu()}"
    )


def test_warp_pyg_parity_so3_chiral():
    """SO(3) TensorNet must give the same energy with Warp and PyG on a chiral structure (#861).

    The Warp radial kernel used to build the transposed antisymmetric edge tensor, so the Warp path evaluated the
    mirror image of the input. O(3) is insensitive to this; SO(3) is not.
    """
    _skip_if_no_warp()
    from pymatgen.core import Lattice, Structure

    from matgl.ext.pymatgen import Structure2Graph

    lattice = Lattice.cubic(5.0)
    species = ["Mo", "S", "S", "Mo", "S"]
    frac_coords = np.array(
        [[0.0, 0.0, 0.0], [0.31, 0.12, 0.05], [0.08, 0.37, 0.21], [0.27, 0.19, 0.43], [0.52, 0.61, 0.17]]
    )
    structure = Structure(lattice, species, frac_coords)
    mirror = Structure(lattice, species, -frac_coords)  # inversion = mirror image up to a proper rotation

    # A small double-precision model gives a reproducible chirality signal well
    # above rounding noise; random scalar readouts can nearly cancel that signal.
    torch.manual_seed(2)
    init_args = {
        "element_types": ("Mo", "S"),
        "is_intensive": False,
        "equivariance_invariance_group": "SO(3)",
        "units": 16,
        "nblocks": 3,
        "dtype": torch.float64,
    }
    model_pyg = TensorNet(**init_args, use_warp=False)
    model_warp = TensorNet(**init_args, use_warp=True)
    model_warp.load_state_dict(model_pyg.state_dict(), strict=False)
    model_pyg.eval()
    model_warp.eval()

    converter = Structure2Graph(element_types=("Mo", "S"), cutoff=model_pyg.cutoff)

    def evaluate(model, struct):
        g, lat, _ = converter.get_graph(struct)
        g.frac_coords = g.frac_coords.double()
        g.pbc_offset = g.pbc_offset.double()
        lat = lat.double()
        _set_pos_and_pbc(g, lat)
        g.pos.requires_grad_(True)
        output = model(g=g)
        forces = -torch.autograd.grad(output, g.pos)[0]
        return output.item(), forces, model.feature_dict["readout"].detach().clone()

    e_pyg, f_pyg, features_pyg = evaluate(model_pyg, structure)
    chirality_gap = abs(e_pyg - evaluate(model_pyg, mirror)[0])
    # Sanity check that the test is sensitive: an SO(3) model distinguishes a chiral structure from its mirror image.
    assert chirality_gap > 1e-8
    e_warp, f_warp, features_warp = evaluate(model_warp, structure)
    assert abs(e_warp - e_pyg) < 0.01 * chirality_gap
    torch.testing.assert_close(features_warp, features_pyg, atol=5e-8, rtol=1e-6)
    torch.testing.assert_close(f_warp, f_pyg, atol=1e-8, rtol=1e-5)


def test_extensive_multitarget_batched(graph_MoS):
    """Extensive readout with ntargets > 1 must give one row of targets per graph."""
    from torch_geometric.data import Batch

    structure, graph, _ = graph_MoS
    lat = torch.tensor(np.array([structure.lattice.matrix]), dtype=matgl.float_th)
    graph.pbc_offshift = torch.matmul(graph.pbc_offset, lat[0])
    graph.pos = graph.frac_coords @ lat[0]
    torch.manual_seed(0)
    model = TensorNet(element_types=["Mo", "S"], is_intensive=False, ntargets=2, use_warp=False)
    single = model(g=graph)
    assert single.shape == (2,)
    batched = model(g=Batch.from_data_list([graph, graph]))
    assert batched.shape == (2, 2)
    assert torch.allclose(batched, single.expand(2, 2), atol=1e-5)


@pytest.mark.parametrize(
    "kwargs",
    [{"is_intensive": False}, {"is_intensive": True}, {"is_intensive": True, "readout_type": "reduce_atom"}],
    ids=["extensive", "weighted_atom", "reduce_atom"],
)
def test_float64_dtype(graph_MoS, kwargs, tmp_path):
    """``dtype=torch.float64`` must apply to every layer, including the readout, and survive save/load."""
    structure, graph, _ = graph_MoS
    lat = torch.tensor(structure.lattice.matrix, dtype=torch.float64)
    graph = graph.clone()
    graph.pos = graph.frac_coords.double() @ lat
    graph.pbc_offshift = graph.pbc_offset.double() @ lat
    model = TensorNet(element_types=["Mo", "S"], dtype=torch.float64, use_warp=False, **kwargs)
    assert {p.dtype for p in model.parameters()} == {torch.float64}
    output = model(g=graph)
    assert output.dtype == torch.float64
    model.save(tmp_path)
    loaded = TensorNet.load(tmp_path)
    assert {p.dtype for p in loaded.parameters()} == {torch.float64}
    assert torch.allclose(loaded(g=graph), output)
