"""Tests for the shared ``MatGLModel.predict_structure`` / ``structure_to_graph``."""

from __future__ import annotations

import pytest
import torch
from pymatgen.core import Molecule
from pymatgen.io.ase import AseAtomsAdaptor

from matgl.ext.pymatgen import Structure2Graph
from matgl.models import CHGNet, M3GNet, MEGNet, SO3Net, TensorNet
from matgl.models._core import structure_to_graph
from matgl.models._qet import QET

ELEMENTS = ("C", "H", "Mo", "S")


def _models():
    return {
        "M3GNet": lambda: M3GNet(element_types=ELEMENTS, is_intensive=False),
        "TensorNet": lambda: TensorNet(element_types=ELEMENTS, is_intensive=False, use_warp=False),
        "CHGNet": lambda: CHGNet(element_types=ELEMENTS),
        "MEGNet": lambda: MEGNet(element_types=ELEMENTS),
        "SO3Net": lambda: SO3Net(element_types=ELEMENTS, is_intensive=False),
        "QET": lambda: QET(element_types=ELEMENTS, use_warp=False),
    }


@pytest.fixture(params=list(_models()))
def model(request):
    torch.manual_seed(0)
    return _models()[request.param]()


def _kwargs(model):
    return {"total_charge": torch.zeros(1)} if isinstance(model, QET) else {}


def test_predict_structure_crystal_molecule_and_atoms(model, MoS, CH4):
    """The default converter follows the input type: crystals, molecules and ASE Atoms all work."""
    crystal = model.predict_structure(MoS, **_kwargs(model))
    assert torch.isfinite(crystal).all()
    molecule = model.predict_structure(CH4, **_kwargs(model))
    assert torch.isfinite(molecule).all()
    atoms = model.predict_structure(AseAtomsAdaptor.get_atoms(MoS), **_kwargs(model))
    assert torch.allclose(atoms, crystal, atol=1e-5)


def test_predict_structure_matches_explicit_converter(model, MoS):
    converter = Structure2Graph(element_types=ELEMENTS, cutoff=model.cutoff)
    explicit = model.predict_structure(MoS, graph_converter=converter, **_kwargs(model))
    assert torch.allclose(model.predict_structure(MoS, **_kwargs(model)), explicit)


def test_predict_structure_follows_model_dtype(MoS):
    model = TensorNet(element_types=ELEMENTS, is_intensive=False, use_warp=False).double()
    assert model.predict_structure(MoS).dtype == torch.float64


def test_predict_structure_return_features(MoS):
    model = TensorNet(element_types=ELEMENTS, is_intensive=False, use_warp=False)
    with pytest.warns(DeprecationWarning, match="return_features"):
        feats = model.predict_structure(MoS, return_features=True, output_layers=["final", "gc_1"])
    assert set(feats) == {"final", "gc_1"}
    with (
        pytest.raises(ValueError, match="Invalid output_layers"),
        pytest.warns(DeprecationWarning, match="return_features"),
    ):
        model.predict_structure(MoS, return_features=True, output_layers=["not_a_layer"])


def test_structure_to_graph_molecule(CH4):
    model = M3GNet(element_types=ELEMENTS, is_intensive=False)
    g, state = structure_to_graph(model, CH4)
    assert torch.allclose(g.pos, torch.tensor(CH4.cart_coords, dtype=g.pos.dtype))
    assert torch.all(g.pbc_offshift == 0)
    assert state.dtype == next(model.parameters()).dtype
    assert isinstance(CH4, Molecule)
