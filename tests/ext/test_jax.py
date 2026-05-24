"""JAX vs PyTorch parity tests for the ``matgl.ext.jax`` inference backend.

Covers TensorNet and QET. All comparisons run in float64 (``jax_enable_x64``) so a
mismatch flags a real algorithmic divergence rather than float32 accumulation
noise. Skipped unless the optional ``jax`` dependency is installed.
"""

from __future__ import annotations

import pytest

jax = pytest.importorskip("jax", reason="matgl.ext.jax requires the optional 'jax' dependency")
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from pymatgen.core import Lattice, Structure  # noqa: E402

from matgl.apps.pes import Potential  # noqa: E402
from matgl.config import DEFAULT_ELEMENTS  # noqa: E402
from matgl.ext._pymatgen import Structure2Graph  # noqa: E402
from matgl.ext.jax import JAXPESCalculator, convert_potential, make_potential_fn  # noqa: E402
from matgl.ext.jax._pad import pad_graph  # noqa: E402
from matgl.models import QET, TensorNet  # noqa: E402

CUTOFF = 5.0

TN_STRUCTURES = {
    "Si2": Structure(Lattice.cubic(3.45), ["Si", "Si"], [[0, 0, 0], [0.5, 0.5, 0.5]]),
    "NaCl-sc": Structure(Lattice.cubic(5.64), ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]]),
    "GaAs": Structure(Lattice.cubic(5.65), ["Ga", "As"], [[0, 0, 0], [0.25, 0.25, 0.25]]),
}
TN_STRUCTURES["NaCl-sc"].make_supercell([2, 2, 2])

TN_CONFIGS = {
    "sb-smooth-extensive": {
        "rbf_type": "SphericalBessel",
        "use_smooth": True,
        "max_n": 8,
        "max_l": 3,
        "is_intensive": False,
    },
    "gaussian-extensive": {"rbf_type": "Gaussian", "num_rbf": 16, "is_intensive": False},
}

QET_STRUCTURES = {
    "GaAs": Structure(Lattice.cubic(5.65), ["Ga", "As"], [[0, 0, 0], [0.25, 0.25, 0.25]]),
    "Mo-single": Structure(Lattice.cubic(3.15), ["Mo"], [[0, 0, 0]]),
}

QET_CONFIGS = {
    "gaussian-default": {"rbf_type": "Gaussian", "num_rbf": 16},
    "sb-smooth": {"rbf_type": "SphericalBessel", "use_smooth": True, "max_n": 8, "max_l": 3},
    "hardness-envs": {"rbf_type": "Gaussian", "num_rbf": 16, "is_hardness_envs": True},
    "sigma-train": {"rbf_type": "Gaussian", "num_rbf": 16, "is_sigma_train": True},
    "with-magmom": {"rbf_type": "Gaussian", "num_rbf": 16, "include_magmom": True},
}


def _build(struct, cfg_kwargs, seed=0):
    """Build a torch Potential (float64) + the input graph for one (config, structure)."""
    torch.manual_seed(seed)
    model = TensorNet(element_types=DEFAULT_ELEMENTS, units=32, nblocks=2, cutoff=CUTOFF, use_warp=False, **cfg_kwargs)
    model.eval()
    rng = np.random.default_rng(seed)
    element_refs = rng.standard_normal(len(DEFAULT_ELEMENTS))
    potential = Potential(
        model=model,
        data_mean=0.37,
        data_std=0.81,
        element_refs=element_refs,
        calc_forces=True,
        calc_stresses=True,
    )
    potential.eval()
    potential.double()

    conv = Structure2Graph(DEFAULT_ELEMENTS, CUTOFF)
    g, lat, _ = conv.get_graph(struct)
    return potential, g, lat.double()


def _torch_outputs(potential, g, lat):
    e, f, s, _ = potential(g, lat)
    return float(e), f.detach().numpy(), s.detach().numpy()


def _jax_inputs(g, lat, pad_to=None):
    lat3 = jnp.asarray(lat[0].detach().numpy())
    frac = jnp.asarray(g.frac_coords.double().numpy())
    pos = frac @ lat3
    z = jnp.asarray(g.node_type.numpy())
    edge_index = jnp.asarray(g.edge_index.numpy())
    pbc_offset = jnp.asarray(g.pbc_offset.double().numpy())
    n = z.shape[0]
    batch = jnp.zeros(n, dtype=jnp.int32)
    strain = jnp.zeros((3, 3))
    if pad_to is not None:
        edge_index, pbc_offset, edge_mask = pad_graph(edge_index, pbc_offset, pad_to)
    else:
        edge_mask = jnp.ones(edge_index.shape[1])
    return pos, strain, frac, lat3, pbc_offset, z, edge_index, batch, edge_mask


@pytest.mark.parametrize("struct_name", list(TN_STRUCTURES))
@pytest.mark.parametrize("cfg_name", list(TN_CONFIGS))
def test_energy_forces_stress_parity(struct_name, cfg_name):
    potential, g, lat = _build(TN_STRUCTURES[struct_name], TN_CONFIGS[cfg_name])
    e_t, f_t, s_t = _torch_outputs(potential, g, lat)

    params, cfg, extras = convert_potential(potential)
    fn = make_potential_fn(params, cfg, extras, num_graphs=1)
    e_j, f_j, s_j = fn(*_jax_inputs(g, lat))
    e_j, f_j, s_j = float(e_j), np.asarray(f_j), np.asarray(s_j)

    assert abs(e_t - e_j) < 1e-6, f"energy: torch={e_t} jax={e_j}"
    assert np.abs(f_t - f_j).max() < 1e-6, f"forces max diff {np.abs(f_t - f_j).max():.2e}"
    assert np.abs(s_t - s_j).max() < 1e-6, f"stress max diff {np.abs(s_t - s_j).max():.2e}"


@pytest.mark.parametrize("struct_name", list(TN_STRUCTURES))
def test_padding_invariance(struct_name):
    """Padded (sentinel) edges must contribute exactly zero — no NaN leak."""
    potential, g, lat = _build(TN_STRUCTURES[struct_name], TN_CONFIGS["sb-smooth-extensive"])
    params, cfg, extras = convert_potential(potential)
    fn = make_potential_fn(params, cfg, extras, num_graphs=1)

    n_edges = g.edge_index.shape[1]
    e0, f0, s0 = fn(*_jax_inputs(g, lat))
    e1, f1, s1 = fn(*_jax_inputs(g, lat, pad_to=n_edges + 257))
    e2, f2, s2 = fn(*_jax_inputs(g, lat, pad_to=n_edges + 1024))

    for e, f, s in [(e1, f1, s1), (e2, f2, s2)]:
        assert np.isfinite(np.asarray(e)).all()
        assert abs(float(e0) - float(e)) < 1e-8
        assert np.abs(np.asarray(f0) - np.asarray(f)).max() < 1e-8
        assert np.abs(np.asarray(s0) - np.asarray(s)).max() < 1e-8


def test_converts_warp_fused_distance_proj():
    """A Warp-enabled TensorNet fuses distance_proj1/2/3 into one distance_proj.

    The Warp ``TensorEmbedding`` registers a single ``Linear(rbf, 3*units)``
    instead of three ``Linear(rbf, units)``; its ``state_dict`` therefore keys
    that layer as ``distance_proj`` (Warp itself is not installable here, but its
    fused weight is exactly the row-concatenation of the three PyG layers).
    ``convert_potential`` must accept that layout and yield identical output.
    """
    potential, g, lat = _build(TN_STRUCTURES["GaAs"], TN_CONFIGS["sb-smooth-extensive"])

    params0, cfg0, extras0 = convert_potential(potential)
    e0, f0, s0 = make_potential_fn(params0, cfg0, extras0, num_graphs=1)(*_jax_inputs(g, lat))

    # Rewrite the embedding to the fused Warp layout.
    emb = potential.model.tensor_embedding
    fused = torch.nn.Linear(emb.distance_proj1.in_features, 3 * emb.distance_proj1.out_features).double()
    with torch.no_grad():
        fused.weight.copy_(
            torch.cat([emb.distance_proj1.weight, emb.distance_proj2.weight, emb.distance_proj3.weight], dim=0)
        )
        fused.bias.copy_(torch.cat([emb.distance_proj1.bias, emb.distance_proj2.bias, emb.distance_proj3.bias], dim=0))
    del emb.distance_proj1, emb.distance_proj2, emb.distance_proj3
    emb.distance_proj = fused

    sd = potential.model.state_dict()
    assert "tensor_embedding.distance_proj.weight" in sd
    assert "tensor_embedding.distance_proj1.weight" not in sd

    params1, cfg1, extras1 = convert_potential(potential)
    e1, f1, s1 = make_potential_fn(params1, cfg1, extras1, num_graphs=1)(*_jax_inputs(g, lat))

    assert abs(float(e0) - float(e1)) < 1e-9
    assert np.abs(np.asarray(f0) - np.asarray(f1)).max() < 1e-9
    assert np.abs(np.asarray(s0) - np.asarray(s1)).max() < 1e-9


def test_calculator_matches_pescalculator():
    """JAXPESCalculator (float32) tracks matgl's PESCalculator within float32 noise."""
    from ase.calculators.calculator import all_changes
    from pymatgen.io.ase import AseAtomsAdaptor

    from matgl.ext.ase import PESCalculator

    torch.manual_seed(1)
    model = TensorNet(
        element_types=DEFAULT_ELEMENTS,
        units=32,
        nblocks=2,
        cutoff=CUTOFF,
        use_warp=False,
        rbf_type="SphericalBessel",
        use_smooth=True,
        max_n=8,
        max_l=3,
        is_intensive=False,
    )
    model.eval()
    potential = Potential(model=model, data_mean=0.1, data_std=0.9, calc_forces=True, calc_stresses=True)
    potential.eval()  # float32

    atoms = AseAtomsAdaptor.get_atoms(TN_STRUCTURES["NaCl-sc"])
    ref = PESCalculator(potential, stress_unit="GPa")
    jax_calc = JAXPESCalculator(potential, stress_unit="GPa")

    ref.calculate(atoms, ["energy"], all_changes)
    jax_calc.calculate(atoms, ["energy"], all_changes)

    de = abs(ref.results["energy"] - jax_calc.results["energy"])
    df = np.abs(ref.results["forces"] - jax_calc.results["forces"]).max()
    ds = np.abs(ref.results["stress"] - jax_calc.results["stress"]).max()
    assert de < 5e-3, f"energy diff {de:.2e}"
    assert df < 5e-3, f"forces diff {df:.2e}"
    assert ds < 5e-2, f"stress diff {ds:.2e}"


def test_calculator_runs_md():
    """A short NVT MD with JAXPESCalculator runs and stays finite."""
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
    from ase.md.verlet import VelocityVerlet
    from pymatgen.io.ase import AseAtomsAdaptor

    torch.manual_seed(2)
    model = TensorNet(
        element_types=DEFAULT_ELEMENTS,
        units=32,
        nblocks=2,
        cutoff=CUTOFF,
        use_warp=False,
        rbf_type="SphericalBessel",
        use_smooth=True,
        max_n=8,
        max_l=3,
        is_intensive=False,
    )
    model.eval()
    potential = Potential(model=model, calc_forces=True, calc_stresses=True)
    potential.eval()

    atoms = AseAtomsAdaptor.get_atoms(TN_STRUCTURES["Si2"])
    atoms.calc = JAXPESCalculator(potential, stress_unit="eV/A3")
    MaxwellBoltzmannDistribution(atoms, temperature_K=300)
    VelocityVerlet(atoms, timestep=1.0).run(10)
    assert np.isfinite(atoms.get_potential_energy())
    assert np.isfinite(atoms.get_forces()).all()


@pytest.mark.parametrize("struct_name", list(QET_STRUCTURES))
@pytest.mark.parametrize("cfg_name", list(QET_CONFIGS))
def test_qet_energy_forces_stress_parity(struct_name, cfg_name):
    torch.manual_seed(0)
    model = QET(
        element_types=DEFAULT_ELEMENTS, units=32, nblocks=2, cutoff=CUTOFF, use_warp=False, **QET_CONFIGS[cfg_name]
    )
    model.eval()
    potential = Potential(model=model, data_mean=0.21, data_std=0.74, calc_forces=True, calc_stresses=True)
    potential.eval()
    potential.double()

    conv = Structure2Graph(DEFAULT_ELEMENTS, CUTOFF)
    g, lat, _ = conv.get_graph(QET_STRUCTURES[struct_name])
    lat = lat.double()
    e_t, f_t, s_t, _ = potential(g, lat)
    e_t, f_t, s_t = float(e_t), f_t.detach().numpy(), s_t.detach().numpy()

    params, cfg, extras = convert_potential(potential)
    fn = make_potential_fn(params, cfg, extras, num_graphs=1)
    e_j, f_j, s_j = fn(*_jax_inputs(g, lat))
    e_j, f_j, s_j = float(e_j), np.asarray(f_j), np.asarray(s_j)

    assert np.isfinite(e_j)
    assert np.isfinite(f_j).all()
    assert np.isfinite(s_j).all()
    assert abs(e_t - e_j) < 1e-6, f"energy: torch={e_t} jax={e_j}"
    assert np.abs(f_t - f_j).max() < 1e-6, f"forces max diff {np.abs(f_t - f_j).max():.2e}"
    assert np.abs(s_t - s_j).max() < 1e-6, f"stress max diff {np.abs(s_t - s_j).max():.2e}"
