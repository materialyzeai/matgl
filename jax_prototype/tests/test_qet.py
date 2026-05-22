"""JAX vs PyTorch parity for the QET model (charge-equilibration head).

Run with:  .venv/bin/python -m pytest jax_prototype/tests/test_qet.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402
from matgl.apps.pes import Potential  # noqa: E402
from matgl.config import DEFAULT_ELEMENTS  # noqa: E402
from matgl.ext._pymatgen_pyg import Structure2Graph  # noqa: E402
from matgl.models import QET  # noqa: E402
from pymatgen.core import Lattice, Structure  # noqa: E402

sys.path.insert(0, str(Path(__file__).parents[1]))
from matgl_jax._convert import convert_potential  # noqa: E402
from matgl_jax._potential import make_potential_fn  # noqa: E402

CUTOFF = 5.0

STRUCTURES = {
    "GaAs": Structure(Lattice.cubic(5.65), ["Ga", "As"], [[0, 0, 0], [0.25, 0.25, 0.25]]),
    "Mo-single": Structure(Lattice.cubic(3.15), ["Mo"], [[0, 0, 0]]),
}

CONFIGS = {
    "gaussian-default": {"rbf_type": "Gaussian", "num_rbf": 16},
    "sb-smooth": {"rbf_type": "SphericalBessel", "use_smooth": True, "max_n": 8, "max_l": 3},
    "hardness-envs": {"rbf_type": "Gaussian", "num_rbf": 16, "is_hardness_envs": True},
    "sigma-train": {"rbf_type": "Gaussian", "num_rbf": 16, "is_sigma_train": True},
    "with-magmom": {"rbf_type": "Gaussian", "num_rbf": 16, "include_magmom": True},
}


def _jax_inputs(g, lat):
    lat3 = jnp.asarray(lat[0].detach().numpy())
    frac = jnp.asarray(g.frac_coords.double().numpy())
    pos = frac @ lat3
    z = jnp.asarray(g.node_type.numpy())
    edge_index = jnp.asarray(g.edge_index.numpy())
    pbc_offset = jnp.asarray(g.pbc_offset.double().numpy())
    n = z.shape[0]
    return (
        pos,
        jnp.zeros((3, 3)),
        frac,
        lat3,
        pbc_offset,
        z,
        edge_index,
        jnp.zeros(n, dtype=jnp.int32),
        jnp.ones(edge_index.shape[1]),
    )


@pytest.mark.parametrize("struct_name", list(STRUCTURES))
@pytest.mark.parametrize("cfg_name", list(CONFIGS))
def test_qet_energy_forces_stress_parity(struct_name, cfg_name):
    torch.manual_seed(0)
    model = QET(element_types=DEFAULT_ELEMENTS, units=32, nblocks=2, cutoff=CUTOFF, use_warp=False, **CONFIGS[cfg_name])
    model.eval()
    potential = Potential(model=model, data_mean=0.21, data_std=0.74, calc_forces=True, calc_stresses=True)
    potential.eval()
    potential.double()

    conv = Structure2Graph(DEFAULT_ELEMENTS, CUTOFF)
    g, lat, _ = conv.get_graph(STRUCTURES[struct_name])
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


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
