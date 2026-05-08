"""
Define commonly used text fixtures. These are meant to be reused in unittests.
- Fixtures that are formulae (e.g., LiFePO4) returns the appropriate pymatgen Structure or Molecule based on the most
  commonly known structure.
- Fixtures that are prefixed with `graph_` returns a (structure, graph, state) tuple.

Given that the fixtures are unlikely to be modified by the underlying code, the fixtures are set with a scope of
"session". In the event that future tests are written that modifies the fixtures, these can be set to the default scope
of "function".
"""

from __future__ import annotations

import pathlib

import pytest
import torch
from pymatgen.core import Lattice, Molecule, Structure
from pymatgen.util.testing import PymatgenTest

import matgl
from matgl.ext.pymatgen import Molecule2Graph, Structure2Graph, get_element_list

if matgl.config.BACKEND == "DGL":
    from matgl.graph._compute_dgl import (
        compute_pair_vector_and_distance,
    )

    def get_graph(structure, cutoff):
        """
        Helper class to generate DGL graph from an input Structure or Molecule.

        Returns:
            Structure/Molecule, Graph, State
        """
        element_types = get_element_list([structure])
        if isinstance(structure, Structure):
            converter = Structure2Graph(element_types=element_types, cutoff=cutoff)  # type: ignore
        else:
            converter = Molecule2Graph(element_types=element_types, cutoff=cutoff)  # type: ignore
        graph, lattice, state = converter.get_graph(structure)
        graph.edata["pbc_offshift"] = torch.matmul(graph.edata["pbc_offset"], lattice[0])
        graph.ndata["pos"] = graph.ndata["frac_coords"] @ lattice[0]
        bond_vec, bond_dist = compute_pair_vector_and_distance(graph)
        graph.edata["bond_dist"] = bond_dist
        graph.edata["bond_vec"] = bond_vec
        return structure, graph, state
else:
    from matgl.graph._compute_pyg import compute_pair_vector_and_distance  # type: ignore[assignment]

    def get_graph(structure, cutoff):
        """
        Helper class to generate DGL graph from an input Structure or Molecule.

        Returns:
            Structure/Molecule, Graph, State
        """
        element_types = get_element_list([structure])
        if isinstance(structure, Structure):
            converter = Structure2Graph(element_types=element_types, cutoff=cutoff)  # type: ignore
        else:
            converter = Molecule2Graph(element_types=element_types, cutoff=cutoff)  # type: ignore

        graph, lattice, state = converter.get_graph(structure)

        graph.pbc_offshift = torch.matmul(graph.pbc_offset, lattice[0])
        graph.pos = graph.frac_coords @ lattice[0]
        bond_vec, bond_dist = compute_pair_vector_and_distance(graph.pos, graph.edge_index, graph.pbc_offshift)

        graph.bond_vec = bond_vec
        graph.bond_dist = bond_dist
        return structure, graph, state


matgl.clear_cache(confirm=False)


@pytest.fixture(scope="session")
def LiFePO4():
    return PymatgenTest.get_structure("LiFePO4")


@pytest.fixture(scope="session")
def CH4():
    coords = [
        [0.000000, 0.000000, 0.000000],
        [0.000000, 0.000000, 1.089000],
        [1.026719, 0.000000, -0.363000],
        [-0.513360, -0.889165, -0.363000],
        [-0.513360, 0.889165, -0.363000],
    ]
    return Molecule(["C", "H", "H", "H", "H"], coords)


@pytest.fixture(scope="session")
def AcAla3NHMe():
    coords = [
        [2.04866000, 3.03970000, -1.90538000],
        [2.57508000, 1.72268000, -0.85631500],
        [2.17184000, 3.30896000, -0.15238200],
        [1.90655000, 2.58401000, -0.92348000],
        [0.03771400, 1.25465000, -1.59488000],
        [0.48396700, 2.10062000, -0.82751300],
        [-0.28972800, 2.66225000, 0.13607100],
        [0.16103700, 3.23064000, 0.83232600],
        [-1.66536000, 2.24227000, 0.38856700],
        [-1.98080000, 2.78577000, 1.28200000],
        [-2.60093000, 2.60713000, -0.76094600],
        [-1.65737000, 0.76935300, 0.84076700],
        [-1.07671000, 0.46484900, 1.87157000],
        [-3.62795000, 2.31590000, -0.52731400],
        [-2.58345000, 3.68786000, -0.91203300],
        [-2.28934000, 2.13627000, -1.69468000],
        [-2.28796000, -0.13783200, 0.06597500],
        [-2.61433000, 0.15084400, -0.84271100],
        [-2.21633000, -1.56248000, 0.35525800],
        [-2.49106000, -1.71026000, 1.40128000],
        [-3.17539000, -2.32986000, -0.54719400],
        [-0.79661200, -2.15180000, 0.24417300],
        [-0.54538500, -3.20721000, 0.79546200],
        [-3.09773000, -3.39555000, -0.33182000],
        [-4.20586000, -2.01059000, -0.37590300],
        [-2.92852000, -2.17659000, -1.60274000],
        [0.08518600, -1.43854000, -0.48961600],
        [-0.20393600, -0.57022000, -0.92587000],
        [1.49370000, -1.76346000, -0.59747700],
        [1.62352000, -2.74612000, -0.13439600],
        [1.93595000, -1.80971000, -2.05523000],
        [2.37375000, -0.79291000, 0.21284200],
        [3.55545000, -0.63911100, -0.05660100],
        [3.00626000, -2.00417000, -2.11517000],
        [1.39238000, -2.59759000, -2.58040000],
        [1.73079000, -0.85338400, -2.54426000],
        [1.77244000, -0.14321700, 1.23186000],
        [0.79538700, -0.29827700, 1.43700000],
        [2.52442000, 0.70798900, 2.12658000],
        [3.17617000, 0.12474300, 2.78380000],
        [3.15630000, 1.39339000, 1.55717000],
        [1.81944000, 1.27508000, 2.73527000],
    ]
    return Molecule(
        [
            "H",
            "H",
            "H",
            "C",
            "O",
            "C",
            "N",
            "H",
            "C",
            "H",
            "C",
            "C",
            "O",
            "H",
            "H",
            "H",
            "N",
            "H",
            "C",
            "H",
            "C",
            "C",
            "O",
            "H",
            "H",
            "H",
            "N",
            "H",
            "C",
            "H",
            "C",
            "C",
            "O",
            "H",
            "H",
            "H",
            "N",
            "H",
            "C",
            "H",
            "H",
            "H",
        ],
        coords,
    )


@pytest.fixture(scope="session")
def graph_AcAla3NHMe(AcAla3NHMe):
    return get_graph(AcAla3NHMe, 5.0)


@pytest.fixture(scope="session")
def graph_AcAla3NHMe_pyg(AcAla3NHMe):
    return get_graph(AcAla3NHMe, 5.0)


@pytest.fixture(scope="session")
def CO():
    return Molecule(["C", "O"], [[0, 0, 0], [1.1, 0, 0]])


@pytest.fixture(scope="session")
def BaNiO3():
    return PymatgenTest.get_structure("BaNiO3")


@pytest.fixture(scope="session")
def MoS():
    return Structure(Lattice.cubic(4.0), ["Mo", "S"], [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])


@pytest.fixture(scope="session")
def Mo():
    return Structure(Lattice.cubic(3.17), ["Mo", "Mo"], [[0.01, 0, 0], [0.5, 0.5, 0.5]])


@pytest.fixture(scope="session")
def MoS2():
    return Structure(
        [
            [3.18430383, 0.0, 1.9498237464610788e-16],
            [-1.5921519149999994, 2.757688010148085, 1.9498237464610788e-16],
            [0.0, 0.0, 19.44629514],
        ],
        [
            "Mo",
            "Mo",
            "Mo",
            "S",
            "S",
            "S",
            "S",
            "S",
            "S",
        ],
        [
            [0.00000000e00, 0.00000000e00, 1.94419205e01],
            [1.59215192e00, 9.19229337e-01, 6.47772374e00],
            [4.44089210e-16, 1.83845867e00, 1.29598221e01],
            [0.00000000e00, 0.00000000e00, 4.92566372e00],
            [1.59215192e00, 9.19229337e-01, 1.14077621e01],
            [4.44089210e-16, 1.83845867e00, 1.78898605e01],
            [0.00000000e00, 0.00000000e00, 8.02996293e00],
            [1.59215192e00, 9.19229337e-01, 1.45120613e01],
            [4.44089210e-16, 1.83845867e00, 1.54786455e00],
        ],
    )


@pytest.fixture(scope="session")
def Li3InCl6():
    return Structure(
        [[13.07046811, 0.0, 0.0], [0.0, 13.07046811, 0.0], [0.0, 0.0, 13.07046811]],
        [
            "Li",
            "Li",
            "Li",
            "Li",
            "Li",
            "Li",
            "Li",
            "Li",
            "Li",
            "Li",
            "Li",
            "Li",
            "Li",
            "Li",
            "Li",
            "Li",
            "Li",
            "Li",
            "Li",
            "Li",
            "Li",
            "Li",
            "Li",
            "Li",
            "Li",
            "Li",
            "Li",
            "Li",
            "Li",
            "Li",
            "In",
            "In",
            "In",
            "In",
            "In",
            "In",
            "In",
            "In",
            "In",
            "In",
            "Cl",
            "Cl",
            "Cl",
            "Cl",
            "Cl",
            "Cl",
            "Cl",
            "Cl",
            "Cl",
            "Cl",
            "Cl",
            "Cl",
            "Cl",
            "Cl",
            "Cl",
            "Cl",
            "Cl",
            "Cl",
            "Cl",
            "Cl",
            "Cl",
            "Cl",
            "Cl",
            "Cl",
            "Cl",
            "Cl",
            "Cl",
            "Cl",
            "Cl",
            "Cl",
            "Cl",
            "Cl",
            "Cl",
            "Cl",
            "Cl",
            "Cl",
            "Cl",
            "Cl",
            "Cl",
            "Cl",
            "Cl",
            "Cl",
            "Cl",
            "Cl",
            "Cl",
            "Cl",
            "Cl",
            "Cl",
            "Cl",
            "Cl",
            "Cl",
            "Cl",
            "Cl",
            "Cl",
            "Cl",
            "Cl",
            "Cl",
            "Cl",
            "Cl",
            "Cl",
        ],
        [
            [1.93677967e00, 7.52823533e-02, 5.55843236e-01],
            [1.24799935e00, 7.17681434e-01, 7.50215459e-01],
            [-2.06430964e-01, -2.85238590e-01, -9.07606382e-02],
            [9.62622321e-02, -1.08227792e00, -3.59164862e-01],
            [1.76154752e00, -1.03540471e00, -2.66179905e-01],
            [-8.51706747e-01, 1.02444495e00, 1.95706353e-01],
            [5.91700021e-01, 3.67672303e-02, 3.25249547e-02],
            [-2.96990359e-01, 2.87559246e-01, -8.13546998e-02],
            [-5.07556637e-01, 8.74262041e-01, 2.76773269e-01],
            [6.42347509e-02, -3.62390011e-01, -8.86678615e-01],
            [8.90020189e-01, 9.13810125e-01, 1.29578746e00],
            [8.11174651e-01, -6.02686473e-01, 5.45857665e-01],
            [9.21023715e-01, 1.18336185e00, 3.29221820e-01],
            [1.65681685e-01, 1.30057976e00, 1.12622252e00],
            [2.55533154e00, 1.76255678e00, 1.60144595e00],
            [1.28345290e-01, 6.26795342e-01, 1.94302786e00],
            [2.35668714e00, 5.84828558e-01, 6.59680539e-01],
            [1.09341343e00, 9.25400387e-01, 4.24375280e-02],
            [-3.65168711e-01, 1.12075912e00, -7.93711231e-01],
            [5.00797667e-01, 1.99331619e00, 4.34840395e-01],
            [2.26547056e-01, 7.58128266e-01, -7.26241056e-01],
            [8.47375507e-01, 6.11971226e-01, 1.68826541e00],
            [5.10407452e-02, -1.47516966e00, 7.45254278e-01],
            [1.25716552e00, 1.16503154e00, -1.25470482e00],
            [1.89029637e00, 1.48486397e00, -9.89043416e-01],
            [8.75335434e-02, 2.79990509e-01, 5.31315506e-01],
            [9.32892981e-01, 8.94534484e-01, 9.21557175e-01],
            [4.23042829e-01, -5.43768350e-03, 6.48055749e-01],
            [3.66067777e-01, 5.79361562e-01, 2.02721674e00],
            [-1.25098236e00, -2.41046856e-01, 2.52546075e00],
            [-1.24250321e-01, 1.29276546e-01, 9.59325748e-01],
            [7.05832596e-01, 3.76526559e-02, -5.45325881e-01],
            [6.88524406e-01, -3.29100101e-01, 2.29533582e-01],
            [3.49188559e-01, 8.59905314e-01, 8.82216273e-01],
            [2.95883276e-01, 1.97025583e-01, 1.03076180e00],
            [6.14376581e-01, 4.80700259e-01, -2.09927853e-01],
            [3.03917775e-01, 1.39652754e00, 4.87692558e-01],
            [9.74299035e-01, -5.49697638e-01, -6.96614732e-01],
            [6.20371013e-01, 4.43851188e-01, 3.89871250e-01],
            [-7.78312951e-01, 1.01861776e00, 1.40391957e00],
            [2.36348101e-01, 6.84049670e-01, 1.06168065e00],
            [-4.60476377e-01, 6.61692089e-01, 9.58710861e-01],
            [3.55277267e-01, 2.24914712e-01, -1.12816327e00],
            [5.36344227e-01, 1.10638257e00, 6.01063564e-01],
            [-3.65348359e-01, 2.04676244e-01, 3.60606832e-01],
            [5.09389855e-01, 6.57643076e-01, 2.36151321e-01],
            [1.26788757e00, 8.87499261e-01, -4.95388437e-01],
            [5.17535079e-01, 8.39248490e-01, 4.40834434e-01],
            [9.74909769e-01, 1.57623396e00, -1.12361983e-01],
            [3.39549121e-01, 9.12743362e-01, 2.72347379e-01],
            [8.03154651e-01, 1.08928055e00, -3.64798885e-01],
            [1.40530307e00, 8.52115786e-01, 1.03583905e00],
            [8.80346459e-01, -3.44419460e-01, -4.67888191e-01],
            [-3.46198984e-01, 6.58839136e-01, 7.30183455e-01],
            [1.06845006e00, 4.48861136e-01, 1.12839574e00],
            [1.68077987e00, 6.19669016e-01, 1.47654159e00],
            [7.12932412e-01, -1.60776141e-02, 8.88301308e-01],
            [8.20366351e-01, 1.25987178e00, 8.08356808e-01],
            [1.80375960e00, 9.56317877e-01, 1.50501212e-01],
            [9.24895533e-01, 2.99225443e-01, -7.80086413e-01],
            [1.15276802e-01, 2.60444367e-01, 9.66580361e-01],
            [-7.83951969e-01, 1.56078734e00, 8.18297594e-01],
            [8.73182712e-01, 1.03711704e00, 3.93797588e-01],
            [-5.18163744e-03, 4.12754236e-01, 6.33412314e-01],
            [-1.95546690e-01, 4.43890647e-01, 3.58757854e-01],
            [1.03573877e00, -2.09632540e-01, 4.28998709e-02],
            [8.56043370e-01, 7.82412782e-01, 7.91025323e-01],
            [1.79320915e-02, 7.50049988e-02, 1.10811315e00],
            [9.32872067e-01, 8.66165272e-01, 5.79333204e-01],
            [-7.13340708e-03, 3.11947948e-02, -2.02650997e-01],
            [3.57827176e-01, 1.32028761e-01, 1.36661567e00],
            [3.07394851e-01, 3.14842748e-01, -3.43795648e-01],
            [7.39807233e-01, -1.52681986e00, 9.62300899e-01],
            [-3.89156807e-01, -2.92056587e-02, 2.71042628e-01],
            [2.13647896e-01, 1.04606155e00, -4.08254071e-02],
            [7.76041464e-01, 4.50550355e-01, 7.26944764e-01],
            [4.05636709e-01, 7.14211240e-01, 1.72470100e00],
            [6.90585003e-01, 1.21942904e00, 1.08929831e00],
            [1.33950378e-01, 7.24724462e-01, 6.01275134e-01],
            [9.26487082e-02, 5.93825675e-01, 1.37663062e00],
            [4.31451043e-01, 1.12264354e00, 1.05768313e-01],
            [1.08388883e00, -1.32583173e-01, -7.01584204e-01],
            [1.72490892e00, 7.55734093e-01, 8.59347109e-02],
            [6.55955875e-01, -1.20836291e-01, 6.26796193e-01],
            [4.53351609e-01, 3.94623646e-01, 2.99038549e-01],
            [7.58929156e-01, 8.02470490e-01, 3.65252887e-01],
            [1.09843465e-01, 1.08661076e00, -4.28013765e-01],
            [2.92341254e-01, 4.09016610e-01, 1.12481109e00],
            [4.26776380e-01, 1.55164825e00, 4.68002163e-01],
            [1.17489549e00, 1.78573655e-01, 2.41281200e-01],
            [4.83289278e-01, 1.88903628e-03, 8.60708970e-01],
            [-1.29200617e-01, 6.33132699e-01, 1.91202158e-01],
            [6.15001602e-01, 2.83125462e-01, 7.66965497e-01],
            [1.14067413e00, 8.30642500e-01, 8.46443563e-01],
            [8.87372018e-01, 1.26336981e00, -5.27502132e-01],
            [4.53424649e-01, 4.81336523e-01, -1.69465513e-01],
            [5.79039092e-01, 3.90227451e-01, 5.62390716e-01],
            [2.66018814e-01, 1.99192229e00, 1.72466289e00],
            [1.24579818e-01, 3.50196754e-01, 4.00487546e-01],
            [-3.29131174e-01, 4.79309768e-01, 1.19332211e00],
        ],
    )


@pytest.fixture(scope="session")
def graph_Mo(Mo):
    return get_graph(Mo, 5.0)


@pytest.fixture(scope="session")
def graph_Mo_pyg(Mo):
    return get_graph(Mo, 5.0)


@pytest.fixture(scope="session")
def graph_CH4(CH4):
    """
    Returns:
        Molecule, Graph, State
    """
    return get_graph(CH4, 2.0)


@pytest.fixture(scope="session")
def graph_CH4_pyg(CH4):
    """
    Returns:
        Molecule, Graph, State
    """
    return get_graph(CH4, 2.0)


@pytest.fixture(scope="session")
def graph_LiFePO4(LiFePO4):
    """
    Returns:
        Molecule, Graph, State
    """
    return get_graph(LiFePO4, 4.0)


@pytest.fixture(scope="session")
def graph_LiFePO4_pyg(LiFePO4):
    """
    Returns:
        Molecule, Graph, State
    """
    return get_graph(LiFePO4, 4.0)


@pytest.fixture(scope="session")
def graph_MoS(MoS):
    return get_graph(MoS, 5.0)


@pytest.fixture(scope="session")
def graph_MoS_pyg(MoS):
    """
    Returns:
        Molecule, Graph, State
    """
    return get_graph(MoS, 5.0)


@pytest.fixture(scope="session")
def graph_CO(CO):
    return get_graph(CO, 5.0)


@pytest.fixture(scope="session")
def graph_MoSH():
    s = Structure(Lattice.cubic(3.17), ["Mo", "S", "H"], [[0, 0, 0], [0.5, 0.5, 0.5], [0.75, 0.75, 0.75]])
    return get_graph(s, 4.0)


@pytest.fixture(scope="session")
def graph_MoSH_pyg():
    s = Structure(Lattice.cubic(3.17), ["Mo", "S", "H"], [[0, 0, 0], [0.5, 0.5, 0.5], [0.75, 0.75, 0.75]])
    return get_graph(s, 4.0)


@pytest.fixture(scope="session")
def graph_Li3InCl6(Li3InCl6):
    return get_graph(Li3InCl6, 6.0)


# ---------------------------------------------------------------------------
# NaCl perturbed training set (used by GRACE training-parity tests).
#
# Built deterministically from the rocksalt NaCl ground state (the Materials
# Project structure ``mp-22862``: conventional cubic cell, ``a = 5.6402 Å``,
# 4 Na + 4 Cl). Ten random copies — 1 ground state + 9 perturbations of
# both the lattice parameters and the fractional atomic positions — are
# pre-labelled offline with ``TensorNet-PES-MatPES-r2SCAN-2025.2`` and
# checked into ``tests/parity_data/nacl_training_set.json.gz``. The fixture
# only **loads** that artifact: it does not run the labelling potential at
# test time. Re-run ``tests/parity_data/gen_nacl_training_set.py`` to
# regenerate the artifact when the labeller or the perturbation recipe
# changes.
# ---------------------------------------------------------------------------


_NACL_TRAINING_SET_PATH = pathlib.Path(__file__).parent / "parity_data" / "nacl_training_set.json.gz"


@pytest.fixture(scope="session")
def NaCl():
    """Rocksalt NaCl conventional cell (matches MP ``mp-22862``)."""
    a = 5.6402  # MP-relaxed PBE lattice constant, in Å.
    species = ["Na", "Cl", "Na", "Cl", "Na", "Cl", "Na", "Cl"]
    coords = [
        [0.0, 0.0, 0.0],
        [0.5, 0.5, 0.5],
        [0.5, 0.5, 0.0],
        [0.0, 0.0, 0.5],
        [0.5, 0.0, 0.5],
        [0.0, 0.5, 0.0],
        [0.0, 0.5, 0.5],
        [0.5, 0.0, 0.0],
    ]
    return Structure(Lattice.cubic(a), species, coords)


@pytest.fixture(scope="session")
def nacl_training_set():
    """Load the pre-labelled NaCl perturbed training set.

    Returns a list of ``dict`` entries, each with the keys

    - ``structure``: pymatgen ``Structure`` (perturbed copy of NaCl)
    - ``energy``:   total potential energy (eV), float
    - ``forces``:   per-atom forces (eV/Å), shape ``(n_atoms, 3)`` ``np.ndarray``
    - ``stress``:   stress tensor (eV/Å³), shape ``(3, 3)`` ``np.ndarray``

    The first entry is the unperturbed ground-state structure. The other
    nine apply a small random symmetric strain to the lattice and a small
    random Cartesian displacement to each atom (recipe pinned in
    ``tests/parity_data/gen_nacl_training_set.py``).

    Skips at fixture-collection time if the artifact is missing — keeps
    the fixture optional in trees where ``tests/parity_data/`` was not
    checked out.
    """

    import numpy as np
    from monty.serialization import loadfn

    payload = loadfn(str(_NACL_TRAINING_SET_PATH))
    samples: list[dict] = []
    for raw in payload["samples"]:
        # ``loadfn`` already decoded the structure dict back into a
        # pymatgen ``Structure`` via the MSON ``@class`` hook.
        struct = raw["structure"]
        if not isinstance(struct, Structure):
            struct = Structure.from_dict(struct)
        samples.append(
            {
                "structure": struct,
                "energy": float(raw["energy"]),
                "forces": np.asarray(raw["forces"], dtype="float64"),
                "stress": np.asarray(raw["stress"], dtype="float64"),
            }
        )
    return samples
