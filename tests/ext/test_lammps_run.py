"""End-to-end LAMMPS runs of ``pair_style matgl`` and its Kokkos variant ``matgl/kk``.

These tests drive a real ``lmp`` binary on the decks in ``lammps/tests/`` and compare against matgl's Python
``PESCalculator``. They are skipped unless the environment points at a LAMMPS build:

* ``MATGL_LAMMPS_BIN`` -- path to an ``lmp`` binary built with ``PKG_ML-MATGL=ON``.
* ``MATGL_LAMMPS_KOKKOS`` -- optional. When the binary also has ``PKG_KOKKOS=ON``, the arguments that follow
  ``-k on`` for this build, e.g. ``"t 2"`` (host: Serial/OpenMP) or ``"g 1"`` (CUDA). Adds the ``kk`` runs
  (``-sf kk -pk kokkos neigh half newton on``).

Example (host-only Kokkos build)::

    MATGL_LAMMPS_BIN=$LAMMPS/build/lmp MATGL_LAMMPS_KOKKOS="t 2" uv run pytest tests/ext/test_lammps_run.py
"""

from __future__ import annotations

import importlib.util
import os
import shlex
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch
from pymatgen.core import Lattice, Structure
from pymatgen.io.ase import AseAtomsAdaptor

from matgl.apps.pes import Potential
from matgl.ext.ase import PESCalculator
from matgl.ext.lammps import export_lammps_model
from matgl.models import TensorNet

LMP_BIN = os.environ.get("MATGL_LAMMPS_BIN", "")
KOKKOS_SPEC = os.environ.get("MATGL_LAMMPS_KOKKOS", "").strip()
DECK_DIR = Path(__file__).resolve().parents[2] / "lammps" / "tests"

pytestmark = pytest.mark.skipif(
    not (LMP_BIN and Path(LMP_BIN).is_file()),
    reason="Set MATGL_LAMMPS_BIN to an lmp binary built with PKG_ML-MATGL to run LAMMPS end-to-end tests.",
)

MODES = ["plain", "kk"] if KOKKOS_SPEC else ["plain"]
needs_kokkos = pytest.mark.skipif(not KOKKOS_SPEC, reason="Set MATGL_LAMMPS_KOKKOS to test pair_style matgl/kk.")

A_CUBIC = 4.5  # Angstrom; must match the lattice in the decks.
GPA_TO_BAR = 1.0e4


def _unit_cell() -> Structure:
    """The 4-atom Mo-S cell built by in.matgl_si (basis order = LAMMPS atom ids 1..4)."""
    return Structure(
        Lattice.cubic(A_CUBIC),
        ["Mo", "S", "Mo", "S"],
        [[0.00, 0.00, 0.00], [0.50, 0.50, 0.50], [0.50, 0.00, 0.25], [0.00, 0.50, 0.75]],
    )


@pytest.fixture(scope="module")
def model_artifact(tmp_path_factory) -> tuple[Path, Potential]:
    """Tiny deterministic TensorNet exported as a LAMMPS TorchScript artifact."""
    torch.manual_seed(0)
    model = TensorNet(
        element_types=("Mo", "S"),
        is_intensive=False,
        units=16,
        nblocks=1,
        num_rbf=8,
        cutoff=4.0,
        use_warp=False,
        rbf_type="Gaussian",
    )
    # An untrained TensorNet gives ~1e-3 eV/A forces on this cell; scale the output so the tolerances below
    # are small relative to the signal (forces ~0.1 eV/A, stresses ~1 GPa).
    potential = Potential(model=model, data_std=100.0, calc_forces=True, calc_stresses=True)
    potential.eval()
    path = tmp_path_factory.mktemp("lammps_model") / "model.pt"
    export_lammps_model(potential, str(path), dtype=torch.float32)
    return path, potential


@pytest.fixture(scope="module")
def reference(model_artifact) -> dict[str, np.ndarray | float]:
    """Python (ASE calculator) energy, forces and stress for the unit cell."""
    _, potential = model_artifact
    atoms = AseAtomsAdaptor().get_atoms(_unit_cell())
    atoms.calc = PESCalculator(potential=potential, stress_unit="GPa", use_voigt=False)
    return {
        "energy": float(atoms.get_potential_energy()),
        "forces": np.asarray(atoms.get_forces()),
        "stress_gpa": np.asarray(atoms.get_stress(voigt=False)),
    }


def _lammps_args(mode: str) -> list[str]:
    if mode == "plain":
        return []
    return ["-k", "on", *shlex.split(KOKKOS_SPEC), "-sf", "kk", "-pk", "kokkos", "neigh", "half", "newton", "on"]


def run_lammps(deck: str, workdir: Path, model: Path, mode: str, **variables: float) -> str:
    """Run ``lmp`` on a deck from lammps/tests inside ``workdir`` and return the log text."""
    workdir.mkdir(parents=True, exist_ok=True)
    target = workdir / "model.pt"
    if not target.exists():
        target.symlink_to(model)
    var_args = [arg for name, value in variables.items() for arg in ("-var", name, str(value))]
    cmd = [LMP_BIN, *_lammps_args(mode), *var_args, "-log", "log.lammps", "-in", str(DECK_DIR / deck)]
    proc = subprocess.run(cmd, cwd=workdir, capture_output=True, text=True, timeout=600, check=False)
    assert proc.returncode == 0, f"{shlex.join(cmd)} failed:\n{proc.stdout[-4000:]}\n{proc.stderr[-4000:]}"
    log = (workdir / "log.lammps").read_text()
    if mode == "kk":
        # Guard against silently testing the plain style: matgl/kk logs where it put the model.
        device = "cuda" if shlex.split(KOKKOS_SPEC)[0] == "g" else "cpu"
        assert f"pair_matgl/kk: model on {device}" in log, "pair_style matgl/kk was not used"
    return log


def parse_thermo(log: str) -> list[dict[str, float]]:
    """Return every thermo row of the (last) run in a LAMMPS log as {column: value}."""
    header: list[str] | None = None
    rows: list[dict[str, float]] = []
    for line in log.splitlines():
        fields = line.split()
        if fields and fields[0] == "Step":
            header, rows = fields, []
            continue
        if header is None or len(fields) != len(header):
            continue
        try:
            rows.append(dict(zip(header, map(float, fields), strict=True)))
        except ValueError:
            continue
    assert rows, "no thermo rows found in LAMMPS log"
    return rows


def parse_dump(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (positions, forces) from the single-frame ``forces.dump`` written by in.matgl_si."""
    lines = path.read_text().splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("ITEM: ATOMS"))
    cols = lines[start].split()[2:]
    data = np.array([[float(v) for v in line.split()] for line in lines[start + 1 :] if line.strip()])
    pos = data[:, [cols.index(c) for c in ("x", "y", "z")]]
    forces = data[:, [cols.index(c) for c in ("fx", "fy", "fz")]]
    return pos, forces


def map_to_unit_cell(positions: np.ndarray) -> np.ndarray:
    """Index of the unit-cell basis atom that each (possibly replicated) position is an image of."""
    basis = _unit_cell().frac_coords
    frac = positions / A_CUBIC
    delta = frac[:, None, :] - basis[None, :, :]
    delta -= np.round(delta)
    dist = np.linalg.norm(delta, axis=-1)
    assert np.all(dist.min(axis=1) < 1e-6), "LAMMPS atom does not map onto a unit-cell basis site"
    return dist.argmin(axis=1)


def pressure_tensor_bar(row: dict[str, float]) -> np.ndarray:
    return np.array(
        [
            [row["Pxx"], row["Pxy"], row["Pxz"]],
            [row["Pxy"], row["Pyy"], row["Pyz"]],
            [row["Pxz"], row["Pyz"], row["Pzz"]],
        ]
    )


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("nrep", [1, 2])
def test_single_point_matches_python(model_artifact, reference, tmp_path, mode, nrep):
    """Energy, forces and pressure tensor from LAMMPS match PESCalculator (and are extensive for nrep > 1)."""
    model, _ = model_artifact
    log = run_lammps("in.matgl_si", tmp_path, model, mode, nrep=nrep)
    row = parse_thermo(log)[-1]
    ncell = nrep**3

    assert row["PotEng"] == pytest.approx(ncell * reference["energy"], abs=1e-4 * 4 * ncell)

    positions, forces = parse_dump(tmp_path / "forces.dump")
    assert len(forces) == 4 * ncell
    expected_forces = reference["forces"][map_to_unit_cell(positions)]
    np.testing.assert_allclose(forces, expected_forces, atol=1e-4)

    # ASE/matgl stress is sigma = (1/V) dE/deps (positive in tension); LAMMPS reports pressure P = -sigma.
    # Atoms have zero velocity, so there is no kinetic contribution.
    expected_pressure = -reference["stress_gpa"] * GPA_TO_BAR
    np.testing.assert_allclose(pressure_tensor_bar(row), expected_pressure, rtol=1e-4, atol=5.0)


@pytest.mark.parametrize("mode", MODES)
def test_pressure_matches_finite_difference(model_artifact, tmp_path, mode):
    """Hydrostatic pressure equals -dE/dV from central differences in the lattice scale."""
    spec = importlib.util.spec_from_file_location("fd_pressure", DECK_DIR / "check_finite_difference_pressure.py")
    assert spec is not None
    assert spec.loader is not None
    fd = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fd)

    model, _ = model_artifact
    eps = 0.01
    rows = {}
    for label, scale in (("minus", 1 - eps), ("zero", 1.0), ("plus", 1 + eps)):
        workdir = tmp_path / label
        run_lammps("in.matgl_finite_strain", workdir, model, mode, scale=scale)
        rows[label] = fd.read_thermo(workdir / "log.lammps")

    zero = rows["zero"]
    de_dlambda = (rows["plus"]["PotEng"] - rows["minus"]["PotEng"]) / (2 * eps)
    fd_pressure = -de_dlambda / (3 * zero["Volume"]) * fd.EV_PER_ANGSTROM3_TO_BAR
    assert zero["Press"] == pytest.approx((zero["Pxx"] + zero["Pyy"] + zero["Pzz"]) / 3, rel=1e-10, abs=1e-6)
    assert zero["Press"] == pytest.approx(fd_pressure, rel=0.05, abs=10.0)


@needs_kokkos
@pytest.mark.parametrize("deck", ["in.matgl_si", "in.matgl_finite_strain"])
def test_kokkos_matches_plain(model_artifact, tmp_path, deck):
    """pair_style matgl/kk reproduces pair_style matgl on the same binary."""
    model, _ = model_artifact
    variables = {"scale": 1.0} if deck == "in.matgl_finite_strain" else {"nrep": 2}
    plain = parse_thermo(run_lammps(deck, tmp_path / "plain", model, "plain", **variables))[-1]
    kk = parse_thermo(run_lammps(deck, tmp_path / "kk", model, "kk", **variables))[-1]

    assert kk["PotEng"] == pytest.approx(plain["PotEng"], rel=1e-6, abs=1e-5)
    for key in ("Pxx", "Pyy", "Pzz"):
        assert kk[key] == pytest.approx(plain[key], rel=1e-5, abs=0.1)
    if deck == "in.matgl_si":
        for key in ("Pyz", "Pxz", "Pxy"):
            assert kk[key] == pytest.approx(plain[key], rel=1e-5, abs=0.1)
        _, f_plain = parse_dump(tmp_path / "plain" / "forces.dump")
        _, f_kk = parse_dump(tmp_path / "kk" / "forces.dump")
        np.testing.assert_allclose(f_kk, f_plain, atol=1e-5)


@pytest.mark.parametrize("mode", MODES)
def test_nve_md_conserves_energy(model_artifact, tmp_path, mode):
    """A short NVE run stays finite and conserves total energy."""
    model, _ = model_artifact
    rows = parse_thermo(run_lammps("in.matgl_md", tmp_path, model, mode, nsteps=50))
    assert rows[-1]["Step"] == 50
    etotal = np.array([r["TotEng"] for r in rows])
    assert np.all(np.isfinite(etotal))
    assert np.ptp(etotal) < 1e-3 * 32, f"total energy drift too large: {etotal}"


@needs_kokkos
def test_kokkos_md_matches_plain(model_artifact, tmp_path):
    """A 50-step NVE trajectory under matgl/kk tracks the plain pair_style step by step."""
    model, _ = model_artifact
    plain = parse_thermo(run_lammps("in.matgl_md", tmp_path / "plain", model, "plain", nsteps=50))
    kk = parse_thermo(run_lammps("in.matgl_md", tmp_path / "kk", model, "kk", nsteps=50))
    assert [r["Step"] for r in kk] == [r["Step"] for r in plain]
    for key in ("PotEng", "KinEng", "TotEng"):
        np.testing.assert_allclose([r[key] for r in kk], [r[key] for r in plain], rtol=1e-5, atol=1e-4)
