from __future__ import annotations

from argparse import Namespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from pymatgen.core import Lattice, Structure

from matgl import cli


@pytest.fixture
def tiny_structure():
    return Structure(Lattice.cubic(3.5), ["Mo", "Mo"], [[0, 0, 0], [0.5, 0.5, 0.5]])


@pytest.fixture
def tiny_cif(tmp_path, tiny_structure):
    path = tmp_path / "Mo.cif"
    tiny_structure.to(filename=str(path))
    return path


@pytest.fixture
def fake_potential():
    pot = MagicMock(name="Potential")
    pot.predict_structure.return_value = torch.tensor([-1.234])
    return pot


def _patch_relaxer(final_structure):
    """Return a context manager that patches Relaxer + load_model in cli."""
    relax_results = {"final_structure": final_structure}
    fake_relaxer = MagicMock()
    fake_relaxer.relax.return_value = relax_results
    return patch.multiple(
        cli,
        _load_potential=MagicMock(return_value=MagicMock(name="Potential")),
        Relaxer=MagicMock(return_value=fake_relaxer),
    )


def test_format_lattice_delta(tiny_structure):
    """_format_lattice_delta yields exactly six 'param: a -> b' strings."""
    lines = list(cli._format_lattice_delta(tiny_structure.lattice, tiny_structure.lattice))
    assert len(lines) == 6
    assert all("->" in line for line in lines)
    # Order: a, b, c, alpha, beta, gamma
    assert lines[0].startswith("a:")
    assert lines[5].startswith("gamma:")


def test_format_site_delta(tiny_structure):
    formatter = lambda fc: np.array2string(fc, formatter={"float_kind": lambda x: f"{x:.3f}"})  # noqa: E731
    out = cli._format_site_delta(formatter, tiny_structure[0], tiny_structure[1])
    assert "->" in out
    assert "Mo" in out


def test_resolve_state_attributes_errors():
    with pytest.raises(ValueError, match="must be supplied"):
        cli._resolve_state_attributes(None, 1)
    with pytest.raises(ValueError, match="must match"):
        cli._resolve_state_attributes(["1"], 2)


def test_resolve_state_attributes_coerces_to_int():
    assert cli._resolve_state_attributes(["0", "1", "2"], 3) == [0, 1, 2]


def test_configure_logging_verbose():
    """_configure_logging only installs handlers when verbose=True."""
    import logging

    root = logging.getLogger()
    saved = list(root.handlers)
    try:
        root.handlers = []
        cli._configure_logging(False)
        assert root.handlers == []
        cli._configure_logging(True)
        # logging.basicConfig is excluded from coverage but should have run.
    finally:
        root.handlers = saved


def test_load_potential_calls_matgl(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(cli.matgl, "load_model", lambda name: sentinel)
    assert cli._load_potential("anything") is sentinel


def test_relax_structure_outfile(tiny_cif, tiny_structure, tmp_path):
    """outfile branch writes to the specified path."""
    out = tmp_path / "out.cif"
    args = Namespace(infile=[str(tiny_cif)], model="x", verbose=False, suffix=None, outfile=str(out))
    with _patch_relaxer(tiny_structure):
        assert cli.relax_structure(args) == 0
    assert out.exists()


def test_relax_structure_suffix(tiny_cif, tiny_structure):
    """suffix branch writes alongside the input with a suffix."""
    args = Namespace(infile=[str(tiny_cif)], model="x", verbose=False, suffix="_relaxed", outfile=None)
    with _patch_relaxer(tiny_structure):
        assert cli.relax_structure(args) == 0
    expected = tiny_cif.with_name(tiny_cif.stem + "_relaxed" + tiny_cif.suffix)
    assert expected.exists()


def test_relax_structure_stdout(tiny_cif, tiny_structure, capsys):
    """No suffix and no outfile: report lattice + per-site deltas to stdout."""
    args = Namespace(infile=[str(tiny_cif)], model="x", verbose=True, suffix=None, outfile=None)
    with _patch_relaxer(tiny_structure):
        assert cli.relax_structure(args) == 0
    captured = capsys.readouterr().out
    assert "Lattice parameters" in captured
    assert "Sites (Fractional coordinates)" in captured
    assert "->" in captured


def test_predict_structure_eform(tiny_cif, fake_potential, capsys):
    """Eform model branch: prints prediction per file."""
    with patch.object(cli, "_load_potential", return_value=fake_potential):
        cli.predict_structure(Namespace(model="EformModel", infile=[str(tiny_cif)], mpids=None, state_attr=None))
    captured = capsys.readouterr().out
    assert "EformModel prediction" in captured
    assert "eV/atom" in captured
    fake_potential.predict_structure.assert_called_once()


def test_predict_structure_bandgap_with_state(tiny_cif, fake_potential, capsys):
    """Bandgap model branch: state attribute selects the functional label."""
    with patch.object(cli, "_load_potential", return_value=fake_potential):
        cli.predict_structure(
            Namespace(
                model="MEGNet-MP-2019.4.1-BandGap-mfi",
                infile=[str(tiny_cif)],
                mpids=None,
                state_attr=["1"],
            )
        )
    captured = capsys.readouterr().out
    # state_attr=1 maps to "GLLB-SC".
    assert "GLLB-SC" in captured
    assert "eV" in captured


def test_predict_structure_with_mpids(fake_potential, capsys, tiny_structure):
    """mpids branch: pulls structures via MPRester and prints predictions.

    ``MPRester`` is lazily imported inside ``predict_structure`` so we install a
    fake ``pymatgen.ext.matproj`` module in ``sys.modules`` rather than patching
    on the cli module directly.
    """
    import sys
    import types

    fake_mpr = MagicMock()
    fake_mpr.get_structure_by_material_id.return_value = tiny_structure
    fake_module = types.ModuleType("pymatgen.ext.matproj")
    fake_module.MPRester = MagicMock(return_value=fake_mpr)
    fake_pkg = sys.modules.get("pymatgen.ext") or types.ModuleType("pymatgen.ext")

    with (
        patch.object(cli, "_load_potential", return_value=fake_potential),
        patch.dict(sys.modules, {"pymatgen.ext": fake_pkg, "pymatgen.ext.matproj": fake_module}),
    ):
        cli.predict_structure(Namespace(model="EformModel", infile=None, mpids=["mp-1234"], state_attr=None))
    captured = capsys.readouterr().out
    assert "mp-1234" in captured
    assert tiny_structure.composition.reduced_formula in captured


def test_molecular_dynamics(tiny_cif):
    """MD command should construct MolecularDynamics with the parsed args and run."""
    args = Namespace(
        infile=[str(tiny_cif)],
        model="x",
        ensemble="nve",
        nsteps=2,
        stepsize=1.0,
        temp=300.0,
        pressure=1.0,
        taut=None,
        taup=None,
        andersen_prob=0.01,
        friction=0.001,
        ttime=25.0,
        pfactor=75.0**2.0,
        external_stress=None,
        compressibility_au=None,
        loginterval=1,
        append_trajectory=False,
        mask=None,
    )
    fake_md = MagicMock()
    with (
        patch.object(cli, "_load_potential", return_value=MagicMock()),
        patch.object(cli, "MolecularDynamics", return_value=fake_md) as mock_md_cls,
        patch.object(cli, "MaxwellBoltzmannDistribution") as mock_boltz,
        patch.object(cli, "AseAtomsAdaptor") as mock_adaptor_cls,
    ):
        mock_adaptor_cls.return_value.get_atoms.return_value = MagicMock(name="atoms")
        assert cli.molecular_dynamics(args) == 0
    mock_md_cls.assert_called_once()
    mock_boltz.assert_called_once()
    fake_md.run.assert_called_once_with(2)


def test_clear_cache_yes_skips_confirm():
    with patch("matgl.clear_cache") as fake:
        cli.clear_cache(Namespace(yes=True))
    fake.assert_called_once_with(False)


def test_clear_cache_default_confirms():
    with patch("matgl.clear_cache") as fake:
        cli.clear_cache(Namespace(yes=False))
    fake.assert_called_once_with(True)


def test_main_dispatches_to_clear(monkeypatch):
    """`main` should parse argv and dispatch to the selected sub-command."""
    called: dict = {}

    def fake_clear(args):
        called["clear"] = args

    monkeypatch.setattr(cli, "clear_cache", fake_clear)
    monkeypatch.setattr("sys.argv", ["mgl", "clear", "--yes"])
    # `get_available_pretrained_models` is invoked at parser-construction time and
    # may hit the network; stub it out so the test stays hermetic.
    monkeypatch.setattr(cli.matgl, "get_available_pretrained_models", lambda: ["FakeModel-PES"])
    cli.main()
    assert called["clear"].yes is True


def test_main_relax_route(monkeypatch, tiny_cif, tiny_structure, tmp_path):
    """End-to-end argv -> relax_structure dispatch with the heavy lifting mocked."""
    out = tmp_path / "out.cif"
    monkeypatch.setattr("sys.argv", ["mgl", "relax", "-i", str(tiny_cif), "-o", str(out)])
    monkeypatch.setattr(cli.matgl, "get_available_pretrained_models", lambda: ["FakeModel-PES"])
    with _patch_relaxer(tiny_structure):
        # The default model is supplied via `default=` so argparse never validates
        # `choices` for a value the user didn't pass; this is enough for `main` to
        # successfully resolve and dispatch.
        cli.main()
    assert out.exists()
