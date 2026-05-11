"""Offline tests for ``matgl.utils.training.MGLDatasetLoader`` and ``MGLPotentialTrainer``.

The HF download is monkeypatched in every test so nothing hits the network.
The NaCl parity payload at ``tests/parity_data/nacl_training_set.json.gz``
shares the per-record schema of the live MatPES JSONs and covers ``{Na, Cl}``;
``MGLDatasetLoader`` expects a flat list of records, so the dict-wrapped
fixture is unwrapped to its ``samples`` array in the per-test patches.
"""

from __future__ import annotations

import gzip
import json
import pathlib

import numpy as np
import pytest

import matgl

if matgl.config.BACKEND != "PYG":
    pytest.skip("Skipping PYG tests", allow_module_level=True)

from matgl.models import TensorNet
from matgl.utils import training as training_mod
from matgl.utils.training import (
    MGLDatasetLoader,
    MGLPotentialTrainer,
    _matpes_atomrefs_filename,
    _matpes_dataset_filename,
    _matpes_parse_version,
)

_NACL_PARITY = pathlib.Path(__file__).parent.parent / "parity_data" / "nacl_training_set.json.gz"


# ---------------------------------------------------------------------------
# Pure-string helpers.
# ---------------------------------------------------------------------------


class TestFilenameHelpers:
    def test_matpes_dataset_lowercase(self):
        assert _matpes_dataset_filename("r2SCAN-2025.2") == "MatPES-R2SCAN-2025.2.json"

    def test_matpes_dataset_uppercase(self):
        assert _matpes_dataset_filename("R2SCAN-2025.2") == "MatPES-R2SCAN-2025.2.json"

    def test_matpes_dataset_pbe(self):
        assert _matpes_dataset_filename("PBE-2025.1") == "MatPES-PBE-2025.1.json"

    def test_atomrefs_filename(self):
        assert _matpes_atomrefs_filename("r2SCAN-2025.2") == "MatPES-R2SCAN-atoms.json"
        assert _matpes_atomrefs_filename("PBE-2025.1") == "MatPES-PBE-atoms.json"

    def test_parse_version_invalid(self):
        with pytest.raises(ValueError, match="Invalid MatPES version"):
            _matpes_parse_version("invalid")
        with pytest.raises(ValueError, match="Invalid MatPES version"):
            _matpes_parse_version("PBE-")
        with pytest.raises(ValueError, match="Invalid MatPES version"):
            _matpes_parse_version("-2025.2")


# ---------------------------------------------------------------------------
# Fixtures / helpers for monkeypatching hf_hub_download.
# ---------------------------------------------------------------------------


def _flat_parity_payload_path(tmp_path: pathlib.Path) -> pathlib.Path:
    """Unwrap the dict-shaped parity payload into the flat list the loader expects."""
    with gzip.open(_NACL_PARITY) as fh:
        wrapped = json.load(fh)
    flat = tmp_path / "MatPES-R2SCAN-2025.2.json"
    flat.write_text(json.dumps(wrapped["samples"]))
    return flat


def _patch_hf_dataset_download(monkeypatch, tmp_path: pathlib.Path) -> pathlib.Path:
    """Make hf_hub_download / try_to_load_from_cache return the flat-list parity payload."""
    flat = _flat_parity_payload_path(tmp_path)

    monkeypatch.setattr(training_mod, "hf_hub_download", lambda **kwargs: str(flat))
    monkeypatch.setattr(training_mod, "try_to_load_from_cache", lambda **kwargs: None)
    return flat


def _patch_hf_atomrefs_download(monkeypatch, tmp_path: pathlib.Path, payload) -> pathlib.Path:
    refs_path = tmp_path / "atomrefs.json"
    refs_path.write_text(json.dumps(payload))

    monkeypatch.setattr(training_mod, "hf_hub_download", lambda **kwargs: str(refs_path))
    monkeypatch.setattr(training_mod, "try_to_load_from_cache", lambda **kwargs: None)
    return refs_path


def _atomrefs_record(symbol: str, energy: float) -> dict:
    """Single-atom MatPES atomrefs record (``chemsys`` / ``energy`` schema)."""
    return {"chemsys": symbol, "energy": energy}


# ---------------------------------------------------------------------------
# MatPES dataset loader (monkeypatched HF).
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _NACL_PARITY.exists(), reason="NaCl parity payload missing")
class TestLoadMatpesDataset:
    def test_returns_mgl_dataset_with_plural_label_keys(self, monkeypatch, tmp_path):
        _patch_hf_dataset_download(monkeypatch, tmp_path)
        ds = MGLDatasetLoader().matpes_dataset(
            version="r2SCAN-2025.2",
            cutoff=4.0,
            save_cache=False,
            root=str(tmp_path / "ds"),
            stress_unit="GPa",  # parity payload is already in matgl-internal units (GPa, compressive negative)
        )
        assert set(ds.labels.keys()) == {"energies", "forces", "stresses"}
        assert len(ds) == len(ds.labels["energies"])
        assert hasattr(ds, "element_types")
        assert set(ds.element_types) == {"Na", "Cl"}

    def test_stress_unit_kbar_scales_to_gpa_with_sign_flip(self, monkeypatch, tmp_path):
        """Default ``stress_unit='kbar'`` applies ``-0.1`` (kbar VASP → GPa matgl).

        matgl's internal stress unit is **GPa** with compressive = negative
        (README, "Model Training"); VASP kbar is compressive = positive, so the
        full conversion to matgl's convention is ``* -0.1``.
        """
        _patch_hf_dataset_download(monkeypatch, tmp_path)
        ds_gpa = MGLDatasetLoader().matpes_dataset(
            version="r2SCAN-2025.2",
            cutoff=4.0,
            save_cache=False,
            root=str(tmp_path / "ds_gpa"),
            stress_unit="GPa",  # identity — payload values already match matgl's GPa convention
        )
        ds_kbar = MGLDatasetLoader().matpes_dataset(
            version="r2SCAN-2025.2",
            cutoff=4.0,
            save_cache=False,  # default stress_unit="kbar"
            root=str(tmp_path / "ds_kbar"),
        )
        gpa = np.asarray(ds_gpa.labels["stresses"], dtype="float64")
        scaled = np.asarray(ds_kbar.labels["stresses"], dtype="float64")
        np.testing.assert_allclose(scaled, gpa * -0.1, rtol=1e-12, atol=0.0)

    def test_stress_unit_ev_per_a3_scales_to_gpa(self, monkeypatch, tmp_path):
        """``stress_unit='eV/A3'`` multiplies by 160.21766208 (eV/Å³ → GPa)."""
        _patch_hf_dataset_download(monkeypatch, tmp_path)
        ds_gpa = MGLDatasetLoader().matpes_dataset(
            version="r2SCAN-2025.2",
            cutoff=4.0,
            save_cache=False,
            root=str(tmp_path / "ds_gpa"),
            stress_unit="GPa",
        )
        ds_eva3 = MGLDatasetLoader().matpes_dataset(
            version="r2SCAN-2025.2",
            cutoff=4.0,
            save_cache=False,
            root=str(tmp_path / "ds_eva3"),
            stress_unit="eV/A3",
        )
        gpa = np.asarray(ds_gpa.labels["stresses"], dtype="float64")
        scaled = np.asarray(ds_eva3.labels["stresses"], dtype="float64")
        np.testing.assert_allclose(scaled, gpa * 160.21766208, rtol=1e-12, atol=0.0)

    def test_invalid_stress_unit_raises(self, monkeypatch, tmp_path):
        """An unknown ``stress_unit`` is rejected by the conversion-factor lookup."""
        _patch_hf_dataset_download(monkeypatch, tmp_path)
        with pytest.raises(KeyError):
            MGLDatasetLoader().matpes_dataset(
                version="r2SCAN-2025.2",
                cutoff=4.0,
                save_cache=False,
                stress_unit="bogus",  # type: ignore[arg-type]
            )

    def test_uses_cached_file_without_calling_hf_hub_download(self, monkeypatch, tmp_path):
        """When the file is already cached, hf_hub_download must not be called."""
        flat = _flat_parity_payload_path(tmp_path)
        monkeypatch.setattr(
            training_mod,
            "try_to_load_from_cache",
            lambda **kwargs: str(flat),
        )

        download_calls: list[dict] = []

        def fail_if_called(**kwargs):
            download_calls.append(kwargs)
            raise AssertionError("hf_hub_download must not be called when the file is cached")

        monkeypatch.setattr(training_mod, "hf_hub_download", fail_if_called)

        ds = MGLDatasetLoader().matpes_dataset(
            version="r2SCAN-2025.2",
            cutoff=4.0,
            save_cache=False,
            root=str(tmp_path / "ds"),
            stress_unit="GPa",
        )
        assert download_calls == []
        assert len(ds) > 0
        assert set(ds.element_types) == {"Na", "Cl"}

    def test_falls_through_to_hf_hub_download_on_cache_miss(self, monkeypatch, tmp_path):
        """A cache miss (try_to_load_from_cache returns None) routes to hf_hub_download."""
        monkeypatch.setattr(training_mod, "try_to_load_from_cache", lambda **kwargs: None)

        flat = _flat_parity_payload_path(tmp_path)
        download_calls: list[dict] = []

        def fake_download(**kwargs):
            download_calls.append(kwargs)
            return str(flat)

        monkeypatch.setattr(training_mod, "hf_hub_download", fake_download)

        ds = MGLDatasetLoader().matpes_dataset(
            version="r2SCAN-2025.2",
            cutoff=4.0,
            save_cache=False,
            root=str(tmp_path / "ds"),
            stress_unit="GPa",
        )
        assert len(download_calls) == 1
        assert download_calls[0]["repo_type"] == "dataset"
        assert download_calls[0]["filename"] == "MatPES-R2SCAN-2025.2.json"
        assert len(ds) > 0


# ---------------------------------------------------------------------------
# MatPES element-refs loader (monkeypatched HF).
# ---------------------------------------------------------------------------


class TestLoadMatpesElementRefs:
    def _payload(self):
        """Live HF schema: a flat list of single-atom records keyed by ``chemsys``."""
        return [_atomrefs_record("Na", -1.0), _atomrefs_record("Cl", -2.0)]

    def test_reorders_to_caller_element_types(self, monkeypatch, tmp_path):
        _patch_hf_atomrefs_download(monkeypatch, tmp_path, self._payload())
        refs = MGLDatasetLoader().matpes_element_refs(version="r2SCAN-2025.2", element_types=("Cl", "Na"))
        np.testing.assert_allclose(refs, [-2.0, -1.0])

    def test_subset_returns_only_requested_elements(self, monkeypatch, tmp_path):
        _patch_hf_atomrefs_download(monkeypatch, tmp_path, self._payload())
        refs = MGLDatasetLoader().matpes_element_refs(version="r2SCAN-2025.2", element_types=("Na",))
        np.testing.assert_allclose(refs, [-1.0])

    def test_missing_element_raises_keyerror(self, monkeypatch, tmp_path):
        _patch_hf_atomrefs_download(monkeypatch, tmp_path, self._payload())
        with pytest.raises(KeyError):
            MGLDatasetLoader().matpes_element_refs(version="r2SCAN-2025.2", element_types=("Li", "Cl"))

    def test_empty_element_types_returns_empty_array(self, monkeypatch, tmp_path):
        """``element_types`` defaults to ``()``; the loader returns a length-0 vector."""
        _patch_hf_atomrefs_download(monkeypatch, tmp_path, self._payload())
        refs = MGLDatasetLoader().matpes_element_refs(version="r2SCAN-2025.2")
        assert refs.shape == (0,)


# ---------------------------------------------------------------------------
# MGLPotentialTrainer init / fit smoke.
# ---------------------------------------------------------------------------


class TestMGLPotentialTrainerInit:
    def test_init_does_not_load_or_train(self, monkeypatch):
        """Constructor stores config; touches no network and instantiates no Lightning."""

        def boom(**kwargs):
            raise AssertionError("hf_hub_download must not be called during __init__.")

        monkeypatch.setattr(training_mod, "hf_hub_download", boom)

        model = TensorNet(
            element_types=("Na", "Cl"),
            cutoff=4.0,
            is_intensive=False,
            use_warp=False,
            units=8,
            ntargets=1,
            num_layers=1,
        )
        trainer = MGLPotentialTrainer(model, accelerator="cpu", max_epochs=2)

        assert trainer.accelerator == "cpu"
        assert trainer.max_epochs == 2
        assert trainer.dataset is None
        assert trainer.loaders is None
        assert trainer.lit_module is None
        assert trainer.trainer is None
        assert trainer.potential is None
        assert trainer.atomrefs is None

    def test_loss_weights_stored(self):
        """All five loss weights (energy/force/stress/magmom/charge) are stored verbatim."""
        model = TensorNet(
            element_types=("Na", "Cl"),
            cutoff=4.0,
            is_intensive=False,
            use_warp=False,
            units=8,
            ntargets=1,
            num_layers=1,
        )
        trainer = MGLPotentialTrainer(
            model,
            energy_weight=1.5,
            force_weight=2.0,
            stress_weight=0.25,
            magmom_weight=0.5,
            charge_weight=0.75,
        )
        assert trainer.energy_weight == 1.5
        assert trainer.force_weight == 2.0
        assert trainer.stress_weight == 0.25
        assert trainer.magmom_weight == 0.5
        assert trainer.charge_weight == 0.75

    def test_loss_weight_defaults_match_matpes_recipe(self):
        """Defaults: energy=1.0, force=1.0, stress=0.1, magmom=0.0, charge=0.0."""
        model = TensorNet(
            element_types=("Na", "Cl"),
            cutoff=4.0,
            is_intensive=False,
            use_warp=False,
            units=8,
            ntargets=1,
            num_layers=1,
        )
        trainer = MGLPotentialTrainer(model)
        assert trainer.energy_weight == 1.0
        assert trainer.force_weight == 1.0
        assert trainer.stress_weight == 0.1
        assert trainer.magmom_weight == 0.0
        assert trainer.charge_weight == 0.0


@pytest.mark.skipif(not _NACL_PARITY.exists(), reason="NaCl parity payload missing")
class TestFit:
    def _make_smart_hub(self, monkeypatch, flat_path, atomrefs_path):
        """Route atomrefs filenames to atomrefs_path, the rest to flat parity."""

        def smart_fake(**kwargs):
            fname = kwargs.get("filename", "")
            if "atoms" in fname:
                return str(atomrefs_path)
            return str(flat_path)

        monkeypatch.setattr(training_mod, "hf_hub_download", smart_fake)
        monkeypatch.setattr(training_mod, "try_to_load_from_cache", lambda **kwargs: None)

    def _make_model(self, element_types):
        return TensorNet(
            element_types=tuple(element_types),
            cutoff=4.0,
            is_intensive=False,
            use_warp=False,
            units=8,
            ntargets=1,
            num_layers=1,
        )

    @staticmethod
    def _trainer(model):
        return MGLPotentialTrainer(
            model,
            energy_weight=1.0,
            force_weight=1.0,
            stress_weight=0.1,
            batch_size=2,
            max_epochs=1,
            accelerator="cpu",
            devices=1,
            seed=42,
            loader_kwargs={"num_workers": 0, "frac_list": (0.6, 0.2, 0.2)},
            trainer_kwargs={
                "logger": False,
                "enable_checkpointing": False,
                "enable_progress_bar": False,
                "enable_model_summary": False,
                "num_sanity_val_steps": 0,
            },
        )

    def test_one_epoch_trains_and_records_state(self, monkeypatch, tmp_path):
        """fit() with a pre-built dataset and ndarray atomrefs populates trainer state."""
        flat = _flat_parity_payload_path(tmp_path)
        atomrefs_path = tmp_path / "atomrefs.json"
        atomrefs_path.write_text(json.dumps([_atomrefs_record("Na", 0.0), _atomrefs_record("Cl", 0.0)]))
        self._make_smart_hub(monkeypatch, flat, atomrefs_path)

        ds = MGLDatasetLoader().matpes_dataset(
            version="r2SCAN-2025.2",
            cutoff=4.0,
            save_cache=False,
            root=str(tmp_path / "ds"),
            stress_unit="GPa",
        )
        refs = MGLDatasetLoader().matpes_element_refs(version="r2SCAN-2025.2", element_types=ds.element_types)

        model = self._make_model(ds.element_types)
        trainer = self._trainer(model)
        potential = trainer.fit(dataset=ds, atomrefs=refs)

        from matgl.apps.pes import Potential

        assert isinstance(potential, Potential)
        assert trainer.potential is potential
        assert trainer.dataset is ds
        assert set(trainer.loaders) == {"train", "val", "test"}
        assert trainer.lit_module is not None
        assert trainer.trainer is not None
        np.testing.assert_allclose(trainer.atomrefs, refs)
        assert "cpu" in type(trainer.trainer.accelerator).__name__.lower()

    def test_fit_atomrefs_none(self, monkeypatch, tmp_path):
        """``atomrefs=None`` disables offsets entirely."""
        _patch_hf_dataset_download(monkeypatch, tmp_path)
        ds = MGLDatasetLoader().matpes_dataset(
            version="r2SCAN-2025.2",
            cutoff=4.0,
            save_cache=False,
            root=str(tmp_path / "ds"),
            stress_unit="GPa",
        )
        model = self._make_model(ds.element_types)
        trainer = self._trainer(model)
        trainer.fit(dataset=ds, atomrefs=None)
        assert trainer.atomrefs is None


# ---------------------------------------------------------------------------
# Sanity: MatPES JSON loadfn handles both .json and .json.gz transparently
# (the parity fixture is .json.gz; the live MatPES files are plain .json).
# ---------------------------------------------------------------------------


def test_matpes_payload_is_loadable_without_gzip_suffix(tmp_path, monkeypatch):
    """Decompress the .json.gz fixture into a plain .json and load via MGLPotentialTrainer."""
    plain = tmp_path / "MatPES-NACL-2025.2.json"
    with gzip.open(_NACL_PARITY, "rb") as src:
        wrapped = json.loads(src.read())
    plain.write_text(json.dumps(wrapped["samples"]))

    monkeypatch.setattr(training_mod, "hf_hub_download", lambda **_: str(plain))
    monkeypatch.setattr(training_mod, "try_to_load_from_cache", lambda **_: None)
    ds = MGLDatasetLoader().matpes_dataset(
        version="r2SCAN-2025.2", cutoff=4.0, save_cache=False, root=str(tmp_path / "ds")
    )
    assert set(ds.labels.keys()) == {"energies", "forces", "stresses"}
