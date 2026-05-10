"""Offline tests for ``matgl.utils.training.MatGLPotentialTrainer``.

The HF download is monkeypatched in every test so nothing hits the network.
Real artefacts substitute for the live MatPES / extxyz files:

- ``tests/parity_data/nacl_training_set.json.gz`` (already in the repo) has
  the same payload schema as the MatPES JSONs and covers ``{Na, Cl}``.
- A tiny synthetic extxyz tarball is written into ``tmp_path`` for the
  extxyz tests so they don't depend on the live HF dataset.
- A tiny synthetic atomrefs JSON covers the element-references path.
"""

from __future__ import annotations

import gzip
import json
import pathlib
import tarfile

import numpy as np
import pytest

import matgl

if matgl.config.BACKEND != "PYG":
    pytest.skip("Skipping PYG tests", allow_module_level=True)

from matgl.models import TensorNet
from matgl.utils import training as training_mod
from matgl.utils.training import (
    MatGLPotentialTrainer,
    _classify_extxyz_split,
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

    def test_matpes_dataset_with_split(self):
        assert _matpes_dataset_filename("R2SCAN-2025.2", split="train") == "MatPES-R2SCAN-2025.2-train.json"

    def test_matpes_dataset_pbe(self):
        assert _matpes_dataset_filename("PBE-2025.1") == "MatPES-PBE-2025.1.json"

    def test_matpes_dataset_invalid_split_raises(self):
        with pytest.raises(ValueError, match="Invalid split"):
            _matpes_dataset_filename("PBE-2025.2", split="dev")

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

    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("vama_dimer_train.extxyz", "train"),
            ("vama_dimer_test.extxyz", "test"),
            ("vama_dimer_valid.extxyz", "valid"),
            ("vama_dimer_val.extxyz", "valid"),
            ("vama-dimer-train.xyz", "train"),
            ("anything_else.extxyz", None),
            ("trainer.extxyz", None),  # ensure ``train`` substring alone doesn't match
        ],
    )
    def test_classify_extxyz_split(self, name, expected):
        assert _classify_extxyz_split(name) == expected

    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("MatPES-R2SCAN-2025.2.json", "matpes"),
            ("MatPES-R2SCAN-2025.2.json.gz", "matpes"),
            ("cp_dimer.tar.gz", "extxyz"),
            ("vama_train.extxyz", "extxyz"),
            ("foo.xyz", "extxyz"),
            ("foo.tar", "extxyz"),
            ("foo.tgz", "extxyz"),
        ],
    )
    def test_detect_data_format_known_suffixes(self, name, expected):
        from matgl.utils.training import _detect_data_format

        assert _detect_data_format(name) == expected

    def test_detect_data_format_unknown_suffix_raises(self):
        from matgl.utils.training import _detect_data_format

        with pytest.raises(ValueError, match="auto-detect"):
            _detect_data_format("dataset.parquet")


# ---------------------------------------------------------------------------
# Helpers for monkeypatching hf_hub_download.
# ---------------------------------------------------------------------------


def _patch_hf_dataset_download(monkeypatch):
    """Make every hf_hub_download call return the NaCl parity payload path."""

    def fake(**kwargs):
        return str(_NACL_PARITY)

    monkeypatch.setattr(training_mod, "hf_hub_download", fake)


def _patch_hf_atomrefs_download(monkeypatch, tmp_path, payload):
    refs_path = tmp_path / "atomrefs.json"
    refs_path.write_text(json.dumps(payload))

    def fake(**kwargs):
        return str(refs_path)

    monkeypatch.setattr(training_mod, "hf_hub_download", fake)


# ---------------------------------------------------------------------------
# extxyz fixtures (synthetic, no network).
# ---------------------------------------------------------------------------


def _write_extxyz_frame(handle, *, energy, n=2, force_value=0.1):
    """Write a single H2 frame (no stress) in extxyz format."""
    handle.write(f"{n}\n")
    handle.write(
        f'Lattice="10.0 0.0 0.0 0.0 10.0 0.0 0.0 0.0 10.0" '
        f'Properties=species:S:1:pos:R:3:forces:R:3 energy={energy} pbc="T T T"\n'
    )
    handle.write(f"H 0.0 0.0 0.0 {force_value:.6f} 0.0 0.0\n")
    handle.write(f"H 0.0 0.0 1.0 {-force_value:.6f} 0.0 0.0\n")


@pytest.fixture
def extxyz_tarball_with_splits(tmp_path):
    """Build a tar.gz with two extxyz files: one '_train', one '_test'."""
    train = tmp_path / "h2_train.extxyz"
    test = tmp_path / "h2_test.extxyz"
    with train.open("w") as f:
        for e in (-1.0, -1.1, -1.2, -1.3):
            _write_extxyz_frame(f, energy=e)
    with test.open("w") as f:
        for e in (-1.05, -1.15):
            _write_extxyz_frame(f, energy=e)

    tar_path = tmp_path / "h2_dimer.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(train, arcname="h2_dimer/h2_train.extxyz")
        tar.add(test, arcname="h2_dimer/h2_test.extxyz")
    return tar_path


@pytest.fixture
def extxyz_plain_file(tmp_path):
    """Build a single extxyz file (no tarball)."""
    path = tmp_path / "h2_single.extxyz"
    with path.open("w") as f:
        for e in (-1.0, -1.1, -1.2):
            _write_extxyz_frame(f, energy=e)
    return path


# ---------------------------------------------------------------------------
# MatPES dataset / refs (monkeypatched HF).
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _NACL_PARITY.exists(), reason="NaCl parity payload missing")
class TestLoadMatpesDataset:
    def test_returns_mgl_dataset_with_plural_label_keys(self, monkeypatch, tmp_path):
        _patch_hf_dataset_download(monkeypatch)
        ds = MatGLPotentialTrainer.load_matpes_dataset(
            version="r2SCAN-2025.2",
            cutoff=4.0,
            cache_dir=tmp_path,
            save_cache=False,
        )
        assert set(ds.labels.keys()) == {"energies", "forces", "stresses"}
        assert len(ds) == len(ds.labels["energies"])
        assert hasattr(ds, "element_types")
        assert set(ds.element_types) == {"Na", "Cl"}


class TestLoadMatpesElementRefs:
    def test_no_reorder_returns_file_order(self, monkeypatch, tmp_path):
        _patch_hf_atomrefs_download(monkeypatch, tmp_path, {"element_types": ["Na", "Cl"], "refs": [-1.0, -2.0]})
        refs = MatGLPotentialTrainer.load_matpes_element_refs(version="r2SCAN-2025.2")
        np.testing.assert_allclose(refs, [-1.0, -2.0])

    def test_reorders_to_caller_element_types(self, monkeypatch, tmp_path):
        _patch_hf_atomrefs_download(monkeypatch, tmp_path, {"element_types": ["Na", "Cl"], "refs": [-1.0, -2.0]})
        refs = MatGLPotentialTrainer.load_matpes_element_refs(version="r2SCAN-2025.2", element_types=("Cl", "Na"))
        np.testing.assert_allclose(refs, [-2.0, -1.0])

    def test_missing_element_raises_keyerror(self, monkeypatch, tmp_path):
        _patch_hf_atomrefs_download(monkeypatch, tmp_path, {"element_types": ["Na", "Cl"], "refs": [-1.0, -2.0]})
        with pytest.raises(KeyError, match=r"\['Li'\]"):
            MatGLPotentialTrainer.load_matpes_element_refs(version="r2SCAN-2025.2", element_types=("Li", "Cl"))


# ---------------------------------------------------------------------------
# extxyz loaders.
# ---------------------------------------------------------------------------


class TestLoadExtxyzDataset:
    def test_local_plain_extxyz(self, extxyz_plain_file):
        ds = MatGLPotentialTrainer.load_extxyz_dataset(path=extxyz_plain_file, cutoff=2.0, save_cache=False)
        # Three frames, no stress key (cluster/dimer extxyz has no stress).
        assert len(ds) == 3
        assert set(ds.labels.keys()) == {"energies", "forces"}
        assert "stresses" not in ds.labels
        np.testing.assert_allclose(ds.labels["energies"], [-1.0, -1.1, -1.2])

    def test_local_tarball_concatenates(self, extxyz_tarball_with_splits):
        ds = MatGLPotentialTrainer.load_extxyz_dataset(path=extxyz_tarball_with_splits, cutoff=2.0, save_cache=False)
        # 4 train + 2 test frames concatenated.
        assert len(ds) == 6
        assert set(ds.labels.keys()) == {"energies", "forces"}

    def test_hub_path_routes_through_hf_hub_download(self, monkeypatch, extxyz_tarball_with_splits):
        # Replace hf_hub_download with a stub that returns our local fixture.
        captured = {}

        def fake(**kwargs):
            captured.update(kwargs)
            return str(extxyz_tarball_with_splits)

        monkeypatch.setattr(training_mod, "hf_hub_download", fake)

        ds = MatGLPotentialTrainer.load_extxyz_dataset(
            repo_id="materialyze/mlip-lr-benchmarks",
            filename="cp_dimer.tar.gz",
            cutoff=2.0,
            save_cache=False,
        )
        assert len(ds) == 6
        assert captured["repo_id"] == "materialyze/mlip-lr-benchmarks"
        assert captured["filename"] == "cp_dimer.tar.gz"

    def test_path_and_repo_id_are_mutually_exclusive(self, extxyz_plain_file):
        with pytest.raises(ValueError, match="not both"):
            MatGLPotentialTrainer.load_extxyz_dataset(
                path=extxyz_plain_file, repo_id="x", filename="y", save_cache=False
            )

    def test_neither_path_nor_repo_raises(self):
        with pytest.raises(ValueError, match="Provide 'path'"):
            MatGLPotentialTrainer.load_extxyz_dataset(save_cache=False)


class TestLoadExtxyzSplits:
    def test_canonical_splits_from_tarball(self, extxyz_tarball_with_splits):
        splits = MatGLPotentialTrainer.load_extxyz_splits(path=extxyz_tarball_with_splits, cutoff=2.0, save_cache=False)
        # Only train and test were present in the fixture; valid is absent.
        assert set(splits.keys()) == {"train", "test"}
        assert len(splits["train"]) == 4
        assert len(splits["test"]) == 2
        # Element types are shared across the splits.
        assert splits["train"].element_types == splits["test"].element_types

    def test_unrecognised_filenames_raises(self, tmp_path):
        # Tarball with a single non-split file.
        plain = tmp_path / "h2_random.extxyz"
        with plain.open("w") as f:
            _write_extxyz_frame(f, energy=-1.0)
        tar_path = tmp_path / "no_splits.tar.gz"
        with tarfile.open(tar_path, "w:gz") as tar:
            tar.add(plain, arcname="bundle/h2_random.extxyz")

        with pytest.raises(ValueError, match="No files inside"):
            MatGLPotentialTrainer.load_extxyz_splits(path=tar_path, save_cache=False)


# ---------------------------------------------------------------------------
# MatGLPotentialTrainer init / save / fit smoke.
# ---------------------------------------------------------------------------


class TestMatGLPotentialTrainerInit:
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
        trainer = MatGLPotentialTrainer(model, accelerator="cpu", max_epochs=2)

        assert trainer.accelerator == "cpu"
        assert trainer.max_epochs == 2
        assert trainer.dataset is None
        assert trainer.loaders is None
        assert trainer.lit_module is None
        assert trainer.trainer is None
        assert trainer.potential is None

    def test_save_before_fit_raises(self, tmp_path):
        model = TensorNet(
            element_types=("Na", "Cl"),
            cutoff=4.0,
            is_intensive=False,
            use_warp=False,
            units=8,
            ntargets=1,
            num_layers=1,
        )
        trainer = MatGLPotentialTrainer(model)
        with pytest.raises(RuntimeError, match="before fit"):
            trainer.save(tmp_path / "nope")


@pytest.mark.skipif(not _NACL_PARITY.exists(), reason="NaCl parity payload missing")
class TestFit:
    def _make_smart_hub(self, monkeypatch, atomrefs_path):
        """Route atoms-shaped filenames to atomrefs_path, everything else to NaCl parity."""

        def smart_fake(**kwargs):
            fname = kwargs.get("filename", "")
            if "atoms" in fname:
                return str(atomrefs_path)
            return str(_NACL_PARITY)

        monkeypatch.setattr(training_mod, "hf_hub_download", smart_fake)

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
        return MatGLPotentialTrainer(
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

    def test_one_epoch_trains_and_persists_state(self, monkeypatch, tmp_path):
        """fit() with a pre-built dataset, ndarray atomrefs, and post-fit save round-trip."""
        _patch_hf_dataset_download(monkeypatch)
        ds = MatGLPotentialTrainer.load_matpes_dataset(
            version="r2SCAN-2025.2",
            cutoff=4.0,
            cache_dir=tmp_path,
            save_cache=False,
        )
        atomrefs_path = tmp_path / "atomrefs.json"
        atomrefs_path.write_text(json.dumps({"element_types": list(ds.element_types), "refs": [0.0, 0.0]}))
        self._make_smart_hub(monkeypatch, atomrefs_path)

        # Fetch refs explicitly to verify the ndarray atomrefs path.
        refs = MatGLPotentialTrainer.load_matpes_element_refs(version="r2SCAN-2025.2", element_types=ds.element_types)

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

        save_dir = tmp_path / "trained"
        save_dir.mkdir()
        trainer.save(save_dir)
        reloaded = matgl.load_model(path=str(save_dir))
        assert isinstance(reloaded, Potential)

    def test_fit_resolves_hf_dataset_tuple_with_auto_format(self, monkeypatch, tmp_path):
        """fit() accepts a (repo_id, filename) tuple and auto-detects MatPES from the .json suffix."""
        captured: list[dict] = []

        def fake(**kwargs):
            captured.append(kwargs)
            fname = kwargs.get("filename", "")
            return str(tmp_path / "atomrefs.json") if "atoms" in fname else str(_NACL_PARITY)

        atomrefs_path = tmp_path / "atomrefs.json"
        atomrefs_path.write_text(json.dumps({"element_types": ["Na", "Cl"], "refs": [-1.0, -2.0]}))
        monkeypatch.setattr(training_mod, "hf_hub_download", fake)

        model = self._make_model(("Na", "Cl"))
        trainer = self._trainer(model)
        trainer.fit(
            dataset=("materialyze/matpes", "MatPES-R2SCAN-2025.2.json"),
            format="auto",  # should infer 'matpes' from .json
            atomrefs=("materialyze/matpes", "MatPES-R2SCAN-atoms.json"),
        )

        # Trainer wrote the resolved atomrefs onto self in (Na, Cl) order.
        np.testing.assert_allclose(trainer.atomrefs, [-1.0, -2.0])
        # hf_hub_download was hit twice: once for the dataset, once for atomrefs.
        filenames = [c["filename"] for c in captured]
        assert "MatPES-R2SCAN-2025.2.json" in filenames
        assert "MatPES-R2SCAN-atoms.json" in filenames

    def test_fit_extxyz_format_with_dict_atomrefs(self, monkeypatch, tmp_path, extxyz_plain_file):
        """fit(format='extxyz') with an in-memory atomrefs dict and a local extxyz path."""

        # No HF traffic at all in this path.
        def boom(**_):
            raise AssertionError("hf_hub_download should not be called for local + dict atomrefs.")

        monkeypatch.setattr(training_mod, "hf_hub_download", boom)

        model = self._make_model(("H",))
        trainer = self._trainer(model)
        trainer.fit(
            dataset=str(extxyz_plain_file),
            format="extxyz",
            atomrefs={"element_types": ["H"], "refs": [-13.6]},
        )
        np.testing.assert_allclose(trainer.atomrefs, [-13.6])
        # The forces-only extxyz (no stress) propagates through the loaders.
        first_batch = next(iter(trainer.loaders["train"]))
        assert len(first_batch) == 6  # (g, lat, state, e, f, s)  — s is zeros via auto-detect

    def test_fit_atomrefs_none(self, monkeypatch, tmp_path, extxyz_plain_file):
        """atomrefs=None disables offsets entirely."""

        def boom(**_):
            raise AssertionError("hf_hub_download should not be called when atomrefs=None.")

        monkeypatch.setattr(training_mod, "hf_hub_download", boom)

        model = self._make_model(("H",))
        trainer = self._trainer(model)
        trainer.fit(dataset=extxyz_plain_file, format="auto", atomrefs=None)
        assert trainer.atomrefs is None

    def test_fit_atomrefs_atomref_instance(self, monkeypatch, extxyz_plain_file):
        """atomrefs accepts a pre-built AtomRef layer instance."""
        import torch

        from matgl.layers._atom_ref_pyg import AtomRef

        def boom(**_):
            raise AssertionError("no HF traffic expected.")

        monkeypatch.setattr(training_mod, "hf_hub_download", boom)

        ref_layer = AtomRef(property_offset=torch.tensor([-7.5], dtype=torch.float32))
        model = self._make_model(("H",))
        trainer = self._trainer(model)
        trainer.fit(dataset=extxyz_plain_file, format="extxyz", atomrefs=ref_layer)
        np.testing.assert_allclose(trainer.atomrefs, [-7.5])


# ---------------------------------------------------------------------------
# Sanity: MatPES JSON loadfn handles both .json and .json.gz transparently
# (the parity fixture is .json.gz; the live MatPES files are .json).
# ---------------------------------------------------------------------------


def test_matpes_payload_is_loadable_without_gzip_suffix(tmp_path, monkeypatch):
    """Decompress the .json.gz fixture into a plain .json and load via MatGLPotentialTrainer."""
    plain = tmp_path / "MatPES-NACL-2025.2.json"
    with gzip.open(_NACL_PARITY, "rb") as src, plain.open("wb") as dst:
        dst.write(src.read())

    monkeypatch.setattr(training_mod, "hf_hub_download", lambda **_: str(plain))
    ds = MatGLPotentialTrainer.load_matpes_dataset(version="r2SCAN-2025.2", cutoff=4.0, save_cache=False)
    assert set(ds.labels.keys()) == {"energies", "forces", "stresses"}
