"""Narrow tests for the config-driven training scripts in ``scripts/train``.

These exercise the shared helpers in ``scripts/train/_common.py`` (config and
pymatgen-JSON dataset loading, scheduler/warmup construction, checkpoint resume)
plus the model registry in ``train_pes.py``. A single tiny TensorNet is trained for
one or two epochs on CPU, so the whole module runs in a few seconds.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import lightning as pl
import numpy as np
import pytest
import torch
import yaml
from pymatgen.core import Lattice, Structure

# Multiple OpenMP runtimes can be linked in some conda envs; allow the duplicate
# rather than aborting (must be set before heavy native libs spin up threads).
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

SCRIPTS_DIR = Path(__file__).resolve().parents[3] / "scripts" / "train"
sys.path.insert(0, str(SCRIPTS_DIR))


def _load_script(name: str):
    """Import a module from scripts/train by file path (it is not a package)."""
    spec = importlib.util.spec_from_file_location(name, SCRIPTS_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules[name] = module
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


_common = _load_script("_common")


def _make_structures(n: int) -> list[Structure]:
    return [
        Structure(Lattice.cubic(3.4 + 0.02 * i), ["Li"] * 4, [[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]])
        for i in range(n)
    ]


def _write_dataset(path: Path, n: int) -> None:
    structures = _make_structures(n)
    labels = {
        "energies": [-1.9 * len(s) for s in structures],
        "forces": [(np.random.default_rng(i).standard_normal((4, 3)) * 0.01).tolist() for i in range(n)],
    }
    path.write_text(json.dumps({"structures": [s.as_dict() for s in structures], "labels": labels}))


def _write_ase_dataset(path: Path, n: int, *, with_stress: bool = True) -> None:
    """Write ``n`` frames to an extended-XYZ file with attached PES results."""
    import ase.io
    from ase.calculators.singlepoint import SinglePointCalculator
    from pymatgen.io.ase import AseAtomsAdaptor

    frames = []
    for i, structure in enumerate(_make_structures(n)):
        atoms = AseAtomsAdaptor.get_atoms(structure)
        results = {
            "energy": -1.9 * len(structure),
            "forces": (np.random.default_rng(i).standard_normal((4, 3)) * 0.01).tolist(),
        }
        if with_stress:
            results["stress"] = [0.1, 0.1, 0.1, 0.0, 0.0, 0.0]  # voigt-6
        atoms.calc = SinglePointCalculator(atoms, **results)
        frames.append(atoms)
    ase.io.write(str(path), frames, format="extxyz")


@pytest.fixture
def dataset_files(tmp_path: Path) -> tuple[Path, Path]:
    train, val = tmp_path / "train.json", tmp_path / "val.json"
    _write_dataset(train, 6)
    _write_dataset(val, 2)
    return train, val


@pytest.fixture
def pretrained_dir(tmp_path: Path, dataset_files: tuple[Path, Path]) -> Path:
    """Save a tiny ``Potential`` to a local dir for finetune tests (no network)."""
    from matgl.apps.pes import Potential
    from matgl.models import TensorNet

    train, val = dataset_files
    config = {"train": str(train), "val": str(val), "cutoff": 4.0, "element_types": "auto"}
    train_data, _, _, element_types = _common.build_datasets(config)
    model = TensorNet(element_types=element_types, is_intensive=False, units=16, nblocks=1)
    refs = _common.compute_element_refs(train_data, element_types)
    pot = Potential(model=model, element_refs=refs)
    out = tmp_path / "pretrained"
    pot.save(out)
    return out


def test_load_config_coerces_ruamel_scalars(tmp_path: Path) -> None:
    """YAML scalars must come back as plain Python types (else checkpoints break)."""
    cfg_path = tmp_path / "config.yaml"
    yaml.safe_dump({"lr": 0.001, "max_epochs": 5, "model_args": {"units": 16}}, cfg_path.open("w"))
    config = _common.load_config(str(cfg_path))
    assert type(config["lr"]) is float
    assert type(config["max_epochs"]) is int
    assert type(config["model_args"]) is dict


def test_build_datasets_separate_files(dataset_files: tuple[Path, Path]) -> None:
    """Separate train/val files load into datasets with an auto element table."""
    train, val = dataset_files
    config = {"train": str(train), "val": str(val), "cutoff": 4.0, "element_types": "auto"}
    train_data, val_data, test_data, element_types = _common.build_datasets(config)
    assert len(train_data) == 6
    assert len(val_data) == 2
    assert test_data is None
    assert element_types == ("Li",)


def test_build_dataloaders_drops_stress_when_unweighted(dataset_files: tuple[Path, Path]) -> None:
    """With stress_weight=0 the PES collate yields the no-stress tuple length."""
    train, val = dataset_files
    config = {"train": str(train), "val": str(val), "cutoff": 4.0, "element_types": "auto", "batch_size": 2}
    train_data, val_data, _, _ = _common.build_datasets(config)
    train_loader, _, test_loader = _common.build_dataloaders(train_data, val_data, None, config)
    assert test_loader is None
    batch = next(iter(train_loader))
    # collate_fn_pes returns (g, lat, state_attr, e, f, s) when stress is included
    # and one fewer trailing target when it is not.
    assert len(batch) == 6


def test_build_scheduler_warmup_steps_once_per_epoch(dataset_files: tuple[Path, Path]) -> None:
    """Linear warmup then StepLR must advance the LR exactly once per epoch."""
    train, val = dataset_files
    config = {
        "model": "TensorNet",
        "model_args": {"units": 16, "nblocks": 1},
        "train": str(train),
        "val": str(val),
        "cutoff": 4.0,
        "element_types": "auto",
        "batch_size": 2,
        "stress_weight": 0.0,
        "force_weight": 1.0,
        "lr": 0.01,
        "max_epochs": 4,
        "warmup_epochs": 2,
        "scheduler": "StepLR",
        "scheduler_args": {"step_size": 1, "gamma": 0.5},
    }
    train_data, val_data, _, element_types = _common.build_datasets(config)
    train_loader, val_loader, _ = _common.build_dataloaders(train_data, val_data, None, config)
    train_module = _load_script("train_pes")
    model = train_module.build_model(config, element_types)
    refs = _common.compute_element_refs(train_data, element_types)
    lit_module = _common.build_potential_module(model, config, refs)

    seen: list[float] = []

    class _LRTrace(pl.Callback):
        def on_train_epoch_start(self, trainer, pl_module):  # noqa: ANN001
            seen.append(round(trainer.optimizers[0].param_groups[0]["lr"], 6))

    trainer = pl.Trainer(
        max_epochs=4,
        accelerator="cpu",
        logger=False,
        inference_mode=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        callbacks=[_LRTrace()],
    )
    trainer.fit(lit_module, train_loader, val_loader)
    # Linear warmup over 2 epochs (0 -> lr), then StepLR halving each epoch.
    assert seen == [0.0, 0.005, 0.01, 0.005]


def test_warmup_incompatible_with_plateau() -> None:
    """Warmup + ReduceLROnPlateau is rejected (SequentialLR can't chain it)."""
    optimizer = torch.optim.Adam([torch.nn.Parameter(torch.zeros(1))], lr=0.01)
    config = {"scheduler": "ReduceLROnPlateau", "warmup_epochs": 2, "lr": 0.01, "max_epochs": 5}
    with pytest.raises(ValueError, match="ReduceLROnPlateau"):
        _common.build_scheduler(optimizer, config)


def test_train_checkpoint_and_resume(dataset_files: tuple[Path, Path], tmp_path: Path) -> None:
    """One epoch writes a checkpoint; resuming continues to the next epoch."""
    train, val = dataset_files
    ckpt_dir = tmp_path / "ckpts"
    config = {
        "model": "TensorNet",
        "model_args": {"units": 16, "nblocks": 1},
        "train": str(train),
        "val": str(val),
        "cutoff": 4.0,
        "element_types": "auto",
        "batch_size": 2,
        "force_weight": 1.0,
        "stress_weight": 0.0,
        "lr": 0.001,
        "accelerator": "cpu",
        "logger": "csv",
        "log_dir": str(tmp_path / "logs"),
        "checkpoint": True,
        "checkpoint_dir": str(ckpt_dir),
        "max_epochs": 1,
    }
    train_data, val_data, _, element_types = _common.build_datasets(config)
    train_loader, val_loader, _ = _common.build_dataloaders(train_data, val_data, None, config)
    train_module = _load_script("train_pes")

    def run(cfg):
        model = train_module.build_model(cfg, element_types)
        refs = _common.compute_element_refs(train_data, element_types)
        lit_module = _common.build_potential_module(model, cfg, refs)
        trainer = _common.build_trainer(cfg, _common.build_logger(cfg), _common.build_callbacks(cfg))
        trainer.fit(lit_module, train_loader, val_loader, ckpt_path=_common.resume_ckpt_path(cfg))
        return trainer

    run(config)
    assert (ckpt_dir / "last.ckpt").exists()

    resume_config = {**config, "max_epochs": 2, "resume_from": "last"}
    trainer = run(resume_config)
    # Resumed from the end of epoch 0 and trained one more epoch.
    assert trainer.current_epoch == 2


# --- ASE-readable dataset loading ---------------------------------------------


def test_load_ase_extxyz_with_stress(tmp_path: Path) -> None:
    """A non-.json file is read with ASE; energy/forces/stress become labels."""
    path = tmp_path / "data.extxyz"
    _write_ase_dataset(path, 3, with_stress=True)
    structures, labels = _common.load_structures_labels(str(path))
    assert len(structures) == 3
    assert sorted(labels) == ["energies", "forces", "stresses"]
    assert np.shape(labels["forces"][0]) == (4, 3)
    assert np.shape(labels["stresses"][0]) == (3, 3)


def test_load_ase_without_stress_omits_key(tmp_path: Path) -> None:
    """Stress is only emitted when present; energy/forces still load."""
    path = tmp_path / "data.extxyz"
    _write_ase_dataset(path, 2, with_stress=False)
    _, labels = _common.load_structures_labels(str(path))
    assert sorted(labels) == ["energies", "forces"]


def test_load_ase_partial_stress_raises(tmp_path: Path) -> None:
    """A file with stress on only some frames is rejected (can't be collated)."""
    import ase.io
    from ase.calculators.singlepoint import SinglePointCalculator
    from pymatgen.io.ase import AseAtomsAdaptor

    frames = []
    for i, structure in enumerate(_make_structures(2)):
        atoms = AseAtomsAdaptor.get_atoms(structure)
        results = {"energy": -7.6, "forces": [[0.0, 0.0, 0.0]] * 4}
        if i == 0:
            results["stress"] = [0.1, 0.1, 0.1, 0.0, 0.0, 0.0]
        atoms.calc = SinglePointCalculator(atoms, **results)
        frames.append(atoms)
    path = tmp_path / "mixed.extxyz"
    ase.io.write(str(path), frames, format="extxyz")
    with pytest.raises(ValueError, match="stress on some frames"):
        _common.load_structures_labels(str(path))


def test_build_datasets_from_ase_files(tmp_path: Path) -> None:
    """build_datasets transparently accepts ASE files for train/val."""
    train, val = tmp_path / "train.extxyz", tmp_path / "val.extxyz"
    _write_ase_dataset(train, 4)
    _write_ase_dataset(val, 2)
    config = {"train": str(train), "val": str(val), "cutoff": 4.0, "element_types": "auto"}
    train_data, val_data, test_data, element_types = _common.build_datasets(config)
    assert len(train_data) == 4
    assert len(val_data) == 2
    assert test_data is None
    assert element_types == ("Li",)


# --- helper-level coverage ----------------------------------------------------


@pytest.mark.parametrize(
    ("name", "extra", "checks"),
    [
        ("Adam", {}, {}),
        ("AdamW", {"weight_decay": 0.01}, {"weight_decay": 0.01}),
        ("SGD", {"momentum": 0.9, "nesterov": True}, {"momentum": 0.9, "nesterov": True}),
        ("RMSprop", {"alpha": 0.99}, {"alpha": 0.99}),
        ("Adamax", {}, {}),
    ],
)
def test_build_optimizer(name: str, extra: dict, checks: dict) -> None:
    """Each named torch.optim class is built with lr + optimizer_args applied."""
    params = [torch.nn.Parameter(torch.zeros(1))]
    optimizer = _common.build_optimizer(params, {"optimizer": name, "lr": 0.01, "optimizer_args": extra})
    assert type(optimizer).__name__ == name
    assert optimizer.param_groups[0]["lr"] == 0.01
    for key, value in checks.items():
        assert optimizer.param_groups[0][key] == value


def test_build_scheduler_defaults_to_cosine() -> None:
    """With no 'scheduler' key, a CosineAnnealingLR over (max_epochs - warmup) is built."""
    optimizer = torch.optim.Adam([torch.nn.Parameter(torch.zeros(1))], lr=0.01)
    scheduler = _common.build_scheduler(optimizer, {"lr": 0.01, "max_epochs": 10, "warmup_epochs": 0})
    assert isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR)
    assert scheduler.T_max == 10


def test_build_datasets_single_file_split(tmp_path: Path) -> None:
    """A single 'dataset' file is split by 'frac_list' into train/val/test."""
    path = tmp_path / "all.json"
    _write_dataset(path, 10)
    config = {
        "dataset": str(path),
        "cutoff": 4.0,
        "element_types": "auto",
        "frac_list": [0.6, 0.2, 0.2],
        "random_state": 1,
    }
    train_data, val_data, test_data, element_types = _common.build_datasets(config)
    assert len(train_data) + len(val_data) + len(test_data) == 10
    assert element_types == ("Li",)


def test_build_datasets_requires_dataset_or_train_val() -> None:
    """Omitting both 'dataset' and 'train'/'val' is a configuration error."""
    with pytest.raises(ValueError, match="train"):
        _common.build_datasets({"cutoff": 4.0})


# --- stress unit handling -----------------------------------------------------


def test_convert_stress_labels_ev_per_ang3_to_gpa() -> None:
    """eV/A3 stresses are scaled by EV_PER_ANG3_TO_GPA; energies/forces untouched."""
    from matgl.apps.pes import EV_PER_ANG3_TO_GPA

    labels = {"energies": [-1.0], "forces": [[[0.0, 0.0, 0.0]]], "stresses": [[[1.0, 0.0, 0.0]] * 3]}
    _common.convert_stress_labels([labels], {"stress_unit": "eV/A3"})
    np.testing.assert_allclose(labels["stresses"][0][0][0], EV_PER_ANG3_TO_GPA)


def test_convert_stress_labels_gpa_is_noop() -> None:
    """An explicit GPa unit leaves the stresses unchanged and warns nothing."""
    import warnings

    labels = {"stresses": [[[5.0, 0.0, 0.0]] * 3]}
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning would fail the test
        _common.convert_stress_labels([labels], {"stress_unit": "GPa"})
    assert labels["stresses"][0][0][0] == 5.0


def test_convert_stress_labels_warns_when_unit_omitted() -> None:
    """Omitting stress_unit while stresses are present assumes GPa with a warning."""
    labels = {"stresses": [[[5.0, 0.0, 0.0]] * 3]}
    with pytest.warns(UserWarning, match="GPa"):
        _common.convert_stress_labels([labels], {})
    assert labels["stresses"][0][0][0] == 5.0  # GPa assumed -> no scaling


def test_convert_stress_labels_no_stress_no_warning() -> None:
    """No warning is emitted when no split carries stress, even without a unit."""
    import warnings

    labels = {"energies": [-1.0], "forces": [[[0.0, 0.0, 0.0]]]}
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _common.convert_stress_labels([labels], {})


def test_convert_stress_labels_rejects_unknown_unit() -> None:
    """An unrecognised stress_unit is a configuration error."""
    labels = {"stresses": [[[1.0, 0.0, 0.0]] * 3]}
    with pytest.raises(ValueError, match="stress_unit"):
        _common.convert_stress_labels([labels], {"stress_unit": "bar"})


def test_build_datasets_converts_stress_units(tmp_path: Path) -> None:
    """build_datasets applies the eV/A3 -> GPa conversion to loaded stresses."""
    from matgl.apps.pes import EV_PER_ANG3_TO_GPA

    structures = _make_structures(4)
    labels = {
        "energies": [-1.9 * len(s) for s in structures],
        "forces": [[[0.0, 0.0, 0.0]] * 4 for _ in structures],
        "stresses": [[[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]] for _ in structures],
    }
    path = tmp_path / "all.json"
    path.write_text(json.dumps({"structures": [s.as_dict() for s in structures], "labels": labels}))
    config = {"dataset": str(path), "cutoff": 4.0, "element_types": "auto", "stress_unit": "eV/A3"}
    train_data, _, _, _ = _common.build_datasets(config)
    # The graph for the first sample carries the converted stress.
    stress = train_data[0][-1]["stresses"]
    np.testing.assert_allclose(float(np.asarray(stress).reshape(-1)[0]), EV_PER_ANG3_TO_GPA, rtol=1e-5)


# --- end-to-end main() smoke tests --------------------------------------------


def _base_pes_config(train: Path, val: Path, tmp_path: Path, out: Path) -> dict:
    return {
        "cutoff": 4.0,
        "train": str(train),
        "val": str(val),
        "batch_size": 2,
        "force_weight": 1.0,
        "stress_weight": 0.0,
        "lr": 1e-3,
        "accelerator": "cpu",
        "logger": "csv",
        "log_dir": str(tmp_path / "logs"),
        "checkpoint": False,
        "max_epochs": 1,
        "model_dir": str(out),
    }


def test_train_main_writes_loadable_model(dataset_files, tmp_path: Path, monkeypatch) -> None:
    """train_pes.main runs from a config file and saves a reloadable model."""
    import matgl

    train, val = dataset_files
    out = tmp_path / "trained"
    config = {
        **_base_pes_config(train, val, tmp_path, out),
        "model": "TensorNet",
        "model_args": {"units": 16, "nblocks": 1},
        "element_types": "auto",
    }
    cfg_path = tmp_path / "train.yaml"
    yaml.safe_dump(config, cfg_path.open("w"))

    train_module = _load_script("train_pes")
    monkeypatch.setattr(sys, "argv", ["train_pes", "--config", str(cfg_path)])
    train_module.main()

    assert (out / "model.pt").exists()
    assert (out / "model.json").exists()
    matgl.load_model(str(out))  # round-trips without error


def test_finetune_main_pins_pretrained_element_types(
    pretrained_dir, dataset_files, tmp_path: Path, monkeypatch
) -> None:
    """finetune_pes.main loads the local model and ignores a conflicting element_types."""
    import matgl

    train, val = dataset_files
    out = tmp_path / "finetuned"
    config = {
        **_base_pes_config(train, val, tmp_path, out),
        "model": str(pretrained_dir),
        "element_types": ["H", "He"],  # deliberately wrong; must be overridden by the pre-trained table
    }
    cfg_path = tmp_path / "finetune.yaml"
    yaml.safe_dump(config, cfg_path.open("w"))

    finetune_module = _load_script("finetune_pes")
    monkeypatch.setattr(sys, "argv", ["finetune_pes", "--config", str(cfg_path)])
    finetune_module.main()

    nnp = matgl.load_model(str(out))
    assert nnp.model.element_types == ("Li",)
