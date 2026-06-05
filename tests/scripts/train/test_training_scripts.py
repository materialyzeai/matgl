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


@pytest.fixture
def dataset_files(tmp_path: Path) -> tuple[Path, Path]:
    train, val = tmp_path / "train.json", tmp_path / "val.json"
    _write_dataset(train, 6)
    _write_dataset(val, 2)
    return train, val


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
