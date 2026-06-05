"""Shared helpers for the MatGL training / finetuning scripts.

Both ``train_pes.py`` (train a potential from scratch) and ``finetune_pes.py``
(continue training a pre-trained potential) read a single YAML/JSON config file and build the
same pieces: datasets loaded from pymatgen-serialized structure JSONs, MatGL
dataloaders, and a PyTorch Lightning trainer. Those pieces live here so the two
entry-point scripts stay thin and consistent.

The flow mirrors ``examples/Training a QET Potential with PyTorch Lightning.ipynb``
(converter -> ``MGLDataset`` -> ``MGLDataLoader`` -> ``PotentialLightningModule``
-> ``lightning.Trainer``) but is driven entirely by a config file and reads
structures + labels from disk rather than from the Materials Project API.

All imports go through MatGL's public APIs (``matgl.ext.pymatgen``,
``matgl.graph.data``, ``matgl.utils.training``) rather than private modules.
"""

from __future__ import annotations

import argparse
import json
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any

import lightning as pl
import numpy as np
import torch
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from monty.serialization import loadfn

from matgl.config import DEFAULT_ELEMENTS
from matgl.ext.pymatgen import Structure2Graph, get_element_list
from matgl.graph.data import MGLDataLoader, MGLDataset, collate_fn_pes, split_dataset
from matgl.layers import AtomRef
from matgl.utils.training import PotentialLightningModule

if TYPE_CHECKING:
    from lightning.pytorch.callbacks import Callback
    from pymatgen.core import Structure
    from torch.utils.data import DataLoader

# Label keys understood by ``collate_fn_pes`` beyond the always-present energies/forces.
OPTIONAL_LABEL_KEYS = ("stresses", "charges", "magmoms")


def parse_args(description: str) -> argparse.Namespace:
    """Parse the command-line arguments shared by the training scripts.

    Args:
        description: Help text describing the specific script.

    Returns:
        Parsed arguments with a single ``config`` attribute (path to the config file).
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "-c",
        "--config",
        required=True,
        help="Path to the YAML (or JSON) training config file.",
    )
    return parser.parse_args()


def load_config(path: str) -> dict[str, Any]:
    """Load a YAML or JSON config file.

    ``monty.serialization.loadfn`` dispatches on the file extension and handles
    both formats, so no explicit YAML dependency is needed.

    Args:
        path: Path to the config file.

    Returns:
        The config as a plain dictionary.
    """
    config = loadfn(path)
    if not isinstance(config, dict):
        raise ValueError(f"Config file {path!r} must deserialize to a mapping, got {type(config).__name__}.")
    # monty.loadfn parses YAML via ruamel, yielding CommentedMap / ScalarFloat etc.
    # Those subclasses leak into save_hyperparameters and break checkpoint (un)pickling
    # under torch's weights_only=True, so coerce everything to plain Python types.
    return json.loads(json.dumps(config))


def load_structures_labels(path: str) -> tuple[list[Structure], dict[str, list]]:
    """Load a dataset JSON of pymatgen structures and their labels.

    The file is expected to deserialize to a mapping of the form::

        {
            "structures": [ <pymatgen Structure dict>, ... ],
            "labels": {"energies": [...], "forces": [...], "stresses": [...], ...},
        }

    ``loadfn`` turns the serialized structure dicts back into ``Structure``
    objects automatically.

    Args:
        path: Path to the dataset JSON file.

    Returns:
        A ``(structures, labels)`` tuple ready to hand to :func:`build_dataset`.
    """
    data = loadfn(path)
    try:
        structures = list(data["structures"])
        labels = dict(data["labels"])
    except (KeyError, TypeError) as exc:
        raise ValueError(
            f"Dataset file {path!r} must be a mapping with 'structures' and 'labels' keys."
        ) from exc
    if len(structures) == 0:
        raise ValueError(f"Dataset file {path!r} contains no structures.")
    return structures, labels


def resolve_element_types(config: dict[str, Any], structures: list[Structure]) -> tuple[str, ...]:
    """Resolve the element table used to build graphs.

    ``config['element_types']`` may be an explicit list of symbols, the string
    ``"auto"`` (derive the table from the provided structures), or be omitted to
    use MatGL's ``DEFAULT_ELEMENTS``.

    Args:
        config: The training config.
        structures: Structures used when ``element_types == "auto"``.

    Returns:
        Tuple of element symbols.
    """
    element_types = config.get("element_types", None)
    if element_types is None:
        return tuple(DEFAULT_ELEMENTS)
    if element_types == "auto":
        return tuple(get_element_list(structures))
    return tuple(element_types)


def build_dataset(
    structures: list[Structure],
    labels: dict[str, list],
    *,
    cutoff: float,
    element_types: tuple[str, ...],
    include_line_graph: bool = False,
) -> MGLDataset:
    """Build an ``MGLDataset`` from structures and labels.

    Mirrors the notebook flow: a ``Structure2Graph`` converter feeds an
    ``MGLDataset``. The on-disk cache is disabled (``save_cache=False``) so
    repeated runs don't read stale graphs.

    Args:
        structures: Pymatgen structures.
        labels: Targets keyed by name (``energies``, ``forces``, and optionally
            ``stresses`` / ``charges`` / ``magmoms``).
        cutoff: Graph radial cutoff in Angstrom.
        element_types: Element table for the converter.
        include_line_graph: Whether to build three-body line graphs.

    Returns:
        The constructed dataset.
    """
    converter = Structure2Graph(element_types=element_types, cutoff=cutoff)
    include_charge = "charges" in labels
    return MGLDataset(
        structures=structures,
        labels=labels,
        converter=converter,
        include_line_graph=include_line_graph,
        include_ref_charge=include_charge,
        save_cache=False,
    )


def build_datasets(
    config: dict[str, Any],
) -> tuple[MGLDataset, MGLDataset, MGLDataset | None, tuple[str, ...]]:
    """Build train/val/test datasets from the config.

    Two layouts are supported:

    * **Separate files** -- ``config['train']`` and ``config['val']`` point at
      dataset JSONs (``config['test']`` optional).
    * **Single file + split** -- ``config['dataset']`` points at one JSON which
      is split via ``config['frac_list']`` (default ``[0.8, 0.1, 0.1]``).

    The element table is resolved once (from the training structures) and shared
    across all splits, then returned so callers don't have to reach into dataset
    internals (split datasets are plain ``Subset`` objects with no structures).

    Args:
        config: The training config.

    Returns:
        ``(train_data, val_data, test_data, element_types)``; ``test_data`` is
        ``None`` when no test set is available.
    """
    cutoff = config["cutoff"]
    include_line_graph = config.get("include_line_graph", False)

    if config.get("dataset"):
        structures, labels = load_structures_labels(config["dataset"])
        element_types = resolve_element_types(config, structures)
        dataset = build_dataset(
            structures,
            labels,
            cutoff=cutoff,
            element_types=element_types,
            include_line_graph=include_line_graph,
        )
        train_data, val_data, test_data = split_dataset(
            dataset,
            frac_list=config.get("frac_list", [0.8, 0.1, 0.1]),
            shuffle=config.get("shuffle", True),
            random_state=config.get("random_state", 42),
        )
        return train_data, val_data, test_data, element_types

    if not (config.get("train") and config.get("val")):
        raise ValueError(
            "Config must provide either 'dataset' (single file + 'frac_list') or both 'train' and 'val' file paths."
        )

    train_structs, train_labels = load_structures_labels(config["train"])
    val_structs, val_labels = load_structures_labels(config["val"])
    # Element table is resolved from the training structures so train/val/test share it.
    element_types = resolve_element_types(config, train_structs)
    build = partial(build_dataset, cutoff=cutoff, element_types=element_types, include_line_graph=include_line_graph)

    train_data = build(train_structs, train_labels)
    val_data = build(val_structs, val_labels)
    test_data = None
    if config.get("test"):
        test_structs, test_labels = load_structures_labels(config["test"])
        test_data = build(test_structs, test_labels)
    return train_data, val_data, test_data, element_types


def build_dataloaders(
    train_data: MGLDataset,
    val_data: MGLDataset,
    test_data: MGLDataset | None,
    config: dict[str, Any],
) -> tuple[DataLoader, DataLoader, DataLoader | None]:
    """Build train/val/test dataloaders.

    The PES collate function is configured from the weights and labels actually
    present: stress is collated when ``stress_weight > 0`` and charge when
    ``charge_weight > 0``.

    Args:
        train_data: Training dataset.
        val_data: Validation dataset.
        test_data: Optional test dataset.
        config: The training config.

    Returns:
        ``(train_loader, val_loader, test_loader)``; ``test_loader`` is ``None``
        when ``test_data`` is ``None``.
    """
    collate_fn = partial(
        collate_fn_pes,
        include_line_graph=config.get("include_line_graph", False),
        include_stress=config.get("stress_weight", 0.0) > 0,
        include_charge=config.get("charge_weight", 0.0) > 0,
        include_magmom=config.get("magmom_weight", 0.0) > 0,
    )
    loaders = MGLDataLoader(
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        collate_fn=collate_fn,
        batch_size=config["batch_size"],
        num_workers=config.get("num_workers", 0),
    )
    if test_data is None:
        train_loader, val_loader = loaders
        return train_loader, val_loader, None
    return loaders


def compute_element_refs(train_data: MGLDataset, element_types: tuple[str, ...]) -> np.ndarray:
    """Fit per-element energy offsets on the training set.

    A least-squares ``AtomRef`` is fit so the model learns residual energies.
    This mirrors the from-scratch training recipe; finetuning instead reuses the
    pre-trained model's existing offsets.

    Args:
        train_data: Training dataset (each item yields the graph at index 0 and
            the labels dict last).
        element_types: Element table, used to size the offset vector.

    Returns:
        The fitted ``property_offset`` as a numpy array.
    """
    from ase.data import atomic_numbers

    graphs = [item[0] for item in train_data]
    energies = torch.tensor([item[-1]["energies"] for item in train_data], dtype=torch.float32)
    max_z = int(np.max([atomic_numbers[el] for el in element_types])) + 1
    atom_ref = AtomRef(max_z=max_z)
    atom_ref.fit(graphs, energies)
    return atom_ref.property_offset


class TrainingLightningModule(PotentialLightningModule):
    """``PotentialLightningModule`` with config-driven optimizer / scheduler + warmup.

    MatGL's mixin steps the scheduler in ``on_train_epoch_end`` *and* returns it
    from ``configure_optimizers`` (which Lightning also steps), advancing the LR
    twice per epoch. We override both so the LR advances exactly once per epoch,
    every scheduler quantity (``warmup_epochs``, ``StepLR.step_size``,
    ``CosineAnnealingLR.T_max``, ``SequentialLR`` milestones) is interpreted in
    epochs, and a ``ReduceLROnPlateau`` scheduler gets its ``monitor`` metric.
    """

    def set_optim(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        monitor: str | None = None,
    ) -> None:
        """Attach the prebuilt optimizer / scheduler used by ``configure_optimizers``.

        Args:
            optimizer: The optimizer.
            scheduler: The (possibly warmup-wrapped) LR scheduler.
            monitor: Metric name for ``ReduceLROnPlateau``.
        """
        self._optimizer = optimizer
        self._scheduler = scheduler
        self._monitor = monitor

    def configure_optimizers(self) -> dict[str, Any]:  # type: ignore[override]
        """Hand Lightning the optimizer + an epoch-interval scheduler config."""
        lr_scheduler_config: dict[str, Any] = {"scheduler": self._scheduler, "interval": "epoch", "frequency": 1}
        if isinstance(self._scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            lr_scheduler_config["monitor"] = self._monitor
        return {"optimizer": self._optimizer, "lr_scheduler": lr_scheduler_config}

    def on_train_epoch_end(self) -> None:
        """No-op: Lightning steps the scheduler (configured above) once per epoch."""
        return


def build_optimizer(params: Any, config: dict[str, Any]) -> torch.optim.Optimizer:
    """Build the optimizer named by ``config['optimizer']`` (default ``Adam``).

    Args:
        params: Parameters to optimize.
        config: The training config (``optimizer``, ``lr``, ``optimizer_args``).

    Returns:
        The optimizer.
    """
    name = config.get("optimizer", "Adam")
    optimizer_args = {"lr": config.get("lr", 1e-3), **config.get("optimizer_args", {})}
    return getattr(torch.optim, name)(params, **optimizer_args)


def build_scheduler(
    optimizer: torch.optim.Optimizer, config: dict[str, Any]
) -> torch.optim.lr_scheduler.LRScheduler:
    """Build an epoch-based LR scheduler with optional linear warmup.

    The decay scheduler is the ``torch.optim.lr_scheduler`` class named by
    ``config['scheduler']`` (with ``config['scheduler_args']``), falling back to
    ``CosineAnnealingLR``. When ``config['warmup_epochs'] > 0`` a linear warmup is
    prepended via ``SequentialLR``.

    Args:
        optimizer: The optimizer to schedule.
        config: The training config.

    Returns:
        The (possibly warmup-wrapped) scheduler.
    """
    warmup_epochs = config.get("warmup_epochs", 0)
    lr = config.get("lr", 1e-3)

    if config.get("scheduler"):
        decay_scheduler = getattr(torch.optim.lr_scheduler, config["scheduler"])(
            optimizer, **config.get("scheduler_args", {})
        )
    else:
        decay_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.get("decay_steps", config["max_epochs"]) - warmup_epochs,
            eta_min=lr * config.get("decay_alpha", 0.0),
        )

    if warmup_epochs > 0:
        if isinstance(decay_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            raise ValueError(
                "warmup_epochs is incompatible with ReduceLROnPlateau (SequentialLR cannot chain a "
                "metric-based scheduler). Use a step-count scheduler such as StepLR or CosineAnnealingLR."
            )
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1e-8, end_factor=1.0, total_iters=warmup_epochs
        )
        return torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_scheduler, decay_scheduler], milestones=[warmup_epochs]
        )
    return decay_scheduler


def build_potential_module(
    model: torch.nn.Module, config: dict[str, Any], element_refs: np.ndarray | None
) -> TrainingLightningModule:
    """Wrap a graph model in a Lightning module with config-driven optim/schedule.

    Args:
        model: The graph network to train (e.g. an ``M3GNet``).
        config: The training config (loss weights, lr, optimizer, scheduler, warmup).
        element_refs: Per-element energy offsets, or ``None`` to skip.

    Returns:
        The configured Lightning module.
    """
    lit_module = TrainingLightningModule(
        model=model,
        element_refs=element_refs,
        include_line_graph=config.get("include_line_graph", False),
        energy_weight=config.get("energy_weight", 1.0),
        force_weight=config.get("force_weight", 1.0),
        stress_weight=config.get("stress_weight", 0.0),
        magmom_weight=config.get("magmom_weight", 0.0),
        charge_weight=config.get("charge_weight", 0.0),
        lr=config.get("lr", 1e-3),
    )
    optimizer = build_optimizer(lit_module.parameters(), config)
    scheduler = build_scheduler(optimizer, config)
    lit_module.set_optim(optimizer, scheduler, monitor=config.get("metric_to_track", "val_Total_Loss"))
    return lit_module


def build_logger(config: dict[str, Any]) -> CSVLogger | WandbLogger:
    """Build the Lightning logger named by ``config['logger']`` (``csv`` or ``wandb``).

    Args:
        config: The training config.

    Returns:
        A ``CSVLogger`` (default) or a ``WandbLogger``.
    """
    logger_name = config.get("logger", "csv")
    if logger_name == "wandb":
        return WandbLogger(
            project=config.get("wandb_project"),
            name=config.get("wandb_name"),
            tags=config.get("tags", []),
        )
    if logger_name == "csv":
        return CSVLogger(config.get("log_dir", "logs"), name=config.get("run_name", "training"))
    raise ValueError(f"Unknown logger {logger_name!r}; expected 'csv' or 'wandb'.")


def build_callbacks(config: dict[str, Any]) -> list[Callback]:
    """Build optional training callbacks (checkpointing).

    When ``config['checkpoint']`` is truthy (default), a ``ModelCheckpoint`` is
    returned that tracks ``config['metric_to_track']`` and always writes
    ``last.ckpt`` for easy resumption. These ``.ckpt`` files capture full training
    state (weights, optimizer, scheduler, epoch) and are distinct from the
    deployable model written by ``lit_module.model.save(...)`` at the end.

    Args:
        config: The training config.

    Returns:
        A list of callbacks (possibly empty).
    """
    if not config.get("checkpoint", True):
        return []
    monitor = config.get("metric_to_track", "val_Total_Loss")
    checkpoint_cb = ModelCheckpoint(
        dirpath=config.get("checkpoint_dir", "checkpoints"),
        filename="{epoch:03d}-{" + monitor + ":.4f}",
        monitor=monitor,
        mode="min",
        save_top_k=config.get("save_top_k", 1),
        save_last=True,
        every_n_epochs=config.get("checkpoint_every_n_epochs"),
    )
    return [checkpoint_cb]


def resume_ckpt_path(config: dict[str, Any]) -> str | None:
    """Resolve the checkpoint path to resume training from.

    ``config['resume_from']`` may be an explicit ``.ckpt`` path or the string
    ``"last"`` (resolves to ``<checkpoint_dir>/last.ckpt`` if it exists). Returns
    ``None`` when there is nothing to resume.

    Args:
        config: The training config.

    Returns:
        Path to a checkpoint, or ``None``.
    """
    resume = config.get("resume_from")
    if not resume:
        return None
    if resume == "last":
        last = Path(config.get("checkpoint_dir", "checkpoints")) / "last.ckpt"
        return str(last) if last.exists() else None
    return resume


def build_trainer(
    config: dict[str, Any],
    logger: CSVLogger | WandbLogger,
    extra_callbacks: list[Callback] | None = None,
) -> pl.Trainer:
    """Build the Lightning trainer.

    ``inference_mode=False`` is required so forces/stresses (computed via
    autograd) are available during test/predict.

    Args:
        config: The training config (``max_epochs``, ``accelerator``, ``devices``).
        logger: The Lightning logger.
        extra_callbacks: Additional callbacks to register (e.g. checkpointing).

    Returns:
        The configured trainer.
    """
    callbacks: list[Callback] = [LearningRateMonitor(logging_interval="epoch")]
    if extra_callbacks:
        callbacks.extend(extra_callbacks)
    return pl.Trainer(
        max_epochs=config["max_epochs"],
        accelerator=config.get("accelerator", "cpu"),
        devices=config.get("devices", 1),
        logger=logger,
        inference_mode=False,
        callbacks=callbacks,
    )
