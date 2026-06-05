"""Shared helpers for the MatGL training / finetuning scripts.

Both ``train.py`` (train a model from scratch) and ``finetune.py`` (continue
training a pre-trained model) read a single YAML/JSON config file and build the
same pieces: datasets loaded from pymatgen-serialized structure JSONs, MatGL
dataloaders, and a PyTorch Lightning trainer. Those pieces live here so the two
entry-point scripts stay thin and consistent.

The flow mirrors ``examples/Training a QET Potential with PyTorch Lightning.ipynb``
(converter -> ``MGLDataset`` -> ``MGLDataLoader`` -> ``PotentialLightningModule``
-> ``lightning.Trainer``) but is driven entirely by a config file and reads
structures + labels from disk rather than from the Materials Project API.

All imports go through MatGL's public, backend-dispatched APIs
(``matgl.ext.pymatgen``, ``matgl.graph.data``), so the scripts follow whatever
``MATGL_BACKEND`` selects (PYG by default).
"""

from __future__ import annotations

import argparse
from functools import partial
from typing import TYPE_CHECKING, Any

from monty.serialization import loadfn

from matgl.config import DEFAULT_ELEMENTS
from matgl.ext.pymatgen import Structure2Graph, get_element_list
from matgl.graph.data import MGLDataLoader, MGLDataset, collate_fn_pes, split_dataset

if TYPE_CHECKING:
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
    return config


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


def build_datasets(config: dict[str, Any]) -> tuple[MGLDataset, MGLDataset, MGLDataset | None]:
    """Build train/val/test datasets from the config.

    Two layouts are supported:

    * **Separate files** -- ``config['train']`` and ``config['val']`` point at
      dataset JSONs (``config['test']`` optional).
    * **Single file + split** -- ``config['dataset']`` points at one JSON which
      is split via ``config['frac_list']`` (default ``[0.8, 0.1, 0.1]``).

    Args:
        config: The training config.

    Returns:
        ``(train_data, val_data, test_data)``; ``test_data`` is ``None`` when no
        test set is available.
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
        return train_data, val_data, test_data

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
    return train_data, val_data, test_data


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
