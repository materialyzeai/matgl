"""Tools to construct a dataset of PYG graphs."""

from __future__ import annotations

import bisect
import hashlib
import json
import logging
import math
import os
import shutil
import uuid
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import lightning as pl
import numpy as np
import torch
from monty.json import MontyDecoder
from torch.utils.data import DataLoader, Sampler, SequentialSampler, Subset
from torch.utils.data import Dataset as TorchDataset
from torch_geometric.data import Batch, Data, Dataset
from tqdm import trange

import matgl

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator

    from matgl.graph.converters import GraphConverter

logger = logging.getLogger(__name__)

# Bump this when the on-disk format changes in a backwards-incompatible way so
# old caches are invalidated automatically (a stricter version of the cutoff
# fingerprint below).
_CACHE_FORMAT_VERSION = 2


def _compute_cache_fingerprint(
    converter: GraphConverter | None,
    include_line_graph: bool,
    include_ref_charge: bool,
    graph_labels: list | None = None,
) -> dict[str, object]:
    """Build a small, JSON-serializable fingerprint of the active dataset config.

    Stored alongside processed graphs so that ``has_cache`` can detect a
    config drift (changed cutoff, different element list, swapped converter
    class, or a change in the ``graph_labels`` that back the state attributes)
    and trigger reprocessing rather than silently returning stale data.
    """
    if converter is None:
        converter_class = None
        cutoff: float | None = None
        element_hash: str | None = None
    else:
        converter_class = type(converter).__name__
        cutoff = float(getattr(converter, "cutoff", float("nan")))
        element_types = getattr(converter, "element_types", None)
        if element_types is None:
            element_hash = None
        else:
            element_hash = hashlib.sha1("|".join(map(str, element_types)).encode("utf-8")).hexdigest()[:16]
    # ``graph_labels`` drives the state attributes (e.g. multi-fidelity ids). A
    # cache built with different (or no) graph_labels must not be silently
    # reused, so fold a hash of them into the fingerprint.
    if graph_labels is None:
        graph_labels_hash: str | None = None
    else:
        payload = json.dumps(graph_labels, sort_keys=True, default=str).encode("utf-8")
        graph_labels_hash = hashlib.sha1(payload).hexdigest()[:16]
    return {
        "format_version": _CACHE_FORMAT_VERSION,
        "converter_class": converter_class,
        "cutoff": cutoff,
        "element_hash": element_hash,
        "graph_labels_hash": graph_labels_hash,
        "include_line_graph": include_line_graph,
        "include_ref_charge": include_ref_charge,
    }


def _default_loader_kwargs(user_kwargs: dict) -> dict:
    """Fill in num_workers / pin_memory / persistent_workers when the caller didn't.

    Most users miss these flags entirely and become CPU-bound when training on
    GPU. Defaults aim to be safe (low worker count, no oversubscription on
    headless CI machines) and only kick in when the user hasn't expressed an
    opinion. ``pin_memory`` only matters when CUDA is available.
    """
    out = dict(user_kwargs)
    if "num_workers" not in out:
        # Conservative default: works on CI single-CPU runners and 32-core
        # workstations. Users with GPU clusters will typically override.
        cpu_count = os.cpu_count() or 1
        out["num_workers"] = min(4, cpu_count)
    if "pin_memory" not in out:
        out["pin_memory"] = torch.cuda.is_available()
    if "persistent_workers" not in out and out.get("num_workers", 0) > 0:
        out["persistent_workers"] = True
    return out


def ensure_batch_attribute(data: Data) -> Data:
    """Ensure a PyG Data object has a batch attribute.

    Args:
        data: PyG Data object.

    Returns:
        Data object with batch attribute set.
    """
    if not hasattr(data, "batch") or data.batch is None:
        data.batch = torch.zeros(data.num_nodes, dtype=torch.long, device=data.x.device)
    return data


def split_dataset(
    self, frac_list: list[float] | None = None, shuffle: bool = False, random_state: int = 42
) -> tuple[Subset, Subset, Subset]:
    """Split a dataset into train/val/test ``Subset``s.

    Args:
        self: Dataset to split (used as a method on ``MGLDataset``).
        frac_list: Fractions for the train/val/test splits. Defaults to ``[0.8, 0.1, 0.1]``.
        shuffle: Whether to shuffle indices before splitting.
        random_state: Seed used when ``shuffle`` is True.

    Returns:
        Tuple of (train, val, test) ``Subset`` views of ``self``.
    """
    if frac_list is None:
        frac_list = [0.8, 0.1, 0.1]
    num_graphs = len(self)
    num_train = int(frac_list[0] * num_graphs)
    num_val = int(frac_list[1] * num_graphs)

    # Split indices are plain integers used only for host-side slicing, so pin them to CPU.
    # This keeps split_dataset working even when torch.set_default_device("mps"/"cuda") is
    # active, which would otherwise make randperm expect a matching-device generator.
    indices = (
        torch.randperm(num_graphs, generator=torch.Generator().manual_seed(random_state), device="cpu")
        if shuffle
        else torch.arange(num_graphs, device="cpu")
    )
    train_idx = indices[:num_train].tolist()
    val_idx = indices[num_train : num_train + num_val].tolist()
    test_idx = indices[num_train + num_val :].tolist()

    return (Subset(self, train_idx), Subset(self, val_idx), Subset(self, test_idx))


def collate_fn_graph(
    batch: list, multiple_values_per_target: bool = False
) -> tuple[Batch | Data, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Merge a list of PyG graphs to form a batch.

    Args:
        batch: List of tuples, each containing (graph, lattice, [line_graph,] state_attr, labels).
        multiple_values_per_target: Whether labels are tensors (True) or scalars (False).

    Returns:
        Tuple containing:
        - g: PyG Data (single graph) or Batch (multiple graphs) object.
        - lat: Lattice tensor (batch_size, 3, 3) or (3, 3) for single graph.
        - state_attr: Stacked state attributes (batch_size, state_dim).
        - labels: Stacked or tensorized labels (batch_size, ...) or (batch_size,).
    """
    graphs, lattices, state_attr, labels = map(list, zip(*batch, strict=False))

    g = Batch.from_data_list(graphs)  # Batch main graphs
    labels_tensor: torch.Tensor = (
        torch.vstack([next(iter(d.values())) for d in labels])  # type:ignore[assignment]
        if multiple_values_per_target
        else torch.tensor([next(iter(d.values())) for d in labels], dtype=matgl.float_th)
    )
    state_attr_tensor: torch.Tensor = torch.stack(state_attr)  # type:ignore[assignment]
    lat: torch.Tensor = lattices[0] if g.batch_size == 1 else torch.squeeze(torch.stack(lattices))  # type: ignore[assignment]

    return g, lat, state_attr_tensor, labels_tensor


def collate_fn_pes(
    batch: list,
    include_stress: bool = True,
    include_line_graph: bool = False,
    include_magmom: bool = False,
    include_charge: bool = False,
) -> tuple:
    """Merge a list of PyG Data objects to form a batch.

    Args:
        batch: List of tuples, each containing (graph, lattices, [line_graphs,] state_attr, labels)
        include_stress (bool): Whether to include stress tensors in the output
        include_line_graph (bool): Whether to include line graphs in the batch
        include_magmom (bool): Whether to include magnetic moments in the output
        include_charge (bool): Whether to include per-atom charges in the output

    Returns:
        Tuple containing:
        - g: Batched PyG graph (Batch object)
        - lat: Stacked lattice tensors (batch_size, ...)
        - state_attr: Stacked state attributes (batch_size, state_dim)
        - e: Energies (batch_size,)
        - f: Forces (num_atoms, 3)
        - s: Stresses (batch_size, 6) or zeros if include_stress=False
        - m: Magnetic moments (batch_size, ...) or zeros if include_magmom=False
        - q: Per-atom charges concatenated across the batch (only when include_charge=True)
    """
    graphs, lattices, state_attr, labels = map(list, zip(*batch, strict=False))

    g = Batch.from_data_list(graphs)  # Batch main graphs
    e = torch.tensor([d["energies"] for d in labels], dtype=matgl.float_th)
    f = torch.vstack([d["forces"] for d in labels])
    s = (
        torch.vstack([d["stresses"] for d in labels])
        if include_stress
        else torch.zeros(e.size(0), dtype=matgl.float_th)
    )
    m = torch.vstack([d["magmoms"] for d in labels]) if include_magmom else torch.zeros(e.size(0), dtype=matgl.float_th)
    q = torch.hstack([d["charges"] for d in labels]) if include_charge else torch.zeros(e.size(0), dtype=matgl.float_th)
    state_attr = torch.stack(state_attr)  # type:ignore[assignment]
    lat = lattices[0] if g.batch_size == 1 else torch.squeeze(torch.stack(lattices))
    if include_magmom:
        return g, lat.squeeze(), state_attr, e, f, s, m
    if include_charge:
        return g, lat.squeeze(), state_attr, e, f, s, q
    return g, lat.squeeze(), state_attr, e, f, s


def _pick_collate_fn(labels: dict) -> Callable:
    """Pick the right collate function from a dataset's label keys.

    The two collate functions return different tuple shapes, so the choice has
    to match the labels actually present in the dataset. Logic:

    - No ``forces`` -> generic property-prediction (``collate_fn_graph``).
    - PES with ``forces`` -> ``collate_fn_pes`` with stress/magmom/charge flags
      enabled based on which optional keys are present.

    ``magmoms`` and ``charges`` are mutually exclusive in ``collate_fn_pes``'s
    return shape; we prefer ``magmoms`` when both happen to be present.
    """
    if "forces" not in labels:
        return collate_fn_graph
    include_stress = "stresses" in labels
    if "magmoms" in labels:
        return partial(collate_fn_pes, include_stress=include_stress, include_magmom=True)
    if "charges" in labels:
        return partial(collate_fn_pes, include_stress=include_stress, include_charge=True)
    return partial(collate_fn_pes, include_stress=include_stress)


def MGLDataLoader(
    train_data: MGLDataset | MGLDiskDataset,
    val_data: MGLDataset | MGLDiskDataset,
    collate_fn: Callable | None = None,
    test_data: MGLDataset | MGLDiskDataset | None = None,
    shard_seed: int = 0,
    **kwargs,
) -> tuple[DataLoader, ...]:
    """Dataloader for MatGL training in PyTorch Geometric.

    Args:
        train_data: Training dataset. Disk datasets automatically use shard-aware batches.
        val_data: Validation dataset. Must use the same storage type as ``train_data``.
        collate_fn (Callable, optional): Collate function for batching. When ``None`` (default),
            one is auto-selected from the training dataset's label keys: ``collate_fn_graph`` for
            single-target property prediction (no ``forces`` key), or ``collate_fn_pes`` with
            stress / magmom / charge flags toggled on based on the keys actually present. Pass
            an explicit callable (e.g. ``partial(collate_fn_pes, include_stress=False)``) to
            override.
        test_data (Dataset, optional): Test dataset (PyG Dataset or subset). Defaults to None.
        shard_seed: Base seed for epoch-wise shuffling of disk-backed shards.
        **kwargs: Pass-through kwargs to torch_geometric.loader.DataLoader. Common ones you may want to set are
            batch_size, num_workers, pin_memory, and generator.

    Returns:
        Tuple[DataLoader, ...]: Train, validation, and test data loaders. Test data loader is None if test_data is None.

    Notes:
        ``num_workers``, ``pin_memory``, and ``persistent_workers`` default to
        sensible values when not supplied (a small worker pool, page-locked
        CUDA transfers when a GPU is visible, and persistent workers when
        ``num_workers > 0`` so the pool isn't torn down between epochs). Pass
        them explicitly to override.
    """
    if isinstance(train_data, MGLDiskDataset):
        if not isinstance(val_data, MGLDiskDataset) or (
            test_data is not None and not isinstance(test_data, MGLDiskDataset)
        ):
            raise TypeError("train, validation, and test datasets must all use MGLDiskDataset")
        if collate_fn is None:
            collate_fn = _pick_collate_fn(train_data.labels)

        loader_kwargs = _default_loader_kwargs(kwargs)
        batch_size = loader_kwargs.pop("batch_size", 1)
        drop_last = loader_kwargs.pop("drop_last", False)
        controlled = {"batch_sampler", "sampler", "shuffle"}.intersection(loader_kwargs)
        if controlled:
            raise ValueError(f"disk-backed loading controls {sorted(controlled)}")
        disk_train_loader = _disk_loader(
            train_data,
            collate_fn=collate_fn,
            batch_size=batch_size,
            shuffle=True,
            drop_last=drop_last,
            seed=shard_seed,
            loader_kwargs=loader_kwargs,
        )
        disk_val_loader = _disk_loader(
            val_data,
            collate_fn=collate_fn,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            seed=shard_seed,
            loader_kwargs=loader_kwargs,
        )
        if test_data is not None:
            disk_test_loader = _disk_loader(
                test_data,
                collate_fn=collate_fn,
                batch_size=batch_size,
                shuffle=False,
                drop_last=False,
                seed=shard_seed,
                loader_kwargs=loader_kwargs,
            )
            return disk_train_loader, disk_val_loader, disk_test_loader
        return disk_train_loader, disk_val_loader

    if isinstance(val_data, MGLDiskDataset) or isinstance(test_data, MGLDiskDataset):
        raise TypeError("train, validation, and test datasets must use the same storage type")

    if collate_fn is None:
        # Peel ``Subset`` (the common shape after ``split_dataset``) to reach
        # the underlying ``MGLDataset`` whose ``labels`` drive the dispatch.
        base = train_data.dataset if isinstance(train_data, Subset) else train_data
        collate_fn = _pick_collate_fn(getattr(base, "labels", {}))

    kwargs = _default_loader_kwargs(kwargs)
    train_loader: DataLoader = DataLoader(train_data, shuffle=True, collate_fn=collate_fn, **kwargs)
    val_loader: DataLoader = DataLoader(val_data, shuffle=False, collate_fn=collate_fn, **kwargs)
    if test_data is not None:
        test_loader: DataLoader = DataLoader(test_data, shuffle=False, collate_fn=collate_fn, **kwargs)
        return train_loader, val_loader, test_loader
    return train_loader, val_loader


class MGLDataset(Dataset):
    """Create a dataset including PyTorch Geometric graphs."""

    def __init__(
        self,
        filename: str = "pyg_graph.pt",
        filename_lattice: str = "lattice.pt",
        filename_line_graph: str = "pyg_line_graph.pt",
        filename_state_attr: str = "state_attr.pt",
        filename_labels: str = "labels.json",
        include_line_graph: bool = False,
        include_ref_charge: bool = False,
        converter: GraphConverter | None = None,
        structures: list | None = None,
        labels: dict[str, list] | None = None,
        root: str = "MGLDataset",
        graph_labels: list[int | float] | None = None,
        clear_processed: bool = False,
        save_cache: bool = True,
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):
        """Initialize the MGLDataset.

        Args:
            filename: File name for storing PyG graphs.
            filename_lattice: File name for storing lattice matrices.
            filename_line_graph: File name for storing PyG line graphs.
            filename_state_attr: File name for storing state attributes.
            filename_labels: File name for storing labels.
            include_line_graph: Whether to include line graphs.
            include_ref_charge: Whether to attach reference charges as ``data.q_ref`` for use by QEq.
            converter: Graph converter for PyG (converts structures to Data objects).
            structures: Pymatgen structures.
            labels: Targets as a dict of {name: list of values}.
            root: Root directory where the dataset should be saved.
            transform: A function/transform that takes in a Data or HeteroData object and returns a transformed version.
            pre_transform: A function/transform that takes in a Data or HeteroData object
                and returns a transformed version.
            pre_filter: A function that takes in a Data or HeteroData object and returns a boolean value.
            directory_name: Name of the directory to store the dataset.
            graph_labels: State attributes.
            clear_processed: Whether to clear stored structures after processing.
            save_cache: Whether to save the processed dataset.
        """
        self.filename = filename
        self.filename_lattice = filename_lattice
        self.filename_line_graph = filename_line_graph
        self.filename_state_attr = filename_state_attr
        self.filename_labels = filename_labels
        self.filename_fingerprint = "fingerprint.json"
        self.include_line_graph = include_line_graph
        self.include_ref_charge = include_ref_charge
        self.converter = converter
        self.structures = structures or []
        self.labels = labels or {}
        for k, v in self.labels.items():
            self.labels[k] = v.tolist() if isinstance(v, np.ndarray) else v
        self.graph_labels = graph_labels
        self.clear_processed = clear_processed
        self.save_cache = save_cache
        self.root = root

        super().__init__(root, transform, pre_transform, pre_filter)

        # Load or process data

        if self.has_cache():
            self.load()

        if self.clear_processed:
            shutil.rmtree(Path(self.root) / "processed", ignore_errors=True)

    def has_cache(self) -> bool:
        """Check if the processed files exist and match the current converter config.

        When a ``converter`` is supplied, the stored fingerprint must match the
        active config (cutoff, element list, converter class, line-graph /
        ref-charge flags). A drift returns False so we reprocess rather than
        silently loading stale graphs.

        The "load-only" flow (no ``converter``, no ``structures``) intentionally
        skips the equality check: the caller is explicitly pointing at a
        pre-built cache directory and saying "load it." We only require the
        four data files to exist in that case.
        """
        root = Path(self.root)
        files_to_check = [
            self.filename,
            self.filename_lattice,
            self.filename_state_attr,
            self.filename_labels,
        ]
        if not all((root / f).exists() for f in files_to_check):
            return False

        # Load-only flow: trust the existing cache when the user did not pass a
        # converter (and thus has nothing to reprocess from anyway).
        if self.converter is None:
            return True

        expected = _compute_cache_fingerprint(
            self.converter, self.include_line_graph, self.include_ref_charge, self.graph_labels
        )
        fingerprint_path = root / self.filename_fingerprint
        if not fingerprint_path.exists():
            logger.warning(
                "MGLDataset cache at %s has no fingerprint; reprocessing to avoid stale graphs.",
                self.root,
            )
            return False
        try:
            stored = json.loads(fingerprint_path.read_text())
        except (json.JSONDecodeError, OSError) as err:
            logger.warning("Unreadable MGLDataset cache fingerprint at %s (%s); reprocessing.", fingerprint_path, err)
            return False
        if stored != expected:
            logger.warning(
                "MGLDataset cache at %s was built with a different converter config "
                "(stored=%s, expected=%s); reprocessing.",
                self.root,
                stored,
                expected,
            )
            return False
        return True

    def process(self) -> None:
        """Convert Pymatgen structures into PyG Data objects."""
        if self.has_cache():
            pass
        else:
            num_graphs = len(self.structures)
            graphs, lattices, state_attrs = [], [], []

            for idx in trange(num_graphs):
                structure = self.structures[idx]
                # Converter returns (Data, lattice, state_attr)
                assert self.converter is not None, "converter must be provided"
                data, lattice, state_attr = self.converter.get_graph(structure)
                data = data.to(device="cpu")
                lattice = lattice.to(device="cpu")

                if self.include_ref_charge:
                    data.q_ref = torch.tensor(self.labels["charges"][idx], dtype=matgl.float_th)

                graphs.append(data)
                lattices.append(lattice)
                state_attrs.append(state_attr)

            state_attrs_tensor: torch.Tensor = (
                torch.tensor(self.graph_labels, dtype=torch.long)
                if self.graph_labels is not None
                else torch.tensor(np.array(state_attrs), dtype=matgl.float_th)
            )

            if self.clear_processed:
                del self.structures
                self.structures = []
            self.graphs = graphs
            self.lattices = lattices
            self.state_attr = state_attrs_tensor

            # Validate loaded or processed data
            if not self.graphs:
                raise ValueError("Dataset is empty after loading or processing")
            self.save()

    def save(self) -> None:
        """Save PyG graphs, labels, and a cache fingerprint to processed_dir."""
        if not self.save_cache:
            return

        root = Path(self.root)
        root.mkdir(parents=True, exist_ok=True)

        if self.labels:
            with (root / self.filename_labels).open("w") as file:
                json.dump(self.labels, file)

        torch.save(self.graphs, root / self.filename)
        torch.save(self.lattices, root / self.filename_lattice)
        torch.save(self.state_attr, root / self.filename_state_attr)

        # Write the fingerprint last so a partial cache (e.g. crash mid-save)
        # is detected as stale on the next run.
        fingerprint = _compute_cache_fingerprint(
            self.converter, self.include_line_graph, self.include_ref_charge, self.graph_labels
        )
        (root / self.filename_fingerprint).write_text(json.dumps(fingerprint, indent=2, sort_keys=True))

    def load(self) -> None:
        """Load PyG graphs from files."""
        root = Path(self.root)
        self.graphs = torch.load(root / self.filename, weights_only=False)
        self.lattices = torch.load(root / self.filename_lattice, weights_only=False)
        self.state_attr = torch.load(root / self.filename_state_attr, weights_only=False)
        with (root / self.filename_labels).open() as f:
            self.labels = json.load(f)

    def __getitem__(self, idx: int) -> tuple:
        """Get graph and associated data with idx."""
        if idx >= len(self.graphs):
            raise IndexError(f"Index {idx} out of range for dataset with {len(self.graphs)} graphs")
        items = [
            self.graphs[idx],
            self.lattices[idx],
            self.state_attr[idx],
            {
                k: torch.tensor(v[idx], dtype=matgl.float_th)
                for k, v in self.labels.items()
                if not isinstance(v[idx], str)
            },
        ]
        return tuple(items)

    def __len__(self) -> int:
        """Get size of dataset."""
        return len(self.graphs)

    @property
    def processed_file_names(self) -> list[str]:
        """List of processed file names."""
        return []

    @property
    def raw_file_names(self) -> list[str]:
        """List of raw file names (not used in this case)."""
        return []


LARGE_DATA_FORMAT_VERSION = 1
LabelLayout = Literal["graph", "node", "unchecked"]

# Unknown properties remain supported and can be classified by callers through
# ``label_layouts``. These defaults cover MatGL's built-in training targets.
DEFAULT_LABEL_LAYOUTS: dict[str, LabelLayout] = {
    "alpha": "graph",
    "charges": "node",
    "energies": "graph",
    "forces": "node",
    "magmoms": "node",
    "spin": "node",
    "stresses": "graph",
    "total_charge": "graph",
}


def _large_state_attr(value: Any, graph_label: Any = None) -> torch.Tensor:
    """Normalize a state attribute or explicit graph label onto the CPU."""
    if graph_label is not None:
        return torch.as_tensor(graph_label, dtype=torch.long, device="cpu")
    return torch.as_tensor(value, dtype=matgl.float_th, device="cpu")


def _atomic_torch_save(value: Any, path: Path) -> None:
    """Write a Torch payload atomically within its destination directory."""
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json_save(value: Any, path: Path) -> None:
    """Write JSON atomically within its destination directory."""
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True))
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _validate_labels(
    labels: dict[str, Any],
    *,
    n_atoms: int | None,
    record_index: int,
    layouts: dict[str, LabelLayout],
    expected_shapes: dict[str, tuple[int, ...] | None],
) -> None:
    """Validate fixed graph shapes and variable-leading-dimension node shapes."""
    for key, value in labels.items():
        if isinstance(value, str):
            shape = None
        else:
            tensor = torch.as_tensor(value)
            layout = layouts.get(key, "unchecked")
            if layout == "node":
                if tensor.ndim == 0:
                    raise ValueError(f"record {record_index}: node label {key!r} must have a leading atom dimension")
                if n_atoms is not None and tensor.shape[0] != n_atoms:
                    raise ValueError(
                        f"record {record_index}: {key!r} has leading dimension {tensor.shape[0]}, expected {n_atoms}"
                    )
                shape = tuple(tensor.shape[1:])
            elif layout == "graph":
                shape = tuple(tensor.shape)
            else:
                shape = None

        if key not in expected_shapes:
            expected_shapes[key] = shape
        elif shape is not None and shape != expected_shapes[key]:
            raise ValueError(
                f"record {record_index}: {key!r} has shape signature {shape}, expected {expected_shapes[key]}"
            )


def _prepare_disk_graph(
    converter: GraphConverter,
    structure: Any,
    labels: dict[str, Any],
    *,
    include_ref_charge: bool,
) -> tuple[Data, torch.Tensor, Any]:
    """Convert one structure using the same graph contract as ``MGLDataset``."""
    graph, lattice, state_attr = converter.get_graph(structure)
    graph = graph.to(device="cpu")
    lattice = torch.as_tensor(lattice, dtype=matgl.float_th, device="cpu")
    if include_ref_charge:
        graph.q_ref = labels["charges"]
    return graph, lattice, state_attr


def write_mgl_shards(
    records: Iterable[dict[str, Any]],
    root: str | Path,
    *,
    converter: GraphConverter | None = None,
    shard_size: int = 1000,
    precomputed: bool = True,
    include_ref_charge: bool = False,
    label_layouts: dict[str, LabelLayout] | None = None,
) -> None:
    """Stream records into transactional, CPU-backed PyG shards.

    Each record must contain ``structure`` and a per-record ``labels`` mapping.
    All records must have the same label keys. The manifest is replaced only
    after every new shard has been written successfully, so readers continue
    to see the prior complete generation if conversion is interrupted.

    Args:
        records: Iterable of structure/label records.
        root: Directory in which to write shards and ``metadata.json``.
        converter: Graph converter, required when ``precomputed=True``.
        shard_size: Maximum records in each independently loadable shard.
        precomputed: Whether to convert structures while writing.
        include_ref_charge: Whether to attach per-atom ``q_ref`` to graphs.
        label_layouts: Optional graph/node/unchecked layout overrides.
    """
    if shard_size <= 0:
        raise ValueError("shard_size must be greater than zero")
    if precomputed and converter is None:
        raise ValueError("converter is required when precomputed=True")

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "metadata.json"
    old_shards: set[str] = set()
    if manifest_path.is_file():
        try:
            old_manifest = json.loads(manifest_path.read_text())
            old_shards = {entry["file"] for entry in old_manifest.get("shards", [])}
        except (json.JSONDecodeError, KeyError, TypeError):
            old_shards = set()

    generation = uuid.uuid4().hex[:12]
    layouts = dict(DEFAULT_LABEL_LAYOUTS)
    if label_layouts:
        invalid = {key: value for key, value in label_layouts.items() if value not in {"graph", "node", "unchecked"}}
        if invalid:
            raise ValueError(f"invalid label layouts: {invalid}")
        layouts.update(label_layouts)
    expected_shapes: dict[str, tuple[int, ...] | None] = {}
    expected_label_keys: tuple[str, ...] | None = None
    shard: list[Any] = []
    shard_entries: list[dict[str, Any]] = []
    created_paths: list[Path] = []
    n = 0

    def flush() -> None:
        nonlocal shard
        if not shard:
            return
        filename = f"shard_{generation}_{len(shard_entries):06d}.pt"
        path = root / filename
        _atomic_torch_save(shard, path)
        created_paths.append(path)
        shard_entries.append({"file": filename, "n": len(shard)})
        shard = []

    try:
        for record_index, record in enumerate(records):
            if "structure" not in record or "labels" not in record:
                raise ValueError(f"record {record_index}: expected 'structure' and 'labels'")
            if not isinstance(record["labels"], dict):
                raise TypeError(f"record {record_index}: 'labels' must be a mapping")
            label_keys = tuple(sorted(record["labels"]))
            if expected_label_keys is None:
                expected_label_keys = label_keys
            elif label_keys != expected_label_keys:
                raise ValueError(
                    f"record {record_index}: label keys {label_keys} do not match expected keys {expected_label_keys}"
                )

            labels = {
                key: value if isinstance(value, str) else torch.as_tensor(value, dtype=matgl.float_th, device="cpu")
                for key, value in record["labels"].items()
            }
            if include_ref_charge and "charges" not in labels:
                raise ValueError(f"record {record_index}: include_ref_charge requires 'charges'")
            structure = record["structure"]
            item: tuple[Any, ...]

            if precomputed:
                if isinstance(structure, dict):
                    structure = MontyDecoder().process_decoded(structure)
                assert converter is not None
                graph, lattice, state_attr = _prepare_disk_graph(
                    converter,
                    structure,
                    labels,
                    include_ref_charge=include_ref_charge,
                )
                _validate_labels(
                    labels,
                    n_atoms=int(graph.num_nodes) if graph.num_nodes is not None else len(structure),
                    record_index=record_index,
                    layouts=layouts,
                    expected_shapes=expected_shapes,
                )
                item = (graph, lattice, _large_state_attr(state_attr, record.get("graph_label")), labels)
            else:
                if isinstance(structure, dict):
                    structure_dict = structure
                elif hasattr(structure, "as_dict"):
                    structure_dict = structure.as_dict()
                else:
                    raise TypeError(f"record {record_index}: structure must be a mapping or implement as_dict()")
                n_atoms = len(structure_dict.get("sites", [])) or None
                _validate_labels(
                    labels,
                    n_atoms=n_atoms,
                    record_index=record_index,
                    layouts=layouts,
                    expected_shapes=expected_shapes,
                )
                item = (structure_dict, labels, record.get("graph_label"))

            shard.append(item)
            n += 1
            if len(shard) >= shard_size:
                flush()

        flush()
        if n == 0:
            raise ValueError("cannot create an empty MGL dataset")

        label_schema = {
            key: {"layout": layouts.get(key, "unchecked"), "shape": list(expected_shapes.get(key) or ())}
            for key in expected_label_keys or ()
        }
        manifest = {
            "format_version": LARGE_DATA_FORMAT_VERSION,
            "backend": "pyg",
            "n": n,
            "shard_size": shard_size,
            "shards": shard_entries,
            "precomputed": precomputed,
            "label_keys": list(expected_label_keys or ()),
            "label_schema": label_schema,
            "include_ref_charge": include_ref_charge,
            "converter": None if converter is None else f"{type(converter).__module__}.{type(converter).__qualname__}",
            "versions": {
                "matgl": getattr(matgl, "__version__", "unknown"),
                "torch": torch.__version__,
            },
        }
        _atomic_json_save(manifest, manifest_path)
    except Exception:
        for path in created_paths:
            path.unlink(missing_ok=True)
        raise

    new_shards = {entry["file"] for entry in shard_entries}
    for filename in old_shards - new_shards:
        (root / filename).unlink(missing_ok=True)


class MGLDiskDataset(TorchDataset):
    """Map-style PyG dataset backed by independently loadable shards."""

    def __init__(self, root: str | Path, converter: GraphConverter | None = None):
        """Open a sharded dataset and validate its manifest.

        Args:
            root: Directory containing ``metadata.json`` and shard files.
            converter: Converter required for non-precomputed datasets.
        """
        self.root = Path(root)
        manifest = json.loads((self.root / "metadata.json").read_text())
        if manifest.get("format_version") != LARGE_DATA_FORMAT_VERSION:
            raise ValueError(f"unsupported dataset format: {manifest.get('format_version')!r}")
        if manifest.get("backend") != "pyg":
            raise ValueError(f"expected a PyG dataset, found {manifest.get('backend')!r}")

        self.n = int(manifest["n"])
        self.shard_size = int(manifest["shard_size"])
        self.precomputed = bool(manifest["precomputed"])
        self.include_ref_charge = bool(manifest["include_ref_charge"])
        self.label_schema = dict(manifest.get("label_schema", {}))
        self.labels = dict.fromkeys(manifest["label_keys"])
        self.converter = converter
        self._shards = list(manifest["shards"])
        self._offsets: list[int] = []
        offset = 0
        for entry in self._shards:
            self._offsets.append(offset)
            offset += int(entry["n"])
            if not (self.root / entry["file"]).is_file():
                raise FileNotFoundError(self.root / entry["file"])
        if offset != self.n:
            raise ValueError(f"manifest contains {offset} items but declares {self.n}")

        self._shard_id: int | None = None
        self._shard: list[Any] | None = None
        if not self.precomputed and converter is None:
            raise ValueError("converter is required for on-the-fly mode")

    @property
    def num_shards(self) -> int:
        """Number of independently loadable shards."""
        return len(self._shards)

    def __len__(self) -> int:
        """Return the number of structures in the dataset."""
        return self.n

    def shard_indices(self, shard_id: int) -> range:
        """Return global sample indices stored in one shard."""
        start = self._offsets[shard_id]
        return range(start, start + int(self._shards[shard_id]["n"]))

    def _location(self, idx: int) -> tuple[int, int]:
        if idx < 0:
            idx += self.n
        if idx < 0 or idx >= self.n:
            raise IndexError(f"index {idx} out of range for dataset of size {self.n}")
        shard_id = bisect.bisect_right(self._offsets, idx) - 1
        return shard_id, idx - self._offsets[shard_id]

    def _load_shard(self, shard_id: int) -> list[Any]:
        if shard_id != self._shard_id:
            path = self.root / self._shards[shard_id]["file"]
            # PyG Data requires pickle-backed loading. Only open trusted datasets.
            loaded = torch.load(path, map_location="cpu", weights_only=False)
            expected = int(self._shards[shard_id]["n"])
            if len(loaded) != expected:
                raise ValueError(f"{path} contains {len(loaded)} items; expected {expected}")
            self._shard = loaded
            self._shard_id = shard_id
        assert self._shard is not None
        return self._shard

    def __getstate__(self) -> dict[str, Any]:
        """Drop the process-local shard cache when spawning loader workers."""
        state = self.__dict__.copy()
        state["_shard_id"] = state["_shard"] = None
        return state

    def __getitem__(self, idx: int) -> tuple[Data, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Load one graph and its labels."""
        shard_id, local_idx = self._location(idx)
        item = self._load_shard(shard_id)[local_idx]

        if self.precomputed:
            graph, lattice, state_attr, labels = item
        else:
            structure_dict, labels, graph_label = item
            structure = MontyDecoder().process_decoded(structure_dict)
            assert self.converter is not None
            graph, lattice, state_attr = _prepare_disk_graph(
                self.converter,
                structure,
                labels,
                include_ref_charge=self.include_ref_charge,
            )
            state_attr = _large_state_attr(state_attr, graph_label)

        tensor_labels = {
            key: torch.as_tensor(value, dtype=matgl.float_th, device="cpu")
            for key, value in labels.items()
            if not isinstance(value, str)
        }
        return graph, lattice, state_attr, tensor_labels


class ShardBatchSampler(Sampler[list[int]]):
    """Produce shard-local batches with deterministic distributed partitioning."""

    def __init__(
        self,
        sampler: MGLDiskDataset | Sampler[int],
        batch_size: int,
        *,
        shuffle: bool | None = None,
        drop_last: bool = False,
        seed: int = 0,
        rank: int | None = None,
        world_size: int | None = None,
    ):
        """Initialize a sampler for shard-efficient single- or multi-rank loading."""
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")
        if (rank is None) != (world_size is None):
            raise ValueError("rank and world_size must be supplied together")
        if isinstance(sampler, MGLDiskDataset):
            self.dataset = sampler
            self.sampler: Sampler[int] = SequentialSampler(sampler)
        else:
            dataset = getattr(sampler, "dataset", getattr(sampler, "data_source", None))
            if not isinstance(dataset, MGLDiskDataset):
                raise TypeError("sampler must address an MGLDiskDataset")
            self.dataset = dataset
            self.sampler = sampler
        self.batch_size = batch_size
        self.shuffle = bool(getattr(sampler, "shuffle", False)) if shuffle is None else shuffle
        self.drop_last = drop_last
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Select the deterministic shuffle for an epoch."""
        self.epoch = epoch
        set_epoch = getattr(self.sampler, "set_epoch", None)
        if callable(set_epoch):
            set_epoch(epoch)

    def _current_epoch(self) -> int:
        return int(getattr(self.sampler, "epoch", self.epoch))

    def _distributed_context(self) -> tuple[int, int]:
        if self.rank is not None and self.world_size is not None:
            rank, world_size = self.rank, self.world_size
        elif hasattr(self.sampler, "rank") and hasattr(self.sampler, "num_replicas"):
            rank = int(self.sampler.rank)  # type: ignore[attr-defined]
            world_size = int(self.sampler.num_replicas)  # type: ignore[attr-defined]
        elif torch.distributed.is_available() and torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()
        else:
            rank, world_size = 0, 1
        if world_size <= 0 or not 0 <= rank < world_size:
            raise ValueError(f"invalid distributed context rank={rank}, world_size={world_size}")
        return rank, world_size

    def _shard_order(self, epoch: int) -> tuple[list[int], torch.Generator]:
        generator = torch.Generator().manual_seed(self.seed + epoch)
        shard_ids = list(range(self.dataset.num_shards))
        if self.shuffle:
            permutation = torch.randperm(len(shard_ids), generator=generator).tolist()
            shard_ids = [shard_ids[index] for index in permutation]
        return shard_ids, generator

    def _batches_in_shard(self, shard_id: int) -> int:
        count = int(self.dataset._shards[shard_id]["n"])
        if self.drop_last:
            return count // self.batch_size
        return math.ceil(count / self.batch_size)

    def _distributed_plan(self, epoch: int) -> tuple[list[int], int, torch.Generator]:
        rank, world_size = self._distributed_context()
        if world_size > self.dataset.num_shards:
            raise ValueError(
                f"world_size={world_size} exceeds num_shards={self.dataset.num_shards}; "
                "create more shards so every rank can read disjoint data"
            )

        shard_ids, generator = self._shard_order(epoch)
        rank_shards = [shard_ids[current_rank::world_size] for current_rank in range(world_size)]
        rank_batch_counts = [sum(self._batches_in_shard(shard_id) for shard_id in assigned) for assigned in rank_shards]
        target = min(rank_batch_counts) if self.drop_last else max(rank_batch_counts)
        if target == 0:
            raise ValueError(
                "at least one rank has no batches; use smaller batches, disable drop_last, "
                "or create fewer/larger shards"
            )
        return rank_shards[rank], target, generator

    def __len__(self) -> int:
        """Return the number of batches produced on this rank."""
        return self._distributed_plan(self._current_epoch())[1]

    def __iter__(self) -> Iterator[list[int]]:
        """Yield one epoch of shard-local batches."""
        epoch = self._current_epoch()
        shard_ids, target, generator = self._distributed_plan(epoch)
        if self.shuffle and not hasattr(self.sampler, "epoch"):
            self.epoch = epoch + 1

        emitted = 0
        last_batch: list[int] | None = None
        for shard_id in shard_ids:
            indices = list(self.dataset.shard_indices(shard_id))
            if self.shuffle:
                permutation = torch.randperm(len(indices), generator=generator).tolist()
                indices = [indices[index] for index in permutation]
            for start in range(0, len(indices), self.batch_size):
                batch = indices[start : start + self.batch_size]
                if len(batch) != self.batch_size and self.drop_last:
                    continue
                if emitted >= target:
                    return
                last_batch = batch
                emitted += 1
                yield batch

        assert last_batch is not None
        while emitted < target:
            emitted += 1
            yield last_batch


def _disk_loader(
    dataset: MGLDiskDataset,
    *,
    collate_fn: Callable,
    batch_size: int,
    shuffle: bool,
    drop_last: bool,
    seed: int,
    loader_kwargs: dict[str, Any],
) -> DataLoader:
    """Build a DataLoader that keeps every batch within one shard."""
    batch_sampler = ShardBatchSampler(
        dataset,
        batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        seed=seed,
    )
    return DataLoader(dataset, batch_sampler=batch_sampler, collate_fn=collate_fn, **loader_kwargs)


class MGLDataModule(pl.LightningDataModule):
    """Lightning DataModule for sharded PyG MatGL datasets."""

    def __init__(
        self,
        root: str | Path,
        *,
        collate_fn: Callable | None = None,
        converter: GraphConverter | None = None,
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool | None = None,
        persistent_workers: bool | None = None,
        drop_last: bool = False,
        seed: int = 0,
        train_split: str = "train",
        val_split: str = "valid",
        test_split: str = "test",
        predict_split: str | None = None,
        **loader_kwargs: Any,
    ):
        """Configure shard-backed datasets and loaders for Lightning.

        Args:
            root: Parent directory containing one directory per split.
            collate_fn: Optional MatGL collator. It is inferred from labels when omitted.
            converter: Converter required for non-precomputed shards.
            batch_size: Structures per batch.
            num_workers: DataLoader worker processes.
            pin_memory: Whether to use page-locked host memory.
            persistent_workers: Whether workers persist between epochs.
            drop_last: Whether to drop incomplete training batches.
            seed: Base seed for deterministic epoch shuffling.
            train_split: Training split directory name.
            val_split: Validation split directory name.
            test_split: Test split directory name.
            predict_split: Prediction split directory name; defaults to ``test_split``.
            **loader_kwargs: Additional uncontrolled DataLoader arguments.
        """
        super().__init__()
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")
        if num_workers < 0:
            raise ValueError("num_workers cannot be negative")
        controlled = {"batch_sampler", "batch_size", "collate_fn", "drop_last", "sampler", "shuffle"}
        overlap = controlled.intersection(loader_kwargs)
        if overlap:
            raise ValueError(f"loader_kwargs cannot override {sorted(overlap)}")

        self.root = Path(root)
        self.collate_fn = collate_fn
        self.converter = converter
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = torch.cuda.is_available() if pin_memory is None else pin_memory
        self.persistent_workers = num_workers > 0 if persistent_workers is None else persistent_workers
        if self.persistent_workers and num_workers == 0:
            raise ValueError("persistent_workers requires num_workers > 0")
        self.drop_last = drop_last
        self.seed = seed
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        self.predict_split = predict_split or test_split
        self.loader_kwargs = loader_kwargs

    def setup(self, stage: str | None = None) -> None:
        """Open only the datasets required by the current Lightning stage."""
        if stage in (None, "fit"):
            self.train_dataset = MGLDiskDataset(self.root / self.train_split, self.converter)
            self.val_dataset = MGLDiskDataset(self.root / self.val_split, self.converter)
        elif stage == "validate":
            self.val_dataset = MGLDiskDataset(self.root / self.val_split, self.converter)
        if stage in (None, "test"):
            self.test_dataset = MGLDiskDataset(self.root / self.test_split, self.converter)
        if stage in (None, "predict"):
            self.predict_dataset = MGLDiskDataset(self.root / self.predict_split, self.converter)

    def _loader(self, dataset: MGLDiskDataset, *, shuffle: bool, drop_last: bool = False) -> DataLoader:
        collate_fn = self.collate_fn or _pick_collate_fn(dataset.labels)
        kwargs = {
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
            "persistent_workers": self.persistent_workers,
            **self.loader_kwargs,
        }
        return _disk_loader(
            dataset,
            collate_fn=collate_fn,
            batch_size=self.batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            seed=self.seed,
            loader_kwargs=kwargs,
        )

    def train_dataloader(self) -> DataLoader:
        """Return the randomized shard-aware training loader."""
        return self._loader(self.train_dataset, shuffle=True, drop_last=self.drop_last)

    def val_dataloader(self) -> DataLoader:
        """Return the deterministic validation loader."""
        return self._loader(self.val_dataset, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        """Return the deterministic test loader."""
        return self._loader(self.test_dataset, shuffle=False)

    def predict_dataloader(self) -> DataLoader:
        """Return the deterministic prediction loader."""
        return self._loader(self.predict_dataset, shuffle=False)
