"""Tools to construct a dataset of PYG graphs."""

from __future__ import annotations

import json
import os
import shutil
from functools import partial
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torch_geometric.data import Batch, Data, Dataset
from tqdm import trange

import matgl

if TYPE_CHECKING:
    from collections.abc import Callable

    from matgl.graph.converters import GraphConverter


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
    if frac_list is None:
        frac_list = [0.8, 0.1, 0.1]
    num_graphs = len(self)
    num_train = int(frac_list[0] * num_graphs)
    num_val = int(frac_list[1] * num_graphs)

    indices = (
        torch.randperm(num_graphs, generator=torch.Generator().manual_seed(random_state))
        if shuffle
        else torch.arange(num_graphs)
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
    return shape; we prefer ``magmoms`` when both happen to be present
    (matching the DGL precedence) and document the override path.
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
    train_data: MGLDataset,
    val_data: MGLDataset,
    collate_fn: Callable | None = None,
    test_data: MGLDataset | None = None,
    **kwargs,
) -> tuple[DataLoader, ...]:
    """Dataloader for MatGL training in PyTorch Geometric.

    Args:
        train_data (Dataset): Training dataset (PyG Dataset or subset).
        val_data (Dataset): Validation dataset (PyG Dataset or subset).
        collate_fn (Callable, optional): Collate function for batching. When ``None`` (default),
            one is auto-selected from the training dataset's label keys: ``collate_fn_graph`` for
            single-target property prediction (no ``forces`` key), or ``collate_fn_pes`` with
            stress / magmom / charge flags toggled on based on the keys actually present. Pass
            an explicit callable (e.g. ``partial(collate_fn_pes, include_stress=False)``) to
            override.
        test_data (Dataset, optional): Test dataset (PyG Dataset or subset). Defaults to None.
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

        shutil.rmtree(self.root + "/processed/")

    def has_cache(self) -> bool:
        """Check if the processed files exist."""
        files_to_check = [
            self.filename,
            self.filename_lattice,
            self.filename_state_attr,
            self.filename_labels,
        ]
        return all(os.path.exists(os.path.join(self.root, f)) for f in files_to_check)

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
        """Save PyG graphs and labels to processed_dir."""
        if not self.save_cache:
            return

        if not os.path.exists(self.root):
            os.makedirs(self.root)

        if self.labels:
            with open(os.path.join(self.root, self.filename_labels), "w") as file:
                json.dump(self.labels, file)

        torch.save(self.graphs, os.path.join(self.root, self.filename))
        torch.save(self.lattices, os.path.join(self.root, self.filename_lattice))
        torch.save(self.state_attr, os.path.join(self.root, self.filename_state_attr))

    def load(self) -> None:
        """Load PyG graphs from files."""
        self.graphs = torch.load(os.path.join(self.root, self.filename), weights_only=False)
        self.lattices = torch.load(os.path.join(self.root, self.filename_lattice), weights_only=False)
        self.state_attr = torch.load(os.path.join(self.root, self.filename_state_attr), weights_only=False)
        with open(os.path.join(self.root, self.filename_labels)) as f:
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
