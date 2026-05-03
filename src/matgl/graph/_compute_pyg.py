"""Computing various graph based operations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

import matgl

if TYPE_CHECKING:
    from collections.abc import Callable

    from torch_geometric.data import Data


def compute_pair_vector_and_distance(
    pos: torch.Tensor,
    edge_index: torch.Tensor,
    pbc_offshift: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Calculate bond vectors and distances.

    Args:
        pos: Node positions, shape (num_nodes, 3)
        edge_index: Edge indices, shape (2, num_edges)
        pbc_offshift: Periodic boundary condition offsets, shape (num_edges, 3)

    Returns:
        bond_vec: Bond vectors, shape (num_edges, 3)
        bond_dist: Bond distances, shape (num_edges,)
    """
    src_idx, dst_idx = edge_index[0], edge_index[1]
    src_pos = pos[src_idx]
    dst_pos = pos[dst_idx]

    if pbc_offshift is not None:
        dst_pos = dst_pos + pbc_offshift

    bond_vec = dst_pos - src_pos
    bond_dist = torch.norm(bond_vec, dim=1)

    return bond_vec, bond_dist


def compute_theta_and_phi(
    bond_vec: torch.Tensor,
    bond_dist: torch.Tensor,
    line_edge_index: torch.Tensor,
    eps: float = 1e-7,
) -> dict[str, torch.Tensor]:
    """Compute the bond angle ``theta`` (cosine) and ``phi`` for each line-graph edge.

    Mirrors :func:`matgl.graph._compute_dgl.compute_theta_and_phi`. ``phi`` is
    fixed to zero in the M3GNet (non-directed) variant.

    Args:
        bond_vec: Per-bond vectors of the parent graph after three-body pruning,
            shape ``(num_bonds, 3)``. Same indexing as line-graph nodes.
        bond_dist: Per-bond distances, shape ``(num_bonds,)``.
        line_edge_index: Line-graph edges as ``(2, num_triples)`` with row 0 =
            source bond, row 1 = destination bond.
        eps: Numerical tolerance for clamping ``cos`` near unity.

    Returns:
        Dict with keys ``cos_theta``, ``phi`` and ``triple_bond_lengths`` ready
        to feed :class:`matgl.layers.SphericalBesselWithHarmonics` (tensor mode).
    """
    src, dst = line_edge_index[0], line_edge_index[1]
    vec1 = bond_vec[src]
    vec2 = bond_vec[dst]
    dot = torch.sum(vec1 * vec2, dim=1)
    n1 = torch.norm(vec1, dim=1)
    n2 = torch.norm(vec2, dim=1)
    cos_theta = dot / (n1 * n2)
    cos_theta = cos_theta.clamp(min=-1 + eps, max=1 - eps)
    phi = torch.zeros_like(cos_theta)
    triple_bond_lengths = bond_dist[dst]
    return {"cos_theta": cos_theta, "phi": phi, "triple_bond_lengths": triple_bond_lengths}


def prune_edges_by_features(
    edge_index: torch.Tensor,
    edge_attrs: dict[str, torch.Tensor],
    feat: torch.Tensor,
    condition: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
    """Drop edges of a PyG-style graph that satisfy a feature-based condition.

    Mirrors :func:`matgl.graph._compute_dgl.prune_edges_by_features` but works
    on raw tensors (no ``Data`` mutation), so the caller can pick what to pass.

    Args:
        edge_index: ``(2, E)`` tensor of edge indices.
        edge_attrs: Dict of per-edge tensors keyed by name.
        feat: Per-edge feature used by ``condition``.
        condition: Callable returning a boolean mask of length ``E``; edges
            where the condition is ``True`` are *removed*.

    Returns:
        Tuple ``(new_edge_index, new_edge_attrs, kept_indices)`` where
        ``kept_indices`` are the original edge ids of the surviving edges.
    """
    valid = ~condition(feat)
    kept = valid.nonzero().squeeze(-1)
    new_edge_index = edge_index[:, valid]
    new_attrs = {k: v[valid] for k, v in edge_attrs.items()}
    return new_edge_index, new_attrs, kept


def _compute_3body_indices(
    edge_index: torch.Tensor,
    num_nodes: int,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Enumerate (bond_i, bond_j) pairs that share a source atom (M3GNet 3-body).

    Direct port of the numpy combinatorics in
    :func:`matgl.graph._compute_dgl._compute_3body` to PyG: the line-graph
    "nodes" are bonds in the parent graph (in their original order), and a
    line-graph edge ``(b_i, b_j)`` exists whenever ``b_i`` and ``b_j`` share a
    source atom and ``b_i != b_j``.

    Args:
        edge_index: ``(2, E)`` parent edge indices (after three-body pruning).
        num_nodes: Number of atoms in the parent graph.
        device: Device for the returned tensors.

    Returns:
        Tuple ``(line_edge_index, n_triple_ij, max_bond_id)``:
            * ``line_edge_index``: shape ``(2, num_triples)``.
            * ``n_triple_ij``: per-bond triple count (one entry per parent
              bond, in original order; length ``E``).
            * ``max_bond_id``: largest bond index appearing in
              ``line_edge_index`` plus 1 (for slicing line-graph node features
              from per-bond tensors).
    """
    src_np = edge_index[0].cpu().numpy()
    n_bond_per_atom = np.bincount(src_np, minlength=num_nodes)

    n_triple = int((n_bond_per_atom * (n_bond_per_atom - 1)).sum())
    n_triple_ij_np = np.repeat(n_bond_per_atom - 1, n_bond_per_atom)

    triple_bond_indices = np.empty((n_triple, 2), dtype=matgl.int_np)
    start = 0
    cs = 0
    for n in n_bond_per_atom:
        if n > 0:
            r = np.arange(n)
            x, y = np.meshgrid(r, r, indexing="xy")
            final = np.stack([y.ravel(), x.ravel()], axis=1)
            mask = final[:, 0] != final[:, 1]
            final = final[mask]
            triple_bond_indices[start : start + n * (n - 1)] = final + cs
            start += n * (n - 1)
            cs += n

    src_bond = torch.tensor(triple_bond_indices[:, 0], dtype=matgl.int_th, device=device)
    dst_bond = torch.tensor(triple_bond_indices[:, 1], dtype=matgl.int_th, device=device)
    line_edge_index = torch.stack([src_bond, dst_bond], dim=0)
    n_triple_ij = torch.tensor(n_triple_ij_np, dtype=matgl.int_th, device=device)

    max_bond_id = int(line_edge_index.max().item()) + 1 if line_edge_index.numel() > 0 else 0
    return line_edge_index, n_triple_ij, max_bond_id


def create_line_graph(
    edge_index: torch.Tensor,
    bond_dist: torch.Tensor,
    bond_vec: torch.Tensor,
    pbc_offset: torch.Tensor | None,
    num_nodes: int,
    threebody_cutoff: float,
) -> dict[str, torch.Tensor]:
    """Build the M3GNet 3-body line graph (PyG variant).

    Equivalent to :func:`matgl.graph._compute_dgl.create_line_graph` for the
    non-directed (M3GNet) case, but returns a tensor bundle (no ``DGLGraph``):

    Args:
        edge_index: Parent ``(2, E)`` edge indices.
        bond_dist: Per-edge distances of the parent graph.
        bond_vec: Per-edge bond vectors of the parent graph.
        pbc_offset: Per-edge PBC offsets of the parent graph (``None`` if not
            available).
        num_nodes: Number of atoms in the parent graph.
        threebody_cutoff: Distance cutoff used to drop edges before forming
            three-body terms.

    Returns:
        Dict with keys:
            * ``edge_index_pruned``: parent edges that survived the cutoff.
            * ``kept_edge_ids``: original parent edge indices of those edges.
            * ``bond_dist`` / ``bond_vec`` / ``pbc_offset``: per-line-graph-node
              tensors (sliced to ``max_bond_id``).
            * ``line_edge_index``: ``(2, num_triples)`` line-graph edges.
            * ``n_triple_ij``: per-line-graph-node count of triples.
    """
    edge_attrs: dict[str, torch.Tensor] = {"bond_dist": bond_dist, "bond_vec": bond_vec}
    if pbc_offset is not None:
        edge_attrs["pbc_offset"] = pbc_offset

    pruned_edge_index, pruned_attrs, kept_edge_ids = prune_edges_by_features(
        edge_index, edge_attrs, bond_dist, lambda x: x > threebody_cutoff
    )

    line_edge_index, n_triple_ij, max_bond_id = _compute_3body_indices(
        pruned_edge_index, num_nodes, device=edge_index.device
    )

    out: dict[str, torch.Tensor] = {
        "edge_index_pruned": pruned_edge_index,
        "kept_edge_ids": kept_edge_ids,
        "bond_dist": pruned_attrs["bond_dist"][:max_bond_id],
        "bond_vec": pruned_attrs["bond_vec"][:max_bond_id],
        "line_edge_index": line_edge_index,
        "n_triple_ij": n_triple_ij[:max_bond_id],
    }
    if "pbc_offset" in pruned_attrs:
        out["pbc_offset"] = pruned_attrs["pbc_offset"][:max_bond_id]
    return out


def ensure_line_graph_compatibility(
    line_graph: dict[str, torch.Tensor],
    bond_dist: torch.Tensor,
    bond_vec: torch.Tensor,
    pbc_offset: torch.Tensor | None,
    threebody_cutoff: float,
) -> dict[str, torch.Tensor]:
    """Refresh per-line-graph-node tensors against an updated parent graph.

    Mirrors the non-directed branch of
    :func:`matgl.graph._compute_dgl.ensure_line_graph_compatibility`.

    Args:
        line_graph: Bundle previously produced by :func:`create_line_graph`.
        bond_dist: Refreshed per-bond distances of the parent graph.
        bond_vec: Refreshed per-bond vectors of the parent graph.
        pbc_offset: Refreshed per-bond PBC offsets (``None`` if not available).
        threebody_cutoff: Same cutoff used to build the original line graph.

    Returns:
        A new bundle whose per-node tensors come from the updated parent graph.
    """
    valid = bond_dist <= threebody_cutoff
    valid_dist = bond_dist[valid]
    valid_vec = bond_vec[valid]

    n_lg_nodes = line_graph["bond_dist"].size(0)
    if n_lg_nodes == valid_dist.size(0):
        new_bond_dist = valid_dist
        new_bond_vec = valid_vec
        new_pbc_offset = pbc_offset[valid] if pbc_offset is not None else None
    else:
        new_bond_dist = bond_dist[:n_lg_nodes]
        new_bond_vec = bond_vec[:n_lg_nodes]
        new_pbc_offset = pbc_offset[:n_lg_nodes] if pbc_offset is not None else None

    new_lg = dict(line_graph)
    new_lg["bond_dist"] = new_bond_dist
    new_lg["bond_vec"] = new_bond_vec
    if new_pbc_offset is not None:
        new_lg["pbc_offset"] = new_pbc_offset
    return new_lg


def separate_node_edge_keys(graph: Data) -> tuple[list[str], list[str], list[str]]:
    """Separates keys in a PyTorch Geometric Data object into node attributes, edge attributes, and other attributes.

    Args:
        graph: PyTorch Geometric Data object.

    Returns:
        tuple: (node_keys, edge_keys, other_keys) where each is a list of attribute names.
    """
    node_keys = []
    edge_keys = []
    other_keys = []

    num_nodes = graph.num_nodes
    num_edges = graph.num_edges

    # PyG's ``Data.__iter__`` yields ``(key, value)`` tuples, so use ``.keys()``
    # explicitly to iterate attribute names.
    for key in graph.keys():  # noqa: SIM118
        value = graph[key]
        if key == "edge_index":
            other_keys.append(key)
            continue
        if isinstance(value, torch.Tensor) and value.dim() > 0:
            first_dim = value.size(0)
            if first_dim == num_nodes:
                node_keys.append(key)
            elif first_dim == num_edges:
                edge_keys.append(key)
            else:
                other_keys.append(key)
        else:
            other_keys.append(key)

    return node_keys, edge_keys, other_keys
