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

    Mirrors the equivalent DGL routine in earlier matgl versions. ``phi`` is
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

    Mirrors the equivalent DGL routine in earlier matgl versions but works
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

    The line-graph "nodes" are bonds in the parent graph (in their original
    order), and a line-graph edge ``(b_i, b_j)`` exists whenever ``b_i`` and
    ``b_j`` share a source atom and ``b_i != b_j``.

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

    Equivalent to the equivalent DGL routine in earlier matgl versions for the
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
    the equivalent DGL routine in earlier matgl versions.

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


# ---------------------------------------------------------------------------
# CHGNet-specific: directed line graph construction and angle utilities
# ---------------------------------------------------------------------------


def create_directed_line_graph(
    edge_index: torch.Tensor,
    pbc_offset: torch.Tensor,
    bond_vec: torch.Tensor,
    bond_dist: torch.Tensor,
    threebody_cutoff: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build a directed line graph for three-body CHGNet interactions (PyG version).

    Mirrors the logic of the legacy DGL ``_create_directed_line_graph`` but
    operates entirely on edge-index tensors — no DGL dependency.

    Convention (same as atom graph):
      - ``src_indices`` / ``edge_index[0]`` = central atom
      - ``dst_indices`` / ``edge_index[1]`` = neighbor atom

    In the line graph:
      - Nodes  = bonds (edges of the atom graph) within ``threebody_cutoff``
      - Edges  = bond-pairs sharing a central atom
      - dst of a line-graph edge = the bond **being updated** (receives messages)
      - src of a line-graph edge = bonds **sending messages** to the dst bond

    Args:
        edge_index: Atom-graph edge indices, shape (2, num_edges).
        pbc_offset: Periodic image offsets per bond, shape (num_edges, 3).
        bond_vec: Bond vectors (central→neighbor), shape (num_edges, 3).
        bond_dist: Bond distances, shape (num_edges,).
        threebody_cutoff: Cutoff radius for three-body interactions.

    Returns:
        lg_edge_index: Line-graph edge indices (2, num_lg_edges).
        lg_bond_vec: Bond vectors for line-graph nodes, shape (num_lg_nodes, 3).
        lg_bond_dist: Bond distances for line-graph nodes, shape (num_lg_nodes,).
        lg_pbc_offset: PBC offsets for line-graph nodes, shape (num_lg_nodes, 3).
        lg_src_bond_sign: Sign correction for src bonds (+1 or -1),
            shape (num_lg_nodes, 1).
    """
    device = edge_index.device

    # --- filter bonds within three-body cutoff ---
    valid = bond_dist <= threebody_cutoff
    edge_ids = valid.nonzero(as_tuple=False).squeeze(1)  # global ids of valid bonds

    if edge_ids.numel() == 0:
        return (
            torch.zeros((2, 0), dtype=matgl.int_th, device=device),
            bond_vec.new_zeros((0, 3)),
            bond_dist.new_zeros(0),
            pbc_offset.new_zeros((0, 3)),
            bond_vec.new_ones((0, 1)),
        )

    src_indices = edge_index[0]  # central atom
    dst_indices = edge_index[1]  # neighbor

    # Map global edge ids → local line-graph node ids
    num_lg_nodes = edge_ids.numel()
    global_to_local = torch.full((edge_index.size(1),), -1, dtype=torch.long, device=device)
    local_ids = torch.arange(num_lg_nodes, dtype=torch.long, device=device)
    global_to_local[edge_ids] = local_ids

    # Sub-graph src/dst for valid bonds
    v_src = src_indices[edge_ids]  # central atoms for valid bonds
    v_dst = dst_indices[edge_ids]  # neighbors for valid bonds
    v_images = pbc_offset[edge_ids]

    is_self_edge = v_src == v_dst

    # Vectorized Line Graph Edge Construction

    # Calculate connections using dense matrices over the reduced (valid) edges
    v_src_j = v_src.unsqueeze(0)  # shape (1, V)
    v_dst_j = v_dst.unsqueeze(0)  # shape (1, V)
    v_images_j = v_images.unsqueeze(0)  # shape (1, V, 3)

    v_src_i = v_src.unsqueeze(1)  # shape (V, 1)
    v_dst_i = v_dst.unsqueeze(1)  # shape (V, 1)
    v_images_i = v_images.unsqueeze(1)  # shape (V, 1, 3)

    # shared_src: src[i] == src[j]
    shared_src = v_src_i == v_src_j

    # incoming_to_ca: dst[i] == src[j]
    incoming_to_ca = v_dst_i == v_src_j

    # Backtracking: incoming & src[i] == dst[j] & images[i] == -images[j]
    is_backtrack = incoming_to_ca & (v_src_i == v_dst_j) & torch.all(-v_images_i == v_images_j, dim=2)

    # Base inclusion for non-self edges (matches DGL logic: incoming & (shared_src | ~backtracking))
    include_mask = incoming_to_ca & (shared_src | ~is_backtrack)

    # For self-edges (is_self_edge[j]), only include incoming_to_ca (no shared_src, no backtrack checks)
    self_edges_j = is_self_edge.unsqueeze(0)  # (1, V)
    include_mask = torch.where(self_edges_j, incoming_to_ca, include_mask)

    # Exclude i == j
    include_mask.fill_diagonal_(False)

    # Get the indices of the connected line graph nodes
    lg_src, lg_dst = include_mask.nonzero(as_tuple=True)
    lg_edge_index = torch.stack([lg_src, lg_dst], dim=0)

    # Line-graph node features = bond properties of the corresponding atom-graph edge
    lg_bond_vec = bond_vec[edge_ids]
    lg_bond_dist = bond_dist[edge_ids]
    lg_pbc_offset = pbc_offset[edge_ids]

    # Sign correction: non-self edges get sign = -1 (bond vector points away from central atom)
    lg_src_bond_sign = torch.ones((num_lg_nodes, 1), dtype=bond_vec.dtype, device=device)

    # Find local ids of non-self edges
    not_self_edge = ~is_self_edge
    ns_local_ids = local_ids[not_self_edge]
    if ns_local_ids.numel() > 0:
        lg_src_bond_sign[ns_local_ids] = -1.0

    return lg_edge_index, lg_bond_vec, lg_bond_dist, lg_pbc_offset, lg_src_bond_sign


def compute_theta(
    bond_vec: torch.Tensor,
    src_bond_sign: torch.Tensor,
    lg_src: torch.Tensor,
    lg_dst: torch.Tensor,
    directed: bool = True,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Compute bond angles (cos theta) for line-graph edges (PyG version).

    Mirrors the legacy DGL ``compute_theta``.

    Args:
        bond_vec: Bond vectors for all line-graph nodes, shape (num_lg_nodes, 3).
        src_bond_sign: Sign correction for src bond vectors, shape (num_lg_nodes, 1).
            ``-1`` for non-self directed bonds (their vector needs flipping to point
            toward the central atom), ``+1`` for self-loop bonds.
        lg_src: Source line-graph node indices (bond sending message), shape (num_lg_edges,).
        lg_dst: Destination line-graph node indices (bond receiving message), shape (num_lg_edges,).
        directed: If True, apply ``src_bond_sign`` correction (for directed line graph).
        eps: Epsilon for numerical stability in acos.

    Returns:
        cos_theta: Cosine of bond angles, shape (num_lg_edges,).
    """
    vec1 = bond_vec[lg_src] * src_bond_sign[lg_src] if directed else bond_vec[lg_src]
    vec2 = bond_vec[lg_dst]
    cos_theta = (vec1 * vec2).sum(dim=1) / (torch.norm(vec1, dim=1) * torch.norm(vec2, dim=1)).clamp(min=eps)
    return cos_theta.clamp(-1 + eps, 1 - eps)


def ensure_directed_line_graph_compatibility(
    bond_dist: torch.Tensor,
    threebody_cutoff: float,
    lg_edge_index: torch.Tensor,
    lg_bond_vec: torch.Tensor,
    lg_bond_dist: torch.Tensor,
    lg_pbc_offset: torch.Tensor,
    lg_src_bond_sign: torch.Tensor,
    edge_ids: torch.Tensor,
    tol: float = 5e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Refresh line-graph node data to match the current atom graph.

    Used in the ``forward`` pass when a pre-built line graph is passed in (e.g. from
    the dataloader) but atomic positions have changed (e.g. during MD / relaxation).
    Updates ``lg_bond_vec``, ``lg_bond_dist``, ``lg_pbc_offset`` in-place to reflect
    new bond vectors computed from current positions.

    Args:
        bond_dist: Current bond distances (atom graph), shape (num_edges,).
        threebody_cutoff: Cutoff for three-body interactions.
        lg_edge_index: Existing line-graph edge index, shape (2, num_lg_edges).
        lg_bond_vec: Existing line-graph node bond vectors (will be replaced).
        lg_bond_dist: Existing line-graph node bond distances (will be replaced).
        lg_pbc_offset: Existing line-graph node PBC offsets.
        lg_src_bond_sign: Existing sign correction tensor.
        edge_ids: Global edge ids that are line-graph nodes, shape (num_lg_nodes,).
        tol: Numerical tolerance added to cutoff when number of nodes exceeds valid count.

    Returns:
        Updated (lg_edge_index, lg_bond_vec, lg_bond_dist, lg_pbc_offset, lg_src_bond_sign, edge_ids).
    """
    valid = bond_dist <= threebody_cutoff
    if lg_bond_dist.size(0) > valid.sum():
        valid = bond_dist <= threebody_cutoff + tol
    if lg_bond_dist.size(0) > valid.sum():
        raise RuntimeError("Line graph is incompatible with atom graph after tolerance adjustment.")
    # Refresh only the node data — topology (edge_index) stays the same
    new_edge_ids = valid.nonzero(as_tuple=False).squeeze(1)[: lg_bond_dist.size(0)]
    return lg_edge_index, lg_bond_vec, lg_bond_dist, lg_pbc_offset, lg_src_bond_sign, new_edge_ids
