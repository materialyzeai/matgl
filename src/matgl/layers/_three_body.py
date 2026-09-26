"""Three-body (bond-angle) interaction terms.

Implements :class:`ThreeBodyInteractions`, the M3GNet/CHGNet angular
update that consumes a line graph (whose edges represent triplets of
neighbours sharing a central atom) and produces an updated edge feature
on the original graph. The combined radial-x-angular basis comes from
:class:`~matgl.layers._basis.SphericalBesselWithHarmonics`, multiplied by
two cosine cutoffs (one per bond in the triplet) and folded back into
the bond messages by :func:`combine_sbf_shf`.

Line-graph construction is handled by :mod:`matgl.graph._compute` upstream.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

import matgl
from matgl.utils.maths import _block_repeat, scatter_sum


class ThreeBodyInteractions(nn.Module):
    """Three-body bond update.

    ``forward`` takes only tensors. The caller is responsible for unpacking
    the parent-graph edges and the line-graph adjacency / per-bond triple
    counts before calling this layer.
    """

    def __init__(self, update_network_atom: nn.Module, update_network_bond: nn.Module, **kwargs):
        """Initialize ThreeBodyInteractions.

        Args:
            update_network_atom: MLP for node features in Eq.2
            update_network_bond: Gated-MLP for edge features in Eq.3
            **kwargs: Kwargs pass-through to nn.Module.__init__().
        """
        super().__init__(**kwargs)
        self.update_network_atom = update_network_atom
        self.update_network_bond = update_network_bond

    def forward(
        self,
        edge_dst_atom: torch.Tensor,
        line_edge_index: torch.Tensor,
        n_triple_ij: torch.Tensor,
        num_bonds: int,
        three_basis: torch.Tensor,
        three_cutoff: torch.Tensor,
        node_feat: torch.Tensor,
        edge_feat: torch.Tensor,
    ):
        """Forward function for ThreeBodyInteractions.

        For each bond ``i->j``, aggregates
        ``sum_k basis(r_ik, theta_jik) * update_network_atom(v_k) * f_c(r_ij) * f_c(r_ik)``
        over the neighbours ``k != j`` of ``i`` (Eq. 2) and adds
        ``update_network_bond`` of the result to ``edge_feat`` (Eq. 3).

        Args:
            edge_dst_atom: For each bond ``b`` in the parent graph, the index
                of the destination atom of that bond. Shape ``(num_bonds,)``.
            line_edge_index: Triplets as ``(2, num_triples)`` parent-graph bond
                ids: row 0 = bond ``i->j`` that receives the message, row 1 =
                bond ``i->k`` that supplies the neighbour ``k``. Must index the
                same bonds as ``edge_dst_atom``, ``three_cutoff`` and
                ``edge_feat`` (see :func:`matgl.graph._compute.create_line_graph`).
            n_triple_ij: Number of triples per parent bond, shape
                ``(num_bonds,)``. Not used by the aggregation, which scatters on
                ``line_edge_index[0]`` directly; retained for API compatibility.
            num_bonds: Total number of bonds in the parent graph (i.e. the
                ``dim_size`` for the per-bond scatter).
            three_basis: three body basis expansion of shape
                ``(num_triples, basis_dim)``.
            three_cutoff: per-bond cutoff weights, shape ``(num_bonds,)``.
            node_feat: node features.
            edge_feat: edge features (one row per bond).
        """
        bond_ij = line_edge_index[0].to(torch.long)
        bond_ik = line_edge_index[1].to(torch.long)

        # Updated features of the neighbour atom k of each triplet
        updated_atoms = self.update_network_atom(node_feat)
        end_atom_features = updated_atoms[edge_dst_atom[bond_ik].to(torch.long)]

        # Basis weighted by the smooth cutoffs of both bonds in the triplet
        weights = three_cutoff[bond_ij] * three_cutoff[bond_ik]
        basis = three_basis * end_atom_features * weights[:, None]

        # Aggregate onto the receiving bond i->j
        new_bonds = scatter_sum(
            basis.to(matgl.float_th),
            segment_ids=bond_ij,
            num_segments=num_bonds,
            dim=0,
        )

        # If no new bonds are generated, return the original edge features
        if new_bonds.shape[0] == 0:
            return edge_feat

        # Update edge features using the bond update network
        return edge_feat + self.update_network_bond(new_bonds)


def combine_sbf_shf(sbf, shf, max_n: int, max_l: int, use_phi: bool):
    """Combine the spherical Bessel function and the spherical Harmonics function.

    For the spherical Bessel function, the column is ordered by
        [n=[0, ..., max_n-1], n=[0, ..., max_n-1], ...], max_l blocks,

    For the spherical Harmonics function, the column is ordered by
        [m=[0], m=[-1, 0, 1], m=[-2, -1, 0, 1, 2], ...] max_l blocks, and each
        block has 2*l + 1
        if use_phi is False, then the columns become
        [m=[0], m=[0], ...] max_l columns

    Args:
        sbf: torch.Tensor spherical bessel function results
        shf: torch.Tensor spherical harmonics function results
        max_n: int, max number of n
        max_l: int, max number of l
        use_phi: whether to use phi
    Returns:
    """
    if sbf.size()[0] == 0:
        return sbf

    if not use_phi:
        repeats_sbf = torch.tensor([1] * max_l * max_n)
        block_size = [1] * max_l
    else:
        # [1, 1, 1, ..., 1, 3, 3, 3, ..., 3, ...]
        repeats_sbf = np.repeat(2 * torch.arange(max_l) + 1, repeats=max_n)  # type:ignore[assignment]
        # tf.repeat(2 * tf.range(max_l) + 1, repeats=max_n)
        block_size = 2 * torch.arange(max_l) + 1  # type: ignore
        # 2 * tf.range(max_l) + 1
    repeats_sbf = repeats_sbf.to(sbf.device)
    expanded_sbf = torch.repeat_interleave(sbf, repeats_sbf, 1)
    expanded_shf = _block_repeat(shf, block_size=block_size, repeats=[max_n] * max_l)
    shape = max_n * max_l
    if use_phi:
        shape *= max_l
    return torch.reshape(expanded_sbf * expanded_shf, [-1, shape])
