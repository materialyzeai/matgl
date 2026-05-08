"""Building blocks for the GRACE (Graph Atomic Cluster Expansion) potential.

The GRACE network is a graph extension of the Atomic Cluster Expansion (ACE)
introduced by Bochkarev, Lysogorskiy and Drautz (Phys. Rev. X 14, 021036
(2024); arXiv:2508.17936). The reference TensorFlow implementation is
``ICAMS/grace-tensorpotential``. This file contains the GRACE-specific
PyG-friendly layers that the model in :mod:`matgl.models._grace`
composes; it deliberately reuses the existing matgl primitives (real
spherical harmonics, real-CG tensor product, polynomial cutoff,
``scatter_add``) so the GRACE PES sits on the same numerical machinery as
``SO3Net`` and ``TensorNet``.

All equivariant tensors follow matgl's lm-major convention
``[N_*, (lmax+1)^2, n_features]``, where the ``(lmax+1)^2`` index runs as
``[(0,0), (1,-1), (1,0), (1,1), (2,-2), ..., (lmax, lmax)]``.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from matgl.layers._so3 import SO3TensorProduct
from matgl.utils.cutoff import polynomial_cutoff
from matgl.utils.maths import scatter_add


def _l_per_lm_index(lmax: int) -> torch.Tensor:
    """Map each flat ``(l, m)`` slot to its ``l`` value.

    Returns a long tensor of length ``(lmax+1)^2`` containing
    ``[0, 1, 1, 1, 2, 2, 2, 2, 2, ...]``.
    """
    parts = [torch.full((2 * l + 1,), l, dtype=torch.long) for l in range(lmax + 1)]
    return torch.cat(parts, dim=0)


class ChebyshevRadialBasis(nn.Module):
    """Squared-rescaled Chebyshev radial basis with polynomial cutoff envelope.

    Distances ``r`` are mapped to ``r̃ = 2 (1 - |1 - r/rcut|) - 1`` and then to
    ``T_1(r̃), T_2(r̃), ..., T_nfunc(r̃)`` (Chebyshev of the first kind). The
    output is multiplied by the smooth polynomial cutoff envelope
    :func:`matgl.utils.cutoff.polynomial_cutoff` so the basis vanishes
    smoothly at ``rcut``. This is the radial basis used by gracemaker.

    Args:
        nfunc: number of Chebyshev basis functions (output channels).
        cutoff: cutoff radius in Å.
        cutoff_exponent: order ``p`` of the polynomial cutoff envelope.
            Larger ``p`` gives more derivatives that vanish at ``rcut``.
    """

    def __init__(self, nfunc: int, cutoff: float, cutoff_exponent: int = 5):
        super().__init__()
        if nfunc < 1:
            raise ValueError(f"nfunc must be >= 1, got {nfunc}")
        if cutoff <= 0:
            raise ValueError(f"cutoff must be > 0, got {cutoff}")
        self.nfunc = int(nfunc)
        self.cutoff = float(cutoff)
        self.cutoff_exponent = int(cutoff_exponent)

    def forward(self, bond_dist: torch.Tensor) -> torch.Tensor:
        """Compute basis values on a 1-D tensor of bond distances.

        Args:
            bond_dist: ``[E]`` (or ``[E, 1]``) bond lengths.

        Returns:
            ``[E, nfunc]`` basis values, zeroed beyond ``cutoff``.
        """
        r = bond_dist.reshape(-1)
        r_safe = torch.where(r == 0.0, r + 1e-10, r)
        x = 2.0 * (1.0 - torch.abs(1.0 - r_safe / self.cutoff)) - 1.0
        # Chebyshev recurrence: T_0=1, T_1=x, T_k=2x T_{k-1} - T_{k-2}.
        # We need T_1 ... T_nfunc, so build nfunc+1 and drop T_0.
        cheb = [torch.ones_like(x), x]
        x2 = 2.0 * x
        for _ in range(2, self.nfunc + 1):
            cheb.append(cheb[-1] * x2 - cheb[-2])
        basis = torch.stack(cheb[1:], dim=-1)
        envelope = polynomial_cutoff(r_safe, self.cutoff, exponent=self.cutoff_exponent)
        return basis * envelope.unsqueeze(-1)


class LinearRadialFunction(nn.Module):
    """Learnable per-``(n, l)`` linear expansion of a radial basis.

    Computes ``R_{nl}(r) = sum_k c_{nlk} g_k(r)`` where ``g_k`` is a fixed
    radial basis (e.g. :class:`ChebyshevRadialBasis`). The per-l radial is
    tiled across the ``2l+1`` magnetic numbers and returned in matgl's
    lm-major convention so it can be elementwise-multiplied with real
    spherical harmonics.

    Args:
        nfunc: number of basis functions of the input radial basis.
        n_rad_max: number of learned radial channels ``n``.
        lmax: angular cutoff.
    """

    def __init__(self, nfunc: int, n_rad_max: int, lmax: int):
        super().__init__()
        self.nfunc = int(nfunc)
        self.n_rad_max = int(n_rad_max)
        self.lmax = int(lmax)
        limit = math.sqrt(2.0 / float(self.n_rad_max + self.nfunc))
        self.crad = nn.Parameter(torch.randn(self.n_rad_max, self.lmax + 1, self.nfunc) * limit)
        self.register_buffer("_l_idx", _l_per_lm_index(self.lmax), persistent=False)

    def forward(self, basis_values: torch.Tensor) -> torch.Tensor:
        """Project the radial basis to per-``(n, l, m)`` channels.

        Args:
            basis_values: ``[E, nfunc]`` radial basis values.

        Returns:
            ``[E, (lmax+1)^2, n_rad_max]`` radial channels broadcast across
            all magnetic quantum numbers per ``l``.
        """
        # [n_rad_max, lmax+1, nfunc] x [E, nfunc] -> [E, n_rad_max, lmax+1]
        y = torch.einsum("nlk,ek->enl", self.crad, basis_values)
        # Expand l → (l, m) by gathering, then permute to lm-major.
        # ``torch.as_tensor`` narrows ``self._l_idx`` from ``Tensor | Module``
        # (the static type seen by mypy for buffers) to ``Tensor``.
        l_idx = torch.as_tensor(self._l_idx)
        y_lm = y.index_select(dim=-1, index=l_idx)  # [E, n_rad_max, (lmax+1)^2]
        return y_lm.transpose(1, 2).contiguous()  # [E, (lmax+1)^2, n_rad_max]


class GraceSPBasis(nn.Module):
    """ACE single-particle basis aggregator.

    Forms ``A_i^{(l, m, n)} = (1 / n_neigh_avg) * sum_{j ~ i} z[mu_j]_n
    * R_{n l}(r_ij) * Y_{l, m}(r̂_ij)`` with a per-element scalar chemical
    indicator ``z`` projected through a learned ``embedding_size → n_rad_max``
    linear layer (matching gracemaker's ``ScalarChemicalEmbedding`` +
    ``DenseLayer`` indicator path).

    Args:
        lmax: angular cutoff.
        n_rad_max: number of radial channels.
        n_elements: number of distinct atomic types.
        embedding_size: width of the chemical embedding ``z``.
        avg_n_neigh: typical neighbor count, used to normalize the sum.
    """

    def __init__(
        self,
        lmax: int,
        n_rad_max: int,
        n_elements: int,
        embedding_size: int,
        avg_n_neigh: float = 1.0,
    ):
        super().__init__()
        self.lmax = int(lmax)
        self.n_rad_max = int(n_rad_max)
        self.n_elements = int(n_elements)
        self.embedding_size = int(embedding_size)
        self.inv_avg_n_neigh = 1.0 / float(avg_n_neigh)
        self.chem_embedding = nn.Parameter(torch.randn(n_elements, embedding_size))
        self.indicator = nn.Linear(embedding_size, n_rad_max, bias=False)

    def forward(
        self,
        radial_nl: torch.Tensor,
        spherical_lm: torch.Tensor,
        edge_index: torch.Tensor,
        node_type: torch.Tensor,
        num_nodes: int,
    ) -> torch.Tensor:
        """Aggregate single-particle basis ``A_i`` from neighbor contributions.

        Args:
            radial_nl: ``[E, (lmax+1)^2, n_rad_max]`` per-edge radial channels.
            spherical_lm: ``[E, (lmax+1)^2]`` real spherical harmonics.
            edge_index: ``[2, E]`` PyG edge index. ``edge_index[0]`` is the
                center atom ``i`` (aggregation target); ``edge_index[1]`` is
                the neighbor ``j``.
            node_type: ``[N]`` atomic-type indices.
            num_nodes: number of atoms ``N``.

        Returns:
            ``[N, (lmax+1)^2, n_rad_max]`` per-atom single-particle basis.
        """
        center_idx = edge_index[0]
        neighbor_idx = edge_index[1]
        z = self.indicator(self.chem_embedding)  # [n_elements, n_rad_max]
        z_j = z[node_type[neighbor_idx]]  # [E, n_rad_max]
        # Combine R_{nl}(r) * Y_{lm}(r̂) and weight by z_j.
        a_edge = radial_nl * spherical_lm.unsqueeze(-1) * z_j.unsqueeze(1)
        a_node = scatter_add(a_edge, center_idx, dim_size=num_nodes, dim=0)
        return a_node * self.inv_avg_n_neigh


class GraceACEStack(nn.Module):
    """Stack of equivariant tensor products forming ``A, A⊗A, A⊗A⊗A, ...``.

    Each subsequent order is produced by coupling the previous order with the
    base ``A`` tensor through :class:`matgl.layers._so3.SO3TensorProduct`,
    which uses real-spherical Clebsch-Gordan coefficients with the natural
    SO(3) parity mask ``(-1)^l1 * (-1)^l2 == (-1)^L``.

    Args:
        lmax: angular cutoff.
        max_order: highest cluster order; ``max_order=3`` yields ``{A, A⊗A,
            A⊗A⊗A}``. Must be ``>= 1``.
    """

    def __init__(self, lmax: int, max_order: int):
        super().__init__()
        if max_order < 1:
            raise ValueError("max_order must be >= 1")
        self.lmax = int(lmax)
        self.max_order = int(max_order)
        # max_order - 1 product layers; each shares the same lmax.
        self.products = nn.ModuleList([SO3TensorProduct(lmax=lmax) for _ in range(self.max_order - 1)])

    def forward(self, a_node: torch.Tensor) -> list[torch.Tensor]:
        """Build the chain of cluster-order equivariant tensors.

        Returns a list ``[A, A⊗A, ..., A^{max_order}]`` of per-atom
        equivariant tensors. Each entry has shape
        ``[N, (lmax+1)^2, n_rad_max]``.
        """
        outputs: list[torch.Tensor] = [a_node]
        prev = a_node
        for product in self.products:
            prev = product(prev, a_node)
            outputs.append(prev)
        return outputs


def collect_invariants(equivariant_tensors: list[torch.Tensor]) -> torch.Tensor:
    """Concatenate ``L=0`` components of every equivariant tensor.

    The ``(l=0, m=0)`` slice of each input is the rotationally invariant
    component, which we concatenate across cluster orders into a single
    per-atom feature vector for the energy readout MLP.

    Args:
        equivariant_tensors: list of ``[N, (lmax+1)^2, n_rad_max]`` tensors,
            one per cluster order.

    Returns:
        ``[N, n_orders * n_rad_max]``.
    """
    parts = [t[:, 0, :] for t in equivariant_tensors]
    return torch.cat(parts, dim=-1)


__all__ = [
    "ChebyshevRadialBasis",
    "GraceACEStack",
    "GraceSPBasis",
    "LinearRadialFunction",
    "collect_invariants",
]
