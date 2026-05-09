"""Main components in SO3Net.

The implementations are taken from Schnetpack2.0
(https://github.com/atomistic-machine-learning/schnetpack in schnetpack/src/schnetpack/nn/so3.py).
"""

from __future__ import annotations

import math

import torch
from torch import nn

import matgl
from matgl.layers import MLP
from matgl.utils.maths import binom, scatter_add
from matgl.utils.so3 import generate_clebsch_gordan_rsh, sh_indices, sparsify_clebsch_gordon

__all__ = [
    "RealSphericalHarmonics",
    "SO3Convolution",
    "SO3GatedNonlinearity",
    "SO3ParametricGatedNonlinearity",
    "SO3TensorProduct",
]


class RealSphericalHarmonics(nn.Module):
    """Generates the real spherical harmonics for a batch of vectors.

    Note:
        The vectors passed to this layer are assumed to be normalized to unit length.

    Spherical harmonics are generated up to angular momentum `lmax` in dimension 1,
    according to the following order:
    - l=0, m=0
    - l=1, m=-1
    - l=1, m=0
    - l=1, m=1
    - l=2, m=-2
    - l=2, m=-1
    - etc.
    """

    def __init__(self, lmax: int):
        """Initialize the RealSphericalHarmonics.

        Args:
            lmax: maximum angular momentum.
        """
        super().__init__()
        self.lmax = lmax

        (
            powers,
            zpow,
            cAm,
            cBm,
            cPi,
        ) = self._generate_Ylm_coefficients(lmax)

        dtype = matgl.float_th
        self.register_buffer("powers", powers.to(dtype=dtype), False)
        self.register_buffer("zpow", zpow.to(dtype=dtype), False)
        self.register_buffer("cAm", cAm.to(dtype=dtype), False)
        self.register_buffer("cBm", cBm.to(dtype=dtype), False)
        self.register_buffer("cPi", cPi.to(dtype=dtype), False)

        ls = torch.arange(0, lmax + 1)
        nls = 2 * ls + 1
        self.lidx = torch.repeat_interleave(ls, nls)
        self.midx = torch.cat([torch.arange(-l_id, l_id + 1) for l_id in ls])

        self.register_buffer("flidx", self.lidx.to(dtype=dtype), False)

    def _generate_Ylm_coefficients(self, lmax: int):
        """Generate the spherical-harmonics coefficient tensors.

        Args:
            lmax (int): maximum angular momentum.

        Returns:
            powers (torch.Tensor): Tensor containing powers for the spherical harmonics.
            zpow (torch.Tensor): Tensor containing powers for Legendre polynomials.
            CAm (torch.Tensor): Coefficients for cosine terms in spherical harmonics.
            CBm (torch.Tensor): Coefficients for sine terms in spherical harmonics.
            CPi (torch.Tensor): Coefficients for Legendre polynomials.
        """
        # see: https://en.wikipedia.org/wiki/Spherical_harmonics#Real_forms

        # calculate Am/Bm coefficients
        m = torch.arange(1, lmax + 1, dtype=matgl.float_th)[:, None]
        p = torch.arange(0, lmax + 1, dtype=matgl.float_th)[None, :]
        mask = p <= m
        mCp = binom(m, p)
        cAm = mCp * torch.cos(0.5 * math.pi * (m - p))
        cBm = mCp * torch.sin(0.5 * math.pi * (m - p))
        cAm *= mask
        cBm *= mask
        powers = torch.stack([torch.broadcast_to(p, cAm.shape), m - p], dim=-1)
        powers *= mask[:, :, None]

        # calculate Pi coefficients
        l_id = torch.arange(0, lmax + 1, dtype=matgl.float_th)[:, None, None]
        m = torch.arange(0, lmax + 1, dtype=matgl.float_th)[None, :, None]
        k = torch.arange(0, lmax // 2 + 1, dtype=matgl.float_th)[None, None, :]
        cPi = torch.sqrt(torch.exp(torch.lgamma(l_id - m + 1) - torch.lgamma(l_id + m + 1)))
        cPi = cPi * (-1) ** k * 2 ** (-l_id) * binom(l_id, k) * binom(2 * l_id - 2 * k, l_id)
        cPi *= torch.exp(torch.lgamma(l_id - 2 * k + 1) - torch.lgamma(l_id - 2 * k - m + 1))
        zpow = l_id - 2 * k - m

        # masking of invalid entries
        cPi = torch.nan_to_num(cPi, 100.0)
        mask1 = k <= torch.floor((l_id - m) / 2)
        mask2 = l_id >= m
        mask = mask1 * mask2
        cPi *= mask
        zpow *= mask

        return powers, zpow, cAm, cBm, cPi

    def forward(self, directions: torch.Tensor) -> torch.Tensor:
        """Compute real spherical harmonics for a batch of unit vectors.

        Args:
            directions: batch of unit-length 3D vectors (Nx3).

        Returns:
            real spherical harmonics up to angular momentum `lmax`.
        """
        powers = torch.as_tensor(self.powers)  # type: ignore[assignment]
        zpow = torch.as_tensor(self.zpow)  # type: ignore[assignment]
        cAm = torch.as_tensor(self.cAm)  # type: ignore[assignment]
        cBm = torch.as_tensor(self.cBm)  # type: ignore[assignment]
        cPi = torch.as_tensor(self.cPi)  # type: ignore[assignment]
        flidx = torch.as_tensor(self.flidx)  # type: ignore[assignment]
        lidx = torch.as_tensor(self.lidx)  # type: ignore[assignment]
        midx = torch.as_tensor(self.midx)  # type: ignore[assignment]

        target_shape = [
            directions.shape[0],
            powers.shape[0],
            powers.shape[1],
            2,
        ]
        Rs = torch.broadcast_to(directions[:, None, None, :2], target_shape)
        pows = torch.broadcast_to(powers[None], target_shape)

        Rs = torch.where(pows == 0, torch.ones_like(Rs), Rs)

        temp = Rs**powers
        monomials_xy = torch.prod(temp, dim=-1)

        Am = torch.sum(monomials_xy * cAm[None], 2)
        Bm = torch.sum(monomials_xy * cBm[None], 2)
        ABm = torch.cat(
            [
                torch.flip(Bm, (1,)),
                math.sqrt(0.5) * torch.ones((Am.shape[0], 1), device=directions.device),
                Am,
            ],
            dim=1,
        )
        ABm = ABm[:, midx + self.lmax]

        target_shape = [
            directions.shape[0],
            zpow.shape[0],
            zpow.shape[1],
            zpow.shape[2],
        ]
        z = torch.broadcast_to(directions[:, 2, None, None, None], target_shape)
        zpows = torch.broadcast_to(zpow[None], target_shape)
        z = torch.where(zpows == 0, torch.ones_like(z), z)
        zk = z**zpows

        Pi = torch.sum(zk * cPi, dim=-1)  # batch x L x M
        Pi_lm = Pi[:, lidx, abs(midx)]
        return torch.sqrt((2 * flidx + 1) / (2 * math.pi)) * Pi_lm * ABm


def scalar2rsh(x: torch.Tensor, lmax: int) -> torch.Tensor:
    """Expand scalar tensor to spherical harmonics shape with angular momentum up to `lmax`.

    Args:
        x: tensor of shape [N, *].
        lmax: maximum angular momentum.

    Returns:
        zero-padded tensor to shape [N, (lmax+1)^2, *].
    """
    return torch.cat(
        [
            x,
            torch.zeros(
                (x.shape[0], int((lmax + 1) ** 2 - 1), x.shape[2]),
                device=x.device,
                dtype=x.dtype,
            ),
        ],
        dim=1,
    )


class SO3TensorProduct(nn.Module):
    r"""SO3-equivariant Clebsch-Gordon tensor product.

    With combined indexing s=(l,m), this can be written as:

    .. math::

        y_{s,f} = \sum_{s_1,s_2} x_{2,s_2,f} x_{1,s_2,f}  C_{s_1,s_2}^{s}.

    """

    def __init__(self, lmax: int, lmax_in_2: int | None = None, lmax_out: int | None = None):
        """Initialize the SO3TensorProduct.

        Args:
            lmax: maximum angular momentum of the first input ``x1`` (and the
                output, when ``lmax_out`` is not given). Also acts as the
                shared ``lmax`` when neither ``lmax_in_2`` nor ``lmax_out`` is
                given (the original symmetric behavior).
            lmax_in_2: maximum angular momentum of the second input ``x2``.
                Defaults to ``lmax`` (symmetric form). Pass a smaller value
                when ``x2`` is known to be zero beyond ``lmax_in_2`` — this
                skips the would-be-zero CG triplets and avoids gathering /
                multiplying the padded slots, which dominates GRACE's first
                inter-block tensor product.
            lmax_out: maximum angular momentum of the output. Defaults to
                ``lmax``. Provided for symmetry; not currently exercised
                outside symmetric use.
        """
        super().__init__()
        self.lmax = lmax
        lmax_in_2_eff = int(lmax) if lmax_in_2 is None else int(lmax_in_2)
        lmax_out_eff = int(lmax) if lmax_out is None else int(lmax_out)
        self.lmax_in_2 = lmax_in_2_eff
        self.lmax_out = lmax_out_eff

        # Need the densest CG over the union of (lmax, lmax_in_2, lmax_out)
        # to populate the output, then we slice down to the actual
        # ``[lmax+1, lmax_in_2+1, lmax_out+1]`` region that's exercised.
        cg_lmax = max(lmax, lmax_in_2_eff, lmax_out_eff)
        cg_full = generate_clebsch_gordan_rsh(cg_lmax).to(matgl.float_th)
        # cg_full has shape [(cg_lmax+1)^2, (cg_lmax+1)^2, (cg_lmax+1)^2].
        # Slice to the actual lmax dims to avoid carrying zero-padded triplets
        # from the larger-lmax block when the operands are smaller.
        d1 = (lmax + 1) ** 2
        d2 = (lmax_in_2_eff + 1) ** 2
        do = (lmax_out_eff + 1) ** 2
        cg_block = cg_full[:d1, :d2, :do].contiguous()
        cg, idx_in_1, idx_in_2, idx_out = sparsify_clebsch_gordon(cg_block)
        self.register_buffer("idx_in_1", idx_in_1, persistent=False)
        self.register_buffer("idx_in_2", idx_in_2, persistent=False)
        self.register_buffer("idx_out", idx_out, persistent=False)
        self.register_buffer("clebsch_gordan", cg, persistent=False)
        self._dim_out = do

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the SO3 tensor product.

        Args:
            x1: atom-wise SO3 features, shape: [n_atoms, (l_max+1)^2, n_features].
            x2: atom-wise SO3 features, shape: [n_atoms, (l_max+1)^2, n_features].

        Returns:
            y: product of SO3 features.
        """
        # NB: profiling on CPU showed that switching the gathers to
        # ``torch.index_select`` slows the *backward* down — its gradient path
        # (``aten::index_add_`` accumulating into [E, K_in, F]) is slower for
        # these shapes than the advanced-indexing backward (``_index_put_impl_``).
        # We keep the advanced-indexing form here.
        idx_in_1 = torch.as_tensor(self.idx_in_1)  # type: ignore[assignment]
        idx_in_2 = torch.as_tensor(self.idx_in_2)  # type: ignore[assignment]
        cg = torch.as_tensor(self.clebsch_gordan)  # type: ignore[assignment]
        x1g = x1[:, idx_in_1, :]
        x2g = x2[:, idx_in_2, :]
        y = x1g * x2g * cg[None, :, None]
        return scatter_add(y, self.idx_out, dim_size=self._dim_out, dim=1)  # type: ignore[arg-type]


class SO3Convolution(nn.Module):
    r"""SO3-equivariant convolution using Clebsch-Gordon tensor product.

    With combined indexing s=(l,m), this can be written as:

    .. math::

        y_{i,s,f} = \sum_{j,s_1,s_2} x_{j,s_2,f} W_{s_1,f}(r_{ij}) Y_{s_1}(r_{ij}) C_{s_1,s_2}^{s}.

    """

    def __init__(self, lmax: int, n_atom_basis: int, n_radial: int):
        """Initialize the SO3Convolution.

        Args:
            lmax: maximum angular momentum.
            n_atom_basis: dim of node features.
            n_radial: dimension of radial basis functions.
        """
        super().__init__()
        self.lmax = lmax
        self.n_atom_basis = n_atom_basis
        self.n_radial = n_radial

        cg = generate_clebsch_gordan_rsh(lmax).to(matgl.float_th)
        cg, idx_in_1, idx_in_2, idx_out = sparsify_clebsch_gordon(cg)
        self.register_buffer("idx_in_1", idx_in_1, persistent=False)
        self.register_buffer("idx_in_2", idx_in_2, persistent=False)
        self.register_buffer("idx_out", idx_out, persistent=False)
        self.register_buffer("clebsch_gordan", cg, persistent=False)

        #        self.filternet = Dense(n_radial, n_atom_basis * (self.lmax + 1), activation=None)
        self.filternet = MLP(
            dims=[n_radial, n_atom_basis * (self.lmax + 1)],
            activation=None,
            activate_last=False,  # No activation in the last layer
            bias_last=True,
        )

        lidx, _ = sh_indices(lmax)
        self.register_buffer("Widx", lidx[idx_in_1])

    def _compute_radial_filter(self, radial_ij: torch.Tensor, cutoff_ij: torch.Tensor) -> torch.Tensor:
        """Compute radial (rotationally-invariant) filter.

        Args:
            radial_ij: radial basis functions with shape [n_neighbors, n_radial_basis]
            cutoff_ij: cutoff function with shape [n_neighbors, 1]

        Returns:
            Wij: radial filters with shape [n_neighbors, n_clebsch_gordon, n_features]
        """
        Wij = self.filternet(radial_ij) * cutoff_ij
        Wij = torch.reshape(Wij, (-1, self.lmax + 1, self.n_atom_basis))
        Widx = torch.as_tensor(self.Widx)  # type: ignore[assignment]
        return Wij[:, Widx]

    def forward(
        self,
        x: torch.Tensor,
        radial_ij: torch.Tensor,
        dir_ij: torch.Tensor,
        cutoff_ij: torch.Tensor,
        idx_i: torch.Tensor,
        idx_j: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the SO3 convolution.

        Args:
            x: atom-wise SO3 features, shape: [n_atoms, (l_max+1)^2, n_atom_basis].
            radial_ij: radial basis functions with shape [n_neighbors, n_radial_basis].
            dir_ij: direction from atom i to atom j, scaled to unit length
                [n_neighbors, (l_max+1)^2].
            cutoff_ij: cutoff function with shape [n_neighbors, 1].
            idx_i: indices for atom i.
            idx_j: indices for atom j.

        Returns:
            y: convolved SO3 features.
        """
        idx_in_1 = torch.as_tensor(self.idx_in_1)  # type: ignore[assignment]
        idx_in_2 = torch.as_tensor(self.idx_in_2)  # type: ignore[assignment]
        idx_out = torch.as_tensor(self.idx_out)  # type: ignore[assignment]
        clebsch_gordan = torch.as_tensor(self.clebsch_gordan)  # type: ignore[assignment]

        xj = x[idx_j[:, None], idx_in_2[None, :], :]
        Wij = self._compute_radial_filter(radial_ij, cutoff_ij)
        v = Wij * dir_ij[:, idx_in_1, None] * clebsch_gordan[None, :, None] * xj
        yij = scatter_add(v, idx_out, dim_size=int((self.lmax + 1) ** 2), dim=1)
        return scatter_add(yij, idx_i, dim_size=x.shape[0])


class SO3ParametricGatedNonlinearity(nn.Module):
    r"""SO3-equivariant parametric gated nonlinearity.

    With combined indexing s=(l,m), this can be written as:

    .. math::

        y_{i,s,f} = x_{j,s,f} * \sigma(f(x_{j,0,\cdot})).

    """

    def __init__(self, n_in: int, lmax: int):
        """Initialize the SO3ParametricGatedNonlinearity.

        Args:
            n_in: number of input channels.
            lmax: maximum angular momentum.
        """
        super().__init__()
        self.lmax = lmax
        self.n_in = n_in
        self.lidx, _ = sh_indices(lmax)
        self.scaling = nn.Linear(n_in, n_in * (lmax + 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s0 = x[:, 0, :]
        h = self.scaling(s0).reshape(-1, self.lmax + 1, self.n_in)
        h = h[:, self.lidx]
        return x * torch.sigmoid(h)


class SO3GatedNonlinearity(nn.Module):
    r"""SO3-equivariant gated nonlinearity.

    With combined indexing s=(l,m), this can be written as:

    .. math::

        y_{i,s,f} = x_{j,s,f} * \sigma(x_{j,0,\cdot})

    """

    def __init__(self, lmax: int):
        super().__init__()
        self.lmax = lmax
        self.lidx, _ = sh_indices(lmax)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s0 = x[:, 0, :]
        return x * torch.sigmoid(s0[:, None, :])
