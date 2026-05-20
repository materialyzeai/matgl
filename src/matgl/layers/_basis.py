"""Radial and angular basis functions used to expand interatomic distances.

This module gathers the bond-distance and angle expansions used by every
matgl architecture. The two most-used radial bases are:

* :class:`SphericalBesselFunction` -- the orthonormal :math:`j_l` basis on
  ``[0, cutoff]`` used by M3GNet, CHGNet and SO3Net.
* :class:`RadialBesselFunction` -- the simpler :math:`l=0` variant from
  https://arxiv.org/abs/2003.03123 used by TensorNet and GRACE; supports
  learnable frequencies.

Alternative radial bases:

* :class:`GaussianExpansion` -- fixed-center Gaussian RBF;
* :class:`ExpNormalFunction` -- exponential-modulated Gaussian RBF used by
  the original PaiNN paper.

Angular expansions:

* :class:`SphericalHarmonicsFunction` -- real spherical harmonics
  evaluated from ``cos(theta)`` (and optionally ``phi``);
* :class:`SphericalBesselWithHarmonics` -- the combined three-body
  expansion used by M3GNet's line graph;
* :class:`FourierExpansion` -- the periodic angular basis used by
  CHGNet's bond-graph terms.

Constants are pre-computed once during ``__init__`` and stored as
non-persistent buffers so they ride along on ``.to(device)`` without
appearing in the saved ``state_dict``.

End users should normally reach for :class:`~matgl.layers.BondExpansion`
in :mod:`matgl.layers._bond`, which selects one of the bases via a string
flag.
"""

from __future__ import annotations

from functools import lru_cache
from math import pi, sqrt

import sympy
import torch
from torch import Tensor, nn

import matgl
from matgl.layers._three_body import combine_sbf_shf
from matgl.utils.cutoff import cosine_cutoff
from matgl.utils.maths import SPHERICAL_BESSEL_ROOTS, _get_lambda_func


class GaussianExpansion(nn.Module):
    """Gaussian Radial Expansion.

    The bond distance is expanded to a vector of shape [m], where m is the number of Gaussian basis centers.
    """

    def __init__(
        self,
        initial: float = 0.0,
        final: float = 4.0,
        num_centers: int = 20,
        width: None | float = 0.5,
    ):
        """Initialize the GaussianExpansion.

        Args:
            initial: Location of initial Gaussian basis center.
            final: Location of final Gaussian basis center
            num_centers: Number of Gaussian Basis functions
            width: Width of Gaussian Basis functions.
        """
        super().__init__()
        self.centers = nn.Parameter(torch.linspace(initial, final, num_centers), requires_grad=False)  # type: ignore
        if width is None:
            self.width = 1.0 / torch.diff(self.centers).mean()
        else:
            self.width = torch.as_tensor(width)

    def reset_parameters(self):
        """Reinitialize model parameters."""
        self.centers = nn.Parameter(self.centers, requires_grad=False)

    def forward(self, bond_dists):
        """Expand distances.

        Args:
            bond_dists :
                Bond (edge) distances between two atoms (nodes)

        Returns:
            A vector of expanded distance with shape [num_centers]
        """
        diff = bond_dists[:, None] - self.centers[None, :]
        return torch.exp(-self.width * (diff**2))


class SphericalBesselFunction(nn.Module):
    """Calculate the spherical Bessel function based on sympy + pytorch implementations."""

    def __init__(self, max_l: int, max_n: int = 5, cutoff: float = 5.0, smooth: bool = False):
        """Initialize the SphericalBesselFunction.

        Args:
            max_l: int, max order (excluding l)
            max_n: int, max number of roots used in each l
            cutoff: float, cutoff radius
            smooth: Whether to smooth the function.
        """
        super().__init__()
        self.max_l = max_l
        self.max_n = max_n
        self.register_buffer("cutoff", torch.tensor(cutoff))
        self.smooth = smooth
        if smooth:
            self.funcs = self._calculate_smooth_symbolic_funcs()
        else:
            self.funcs = self._calculate_symbolic_funcs()
            # Pre-compute non-smooth basis constants once. ``roots_slice`` holds
            # the Bessel zeros used in ``_call_sbf``; ``inv_norm`` collapses the
            # ``sqrt(2/c^3) / |j_{l+1}(root)|`` per-edge normalisation into a
            # single broadcastable tensor of shape (max_l, max_n). Storing them
            # as non-persistent buffers means they ride along on ``.to(device)``
            # but never touch the saved ``state_dict`` (preserves checkpoint
            # compatibility).
            roots_slice = SPHERICAL_BESSEL_ROOTS[:max_l, :max_n].clone()
            factor = sqrt(2.0 / float(cutoff) ** 3)
            inv_norm_rows = [factor / torch.abs(self.funcs[i + 1](roots_slice[i])) for i in range(max_l)]
            inv_norm = torch.stack(inv_norm_rows, dim=0)
            self.register_buffer("roots_slice", roots_slice, persistent=False)
            self.register_buffer("inv_norm", inv_norm, persistent=False)

    @lru_cache(maxsize=128)
    def _calculate_symbolic_funcs(self) -> list:
        """Generate spherical basis functions based on Rayleigh formula.

        Returns:
            list of symbolic functions
        """
        x = sympy.symbols("x")
        funcs = [sympy.expand_func(sympy.functions.special.bessel.jn(i, x)) for i in range(self.max_l + 1)]
        return [sympy.lambdify(x, func, torch) for func in funcs]

    @lru_cache(maxsize=128)
    def _calculate_smooth_symbolic_funcs(self) -> list:
        return _get_lambda_func(max_n=self.max_n, cutoff=self.cutoff)

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        """Compute the spherical Bessel function values.

        Args:
            r: torch.Tensor, distance tensor, 1D.

        Returns:
            torch.Tensor: [n, max_n * max_l] spherical Bessel function results
        """
        if self.smooth:
            return self._call_smooth_sbf(r)
        return self._call_sbf(r)

    def _call_smooth_sbf(self, r):
        results = [i(r) for i in self.funcs]
        return torch.t(torch.stack(results))

    def _call_sbf(self, r):
        # ``r`` is per-edge distance. The non-smooth spherical Bessel basis is
        # j_l(root_{l,n} * r / cutoff) * sqrt(2/cutoff^3) / |j_{l+1}(root_{l,n})|
        # for l in [0, max_l) and n in [0, max_n). All ``l``-independent
        # constants are precomputed and stored as buffers in ``__init__``; the
        # remaining work per call is one Bessel-function evaluation per ``l``.
        scaled = r.clamp(max=self.cutoff).unsqueeze(-1) / self.cutoff  # (N, 1)
        results = []
        for i in range(self.max_l):
            x = scaled * self.roots_slice[i]  # (N, max_n)
            results.append(self.funcs[i](x) * self.inv_norm[i])
        return torch.cat(results, dim=1)

    @staticmethod
    def rbf_j0(r, cutoff: float = 5.0, max_n: int = 3):
        """Spherical Bessel function of order 0.

        Ensures the function value vanishes at cutoff.

        Args:
            r: torch.Tensor pytorch tensors
            cutoff: float, the cutoff radius
            max_n: int max number of basis

        Returns:
            basis function expansion using first spherical Bessel function
        """
        n = (torch.arange(1, max_n + 1)).type(dtype=matgl.float_th)[None, :]
        r = r[:, None]
        return sqrt(2.0 / cutoff) * torch.sin(n * pi / cutoff * r) / r


class RadialBesselFunction(nn.Module):
    """Zeroth order bessel function of the first kind.

    Implements the proposed 1D radial basis function in terms of zeroth order bessel function of the first kind with
    increasing number of roots and a given cutoff.

    Details are given in: https://arxiv.org/abs/2003.03123

    This is equivalent to SphericalBesselFunction class with max_l=1, i.e. only l=0 bessel functions), but with
    optional learnable frequencies.
    """

    def __init__(self, max_n: int, cutoff: float, learnable: bool = False):
        """Initialize the RadialBesselFunction.

        Args:
            max_n: int, max number of roots (including max_n)
            cutoff: float, cutoff radius
            learnable: bool, whether to learn the location of roots.
        """
        super().__init__()
        self.max_n = max_n
        self.inv_cutoff = 1 / cutoff
        self.norm_const = (2 * self.inv_cutoff) ** 0.5

        if learnable:
            self.frequencies = torch.nn.Parameter(
                data=torch.Tensor(pi * torch.arange(1, self.max_n + 1, dtype=matgl.float_th)),
                requires_grad=True,
            )
        else:
            self.register_buffer(
                "frequencies",
                pi * torch.arange(1, self.max_n + 1, dtype=matgl.float_th),
            )

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        r = r[:, None]  # (nEdges,1)
        d_scaled = r * self.inv_cutoff
        return self.norm_const * torch.sin(self.frequencies * d_scaled) / r


class FourierExpansion(nn.Module):
    """Fourier Expansion of a (periodic) scalar feature."""

    def __init__(self, max_f: int = 5, interval: float = pi, scale_factor: float = 1.0, learnable: bool = False):
        """Initialize the FourierExpansion.

        Args:
            max_f (int): the maximum frequency of the Fourier expansion.
                Default = 5
            interval (float): the interval of the Fourier expansion, such that functions
                are orthonormal over [-interval, interval]. Default = pi
            scale_factor (float): pre-factor to scale all values.
            learnable (bool): whether to set the frequencies as learnable parameters.
                Default = False.
        """
        super().__init__()
        self.max_f = max_f
        self.interval = interval
        self.scale_factor = scale_factor
        # Initialize frequencies at canonical
        if learnable:
            self.frequencies = torch.nn.Parameter(
                data=torch.arange(0, max_f + 1, dtype=matgl.float_th),
                requires_grad=True,
            )
        else:
            self.register_buffer("frequencies", torch.arange(0, max_f + 1, dtype=matgl.float_th))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Expand x into cos and sin functions."""
        result = x.new_zeros(x.shape[0], 1 + 2 * self.max_f)
        tmp = torch.outer(x, self.frequencies)
        result[:, ::2] = torch.cos(tmp * pi / self.interval)
        result[:, 1::2] = torch.sin(tmp[:, 1:] * pi / self.interval)
        return result / self.interval * self.scale_factor


class SphericalHarmonicsFunction(nn.Module):
    """Spherical Harmonics function."""

    def __init__(self, max_l: int, use_phi: bool = True):
        """Initialize the SphericalHarmonicsFunction.

        Args:
            max_l: int, max l (excluding l)
            use_phi: bool, whether to use the polar angle. If not,
                the function will compute `Y_l^0`.
        """
        super().__init__()
        self.max_l = max_l
        self.use_phi = use_phi
        funcs = []
        theta, phi = sympy.symbols("theta phi")
        for lval in range(self.max_l):
            m_list = range(-lval, lval + 1) if self.use_phi else [0]  # type: ignore
            for m in m_list:
                func = sympy.functions.special.spherical_harmonics.Znm(lval, m, theta, phi).expand(func=True)
                funcs.append(func)
        # replace all theta with cos(theta)
        cos_theta = sympy.symbols("costheta")
        funcs = [i.subs({theta: sympy.acos(cos_theta)}) for i in funcs]
        self.orig_funcs = [sympy.simplify(i).evalf() for i in funcs]
        self.funcs = [sympy.lambdify([cos_theta, phi], i, [{"conjugate": torch.conj}, torch]) for i in self.orig_funcs]
        self.funcs[0] = _y00

    def __call__(self, cos_theta, phi=None):
        """Compute spherical harmonics values.

        Args:
            cos_theta: Cosine of the azimuthal angle
            phi: torch.Tensor, the polar angle.

        Returns:
            torch.Tensor: [n, m] spherical harmonic results, where n is the number
            of angles. The column is arranged following
            `[Y_0^0, Y_1^{-1}, Y_1^{0}, Y_1^1, Y_2^{-2}, ...]`
        """
        # cos_theta = torch.tensor(cos_theta, dtype=torch.complex64)
        # phi = torch.tensor(phi, dtype=torch.complex64)
        return torch.stack([func(cos_theta, phi) for func in self.funcs], axis=1)
        # results = results.type(dtype=DataType.torch_float)
        # return results


def _y00(theta, phi):
    r"""Spherical Harmonics with `l=m=0`.

    ..math::
        Y_0^0 = \frac{1}{2} \sqrt{\frac{1}{\pi}}

    Args:
        theta: torch.Tensor, the azimuthal angle
        phi: torch.Tensor, the polar angle

    Returns: `Y_0^0` results
    """
    return 0.5 * torch.ones_like(theta) * sqrt(1.0 / pi)


def spherical_bessel_smooth(r: Tensor, cutoff: float = 5.0, max_n: int = 10) -> Tensor:
    """Orthogonal basis with first and second derivative vanishing at the cutoff.

    The function was derived from the order 0 spherical Bessel function, and was
    expanded by the different zero roots.

    Ref:
        https://arxiv.org/pdf/1907.02374.pdf

    Args:
        r: torch.Tensor distance tensor
        cutoff: float, cutoff radius
        max_n: int, max number of basis, expanded by the zero roots

    Returns:
        expanded spherical harmonics with derivatives smooth at boundary
    """
    n = torch.arange(max_n).type(dtype=matgl.float_th)[None, :]
    r = r[:, None]
    fnr = (
        (-1) ** n
        * sqrt(2.0)
        * pi
        / cutoff**1.5
        * (n + 1)
        * (n + 2)
        / torch.sqrt(2 * n**2 + 6 * n + 5)
        * (_sinc(r * (n + 1) * pi / cutoff) + _sinc(r * (n + 2) * pi / cutoff))
    )
    en = n**2 * (n + 2) ** 2 / (4 * (n + 1) ** 4 + 1)
    dn = [torch.tensor(1.0)]
    for i in range(1, max_n):
        dn_value = 1 - en[0, i] / dn[-1]
        dn.append(dn_value)
    dn = torch.stack(dn)  # type: ignore
    gn = [fnr[:, 0]]
    for i in range(1, max_n):
        gn_value = 1 / torch.sqrt(dn[i]) * (fnr[:, i] + torch.sqrt(en[0, i] / dn[i - 1]) * gn[-1])
        gn.append(gn_value)

    return torch.t(torch.stack(gn))


def _sinc(x):
    return torch.sin(x) / x


class SphericalBesselWithHarmonics(nn.Module):
    """Expansion of basis using Spherical Bessel and Harmonics."""

    def __init__(self, max_n: int, max_l: int, cutoff: float, use_smooth: bool, use_phi: bool):
        """Init SphericalBesselWithHarmonics.

        Args:
            max_n: Degree of radial basis functions.
            max_l: Degree of angular basis functions.
            cutoff: Cutoff sphere.
            use_smooth: Whether using smooth version of SBFs or not.
            use_phi: Using phi as angular basis functions.
        """
        super().__init__()

        assert max_n <= 64
        self.max_n = max_n
        self.max_l = max_l
        self.cutoff = cutoff
        self.use_phi = use_phi
        self.use_smooth = use_smooth

        # retrieve formulas
        self.shf = SphericalHarmonicsFunction(self.max_l, self.use_phi)
        if self.use_smooth:
            self.sbf = SphericalBesselFunction(self.max_l, self.max_n * self.max_l, self.cutoff, self.use_smooth)
        else:
            self.sbf = SphericalBesselFunction(self.max_l, self.max_n, self.cutoff, self.use_smooth)

    def forward(self, triple_bond_lengths, cos_theta, phi):
        """Compute the spherical-bessel * spherical-harmonics basis.

        Args:
            triple_bond_lengths: bond-length tensor for each triple.
            cos_theta: cosine of the bond-bond angle for each triple.
            phi: azimuthal angle for each triple.
        """
        sbf = self.sbf(triple_bond_lengths)
        shf = self.shf(cos_theta, phi)
        return combine_sbf_shf(sbf, shf, max_n=self.max_n, max_l=self.max_l, use_phi=self.use_phi)


class ExpNormalFunction(nn.Module):
    """Implementation of radial basis function using exponential normal smearing."""

    def __init__(self, cutoff: float = 5.0, num_rbf: int = 50, learnable: bool = True):
        """Initialize ExpNormalSmearing.

        Args:
            cutoff (float): The cutoff distance beyond which interactions are considered negligible. Default is 5.0.
            num_rbf (int): The number of radial basis functions (RBF) to use. Default is 50.
            learnable (bool): If True, the means and betas parameters are learnable.
                              If False, they are fixed. Default is True.
        """
        super().__init__()
        self.cutoff = cutoff
        self.num_rbf = num_rbf
        self.learnable = learnable

        self.alpha = 5.0 / cutoff

        means, betas = self._initial_params()
        if learnable:
            self.register_parameter("means", nn.Parameter(means))
            self.register_parameter("betas", nn.Parameter(betas))
        else:
            self.register_buffer("means", means)
            self.register_buffer("betas", betas)

    def _initial_params(self):
        """Initialize the means and betas parameters."""
        start_value = torch.exp(torch.tensor(-self.cutoff, dtype=matgl.float_th))
        means = torch.linspace(start_value, 1, self.num_rbf)
        betas = torch.tensor([(2 / self.num_rbf * (1 - start_value)) ** -2] * self.num_rbf)
        return means, betas

    def forward(self, r: torch.Tensor):
        """Compute the radial basis function for the input distances.

        Args:
            r (torch.Tensor): Input distances.

        Returns:
            torch.Tensor: Smearing function applied to the input distances.
        """
        r = r.unsqueeze(-1)
        cutoff = torch.as_tensor(self.cutoff).item()  # type: ignore[assignment]
        betas = torch.as_tensor(self.betas)  # type: ignore[assignment]
        means = torch.as_tensor(self.means)  # type: ignore[assignment]
        return cosine_cutoff(r, cutoff) * torch.exp(-betas * (torch.exp(self.alpha * (-r)) - means) ** 2)
