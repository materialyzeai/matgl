"""Zbl repulsive potential.The official implementation can be found in https://github.com/atomistic-machine-learning/schnetpack."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from ase.data import atomic_numbers
from torch_geometric.nn import MessagePassing, global_add_pool
from torch_geometric.utils import scatter

import matgl
from matgl.layers._activations import softplus_inverse
from matgl.utils.cutoff import polynomial_cutoff


class NuclearRepulsion(MessagePassing):
    """Ziegler-Biersack-Littmark style repulsion energy.

    Args:
        r_cut (float): cutoff for interaction range
        a0 (float): distance unit conversion from bohr into angstrom.
        ke (float): coulomb constant unit conversion from Ha into eV.
        trainable (bool):  whether the parameters are trainable.
    """

    def __init__(
        self,
        r_cut: float,
        a0: float = 0.5291772105638411,
        ke: float = 14.399645351950548,
        trainable: bool = False,
    ):
        super().__init__(aggr="add")
        self.r_cut = r_cut
        self.register_buffer("ke", torch.tensor(ke, dtype=matgl.float_th))

        a_div = softplus_inverse(torch.tensor([1.0 / (a0 * 0.8854)], dtype=matgl.float_th))
        a_pow = softplus_inverse(torch.tensor([0.23], dtype=matgl.float_th))
        exponents = softplus_inverse(torch.tensor([3.19980, 0.94229, 0.40290, 0.20162], dtype=matgl.float_th))
        coefficients = softplus_inverse(torch.tensor([0.18175, 0.50986, 0.28022, 0.02817], dtype=matgl.float_th))

        # Initialize network parameters
        self.a_pow = nn.Parameter(a_pow, requires_grad=trainable)
        self.a_div = nn.Parameter(a_div, requires_grad=trainable)
        self.coefficients = nn.Parameter(coefficients, requires_grad=trainable)
        self.exponents = nn.Parameter(exponents, requires_grad=trainable)

    def forward(self, element_types: tuple, graph):
        """Compute the ZBL nuclear repulsion energy.

        Args:
            element_types: A tuple of element types defined in the model class.
            graph: PyG graph.

        Returns:
            energy: Pairwise ZBL nuclear repulsive energy.
        """
        z_list = torch.tensor([atomic_numbers[i] for i in element_types], dtype=matgl.float_th)
        z = z_list[graph.node_type]

        idx_i = graph.edge_index[0]
        idx_j = graph.edge_index[1]

        r_ij = graph.bond_dist

        # Construct screening function
        a = z ** F.softplus(self.a_pow)
        a_ij = (a[idx_i] + a[idx_j]) * F.softplus(self.a_div)
        # Get exponents and coefficients, normalize the latter
        exponents = a_ij[..., None] * F.softplus(self.exponents)[None, ...]
        coefficients = F.softplus(self.coefficients)[None, ...]
        coefficients = F.normalize(coefficients, p=1.0, dim=1)

        screening = torch.sum(coefficients * torch.exp(-exponents * r_ij[:, None]), dim=1)

        # Compute nuclear repulsion
        eij_repuls = (z[idx_i] * z[idx_j]) * polynomial_cutoff(r_ij, self.r_cut) * screening / r_ij

        # Aggregate edge contributions to nodes
        e_repuls = scatter(eij_repuls, idx_i.to(torch.long))

        # Global summation of node energies
        energy = 0.5 * self.ke * global_add_pool(e_repuls, graph.batch)

        return torch.squeeze(energy)

    def message(self, eij_repuls):
        return eij_repuls

    def update(self, aggr_out):
        return aggr_out
