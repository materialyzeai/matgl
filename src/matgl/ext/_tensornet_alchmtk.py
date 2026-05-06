# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors
#    may be used to endorse or promote products derived from this software
#    without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
"""nvalchemi-toolkit wrapper for TensorNet (PyG backend).

Wraps :class:`~matgl.models._tensornet_pyg.TensorNet` as a
:class:`~nvalchemi.models.base.BaseModelMixin`-compatible model for use in
nvalchemi dynamics pipelines (molecular dynamics and geometry optimization).

Usage
-----
Load a pre-trained TensorNet potential and wrap it::

    import matgl
    from matgl.ext.alchmtk import TensorNetWrapper

    potential = matgl.load_model("TensorNet-MatPES-PBE-v2025.1-PES")
    model = TensorNetWrapper.from_potential(potential)

    from nvalchemi.hooks import NeighborListHook
    from nvalchemi.dynamics.base import DynamicsStage

    nl_hook = NeighborListHook(model.model_config.neighbor_config, stage=DynamicsStage.BEFORE_COMPUTE)
    dynamics.register_hook(nl_hook)
    dynamics.model = model

Notes:
-----
* Energy is the primitive output. Forces and stresses are derived via
  autograd (``autograd_outputs={"forces", "stress"}``).
* Stresses use the affine strain trick via
  :func:`~nvalchemi.models._utils.prepare_strain` and
  :func:`~nvalchemi.models._utils.autograd_stresses`.
  Output unit is eV/A^3, Cauchy convention (``-dE/d(strain)/V``).
* Requires :class:`~nvalchemi.hooks.NeighborListHook` with
  ``format=NeighborListFormat.COO`` to populate ``neighbor_list`` and
  ``neighbor_list_shifts`` before each model call.
* Only ``is_intensive=False`` models are supported (total energy, not
  per-atom properties).
* ZBL nuclear repulsion is optionally included via ``calc_repuls``.
"""

from __future__ import annotations

import types
from typing import TYPE_CHECKING, Any

import torch
from pymatgen.core.periodic_table import Element
from torch import nn

from matgl.models._tensornet_pyg import TensorNet

try:
    from nvalchemi.data import AtomicData, Batch
    from nvalchemi.models._utils import (
        autograd_forces,
        autograd_stresses,
        prepare_strain,
    )
    from nvalchemi.models.base import (
        BaseModelMixin,
        ModelConfig,
        NeighborConfig,
        NeighborListFormat,
    )

    _NVALCHEMI_AVAILABLE = True
except ImportError:
    _NVALCHEMI_AVAILABLE = False
    BaseModelMixin = object  # type: ignore[misc,assignment]

if TYPE_CHECKING:
    from pathlib import Path

    from nvalchemi._typing import ModelOutputs

    from matgl.apps._pes_pyg import Potential

__all__ = ["TensorNetWrapper"]


class TensorNetWrapper(nn.Module, BaseModelMixin):  # type: ignore[misc]
    """nvalchemi-toolkit wrapper for TensorNet (PyG backend).

    Parameters
    ----------
    model : TensorNet
        An instantiated TensorNet model.  Must have ``is_intensive=False``.
    data_mean : float or torch.Tensor, optional
        Training-target mean for energy un-normalization.  Default: 0.0.
    data_std : float or torch.Tensor, optional
        Training-target standard deviation.  Default: 1.0.
    element_refs : torch.Tensor or None, optional
        Per-element-type energy offsets, shape ``[num_element_types]``.
        Added to the total energy after un-normalization.  Default: None.
    calc_repuls : bool, optional
        Include ZBL nuclear repulsion.  Default: False.
    """

    model: TensorNet

    def __init__(
        self,
        model: TensorNet,
        data_mean: float | torch.Tensor = 0.0,
        data_std: float | torch.Tensor = 1.0,
        element_refs: torch.Tensor | None = None,
        calc_repuls: bool = False,
    ) -> None:
        if not _NVALCHEMI_AVAILABLE:
            raise ImportError(
                "nvalchemi-toolkit is required to use TensorNetWrapper. Install it with: pip install nvalchemi-toolkit"
            )
        super().__init__()

        if model.is_intensive:
            raise ValueError(
                "TensorNetWrapper requires is_intensive=False (total-energy prediction). "
                "Intensive models (per-atom properties) are not supported for MD/relaxation."
            )

        self.model = model

        # ZBL nuclear repulsion (fixed analytical potential)
        self.repuls: NuclearRepulsion | None = None  # type: ignore[name-defined]
        if calc_repuls:
            from matgl.layers._zbl_pyg import NuclearRepulsion

            self.repuls = NuclearRepulsion(float(model.cutoff))

        self.model_config = ModelConfig(
            outputs=frozenset({"energy", "forces", "stress"}),
            active_outputs={"energy", "forces", "stress"},
            autograd_outputs=frozenset({"forces", "stress"}),
            autograd_inputs=frozenset({"positions"}),
            required_inputs=frozenset(),
            optional_inputs=frozenset({"neighbor_list_shifts", "cell"}),
            supports_pbc=True,
            needs_pbc=False,
            neighbor_config=NeighborConfig(
                cutoff=float(self.model.cutoff),
                format=NeighborListFormat.COO,
                half_list=False,
            ),
        )

        if not isinstance(data_mean, torch.Tensor):
            data_mean = torch.tensor(data_mean, dtype=torch.float32)
        if not isinstance(data_std, torch.Tensor):
            data_std = torch.tensor(data_std, dtype=torch.float32)
        self.data_mean: torch.Tensor
        self.data_std: torch.Tensor
        self.register_buffer("data_mean", data_mean)
        self.register_buffer("data_std", data_std)

        if element_refs is not None:
            if not isinstance(element_refs, torch.Tensor):
                element_refs = torch.tensor(element_refs, dtype=torch.float32)
            self.register_buffer("_element_ref_offset", element_refs)
        else:
            self._element_ref_offset: torch.Tensor | None = None

        self._build_z_to_type_table()

    def _build_z_to_type_table(self) -> None:
        """Build a ``[max_z + 1]`` lookup: atomic number -> type index."""
        symbol_to_idx = {sym: i for i, sym in enumerate(self.model.element_types)}
        max_z = max(Element(sym).Z for sym in self.model.element_types)
        table = torch.full((max_z + 1,), -1, dtype=torch.long)
        for sym, idx in symbol_to_idx.items():
            table[Element(sym).Z] = idx
        self._z_to_type: torch.Tensor
        self.register_buffer("_z_to_type", table, persistent=False)

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_potential(cls, potential: Potential) -> TensorNetWrapper:
        """Construct from a matgl :class:`~matgl.apps._pes_pyg.Potential`.

        Parameters
        ----------
        potential : Potential
            A matgl ``Potential`` wrapping a ``TensorNet`` model.

        Returns:
        -------
        TensorNetWrapper
        """
        if not isinstance(potential.model, TensorNet):
            raise TypeError(f"Expected potential.model to be TensorNet, got {type(potential.model).__name__}.")

        element_refs = None
        if potential.element_refs is not None:
            element_refs = potential.element_refs.property_offset.clone()

        data_mean = potential.data_mean
        data_std = potential.data_std
        assert isinstance(data_mean, torch.Tensor)
        assert isinstance(data_std, torch.Tensor)
        return cls(
            model=potential.model,
            data_mean=data_mean.clone(),
            data_std=data_std.clone(),
            element_refs=element_refs,
            calc_repuls=getattr(potential, "calc_repuls", False),
        )

    # ------------------------------------------------------------------
    # BaseModelMixin interface
    # ------------------------------------------------------------------

    @property
    def embedding_shapes(self) -> dict[str, tuple[int, ...]]:
        """Return ``{node_embeddings: (units,), graph_embeddings: (units,)}``."""
        units: int = self.model.units
        return {
            "node_embeddings": (units,),
            "graph_embeddings": (units,),
        }

    def _model_dtype(self) -> torch.dtype:
        """Return the dtype of TensorNet parameters."""
        try:
            return next(self.model.parameters()).dtype
        except StopIteration:
            return torch.float32

    # ------------------------------------------------------------------
    # adapt_input
    # ------------------------------------------------------------------

    def adapt_input(self, data: AtomicData | Batch, **kwargs: Any) -> dict[str, Any]:
        """Convert nvalchemi ``Batch`` to TensorNet inputs.

        Reads ``neighbor_list`` and ``neighbor_list_shifts`` directly
        from the batch (populated by :class:`~nvalchemi.hooks.NeighborListHook`).
        Does **not** call ``super().adapt_input()`` — ``Batch`` lacks
        ``model_dump()`` which the base implementation requires.
        Gradient setup is handled by ``forward()`` before this call.

        Parameters
        ----------
        data : AtomicData | Batch
            A bare ``AtomicData`` is promoted to a single-graph ``Batch``.

        Returns:
        -------
        dict[str, Any]
        """
        if isinstance(data, AtomicData):
            data = Batch.from_data_list([data])

        dtype = self._model_dtype()
        device = data.positions.device
        B: int = data.num_graphs

        node_type = self._z_to_type[data.atomic_numbers]

        # nvalchemi (E, 2) -> TensorNet/PyG (2, E)
        edge_index = data.neighbor_list.T  # [2, E]
        E: int = edge_index.shape[1]

        shifts_raw = getattr(data, "neighbor_list_shifts", None)
        if shifts_raw is None:
            shifts = torch.zeros(E, 3, dtype=dtype, device=device)
        else:
            shifts = shifts_raw.to(dtype=dtype, device=device)

        cell_raw = getattr(data, "cell", None)
        if cell_raw is None:
            cell = torch.eye(3, dtype=dtype, device=device).unsqueeze(0).expand(B, -1, -1)
        else:
            cell = cell_raw.to(dtype=dtype, device=device)

        positions = data.positions.to(dtype=dtype)
        edge_batch = data.batch_idx[edge_index[0]]
        pbc_offshift = torch.einsum("eb,ebc->ec", shifts, cell[edge_batch])

        return {
            "node_type": node_type,
            "pos": positions,
            "edge_index": edge_index,
            "pbc_offshift": pbc_offshift,
            "batch": data.batch_idx,
            "num_graphs": B,
        }

    # ------------------------------------------------------------------
    # adapt_output
    # ------------------------------------------------------------------

    def adapt_output(
        self,
        model_output: dict[str, Any],
        data: AtomicData | Batch,
    ) -> ModelOutputs:
        """Map results to nvalchemi ``ModelOutputs``."""
        output = super().adapt_output(model_output, data)
        output["energy"] = model_output["energy"]
        if model_output.get("forces") is not None:
            output["forces"] = model_output["forces"]
        if model_output.get("stress") is not None:
            output["stress"] = model_output["stress"]
        return output

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------

    def forward(self, data: AtomicData | Batch, **kwargs: Any) -> ModelOutputs:
        """Run TensorNet and return energy, forces, and optionally stress.

        Parameters
        ----------
        data : AtomicData | Batch
            Input data with ``neighbor_list`` and ``neighbor_list_shifts``
            populated by :class:`~nvalchemi.hooks.NeighborListHook`.

        Returns:
        -------
        ModelOutputs
        """
        if isinstance(data, AtomicData):
            data = Batch.from_data_list([data])

        compute_forces = "forces" in (self.model_config.active_outputs & self.model_config.outputs)
        compute_stresses = "stress" in (self.model_config.active_outputs & self.model_config.outputs)

        # Affine strain for stress (before adapt_input so scaled positions
        # flow through the model).
        displacement = None
        orig_positions = None
        orig_cell = None
        if compute_stresses and hasattr(data, "cell") and data.cell is not None:
            scaled_pos, scaled_cell, displacement = prepare_strain(data.positions, data.cell, data.batch_idx)
            orig_positions = data.positions
            orig_cell = data.cell
            data["positions"] = scaled_pos
            data["cell"] = scaled_cell

        # Clone + enable grad (avoids mutating the original batch).
        if compute_forces or compute_stresses:
            pos = data.positions.clone()
            pos.requires_grad_(True)
            data["positions"] = pos

        inputs = self.adapt_input(data, **kwargs)

        g = types.SimpleNamespace(
            node_type=inputs["node_type"],
            pos=inputs["pos"],
            edge_index=inputs["edge_index"],
            pbc_offshift=inputs["pbc_offshift"],
            batch=inputs["batch"],
            num_graphs=inputs["num_graphs"],
        )

        raw_energy = self.model(g=g)
        total_energy = self.data_std * raw_energy + self.data_mean

        # ZBL repulsion
        if self.repuls is not None:
            from matgl.graph._compute_pyg import compute_pair_vector_and_distance

            _, bond_dist = compute_pair_vector_and_distance(
                inputs["pos"],
                inputs["edge_index"],
                inputs["pbc_offshift"],
            )
            g.bond_dist = bond_dist
            total_energy = total_energy + self.repuls(self.model.element_types, g)

        # Element reference offsets
        if self._element_ref_offset is not None:
            atomic_offset = self._element_ref_offset[inputs["node_type"]]
            graph_offset = torch.zeros(inputs["num_graphs"], device=atomic_offset.device, dtype=atomic_offset.dtype)
            graph_offset.scatter_add_(0, inputs["batch"], atomic_offset)
            total_energy = total_energy + graph_offset

        # Reshape to [B, 1]
        if total_energy.dim() == 0:
            total_energy = total_energy.unsqueeze(0)
        if total_energy.dim() == 1:
            total_energy = total_energy.unsqueeze(-1)

        result: dict[str, torch.Tensor | None] = {"energy": total_energy}

        if compute_forces:
            result["forces"] = autograd_forces(
                total_energy,
                data.positions,
                training=False,
                retain_graph=compute_stresses,
            )

        if compute_stresses and displacement is not None:
            result["stress"] = autograd_stresses(
                total_energy,
                displacement,
                orig_cell,
                data.num_graphs,
            )

        # Restore originals after strain
        if orig_positions is not None:
            data["positions"] = orig_positions
            data["cell"] = orig_cell

        return self.adapt_output(result, data)

    # ------------------------------------------------------------------
    # compute_embeddings
    # ------------------------------------------------------------------

    def compute_embeddings(self, data: AtomicData | Batch, **kwargs: Any) -> AtomicData | Batch:
        """Compute node and graph embeddings (no forces/stress).

        Writes ``node_embeddings`` ([V, units]) and ``graph_embeddings``
        ([B, units]) into *data* in-place.
        """
        if isinstance(data, AtomicData):
            data = Batch.from_data_list([data])

        inputs = self.adapt_input(data, **kwargs)
        g = types.SimpleNamespace(
            node_type=inputs["node_type"],
            pos=inputs["pos"],
            edge_index=inputs["edge_index"],
            pbc_offshift=inputs["pbc_offshift"],
            batch=inputs["batch"],
            num_graphs=inputs["num_graphs"],
        )

        fea_dict = self.model(g=g, return_all_layer_output=True)
        node_embeddings = fea_dict["readout"]  # [V, units]

        units = node_embeddings.shape[-1]
        graph_embeddings = torch.zeros(
            inputs["num_graphs"],
            units,
            device=node_embeddings.device,
            dtype=node_embeddings.dtype,
        )
        graph_embeddings.scatter_add_(
            0,
            inputs["batch"].unsqueeze(-1).expand(-1, units),
            node_embeddings,
        )

        data.node_embeddings = node_embeddings
        data.graph_embeddings = graph_embeddings
        return data

    # ------------------------------------------------------------------
    # export_model
    # ------------------------------------------------------------------

    def export_model(self, path: Path, as_state_dict: bool = False) -> None:
        """Save the underlying TensorNet (without the nvalchemi wrapper).

        Not compatible with ``matgl.load_model``.  To get that format,
        save the original ``Potential`` before wrapping.
        """
        if as_state_dict:
            torch.save(self.model.state_dict(), path)
        else:
            torch.save(self.model, path)
