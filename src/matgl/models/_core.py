from __future__ import annotations

import warnings
from abc import ABCMeta
from typing import TYPE_CHECKING, Any

import torch
from torch import nn

from matgl.utils.io import IOMixIn

if TYPE_CHECKING:
    from torch_geometric.data import Data

    from matgl.graph.converters import GraphConverter


def _warn_feature_dict_kwarg(name: str) -> None:
    """Emit a DeprecationWarning for legacy feature-dict flags.

    ``return_all_layer_output`` / ``return_features`` will be removed in matgl v5;
    access ``model.feature_dict`` after calling ``forward`` / ``predict_structure``
    instead.
    """
    warnings.warn(
        f"`{name}` is deprecated and will be removed in matgl v5. "
        "Intermediate layer outputs are now always stored on `model.feature_dict` "
        "after each forward pass — use that instead.",
        DeprecationWarning,
        stacklevel=3,
    )


def structure_to_graph(
    model: nn.Module,
    structure: Any,
    graph_converter: GraphConverter | None = None,
) -> tuple[Data, torch.Tensor]:
    """Convert a structure into a graph ready for ``model.forward``.

    The default converter is chosen from the input type: ``Molecule2Graph`` for a pymatgen ``Molecule``,
    ``Atoms2Graph`` for an ASE ``Atoms`` and ``Structure2Graph`` otherwise, using ``model.element_types`` and
    ``model.cutoff``. Positions and periodic offsets are attached, and floating-point graph tensors are moved to the
    model's device and dtype.

    Args:
        model: Model providing ``element_types`` and ``cutoff`` (and whose parameters set device/dtype).
        structure: pymatgen ``Structure`` / ``Molecule`` or ASE ``Atoms``.
        graph_converter: Optional converter; overrides the type-based default.

    Returns:
        (graph, default state attributes as a tensor).
    """
    if graph_converter is None:
        from pymatgen.core import Molecule

        element_types: tuple[str, ...] = tuple(getattr(model, "element_types"))  # noqa: B009
        cutoff = float(getattr(model, "cutoff"))  # noqa: B009
        if isinstance(structure, Molecule):
            from matgl.ext.pymatgen import Molecule2Graph

            graph_converter = Molecule2Graph(element_types=element_types, cutoff=cutoff)
        elif hasattr(structure, "get_positions"):  # ASE Atoms, without importing ase eagerly
            from matgl.ext.ase import Atoms2Graph

            graph_converter = Atoms2Graph(element_types=element_types, cutoff=cutoff)
        else:
            from matgl.ext.pymatgen import Structure2Graph

            graph_converter = Structure2Graph(element_types=element_types, cutoff=cutoff)

    g, lat, state_default = graph_converter.get_graph(structure)
    param = next(model.parameters(), None)
    device = param.device if param is not None else g.frac_coords.device
    dtype = param.dtype if param is not None and param.is_floating_point() else g.frac_coords.dtype
    g = g.to(device)
    lat = torch.as_tensor(lat, device=device, dtype=dtype)
    g.frac_coords = g.frac_coords.to(dtype)
    g.pbc_offset = g.pbc_offset.to(dtype)
    g.pbc_offshift = torch.matmul(g.pbc_offset, lat[0])
    g.pos = g.frac_coords @ lat[0]
    return g, torch.as_tensor(state_default, dtype=dtype, device=device)


class MatGLModel(nn.Module, IOMixIn, metaclass=ABCMeta):
    def __init__(self) -> None:
        super().__init__()
        # Populated by ``forward`` with the intermediate layer features
        # (e.g. ``embedding``, ``gc_<i>``, ``readout``, ``final``).
        # Always refreshed on every ``forward`` call. Use this in place of the
        # deprecated ``return_all_layer_output`` / ``return_features`` flags.
        self.feature_dict: dict = {}

    def predict_structure(
        self,
        structure: Any,
        state_feats: torch.Tensor | None = None,
        graph_converter: GraphConverter | None = None,
        output_layers: list | None = None,
        return_features: bool = False,
        **kwargs: Any,
    ) -> torch.Tensor | dict:
        """Convenience method to directly predict a property from a structure.

        Args:
            structure: pymatgen ``Structure`` / ``Molecule`` or ASE ``Atoms``. The default graph converter is picked
                from the input type (see :func:`structure_to_graph`).
            state_feats: Optional state attributes. Defaults to the converter's default state.
            graph_converter: Optional graph converter overriding the type-based default.
            output_layers: Names of ``feature_dict`` entries to return when ``return_features=True``; all entries
                when ``None``.
            return_features: **Deprecated.** Use ``model.feature_dict`` after calling ``predict_structure``
                instead. Will be removed in matgl v5.
            **kwargs: Extra keyword arguments forwarded to ``forward`` (e.g. ``total_charge`` for QET).

        Returns:
            Predicted property tensor, or a dict of intermediate features when ``return_features=True``.
        """
        if return_features:
            _warn_feature_dict_kwarg("return_features")
        g, state_default = structure_to_graph(self, structure, graph_converter)
        if state_feats is None:
            state_feats = state_default
        output = self(g=g, state_attr=state_feats, **kwargs)
        if not return_features:
            return output.detach()
        if output_layers is None:
            output_layers = list(self.feature_dict)
        elif not isinstance(output_layers, list) or set(output_layers).difference(self.feature_dict):
            raise ValueError(f"Invalid output_layers. Must be a sublist of {list(self.feature_dict)}.")
        return {
            k: v.detach() if isinstance(v, torch.Tensor) else v
            for k, v in self.feature_dict.items()
            if k in output_layers
        }
