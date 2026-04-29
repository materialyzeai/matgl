"""Tests for the backend-dispatching ``matgl.graph.converters`` shim."""

from __future__ import annotations

import matgl
from matgl.graph.converters import GraphConverter


def test_converters_shim_dispatches_to_active_backend():
    """``matgl.graph.converters.GraphConverter`` must alias the active backend's
    converter base class.

    The shim is a thin re-export and is imported by other modules (notably for
    type-checking), so we just need to make sure the alias works on either backend.
    """
    if matgl.config.BACKEND == "DGL":
        from matgl.graph._converters_dgl import GraphConverter as Expected
    else:
        from matgl.graph._converters_pyg import GraphConverter as Expected

    assert GraphConverter is Expected
