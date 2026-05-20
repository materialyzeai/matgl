"""Tests for the public ``matgl.graph.converters`` re-export."""

from __future__ import annotations

from matgl.graph._converters import GraphConverter as _PrivateGraphConverter
from matgl.graph.converters import GraphConverter


def test_converters_public_alias():
    """``matgl.graph.converters.GraphConverter`` must alias the private implementation."""
    assert GraphConverter is _PrivateGraphConverter
