from __future__ import annotations

import importlib.util
import os.path
from unittest.mock import patch

import pytest

import matgl
from matgl.config import MATGL_CACHE, clear_cache, ensure_backend


def test_clear_cache():
    clear_cache(False)
    assert not os.path.exists(MATGL_CACHE)


def test_clear_cache_missing_dir(caplog):
    """A second ``clear_cache`` call after the cache was already deleted must not raise."""
    import logging

    clear_cache(False)
    with caplog.at_level(logging.WARNING, logger="matgl.config"):
        clear_cache(False)
    assert any("not found" in rec.message for rec in caplog.records)


def test_clear_cache_no_when_user_says_no(monkeypatch):
    """If the user answers 'n', the cache directory must remain untouched."""
    os.makedirs(MATGL_CACHE, exist_ok=True)
    answers = iter(["n"])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(answers))
    clear_cache(confirm=True)
    assert os.path.exists(MATGL_CACHE)


def test_ensure_backend_dgl_missing_raises_runtime_error():
    """Mocked-missing DGL must raise a ``RuntimeError`` from ``ensure_backend('DGL')``."""
    with (
        patch.object(importlib.util, "find_spec", side_effect=ImportError("nope")),
        pytest.raises(RuntimeError, match="Please install DGL"),
    ):
        ensure_backend("DGL")


def test_ensure_backend_pyg_missing_raises_runtime_error():
    """Mocked-missing PyG must raise a ``RuntimeError`` from ``ensure_backend('PYG')``."""
    with (
        patch.object(importlib.util, "find_spec", side_effect=ImportError("nope")),
        pytest.raises(RuntimeError, match="Please install torch_geometric"),
    ):
        ensure_backend("PYG")


def test_set_backend():
    with pytest.raises(ValueError, match="Invalid backend"):
        matgl.set_backend("nonsense")
