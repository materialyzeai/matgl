from __future__ import annotations

import os.path

import pytest

import matgl
from matgl.config import MATGL_CACHE, clear_cache


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


def test_set_backend_dgl_warns():
    """`set_backend("DGL")` is a no-op stub that warns the DGL backend is gone."""
    with pytest.warns(DeprecationWarning, match="DGL backend no longer exists"):
        matgl.set_backend("DGL")
    # Case-insensitive: the legacy env-var convention accepted any case.
    with pytest.warns(DeprecationWarning, match="DGL backend no longer exists"):
        matgl.set_backend("dgl")


def test_set_backend_pyg_silent(recwarn):
    """`set_backend("PYG")` and the no-arg default are silent no-ops."""
    matgl.set_backend("PYG")
    matgl.set_backend()
    assert not recwarn.list
