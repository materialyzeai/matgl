from __future__ import annotations

import os.path

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
