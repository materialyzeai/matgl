"""This is an integration test file that checks on pre-trained models to ensure they still work."""

from __future__ import annotations

import os

import pytest

import matgl


@pytest.mark.skipif(os.getenv("CI") == "true" or matgl.config.BACKEND != "DGL", reason="Unreliable in CI environments.")
def test_loading_all_models():
    """
    Test that all pre-trained models at least load.
    """
    for m in matgl.get_available_pretrained_models():
        assert matgl.load_model(m) is not None
