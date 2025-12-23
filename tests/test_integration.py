"""This is an integration test file that checks on pre-trained models to ensure they still work."""

from __future__ import annotations

import os

import pytest

import matgl


@pytest.mark.skipif(matgl.config.BACKEND != "DGL", reason="Only works with DGL.")
def test_form_e(LiFePO4):
    model = matgl.load_model("M3GNet-MP-2018.6.1-Eform")
    for _i in range(3):
        # This loop ensures there is no stochasticity in the prediction, a problem in v1 of the models.
        assert model.predict_structure(LiFePO4) == pytest.approx(-2.5489, 3)


@pytest.mark.skipif(matgl.config.BACKEND != "DGL", reason="Only works with DGL.")
def test_chgnet_loading():
    matgl.load_model("CHGNet-MatPES-PBE-2025.2.10-2.7M-PES")


@pytest.mark.skipif(os.getenv("CI") == "true" or matgl.config.BACKEND != "DGL", reason="Unreliable in CI environments.")
def test_loading_all_models():
    """
    Test that all pre-trained models at least load.
    """
    for m in matgl.get_available_pretrained_models():
        assert matgl.load_model(m) is not None
