from __future__ import annotations

import pytest
import torch.nn

from matgl.data.transformer import Normalizer
from matgl.models._wrappers import TransformedTargetModel


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self):
        return torch.ones(1)

    def predict_structure(self, structure):
        return torch.ones(1)


def test_forward():
    model = TransformedTargetModel(DummyModel(), Normalizer(1, 2))
    assert float(model.forward()), pytest.approx(3)


def test_predict_structure(LiFePO4):
    model = TransformedTargetModel(DummyModel(), Normalizer(1, 2))
    assert float(model.predict_structure(LiFePO4)), pytest.approx(3)


def test_repr():
    model = TransformedTargetModel(DummyModel(), Normalizer(1, 2))
    assert repr(model) == "TransformedTargetModel:\n\tModel: DummyModel()\n\tTransformer: Normalizer(mean=1, std=2)"
