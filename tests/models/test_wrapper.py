from __future__ import annotations

import pytest
import torch.nn

from matgl.data.transformer import LogTransformer, Normalizer
from matgl.models import TensorNet
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
    assert float(model.forward()) == pytest.approx(3)


def test_predict_structure(LiFePO4):
    model = TransformedTargetModel(DummyModel(), Normalizer(1, 2))
    assert float(model.predict_structure(LiFePO4)) == pytest.approx(3)


def test_repr():
    model = TransformedTargetModel(DummyModel(), Normalizer(1, 2))
    assert repr(model) == "TransformedTargetModel:\n\tModel: DummyModel()\n\tTransformer: Normalizer(mean=1, std=2)"


@pytest.mark.parametrize(
    "transformer",
    [Normalizer(torch.tensor(1.5), torch.tensor(2.0)), Normalizer(-1.0, 0.5), LogTransformer()],
    ids=["normalizer-tensor", "normalizer-float", "log"],
)
def test_save_load_roundtrip(tmp_path, graph_MoS, transformer):
    """``save`` must produce files that ``load`` (restricted unpickler) accepts, with identical predictions."""
    structure, graph, _ = graph_MoS
    lat = torch.tensor(structure.lattice.matrix, dtype=torch.float32)
    graph.pbc_offshift = graph.pbc_offset @ lat
    graph.pos = graph.frac_coords @ lat
    torch.manual_seed(0)
    model = TransformedTargetModel(TensorNet(element_types=("Mo", "S"), use_warp=False), transformer)
    model.save(tmp_path)
    loaded = TransformedTargetModel.load(tmp_path)
    assert type(loaded.transformer) is type(transformer)
    assert torch.allclose(loaded(g=graph), model(g=graph))


def test_property_training_normalization_export(MoS, tmp_path):
    """Export the loss-space predictions with their normalization rather than saving a raw model (#862)."""
    from matgl.utils.training import ModelLightningModule

    model = TensorNet(element_types=("Mo", "S"), use_warp=False)
    lit = ModelLightningModule(model, data_mean=-5.0, data_std=2.5)
    raw = model.predict_structure(MoS)
    target = 2.5 * raw - 5.0
    losses = lit.loss_fn(torch.nn.functional.mse_loss, target.reshape(1), raw.reshape(1))
    assert losses["Total_Loss"].item() == pytest.approx(0.0)
    exported = TransformedTargetModel(lit.model, Normalizer(lit.data_mean, lit.data_std))
    exported.save(tmp_path)
    loaded = TransformedTargetModel.load(tmp_path)
    torch.testing.assert_close(loaded.predict_structure(MoS), target)
