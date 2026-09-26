from __future__ import annotations

import json

import lightning as pl
import pytest
import torch
from pymatgen.core import Lattice, Structure
from torch.utils.data import SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torch_geometric.data import Data

from matgl.ext.pymatgen import Structure2Graph
from matgl.graph.data import MGLDataLoader, MGLDataModule, MGLDiskDataset, ShardBatchSampler, write_mgl_shards
from matgl.models import QET
from matgl.utils.training import PotentialLightningModule


class _Converter:
    cutoff = 5.0

    def get_graph(self, structure):
        num_nodes = len(structure)
        edge_index = torch.tensor([[0, 1], [1, 0]]) if num_nodes > 1 else torch.empty((2, 0), dtype=torch.long)
        graph = Data(
            num_nodes=num_nodes,
            node_type=torch.zeros(num_nodes, dtype=torch.long),
            frac_coords=torch.as_tensor(structure.frac_coords, dtype=torch.float32),
            edge_index=edge_index,
            pbc_offset=torch.zeros((edge_index.shape[1], 3)),
        )
        return graph, torch.tensor(structure.lattice.matrix).unsqueeze(0), torch.tensor([0.0, 1.0])


def _structure() -> Structure:
    return Structure(Lattice.cubic(4), ["Li", "F"], [[0, 0, 0], [0.5, 0.5, 0.5]])


def _record(index: int) -> dict:
    return {
        "structure": _structure(),
        "labels": {
            "charges": torch.tensor([0.25, -0.25]),
            "energies": float(index),
            "forces": torch.full((2, 3), float(index)),
            "stresses": torch.arange(6, dtype=torch.float32),
        },
    }


def _write_split(root, name: str, count: int = 5) -> MGLDiskDataset:
    path = root / name
    write_mgl_shards(
        (_record(index) for index in range(count)),
        path,
        converter=_Converter(),  # type: ignore[arg-type]
        shard_size=2,
        include_ref_charge=True,
    )
    return MGLDiskDataset(path)


def test_disk_dataset_loader_and_qeq_charge(tmp_path):
    dataset = _write_split(tmp_path, "train", count=10)
    manifest = json.loads((tmp_path / "train" / "metadata.json").read_text())

    assert [entry["n"] for entry in manifest["shards"]] == [2, 2, 2, 2, 2]
    assert manifest["label_schema"]["forces"] == {"layout": "node", "shape": [3]}
    assert manifest["label_schema"]["stresses"] == {"layout": "graph", "shape": [6]}
    assert len(dataset) == 10
    assert dataset[-1][3]["energies"].item() == 9
    assert dataset[0][0].q_ref.shape == (2,)

    train_loader, val_loader = MGLDataLoader(
        dataset,
        dataset,
        batch_size=2,
        num_workers=2,
        shard_seed=17,
    )
    graph, _, _, energy, forces, stress, charges = next(iter(train_loader))
    assert graph.q_ref.shape == (4,)
    assert energy.shape == (2,)
    assert forces.shape == (4, 3)
    assert stress.shape == (2, 6)
    assert charges.shape == (4,)
    assert len(list(val_loader)) == 5


def test_shard_sampler_shuffle_and_distributed_partition(tmp_path):
    dataset = _write_split(tmp_path, "train", count=10)
    sampler = ShardBatchSampler(dataset, 3, shuffle=True, seed=9)

    epoch_zero = list(sampler)
    epoch_one = list(sampler)
    assert epoch_zero != epoch_one
    assert sorted(index for batch in epoch_zero for index in batch) == list(range(10))
    assert all(len({dataset._location(index)[0] for index in batch}) == 1 for batch in epoch_zero)

    rank_batches = [
        list(ShardBatchSampler(DistributedSampler(dataset, num_replicas=2, rank=rank), 3)) for rank in range(2)
    ]
    assert len(rank_batches[0]) == len(rank_batches[1])
    rank_shards = [{dataset._location(index)[0] for batch in batches for index in batch} for batches in rank_batches]
    assert rank_shards[0].isdisjoint(rank_shards[1])


def test_interrupted_rebuild_keeps_manifest_and_removes_orphans(tmp_path):
    dataset_root = tmp_path / "train"
    write_mgl_shards((_record(index) for index in range(4)), dataset_root, converter=_Converter(), shard_size=2)
    committed_manifest = (dataset_root / "metadata.json").read_text()
    committed_files = {path.name for path in dataset_root.iterdir()}

    invalid = [_record(index) for index in range(3)]
    invalid[-1]["labels"].pop("stresses")
    with pytest.raises(ValueError, match="label keys"):
        write_mgl_shards(invalid, dataset_root, converter=_Converter(), shard_size=2)

    assert (dataset_root / "metadata.json").read_text() == committed_manifest
    assert {path.name for path in dataset_root.iterdir()} == committed_files
    assert len(MGLDiskDataset(dataset_root)) == 4


def test_successful_rebuild_replaces_old_shards(tmp_path):
    dataset_root = tmp_path / "train"
    write_mgl_shards((_record(index) for index in range(4)), dataset_root, converter=_Converter(), shard_size=2)
    old_shards = set(dataset_root.glob("shard_*.pt"))

    write_mgl_shards((_record(index) for index in range(3)), dataset_root, converter=_Converter(), shard_size=3)

    assert not old_shards.intersection(dataset_root.glob("shard_*.pt"))
    assert len(MGLDiskDataset(dataset_root)) == 3


def test_on_the_fly_dataset(tmp_path):
    dataset_root = tmp_path / "train"
    write_mgl_shards((_record(index) for index in range(3)), dataset_root, precomputed=False, shard_size=2)

    with pytest.raises(ValueError, match="converter is required"):
        MGLDiskDataset(dataset_root)

    dataset = MGLDiskDataset(dataset_root, converter=_Converter())
    graph, lattice, state_attr, labels = dataset[1]
    assert graph.num_nodes == 2
    assert lattice.shape == (1, 3, 3)
    assert state_attr.shape == (2,)
    assert labels["energies"].item() == 1


def test_data_module_all_stages_and_auto_collate(tmp_path):
    for split in ("train", "valid", "test"):
        _write_split(tmp_path, split)

    data_module = MGLDataModule(tmp_path, batch_size=2, num_workers=0, seed=7)
    data_module.setup()

    assert len(data_module.train_dataloader()) == 3
    assert len(data_module.val_dataloader()) == 3
    assert len(data_module.test_dataloader()) == 3
    assert len(data_module.predict_dataloader()) == 3
    assert data_module.predict_dataset.root == tmp_path / "test"


def test_qet_training_with_disk_data_module(tmp_path):
    """Run a real QET optimization step through shards, sampler, and DataModule."""
    torch.manual_seed(0)
    converter = Structure2Graph(element_types=("Li", "F"), cutoff=5.0)
    for split in ("train", "valid"):
        write_mgl_shards(
            (_record(index) for index in range(2)),
            tmp_path / split,
            converter=converter,
            shard_size=2,
            include_ref_charge=True,
        )

    data_module = MGLDataModule(tmp_path, batch_size=2, num_workers=0, seed=7)
    data_module.setup("fit")
    assert isinstance(data_module.train_dataset, MGLDiskDataset)
    assert isinstance(data_module.train_dataloader().batch_sampler, ShardBatchSampler)
    model = QET(
        element_types=("Li", "F"),
        units=8,
        nblocks=1,
        num_rbf=8,
        cutoff=5.0,
        use_warp=False,
    )
    module = PotentialLightningModule(
        model=model,
        stress_weight=0.0,
        charge_weight=0.1,
        loss="mse_loss",
    )
    parameters_before = [parameter.detach().clone() for parameter in model.parameters() if parameter.requires_grad]
    trainer = pl.Trainer(
        fast_dev_run=True,
        accelerator="cpu",
        inference_mode=False,
        logger=False,
        enable_checkpointing=False,
        default_root_dir=tmp_path,
    )

    trainer.fit(module, datamodule=data_module)

    assert trainer.global_step == 1
    assert module._last_preds is not None
    assert torch.isfinite(module._last_preds[0]).all()
    assert torch.isfinite(module._last_preds[3]).all()
    assert torch.allclose(module._last_preds[3].reshape(2, 2).sum(dim=1), torch.zeros(2), atol=1e-6)
    assert any(
        not torch.equal(before, after)
        for before, after in zip(parameters_before, (p for p in model.parameters() if p.requires_grad), strict=True)
    )


def test_writer_rejects_invalid_records_and_layouts(tmp_path):
    with pytest.raises(ValueError, match="shard_size"):
        write_mgl_shards([], tmp_path / "size", converter=_Converter(), shard_size=0)
    with pytest.raises(ValueError, match="converter is required"):
        write_mgl_shards([_record(0)], tmp_path / "converter")
    with pytest.raises(ValueError, match="empty"):
        write_mgl_shards([], tmp_path / "empty", converter=_Converter())
    with pytest.raises(ValueError, match="invalid label layouts"):
        write_mgl_shards(
            [_record(0)],
            tmp_path / "layout",
            converter=_Converter(),
            label_layouts={"forces": "invalid"},  # type: ignore[dict-item]
        )

    wrong_atoms = _record(0)
    wrong_atoms["labels"]["forces"] = torch.zeros((3, 3))
    with pytest.raises(ValueError, match="leading dimension"):
        write_mgl_shards([wrong_atoms], tmp_path / "shape", converter=_Converter())

    missing_charge = _record(0)
    missing_charge["labels"].pop("charges")
    with pytest.raises(ValueError, match="requires 'charges'"):
        write_mgl_shards(
            [missing_charge],
            tmp_path / "charge",
            converter=_Converter(),
            include_ref_charge=True,
        )


def test_dataset_and_sampler_reject_invalid_configuration(tmp_path):
    dataset = _write_split(tmp_path, "train", count=4)

    with pytest.raises(IndexError, match="out of range"):
        _ = dataset[4]
    with pytest.raises(ValueError, match="batch_size"):
        ShardBatchSampler(dataset, 0)
    with pytest.raises(ValueError, match="rank and world_size"):
        ShardBatchSampler(dataset, 2, rank=0)
    with pytest.raises(ValueError, match="exceeds num_shards"):
        list(ShardBatchSampler(dataset, 2, rank=0, world_size=3))
    with pytest.raises(ValueError, match="no batches"):
        list(ShardBatchSampler(dataset, 10, drop_last=True))


def test_loader_rejects_mixed_storage_types(tmp_path):
    dataset = _write_split(tmp_path, "train")

    with pytest.raises(TypeError, match="must all use MGLDiskDataset"):
        MGLDataLoader(dataset, [])  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="same storage type"):
        MGLDataLoader([], dataset)  # type: ignore[arg-type]


def test_disk_loader_supports_test_data_and_rejects_sampler_controls(tmp_path):
    dataset = _write_split(tmp_path, "train")

    loaders = MGLDataLoader(dataset, dataset, test_data=dataset, batch_size=2, num_workers=0)
    assert len(loaders) == 3
    assert all(len(loader) == 3 for loader in loaders)

    with pytest.raises(ValueError, match="disk-backed loading controls"):
        MGLDataLoader(dataset, dataset, batch_size=2, num_workers=0, shuffle=False)


def test_writer_accepts_serialized_structures_and_graph_labels(tmp_path):
    record = _record(0)
    record["structure"] = record["structure"].as_dict()
    record["graph_label"] = 3
    record["labels"]["source"] = "calculation-1"
    record["labels"]["custom"] = torch.tensor([1.0, 2.0])

    write_mgl_shards(
        [record],
        tmp_path / "serialized",
        converter=_Converter(),
        label_layouts={"custom": "unchecked"},
    )
    dataset = MGLDiskDataset(tmp_path / "serialized")

    assert dataset[0][2].item() == 3
    assert "source" not in dataset[0][3]

    write_mgl_shards([record], tmp_path / "on-the-fly", precomputed=False)
    assert len(MGLDiskDataset(tmp_path / "on-the-fly", converter=_Converter())) == 1


def test_writer_rejects_malformed_records_and_label_shapes(tmp_path):
    with pytest.raises(ValueError, match="expected 'structure' and 'labels'"):
        write_mgl_shards([{"structure": _structure()}], tmp_path / "missing", converter=_Converter())
    with pytest.raises(TypeError, match=r"labels.*mapping"):
        write_mgl_shards(
            [{"structure": _structure(), "labels": []}],  # type: ignore[dict-item]
            tmp_path / "labels",
            converter=_Converter(),
        )

    scalar_node = _record(0)
    scalar_node["labels"]["forces"] = 1.0
    with pytest.raises(ValueError, match="leading atom dimension"):
        write_mgl_shards([scalar_node], tmp_path / "scalar", converter=_Converter())

    mismatched = [_record(0), _record(1)]
    mismatched[1]["labels"]["stresses"] = torch.zeros(3)
    with pytest.raises(ValueError, match="shape signature"):
        write_mgl_shards(mismatched, tmp_path / "shape-signature", converter=_Converter())

    invalid_structure = {"structure": object(), "labels": _record(0)["labels"]}
    with pytest.raises(TypeError, match="structure must be a mapping"):
        write_mgl_shards([invalid_structure], tmp_path / "structure", precomputed=False)


def test_writer_recovers_from_invalid_old_manifest(tmp_path):
    root = tmp_path / "train"
    root.mkdir()
    (root / "metadata.json").write_text("not json")

    write_mgl_shards([_record(0)], root, converter=_Converter())

    assert len(MGLDiskDataset(root)) == 1


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("format_version", -1, "unsupported dataset format"),
        ("backend", "dgl", "expected a PyG dataset"),
        ("n", 2, "manifest contains"),
    ],
)
def test_disk_dataset_rejects_invalid_manifest(tmp_path, field, value, message):
    root = tmp_path / field
    _write_split(tmp_path, field, count=1)
    manifest_path = root / "metadata.json"
    manifest = json.loads(manifest_path.read_text())
    manifest[field] = value
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match=message):
        MGLDiskDataset(root)


def test_disk_dataset_rejects_missing_or_truncated_shards(tmp_path):
    missing_root = tmp_path / "missing"
    missing = _write_split(tmp_path, "missing", count=1)
    (missing_root / missing._shards[0]["file"]).unlink()
    with pytest.raises(FileNotFoundError):
        MGLDiskDataset(missing_root)

    truncated_root = tmp_path / "truncated"
    truncated = _write_split(tmp_path, "truncated", count=2)
    shard_path = truncated_root / truncated._shards[0]["file"]
    torch.save(torch.load(shard_path, weights_only=False)[:1], shard_path)
    with pytest.raises(ValueError, match="contains 1 items; expected 2"):
        MGLDiskDataset(truncated_root)[0]


def test_disk_dataset_worker_state_discards_cached_shard(tmp_path):
    dataset = _write_split(tmp_path, "train", count=1)
    _ = dataset[0]

    state = dataset.__getstate__()

    assert state["_shard_id"] is None
    assert state["_shard"] is None


def test_sampler_validation_epoch_and_distributed_context(tmp_path, monkeypatch):
    dataset = _write_split(tmp_path, "train", count=5)
    with pytest.raises(TypeError, match="MGLDiskDataset"):
        ShardBatchSampler(SequentialSampler([]), 2)
    with pytest.raises(ValueError, match="invalid distributed context"):
        len(ShardBatchSampler(dataset, 2, rank=1, world_size=1))

    distributed = DistributedSampler(dataset, num_replicas=2, rank=0)
    sampler = ShardBatchSampler(distributed, 2)
    sampler.set_epoch(4)
    assert distributed.epoch == 4

    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 1)
    assert ShardBatchSampler(dataset, 2)._distributed_context() == (0, 1)

    batches = list(ShardBatchSampler(dataset, 2, drop_last=True))
    assert all(len(batch) == 2 for batch in batches)


def test_data_module_validate_and_custom_predict_split(tmp_path):
    _write_split(tmp_path, "valid", count=1)
    _write_split(tmp_path, "inference", count=1)
    data_module = MGLDataModule(tmp_path, batch_size=1, num_workers=0, predict_split="inference")

    data_module.setup("validate")
    assert len(data_module.val_dataset) == 1
    data_module.setup("predict")
    assert data_module.predict_dataset.root == tmp_path / "inference"


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"batch_size": 0}, "batch_size"),
        ({"num_workers": -1}, "num_workers"),
        ({"num_workers": 0, "persistent_workers": True}, "persistent_workers"),
        ({"shuffle": True}, "cannot override"),
    ],
)
def test_data_module_rejects_invalid_loader_configuration(tmp_path, kwargs, message):
    with pytest.raises(ValueError, match=message):
        MGLDataModule(tmp_path, **kwargs)
