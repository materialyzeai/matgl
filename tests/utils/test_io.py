from __future__ import annotations

import os
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

import matgl
from matgl.utils import io as matgl_io
from matgl.utils.io import IOMixIn, get_available_pretrained_models, load_model

this_dir = Path(os.path.abspath(os.path.dirname(__file__)))


class OldModel(torch.nn.Module, IOMixIn):
    __version__ = 1

    def __init__(self, n, **kwargs):
        super().__init__()
        self.save_args(locals(), kwargs)
        self.n = n


class NewModel(torch.nn.Module, IOMixIn):
    __version__ = 100000

    def __init__(self, n, **kwargs):
        super().__init__()
        self.save_args(locals(), kwargs)
        self.n = n


def test_model_versioning():
    model = OldModel(1, k=2)
    model.save("OldModel")
    with pytest.warns(UserWarning, match="Incompatible model version detected!"):
        model2 = NewModel.load("OldModel")
    # Model will still load since there are no incompatibilities. Check the properties are reloaded.
    assert isinstance(model2, NewModel)
    assert model2.n == 1
    assert model2._init_args["k"] == 2
    shutil.rmtree("OldModel")


@pytest.mark.skipif(os.getenv("CI") == "true", reason="Unreliable in CI environments.")
def test_get_available_pretrained_models():
    model_names = get_available_pretrained_models()
    assert len(model_names) > 1
    assert "M3GNet-MP-2021.2.8-PES" in model_names


@pytest.mark.skipif(matgl.config.BACKEND != "DGL", reason="Only works with DGL.")
def test_load_model():
    # Load model from name.

    model = load_model("M3GNet-MP-2021.2.8-DIRECT-PES")
    assert issubclass(model.__class__, torch.nn.Module)

    # Load model from a full path.
    model = load_model(this_dir / ".." / ".." / "pretrained_models" / "MEGNet-MP-2018.6.1-Eform")
    assert issubclass(model.__class__, torch.nn.Module)
    model = load_model(this_dir / ".." / ".." / "pretrained_models" / "CHGNet-MPtrj-2024.2.13-11M-PES")
    assert issubclass(model.__class__, torch.nn.Module)


def test_from_pretrained_uses_hf_hub(tmp_path):
    """from_pretrained should download via huggingface_hub and delegate to load()."""
    model = OldModel(3, k=5)
    serialized_dir = tmp_path / "serialized"
    model.save(serialized_dir)

    def fake_hf_hub_download(repo_id, filename, **kwargs):
        assert repo_id == "owner/repo"
        return str(serialized_dir / filename)

    with (
        patch.object(matgl_io, "hf_hub_download", side_effect=fake_hf_hub_download),
        pytest.warns(UserWarning, match="Incompatible model version detected!"),
    ):
        loaded = NewModel.from_pretrained("owner/repo", revision="main")

    assert isinstance(loaded, NewModel)
    assert loaded.n == 3
    assert loaded._init_args["k"] == 5


def test_load_model_falls_back_to_hf_hub(tmp_path):
    """load_model should try the HF Hub when the name looks like an owner/repo id."""
    model = OldModel(7, flag=True)
    serialized_dir = tmp_path / "serialized"
    model.save(serialized_dir)

    def fake_hf_hub_download(repo_id, filename, **kwargs):
        assert repo_id == "owner/hf-only-model"
        return str(serialized_dir / filename)

    with patch.object(matgl_io, "hf_hub_download", side_effect=fake_hf_hub_download):
        loaded = load_model("owner/hf-only-model")

    assert isinstance(loaded, torch.nn.Module)
    assert loaded._init_args["flag"] is True


def test_push_to_hub_invokes_hub_api(tmp_path):
    """push_to_hub should serialize the model and upload the folder via HfApi."""
    model = OldModel(2)
    captured = {}

    fake_commit = MagicMock(commit_url="https://huggingface.co/owner/repo/commit/abcdef")

    def fake_upload_folder(*, folder_path, repo_id, **kwargs):
        captured["folder_path"] = folder_path
        captured["repo_id"] = repo_id
        # All matgl artifacts plus a generated README must be present at upload time.
        files = set(os.listdir(folder_path))
        assert {"model.pt", "state.pt", "model.json", "README.md"}.issubset(files)
        return fake_commit

    with (
        patch.object(matgl_io, "create_repo") as fake_create,
        patch.object(matgl_io, "HfApi") as fake_api_cls,
    ):
        fake_api = MagicMock()
        fake_api.upload_folder.side_effect = fake_upload_folder
        fake_api_cls.return_value = fake_api

        url = model.push_to_hub("owner/repo", metadata={"note": "test"})

    assert url == "https://huggingface.co/owner/repo/commit/abcdef"
    assert captured["repo_id"] == "owner/repo"
    fake_create.assert_called_once()


def test_load_bad_model():
    with pytest.raises(ValueError, match=r"Bad serialized model or bad model name."):
        load_model("badbadmodelname")

    try:
        os.makedirs("bad_serialized_model")
        with open("bad_serialized_model/model.json", "w") as f:
            f.write("hello")
        with open("bad_serialized_model/model.pt", "w") as f:
            f.write("hello")
        with open("bad_serialized_model/state.pt", "w") as f:
            f.write("hello")

        with pytest.raises(ValueError, match="Bad serialized model"):
            load_model("bad_serialized_model")
    finally:
        shutil.rmtree("bad_serialized_model")
