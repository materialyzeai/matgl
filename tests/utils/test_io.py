from __future__ import annotations

import os
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

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


class ModelWithParams(torch.nn.Module, IOMixIn):
    """Minimal IOMixIn model with a known parameter count for ``num_parameters`` tests."""

    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(5, 3)
        self.save_args(locals())


def test_iomixin_num_parameters():
    model = ModelWithParams()
    # Linear(5, 3): weight 5 * 3 + bias 3
    assert model.num_parameters == 5 * 3 + 3


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
    # All names should be bare model names (no "owner/" prefix).
    assert all("/" not in name for name in model_names)


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


def test_load_model_resolves_bare_name_via_materialyze_hf_org(tmp_path):
    """A bare name should be resolved via the materialyze HF org."""
    model = OldModel(11, source="materialyze")
    serialized_dir = tmp_path / "serialized"
    model.save(serialized_dir)

    calls: list[str] = []

    def fake_hf_hub_download(repo_id, filename, **kwargs):
        calls.append(repo_id)
        assert repo_id == f"{matgl_io.HF_MATGL_ORG}/BareModelName"
        return str(serialized_dir / filename)

    with patch.object(matgl_io, "hf_hub_download", side_effect=fake_hf_hub_download):
        loaded = load_model("BareModelName")

    assert isinstance(loaded, torch.nn.Module)
    assert loaded._init_args["source"] == "materialyze"
    assert calls == [f"{matgl_io.HF_MATGL_ORG}/BareModelName"] * 3


def test_load_model_raises_when_bare_name_missing_on_hf(tmp_path):
    """If a bare name is not on the materialyze HF org, loading should fail with a ValueError."""

    def fake_hf_hub_download(repo_id, filename, **kwargs):
        raise RuntimeError(f"missing {repo_id}")

    with (
        patch.object(matgl_io, "hf_hub_download", side_effect=fake_hf_hub_download),
        pytest.raises(ValueError, match=r"Bad serialized model or bad model name\."),
    ):
        load_model("LegacyOnlyModel")


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
        fake_api.repo_exists.return_value = False
        fake_api.upload_folder.side_effect = fake_upload_folder
        fake_api_cls.return_value = fake_api

        url = model.push_to_hub("owner/repo", metadata={"note": "test"})

    assert url == "https://huggingface.co/owner/repo/commit/abcdef"
    assert captured["repo_id"] == "owner/repo"
    fake_create.assert_called_once()


def test_push_to_hub_retains_existing_readme(tmp_path):
    """When the repo already exists, push_to_hub must not generate a README."""
    model = OldModel(2)

    def fake_upload_folder(*, folder_path, repo_id, **kwargs):
        files = set(os.listdir(folder_path))
        assert {"model.pt", "state.pt", "model.json"}.issubset(files)
        assert "README.md" not in files
        return MagicMock(commit_url="https://huggingface.co/owner/repo/commit/deadbeef")

    with (
        patch.object(matgl_io, "create_repo"),
        patch.object(matgl_io, "HfApi") as fake_api_cls,
    ):
        fake_api = MagicMock()
        fake_api.repo_exists.return_value = True
        fake_api.upload_folder.side_effect = fake_upload_folder
        fake_api_cls.return_value = fake_api

        model.push_to_hub("owner/repo", metadata={"note": "test"})


def test_push_to_hub_uploads_explicit_readme_for_existing_repo(tmp_path):
    """An explicit ``readme_text`` must be uploaded even when the repo already exists."""
    model = OldModel(2)

    def fake_upload_folder(*, folder_path, repo_id, **kwargs):
        files = set(os.listdir(folder_path))
        assert "README.md" in files
        contents = (Path(folder_path) / "README.md").read_text()
        assert "Custom README content" in contents
        return MagicMock(commit_url="https://huggingface.co/owner/repo/commit/cafebabe")

    with (
        patch.object(matgl_io, "create_repo"),
        patch.object(matgl_io, "HfApi") as fake_api_cls,
    ):
        fake_api = MagicMock()
        fake_api.repo_exists.return_value = True
        fake_api.upload_folder.side_effect = fake_upload_folder
        fake_api_cls.return_value = fake_api

        model.push_to_hub("owner/repo", readme_text="Custom README content")


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


def test_load_model_wraps_unknown_error_as_runtime_error(tmp_path):
    """Non-(ImportError, ValueError) exceptions should be re-raised as ``RuntimeError``."""
    serialized_dir = tmp_path / "weird_model"
    serialized_dir.mkdir()
    for fn in ("model.pt", "state.pt", "model.json"):
        (serialized_dir / fn).write_text("doesn't matter")

    def boom(*_args, **_kwargs):
        raise KeyboardInterrupt("boom")

    with (
        patch.object(matgl_io, "_get_file_paths", side_effect=boom),
        pytest.raises(RuntimeError, match="Unknown error occurred while loading model"),
    ):
        load_model(serialized_dir)


def test_get_file_paths_malformed_identifier_raises_value_error(tmp_path):
    """An identifier that contains ``/`` but doesn't match a valid HF repo id must
    fail with a clear ``ValueError`` rather than attempting a network call."""
    from matgl.utils.io import _get_file_paths

    # Doesn't exist locally and doesn't match ``owner/name`` (leading slash, double slash).
    bogus = "//bad//repo//id"
    with pytest.raises(ValueError, match=r"No valid model found locally or at Hugging Face Hub"):
        _get_file_paths(tmp_path / bogus, str_path=bogus)


def test_get_file_paths_bare_name_hub_failure_raises_value_error(tmp_path):
    """When a bare name fails to download from the materialyze HF org, raise ``ValueError``."""
    from matgl.utils.io import _get_file_paths

    def boom(*_args, **_kwargs):
        raise RuntimeError("simulated hub failure")

    with (
        patch.object(matgl_io, "_download_from_hf_hub", side_effect=boom),
        pytest.raises(ValueError, match=r"No valid model found locally or at Hugging Face repo"),
    ):
        _get_file_paths(tmp_path / "BareName", str_path="BareName")


def test_resolve_module_aliases_private_modules_to_public_packages():
    """Private ``matgl.models._*`` and ``matgl.apps._pes_*`` modules must alias to
    ``matgl.models`` / ``matgl.apps.pes`` so backend-aware loaders pick the right
    implementation transparently."""
    from matgl.utils.io import _resolve_module

    assert _resolve_module("matgl.models._m3gnet_dgl") == "matgl.models"
    assert _resolve_module("matgl.models._m3gnet") == "matgl.models"
    assert _resolve_module("matgl.models._megnet_dgl") == "matgl.models"
    assert _resolve_module("matgl.models._tensornet") == "matgl.models"
    assert _resolve_module("matgl.models._chgnet_dgl") == "matgl.models"
    assert _resolve_module("matgl.apps._pes_dgl") == "matgl.apps.pes"
    assert _resolve_module("matgl.apps._pes") == "matgl.apps.pes"
    # Public modules pass through unchanged.
    assert _resolve_module("matgl.models") == "matgl.models"
    assert _resolve_module("matgl.apps.pes") == "matgl.apps.pes"
    # Out-of-scope modules pass through unchanged.
    assert _resolve_module("some.other.pkg._internal") == "some.other.pkg._internal"


def test_generate_hf_model_card_with_unserializable_metadata():
    """``_generate_hf_model_card`` must swallow ``TypeError`` from non-serializable metadata.

    Forces the ``json.dumps`` fallback to fail even with ``default=str`` by using an
    object whose ``__repr__`` raises (and therefore so does ``str(obj)``).
    """
    from matgl.utils.io import _generate_hf_model_card

    class Unserializable:
        def __repr__(self):
            raise TypeError("repr exploded")

    model = OldModel(1)
    card = _generate_hf_model_card(model, metadata={"oops": Unserializable()})

    assert "## Metadata" not in card
    assert "OldModel" in card


def test_get_available_pretrained_models_handles_hub_errors():
    """If the HF hub call fails, ``get_available_pretrained_models`` returns an empty list."""

    class _BoomApi:
        def list_models(self, **_kwargs):
            raise RuntimeError("network is down")

    with patch.object(matgl_io, "HfApi", return_value=_BoomApi()):
        names = get_available_pretrained_models()

    assert names == []


def test_get_available_pretrained_models_strips_owner_prefix():
    """Returned names should be bare (no ``"owner/"`` prefix) and sorted."""

    class _FakeModelInfo:
        def __init__(self, repo_id: str):
            self.id = repo_id

    class _FakeApi:
        def list_models(self, **_kwargs):
            return [
                _FakeModelInfo("materialyze/Zeta"),
                _FakeModelInfo("materialyze/Alpha"),
                _FakeModelInfo("no-slash-id"),  # malformed entries are silently skipped
            ]

    with patch.object(matgl_io, "HfApi", return_value=_FakeApi()):
        names = get_available_pretrained_models()

    assert names == ["Alpha", "Zeta"]
