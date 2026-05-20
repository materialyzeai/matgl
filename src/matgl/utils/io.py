"""Provides utilities for managing models and data."""

from __future__ import annotations

import difflib
import inspect
import json
import logging
import os
import re
import tempfile
import warnings
from pathlib import Path
from typing import cast

import torch
from huggingface_hub import HfApi, create_repo, hf_hub_download

from matgl.config import MATGL_CACHE

logger = logging.getLogger(__name__)

# Files that comprise a serialized matgl model on disk and on Hugging Face Hub.
_MODEL_FILES = ("model.pt", "state.pt", "model.json")


def _resolve_module(modname: str) -> str:
    """Map a private ``matgl.models._*`` / ``matgl.apps._pes*`` ``@module`` string to its public package.

    Resolves to ``matgl.models`` or ``matgl.apps.pes`` as appropriate.
    """
    if modname.startswith("matgl.models._"):
        return "matgl.models"
    if modname.startswith("matgl.apps._pes"):
        return "matgl.apps.pes"
    return modname


# Loose validation pattern for a Hugging Face repo_id ("owner/name" or "owner/name/subfolder"-like).
_HF_REPO_ID_RE = re.compile(r"^[A-Za-z0-9][\w\-.]*/[\w\-.]+$")

# Default Hugging Face organization searched when a bare model name is supplied to
# ``load_model`` (i.e. no "owner/" prefix). Mirrors the official matgl model zoo.
HF_MATGL_ORG = "materialyze"


class IOMixIn:
    """Mixin class for handling input/output operations for model saving and loading.

    This class provides methods to save initialization arguments and model states,
    serialize the model into files, and load a model from specified paths. It helps
    manage the lifecycle of model-related data, including initialization parameters,
    state dictionaries, and metadata, in a structured and reusable manner.

    The functionality includes:
    - Saving initialization arguments for reproducibility.
    - Serializable save of model state and initialization arguments to disk.
    - Support for saving and populating additional metadata.
    - Loading models and their states from specified paths or pre-trained model sources.

    Usage of this mixin is intended for models requiring serialization of their
    initialization arguments and runtime states. It assumes the model class
    implements and supports PyTorch's state_dict and load_state_dict methods.
    """

    def save_args(self, locals: dict, kwargs: dict | None = None) -> None:
        """Save the arguments passed to the class initializer.

        Collects the arguments from the `__init__` method of the class, excluding `self` and
        `__class__`. If an additional `kwargs` dictionary is provided, it is merged into the
        collected arguments. If any of the collected arguments are instances of a subclass of
        `IOMixIn`, those arguments are serialized into a dictionary representation that includes
        class metadata and initialization arguments. Finally, the arguments are stored as an
        instance variable `_init_args`.

        Args:
            locals (dict): A dictionary containing the local variables passed to the class
                initializer.
            kwargs (dict | None): An optional dictionary containing additional keyword
                arguments to include in the initialization arguments.

        Returns:
            None
        """
        args = inspect.getfullargspec(self.__class__.__init__).args
        d = {k: v for k, v in locals.items() if k in args and k not in ("self", "__class__")}
        if kwargs is not None:
            d.update(kwargs)

        # If one of the args is a subclass of IOMixIn, we will serialize that class.
        for k, v in d.items():
            if issubclass(v.__class__, IOMixIn):
                d[k] = {
                    "@class": v.__class__.__name__,
                    "@module": v.__class__.__module__,
                    "@model_version": getattr(v, "__version__", 0),
                    "init_args": v._init_args,
                }
        self._init_args = d

    def save(self, path: str | Path = ".", metadata: dict | None = None, makedirs: bool = True):
        """Saves the state and configuration of the model to the specified path.

        This method saves the model's initialization arguments, model weights,
        and additional metadata into the specified directory. It also creates
        necessary directories if they don't exist when `makedirs` is True.

        Args:
            path (str | Path, optional): Target path or directory where the model
                data will be saved. Defaults to ".".
            metadata (dict | None, optional): Additional metadata to save along
                with the model. Defaults to None.
            makedirs (bool, optional): Whether to create the necessary directories
                if they do not exist. Defaults to True.

        Raises:
            OSError: Raised if the directory creation fails when `makedirs` is
                set to True.
        """
        path = Path(path)
        if makedirs:
            os.makedirs(path, exist_ok=True)

        torch.save(self._init_args, path / "model.pt")  # type: ignore
        torch.save(self.state_dict(), path / "state.pt")  # type: ignore
        d = {
            "@class": self.__class__.__name__,
            "@module": self.__class__.__module__,
            "@model_version": getattr(self, "__version__", 0),
            "metadata": metadata,
            "kwargs": self._init_args,
        }  # type: ignore
        with open(path / "model.json", "w") as f:
            json.dump(d, f, default=str, indent=4)

    @classmethod
    def load(cls, path: str | Path | dict, **kwargs):
        """Load the model weights from a directory, pre-trained name, or Hugging Face repo.

        Args:
            path (str|path|dict): Path to saved model, name of a pre-trained model, or
                Hugging Face Hub repo id in ``"owner/name"`` form. If it is a dict, it is
                assumed to be of the form::

                    {
                        "model.pt": path to model.pt file,
                        "state.pt": path to state file,
                        "model.json": path to model.json file
                    }

                Otherwise the identifier is resolved via :func:`_get_file_paths`, whose
                search order is:

                1. Local filesystem path.
                2. If the identifier already matches an ``"owner/name"`` pattern, the
                   Hugging Face Hub directly.
                3. If the identifier is a bare model name, the official ``materialyze``
                   Hugging Face org (``materialyze/<name>``).
            **kwargs: Additional kwargs forwarded to ``huggingface_hub.hf_hub_download``
                (e.g. ``force_download``, ``revision``, ``token``, ``cache_dir``).

        Returns: model_object.
        """
        fpaths = path if isinstance(path, dict) else _get_file_paths(Path(path), **kwargs)

        with open(fpaths["model.json"]) as f:
            model_data = json.load(f)

        _check_ver(cls, model_data)

        map_location = torch.device("cpu") if not torch.cuda.is_available() else None
        state = torch.load(fpaths["state.pt"], map_location=map_location, weights_only=False)
        d = torch.load(fpaths["model.pt"], map_location=map_location, weights_only=False)

        # Deserialize any args that are IOMixIn subclasses.
        for k, v in d.items():
            if isinstance(v, dict) and "@class" in v and "@module" in v:
                modname = _resolve_module(v["@module"])
                classname = v["@class"]
                mod = __import__(modname, globals(), locals(), [classname], 0)
                cls_ = getattr(mod, classname)
                _check_ver(cls_, v)  # Check version of any subclasses too.
                d[k] = cls_(**v["init_args"])
        d = {k: v for k, v in d.items() if not k.startswith("@")}
        model = cls(**d)
        model.load_state_dict(state, strict=False)  # type: ignore
        return model

    def push_to_hub(
        self,
        repo_id: str,
        *,
        commit_message: str = "Upload matgl model",
        private: bool = False,
        token: str | None = None,
        metadata: dict | None = None,
        revision: str | None = None,
        create_pr: bool = False,
        repo_type: str = "model",
        readme_text: str | None = None,
    ) -> str:
        """Serialize the model and upload it to the Hugging Face Hub.

        The model is saved using the standard matgl serialization (``model.pt``, ``state.pt``
        and ``model.json``) to a temporary directory, then uploaded to the specified
        repository. A ``README.md`` model card is generated only when the repository is
        being created for the first time; if the repository already exists on the Hub, its
        existing ``README.md`` is left untouched (unless ``readme_text`` is explicitly
        provided, in which case the supplied content is uploaded).

        Args:
            repo_id: Target repository in ``"owner/name"`` form. The repo will be created
                if it does not already exist.
            commit_message: Commit message to use for the upload.
            private: Whether to create the repository as private when it does not yet
                exist. Ignored for existing repos.
            token: Optional Hugging Face authentication token. Falls back to the token
                cached by ``huggingface_hub`` (e.g. via ``huggingface-cli login``).
            metadata: Optional metadata dict forwarded to :meth:`IOMixIn.save`.
            revision: Optional branch or revision to upload to.
            create_pr: If True, a pull request will be opened instead of committing to
                the branch directly.
            repo_type: Repo type, typically ``"model"``.
            readme_text: Optional README.md text to upload. When provided, the supplied
                content is always uploaded (overwriting any existing README on the Hub).
                When omitted, a default README is generated only for newly-created repos.

        Returns:
            The URL of the resulting commit on the Hugging Face Hub.
        """
        api = HfApi(token=token)
        repo_existed = api.repo_exists(repo_id=repo_id, repo_type=repo_type, token=token)
        create_repo(repo_id, private=private, token=token, repo_type=repo_type, exist_ok=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            self.save(tmp_path, metadata=metadata, makedirs=True)  # type: ignore[attr-defined]

            readme = tmp_path / "README.md"
            # Only write a README when (a) the user explicitly supplied one, or (b) the
            # repo is being created fresh. For pre-existing repos with no explicit
            # readme_text, the existing README on the Hub is retained.
            if not readme.exists() and (readme_text is not None or not repo_existed):
                readme.write_text(_generate_hf_model_card(self, metadata=metadata, readme_text=readme_text))

            commit = api.upload_folder(
                folder_path=str(tmp_path),
                repo_id=repo_id,
                repo_type=repo_type,
                commit_message=commit_message,
                revision=revision,
                create_pr=create_pr,
                token=token,
            )

        return commit.commit_url if hasattr(commit, "commit_url") else str(commit)

    @classmethod
    def from_pretrained(
        cls,
        repo_id: str,
        *,
        revision: str | None = None,
        token: str | None = None,
        cache_dir: str | Path | None = None,
        force_download: bool = False,
        **kwargs,
    ):
        """Load a matgl model from a Hugging Face Hub repository.

        This is a thin convenience wrapper around :meth:`IOMixIn.load` that downloads the
        serialized model artifacts from the Hugging Face Hub and then delegates to the
        standard matgl loading path.

        Args:
            repo_id: Repository id on the Hugging Face Hub (e.g. ``"owner/matgl-model"``).
            revision: Optional branch, tag or commit hash to download from.
            token: Optional Hugging Face authentication token for private repos.
            cache_dir: Directory to cache downloaded files. Defaults to
                ``MATGL_CACHE`` (i.e. ``~/.cache/matgl``) so that matgl shares a single
                cache location for both the legacy GitHub mirror and the Hugging Face Hub.
            force_download: If True, re-download files even if they are cached.
            **kwargs: Additional kwargs forwarded to :meth:`IOMixIn.load`.

        Returns:
            An instance of ``cls`` with weights loaded from the Hub.
        """
        fpaths = _download_from_hf_hub(
            repo_id,
            revision=revision,
            token=token,
            cache_dir=cache_dir,
            force_download=force_download,
        )
        return cls.load(fpaths, **kwargs)

    @property
    def num_parameters(self) -> int:
        """Return the number of parameters in the model."""
        return sum(p.numel() for p in cast("torch.nn.Module", self).parameters())


def load_model(path: str | Path, **kwargs):
    r"""Convenience method to load a model from a directory or name.

    Args:
        path (str|path): Path to saved model, name of pre-trained model, or Hugging Face Hub
            repo id in ``"owner/name"`` form. The search order is:

            1. Local filesystem path.
            2. If ``path`` already matches an ``"owner/name"`` pattern, the Hugging Face Hub
               directly.
            3. If ``path`` is a bare model name (no ``"/"``), the official ``materialyze``
               Hugging Face org (``materialyze/<name>``).
        **kwargs: Additional kwargs forwarded to ``huggingface_hub.hf_hub_download``
            (e.g. ``revision``, ``token``, ``force_download``, ``cache_dir``).

    Returns:
        model_object.
    """
    str_path = str(path)
    path = Path(path)

    try:
        fpaths = _get_file_paths(path, str_path=str_path, **kwargs)
        with open(fpaths["model.json"]) as f:
            d = json.load(f)
            modname = _resolve_module(d["@module"])
            classname = d["@class"]
            mod = __import__(modname, globals(), locals(), [classname], 0)
            cls_ = getattr(mod, classname)
            return cls_.load(fpaths, **kwargs)
    except (ImportError, ValueError) as ex:
        raise ValueError(
            "Bad serialized model or bad model name. If you have an outdated cached model,"
            " please 'clear your cache by running `python -c 'import matgl; matgl.clear_cache()'`"
        ) from ex
    except BaseException as ex:
        # The chained ``from ex`` already carries the original traceback to the
        # caller; no need for a redundant ``traceback.print_exc()`` to stderr.
        raise RuntimeError(
            "Unknown error occurred while loading model. Please review the traceback for more information."
        ) from ex


def _get_file_paths(path: Path, str_path: str | None = None, **kwargs):
    """Search path for files.

    Args:
        path (Path): Path to saved model, name of a pre-trained model, or Hugging Face
            Hub repo id. The search order is:

            1. Local filesystem path.
            2. If the identifier already matches an ``"owner/name"`` repo id pattern, the
               Hugging Face Hub directly.
            3. If the identifier is a bare model name, the official ``materialyze`` Hugging
               Face org (``materialyze/<name>``).
        str_path (str | None): The original string form of ``path``, preserved so that
            Hugging Face repo ids (which use ``/``) are not mangled by ``Path``.
        **kwargs: Additional kwargs forwarded to ``huggingface_hub.hf_hub_download``.
            Supported kwargs include ``revision``, ``token``, ``cache_dir``, and
            ``force_download``.

    Returns:
        {
            "model.pt": path to model.pt file,
            "state.pt": path to state file,
            "model.json": path to model.json file
        }
    """
    fnames = _MODEL_FILES

    if all((path / fn).exists() for fn in fnames):
        return {fn: path / fn for fn in fnames}

    str_path = str_path if str_path is not None else str(path)

    if _HF_REPO_ID_RE.match(str_path):
        return _download_from_hf_hub(str_path, **kwargs)

    if "/" not in str_path:
        hf_repo_id = f"{HF_MATGL_ORG}/{str_path}"
        try:
            return _download_from_hf_hub(hf_repo_id, **kwargs)
        except Exception as err:
            raise ValueError(_format_unknown_model_message(str_path, hf_repo_id=hf_repo_id)) from err

    raise ValueError(_format_unknown_model_message(str_path))


def _format_unknown_model_message(identifier: str, *, hf_repo_id: str | None = None) -> str:
    """Build a helpful 'model not found' message with nearest-match suggestion.

    When the Hub is reachable, queries ``get_available_pretrained_models`` to build
    a "Did you mean: ..." hint via ``difflib.get_close_matches``. When the Hub call
    fails the candidate list is empty and the hint is simply omitted — there is no
    static fallback because, with the Hub down, the user could not download the
    model anyway.
    """
    candidates = get_available_pretrained_models()
    suggestions = difflib.get_close_matches(identifier, candidates, n=3, cutoff=0.5)
    parts = [
        f"No valid model found locally or at Hugging Face repo '{hf_repo_id}'."
        if hf_repo_id
        else f"No valid model found locally or at Hugging Face Hub for identifier '{identifier}'.",
    ]
    if suggestions:
        parts.append(f"Did you mean: {', '.join(suggestions)}?")
    parts.append("Run `matgl.get_available_pretrained_models()` to list all available models.")
    return " ".join(parts)


def _download_from_hf_hub(
    repo_id: str,
    *,
    revision: str | None = None,
    token: str | None = None,
    cache_dir: str | Path | None = None,
    force_download: bool = False,
    **_ignored,
) -> dict[str, Path]:
    """Download the three matgl model artifacts from a Hugging Face Hub repo.

    Downloads are cached under ``MATGL_CACHE`` by default so that matgl shares a single
    cache location for both the legacy GitHub mirror and the Hugging Face Hub.

    Args:
        repo_id: Hugging Face repo id of the form ``"owner/name"``.
        revision: Optional branch, tag, or commit hash.
        token: Optional authentication token.
        cache_dir: Optional cache directory override. Defaults to ``MATGL_CACHE``.
        force_download: If True, re-download files even if cached.
        **_ignored: Swallows unrelated kwargs so this helper can be used as a drop-in
            fallback inside ``_get_file_paths``.

    Returns:
        Mapping of filename to local path for the downloaded artifacts.
    """
    resolved_cache_dir = str(cache_dir) if cache_dir is not None else str(MATGL_CACHE)
    return {
        fn: Path(
            hf_hub_download(
                repo_id=repo_id,
                filename=fn,
                revision=revision,
                token=token,
                cache_dir=resolved_cache_dir,
                force_download=force_download,
            )
        )
        for fn in _MODEL_FILES
    }


def _generate_hf_model_card(model, metadata: dict | None = None, readme_text: str | None = None) -> str:
    """Generate a minimal README.md model card for a matgl model uploaded to the Hub.

    Args:
        model: The model instance being uploaded.
        metadata: Optional user-supplied metadata dict.
        readme_text: Optional README.md text to upload. If None, a default README.md text will be generated.

    Returns:
        A markdown string suitable for use as ``README.md`` on the Hugging Face Hub.
    """
    cls_name = model.__class__.__name__
    model_version = getattr(model, "__version__", 0)
    meta_block = ""
    if metadata:
        try:
            meta_block = "\n\n## Metadata\n\n```json\n" + json.dumps(metadata, indent=2, default=str) + "\n```\n"
        except (TypeError, ValueError):
            meta_block = ""

    if readme_text is None:
        readme_text = (
            f"# {cls_name}\n\n"
            f"This is a [matgl](https://github.com/materialsvirtuallab/matgl) `{cls_name}` "
            f"model (version {model_version}).\n\n"
            "## Usage\n\n"
            "```python\n"
            "import matgl\n\n"
            'model = matgl.load_model("<owner>/<repo>")\n'
            "```\n"
        )

    return (
        "---\n"
        "library_name: matgl\n"
        "tags:\n"
        "- matgl\n"
        "- materials-science\n"
        "- graph-neural-network\n"
        "---\n\n"
        f"{readme_text}"
        f"{meta_block}"
    )


def _check_ver(cls_, d: dict):
    """Check version of cls_ in current matgl against those noted in a model.json dict.

    Args:
        cls_: Class object.
        d: Dict from serialized json.

    Raises:
        Deprecation warning if the code is
    """
    if getattr(cls_, "__version__", 0) > d.get("@model_version", 0):
        warnings.warn(
            "Incompatible model version detected! The code will continue to load the model but it is "
            "recommended that you provide a path to an updated model, increment your @model_version in model.json "
            "if you are confident that the changes are not problematic, or clear your ~/.matgl cache using "
            '`python -c "import matgl; matgl.clear_cache()"`',
            UserWarning,
            stacklevel=2,
        )


def get_available_pretrained_models() -> list[str]:
    """Get the list of pre-trained models available for download via ``load_model``.

    Queries the official ``materialyze`` Hugging Face org and returns the bare model
    names (i.e. without the ``"materialyze/"`` prefix) so they can be passed directly
    to ``load_model``. Returns an empty list when the Hub is unreachable — there is
    no point shipping a static fallback because, without Hub access, the user could
    not download any of those models either.

    Returns:
        Sorted list of available model names.
    """
    names: set[str] = set()
    try:
        for model_info in HfApi().list_models(author=HF_MATGL_ORG):
            repo_id = str(getattr(model_info, "id", "") or getattr(model_info, "modelId", "") or "")
            if "/" in repo_id:
                names.add(repo_id.split("/", 1)[1])
    except Exception as err:
        logger.warning("Failed to list models from Hugging Face org '%s': %s", HF_MATGL_ORG, err)

    return sorted(names)
