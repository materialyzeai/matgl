"""Provides utilities for managing models and data."""

from __future__ import annotations

import inspect
import json
import logging
import os
import warnings
from pathlib import Path

import requests
import torch

import matgl
from matgl.config import MATGL_CACHE, PRETRAINED_MODELS_BASE_URL

logger = logging.getLogger(__file__)


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
        """
        This method saves the arguments passed to the class initializer. It collects the arguments
        from the `__init__` method of the class, excluding `self` and `__class__`. If an additional
        `kwargs` dictionary is provided, it is merged into the collected arguments. If any of the
        collected arguments are instances of a subclass of `IOMixIn`, those arguments are serialized
        into a dictionary representation that includes class metadata and initialization arguments.
        Finally, the arguments are stored as an instance variable `_init_args`.

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
        """
        Saves the state and configuration of the model to the specified path.

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
            json.dump(d, f, default=lambda o: str(o), indent=4)

    @classmethod
    def load(cls, path: str | Path | dict, **kwargs):
        """Load the model weights from a directory.

        Args:
            path (str|path|dict): Path to saved model or name of pre-trained model. If it is a dict, it is assumed to
                be of the form::

                    {
                        "model.pt": path to model.pt file,
                        "state.pt": path to state file,
                        "model.json": path to model.json file
                    }

                Otherwise, the search order is path, followed by download from PRETRAINED_MODELS_BASE_URL
                (with caching).
            **kwargs: Additional kwargs passed to RemoteFile class. E.g., a useful one might be force_download if you
                want to update the model.

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
                modname = v["@module"]
                classname = v["@class"]
                cls_lower = classname.lower()

                if (
                    "m3gnet" in cls_lower or "megnet" in cls_lower or "chgnet" in cls_lower or "qet" in cls_lower
                ) and matgl.config.BACKEND == "PYG":
                    warnings.warn(
                        f"Model {classname} is a DGL model, but the backend is PYG. Setting the backend to DGL.",
                        stacklevel=2,
                    )
                    matgl.set_backend("DGL")
                mod = __import__(modname, globals(), locals(), [classname], 0)
                cls_ = getattr(mod, classname)
                _check_ver(cls_, v)  # Check version of any subclasses too.
                d[k] = cls_(**v["init_args"])
        d = {k: v for k, v in d.items() if not k.startswith("@")}
        model = cls(**d)
        model.load_state_dict(state, strict=False)  # type: ignore

        return model


class RemoteFile:
    """Handling of download of remote files to a local cache."""

    def __init__(self, uri: str, cache_location: str | Path = MATGL_CACHE, force_download: bool = False):
        """
        Args:
            uri: Uniform resource identifier.
            cache_location: Directory to cache downloaded RemoteFile. By default, downloaded models are saved at
            $HOME/.matgl.
            force_download: To speed up access, a model with the same name in the cache location will be used if
            present. If you want to force a re-download, set this to True.
        """
        self.uri = uri
        toks = uri.split("/")
        self.model_name = toks[-2]
        self.fname = toks[-1]
        self.cache_location = Path(cache_location)
        self.local_path = self.cache_location / self.model_name / self.fname
        if (not self.local_path.exists()) or force_download:
            logger.info("Downloading from remote location...")
            self._download()
        else:
            logger.info(f"Using cached local file at {self.local_path}...")

    def _download(self):
        r = requests.get(self.uri)
        if r.status_code == 200:
            os.makedirs(self.cache_location / self.model_name, exist_ok=True)
            with open(self.local_path, "wb") as f:
                f.write(r.content)
        else:
            raise requests.RequestException(f"Bad uri: {self.uri}")

    def __enter__(self):
        """Support with context.

        Returns:
            Stream on local path.
        """
        self.stream = open(self.local_path, "rb")
        return self.stream

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the with context.

        Args:
            exc_type: Usual meaning in __exit__.
            exc_val: Usual meaning in __exit__.
            exc_tb: Usual meaning in __exit__.
        """
        self.stream.close()


def load_model(path: str | Path, **kwargs):
    r"""Convenience method to load a model from a directory or name.

    Args:
        path (str|path): Path to saved model or name of pre-trained model. The search order is path, followed by
            download from PRETRAINED_MODELS_BASE_URL (with caching).
        **kwargs: Additional kwargs passed to RemoteFile class. E.g., a useful one might be force_download if you
            want to update the model.

    Returns:
        Returns: model_object if include_json is false. (model_object, dict) if include_json is True.
    """
    path = Path(path)

    try:
        fpaths = _get_file_paths(path, **kwargs)
        with open(fpaths["model.json"]) as f:
            d = json.load(f)
            modname = d["@module"]
            classname = d["@class"]

            mod = __import__(modname, globals(), locals(), [classname], 0)
            cls_ = getattr(mod, classname)
            return cls_.load(fpaths, **kwargs)
    except (ImportError, ValueError) as ex:
        raise ValueError(
            "Bad serialized model or bad model name. There are several possible causes. If you are using a DGL model, "
            "you need to set the backend to DGL using `matgl.set_backend('DGL')`. If you have an outdated cached model,"
            " please 'clear your cache by running `python -c 'import matgl; matgl.clear_cache()'`"
        ) from ex
    except BaseException as ex:
        import traceback

        traceback.print_exc()
        raise RuntimeError(
            "Unknown error occurred while loading model. Please review the traceback for more information."
        ) from ex


def _get_file_paths(path: Path, **kwargs):
    """Search path for files.

    Args:
        path (Path): Path to saved model or name of pre-trained model. The search order is path, followed by
            download from PRETRAINED_MODELS_BASE_URL (with caching).
        **kwargs: Additional kwargs passed to RemoteFile class. E.g., a useful one might be force_download if you
            want to update the model.

    Returns:
        {
            "model.pt": path to model.pt file,
            "state.pt": path to state file,
            "model.json": path to model.json file
        }
    """
    fnames = ("model.pt", "state.pt", "model.json")

    if all((path / fn).exists() for fn in fnames):
        return {fn: path / fn for fn in fnames}

    try:
        return {fn: RemoteFile(f"{PRETRAINED_MODELS_BASE_URL}{path}/{fn}", **kwargs).local_path for fn in fnames}
    except requests.RequestException:
        import traceback

        traceback.print_exc()
        raise ValueError(f"No valid model found in pre-trained_models at {PRETRAINED_MODELS_BASE_URL}.") from None


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
    """Checks Github for available pretrained_models for download. These can be used with load_model.

    Returns:
        List of available models.
    """
    r = requests.get("http://api.github.com/repos/materialsvirtuallab/matgl/contents/pretrained_models")
    return [d["name"] for d in json.loads(r.content.decode("utf-8")) if d["type"] == "dir"]
