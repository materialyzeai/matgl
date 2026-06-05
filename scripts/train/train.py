#!/usr/bin/env python
"""Train a MatGL interatomic potential from scratch from a config file.

Reads a YAML/JSON config (see ``scripts/train/_common.py`` for the dataset JSON
format and the recognised keys), builds the requested graph model, fits
per-element energy offsets on the training set, and trains a
``PotentialLightningModule`` with PyTorch Lightning. The flow mirrors
``examples/Training a QET Potential with PyTorch Lightning.ipynb`` but is driven
entirely by the config file.

Usage::

    python scripts/train/train.py --config config.yaml
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any

import _common

from matgl.models import CHGNet, M3GNet, QET, SO3Net, TensorNet

if TYPE_CHECKING:
    from torch import nn

warnings.simplefilter("ignore")

# Model name -> class. Each is built with ``is_intensive=False`` (PES models
# predict an extensive total energy) plus the user's ``model_args``.
MODEL_REGISTRY: dict[str, type] = {
    "M3GNet": M3GNet,
    "CHGNet": CHGNet,
    "TensorNet": TensorNet,
    "SO3Net": SO3Net,
    "QET": QET,
}


def build_model(config: dict[str, Any], element_types: tuple[str, ...]) -> nn.Module:
    """Instantiate the graph model named by ``config['model']``.

    Args:
        config: The training config (``model`` name and ``model_args``).
        element_types: Element table the model embeds.

    Returns:
        The instantiated model.
    """
    name = config["model"]
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model {name!r}; choose from {sorted(MODEL_REGISTRY)}.")
    return MODEL_REGISTRY[name](
        element_types=element_types,
        is_intensive=False,
        **config.get("model_args", {}),
    )


def main() -> None:
    """Run training from a config file."""
    args = _common.parse_args(__doc__ or "Train a MatGL potential from scratch.")
    config = _common.load_config(args.config)

    print("Preparing datasets...")
    train_data, val_data, test_data, element_types = _common.build_datasets(config)
    train_loader, val_loader, test_loader = _common.build_dataloaders(train_data, val_data, test_data, config)

    print("Building model...")
    model = build_model(config, element_types)

    print("Fitting per-element energy offsets...")
    element_refs = _common.compute_element_refs(train_data, element_types)

    lit_module = _common.build_potential_module(model, config, element_refs)

    logger = _common.build_logger(config)
    trainer = _common.build_trainer(config, logger)

    print("Start training...")
    trainer.fit(model=lit_module, train_dataloaders=train_loader, val_dataloaders=val_loader)

    if test_loader is not None:
        print("Testing model...")
        # Pass the in-memory module so the just-trained weights are evaluated
        # rather than Lightning's default ckpt_path="best".
        trainer.test(model=lit_module, dataloaders=test_loader)

    model_dir = Path(config.get("model_dir", "./trained_model"))
    os.makedirs(model_dir, exist_ok=True)
    lit_module.model.save(model_dir)
    print(f"Saved trained model to {model_dir}")


if __name__ == "__main__":
    main()
