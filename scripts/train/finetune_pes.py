#!/usr/bin/env python
"""Finetune a pre-trained MatGL potential from a config file.

Loads a pre-trained potential (a Hugging Face Hub name like ``M3GNet-MP-2021.2.8-PES``
or a local model directory) via ``matgl.load_model``, then continues training it
on the datasets named in the config. The element table and per-element energy
offsets are taken from the pre-trained model, so the only difference from
``train_pes.py`` is that the model and its ``element_refs`` come from disk rather
than being built and fit from scratch.

Usage::

    python scripts/train/finetune_pes.py --config config.yaml
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path

import _common

import matgl

warnings.simplefilter("ignore")


def main() -> None:
    """Run finetuning from a config file."""
    args = _common.parse_args(__doc__ or "Finetune a pre-trained MatGL potential.")
    config = _common.load_config(args.config)

    print(f"Loading pre-trained model {config['model']!r}...")
    nnp = matgl.load_model(config["model"])
    model = nnp.model
    element_refs = nnp.element_refs.property_offset
    # The dataset graphs must use the pre-trained model's element table, so pin it
    # in the config before building datasets (overriding any user-provided value).
    config["element_types"] = list(model.element_types)

    print("Preparing datasets...")
    train_data, val_data, test_data, _ = _common.build_datasets(config)
    train_loader, val_loader, test_loader = _common.build_dataloaders(train_data, val_data, test_data, config)

    lit_module = _common.build_potential_module(model, config, element_refs)

    logger = _common.build_logger(config)
    trainer = _common.build_trainer(config, logger, _common.build_callbacks(config))

    print("Start finetuning...")
    trainer.fit(
        model=lit_module,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=_common.resume_ckpt_path(config),
    )

    if test_loader is not None:
        print("Testing model...")
        # Pass the in-memory module so the just-trained weights are evaluated
        # rather than Lightning's default ckpt_path="best".
        trainer.test(model=lit_module, dataloaders=test_loader)

    model_dir = Path(config.get("model_dir", "./finetuned_model"))
    os.makedirs(model_dir, exist_ok=True)
    lit_module.model.save(model_dir)
    print(f"Saved finetuned model to {model_dir}")


if __name__ == "__main__":
    main()
