"""Lightning callbacks for matgl training."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import lightning as pl
import torch

import matgl


def add_sample_indices(dataset: Any, start: int = 0) -> None:
    """Stamp a unique global index onto every sample's graph in ``dataset``.

    The index is what :class:`PredictionLogger` uses to keep per-epoch logs sorted under a
    shuffled training dataloader: column ``k`` of the saved energy / force arrays is always
    the prediction for the configuration whose stamped index is ``k``.

    For PYG, the index is stored as ``data.sample_idx`` (a ``(1,)`` long tensor) on each
    underlying ``torch_geometric.data.Data`` graph. ``Batch.from_data_list`` then collates
    it automatically into a ``(B,)`` tensor on the batched ``Batch``.

    For DGL, the index is replicated per-atom into ``g.ndata["sample_idx"]``. ``dgl.batch``
    concatenates ``ndata`` so the per-graph index can be recovered downstream from the
    batch boundaries given by ``g.batch_num_nodes()``.

    Works with both raw ``MGLDataset`` instances and ``torch.utils.data.Subset`` /
    ``dgl.data.utils.Subset`` returned by ``split_dataset``. Mutation is in-place: indices
    are written onto the shared underlying graph objects, so call this after splitting and
    only on the subset(s) you want logged.

    Args:
        dataset: An iterable that yields ``(graph, ...)`` tuples — typically an MGLDataset
            or a Subset thereof.
        start: First index to assign. Defaults to 0.
    """
    for k, item in enumerate(dataset):
        graph = item[0]
        idx = start + k
        if matgl.config.BACKEND == "PYG":
            graph.sample_idx = torch.tensor([idx], dtype=torch.long)
        else:
            graph.ndata["sample_idx"] = torch.full((graph.num_nodes(),), idx, dtype=torch.long)


class PredictionLogger(pl.Callback):
    """Lightning callback that logs per-epoch energy (and optionally force) predictions to disk.

    At the end of each validation epoch the callback overwrites a single ``val_predictions.pt``
    file containing all epochs seen so far, so the file is always recoverable after a walltime
    cut.  An analogous ``train_predictions.pt`` is written when ``log_train=True``, and a
    ``test_predictions.pt`` is written once when ``trainer.test()`` is called.

    The dataset(s) being logged must be stamped with global indices via
    :func:`add_sample_indices` before training so that column ``k`` of the saved arrays
    always refers to the same sample, even with a shuffled training dataloader. Without
    indices the callback raises at the first batch end.

    The saved dict has the following keys (force keys only present when ``log_forces=True``):

    .. code-block:: text

        {
            "energy_pred":  (n_epochs, n_structures)   # predicted energies per epoch
            "energy_true":  (n_structures,)             # DFT reference energies (fixed)
            "num_atoms":    (n_structures,)             # atoms per structure (fixed)
            "force_pred":   (n_epochs, total_atoms, 3) # predicted forces per epoch
            "force_true":   (total_atoms, 3)            # DFT reference forces (fixed)
        }

    Example::

        from matgl.utils.callbacks import PredictionLogger, add_sample_indices

        add_sample_indices(val_data)
        logger = PredictionLogger("predictions/", log_forces=True)
        trainer = pl.Trainer(callbacks=[logger], ...)
        trainer.fit(lit_model, train_loader, val_loader)

    Args:
        save_dir: Directory where prediction files are written.
        log_train: If True, also log training-set predictions each epoch.
        log_forces: If True, also log per-atom force predictions and ground truth.
    """

    def __init__(
        self,
        save_dir: str | Path,
        log_train: bool = False,
        log_forces: bool = False,
    ) -> None:
        """Initialise PredictionLogger."""
        super().__init__()
        self.save_dir = Path(save_dir)
        self.log_train = log_train
        self.log_forces = log_forces

        # current-epoch buffers: idx -> {e_pred, e_true, num_atoms, f_pred?, f_true?}
        self._val_buf: dict[int, dict[str, torch.Tensor]] = {}
        self._train_buf: dict[int, dict[str, torch.Tensor]] = {}
        self._test_buf: dict[int, dict[str, torch.Tensor]] = {}

        # accumulated across all epochs (stacked sorted-by-idx tensors)
        self._val_e_pred: list[torch.Tensor] = []
        self._val_f_pred: list[torch.Tensor] = []
        self._train_e_pred: list[torch.Tensor] = []
        self._train_f_pred: list[torch.Tensor] = []

        # ground truth + num_atoms (set once, from first epoch)
        self._val_e_true: torch.Tensor | None = None
        self._val_f_true: torch.Tensor | None = None
        self._val_num_atoms: torch.Tensor | None = None
        self._train_e_true: torch.Tensor | None = None
        self._train_f_true: torch.Tensor | None = None
        self._train_num_atoms: torch.Tensor | None = None

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _absorb(self, pl_module: Any, buf: dict[int, dict[str, torch.Tensor]]) -> None:
        """Capture per-sample preds from the latest step and stash by sample idx."""
        out = getattr(pl_module, "_step_output", None)
        if out is None:
            return
        sample_idx = out.get("sample_idx")
        if sample_idx is None:
            raise RuntimeError(
                "PredictionLogger could not find per-sample indices on the batch. Call "
                "`matgl.utils.callbacks.add_sample_indices(dataset)` on the dataset (or its "
                "subset) you are logging before constructing the dataloader."
            )
        e_pred = out["energies_pred"]
        e_true = out["energies_true"]
        num_atoms = out["num_atoms"]
        f_pred = out["forces_pred"] if self.log_forces else None
        f_true = out["forces_true"] if self.log_forces else None
        offset = 0
        for i, (idx, n) in enumerate(zip(sample_idx.tolist(), num_atoms.tolist(), strict=False)):
            entry: dict[str, torch.Tensor] = {
                "e_pred": e_pred[i].reshape(()),
                "e_true": e_true[i].reshape(()),
                "num_atoms": torch.tensor(int(n), dtype=torch.long),
            }
            if self.log_forces and f_pred is not None and f_true is not None:
                entry["f_pred"] = f_pred[offset : offset + int(n)]
                entry["f_true"] = f_true[offset : offset + int(n)]
            buf[int(idx)] = entry
            offset += int(n)

    def _stack_epoch(
        self, buf: dict[int, dict[str, torch.Tensor]]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """Stack per-sample buffer in sorted-idx order into epoch tensors."""
        sorted_idx = sorted(buf.keys())
        e_pred = torch.stack([buf[i]["e_pred"] for i in sorted_idx])
        e_true = torch.stack([buf[i]["e_true"] for i in sorted_idx])
        num_atoms = torch.stack([buf[i]["num_atoms"] for i in sorted_idx])
        f_pred: torch.Tensor | None = None
        f_true: torch.Tensor | None = None
        if self.log_forces:
            f_pred = torch.cat([buf[i]["f_pred"] for i in sorted_idx], dim=0)
            f_true = torch.cat([buf[i]["f_true"] for i in sorted_idx], dim=0)
        return e_pred, e_true, num_atoms, f_pred, f_true

    def _save(self, path: Path, payload: dict) -> None:
        """Write payload to ``path``, creating parent dirs if needed."""
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, path)

    # ------------------------------------------------------------------
    # fit_start — validate configuration
    # ------------------------------------------------------------------

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Raise early if multiple val dataloaders are configured."""
        val_dls = trainer.val_dataloaders
        if isinstance(val_dls, (list, tuple)) and len(val_dls) > 1:
            raise ValueError(
                "PredictionLogger does not support multiple validation dataloaders. "
                "Pass a single val dataloader to trainer.fit()."
            )

    # ------------------------------------------------------------------
    # validation
    # ------------------------------------------------------------------

    def on_validation_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Reset val buffer at the start of each validation epoch."""
        if not trainer.sanity_checking:
            self._val_buf = {}

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Accumulate predictions from each validation batch."""
        if not trainer.sanity_checking:
            self._absorb(pl_module, self._val_buf)

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Stack and save val predictions, overwriting the file each epoch."""
        if trainer.sanity_checking or not self._val_buf:
            return
        e_pred, e_true, num_atoms, f_pred, f_true = self._stack_epoch(self._val_buf)
        self._val_e_pred.append(e_pred)
        if self._val_e_true is None:
            self._val_e_true = e_true
            self._val_num_atoms = num_atoms
        if self.log_forces and f_pred is not None and f_true is not None:
            self._val_f_pred.append(f_pred)
            if self._val_f_true is None:
                self._val_f_true = f_true

        assert self._val_e_true is not None
        assert self._val_num_atoms is not None
        payload: dict[str, torch.Tensor] = {
            "energy_pred": torch.stack(self._val_e_pred),
            "energy_true": self._val_e_true,
            "num_atoms": self._val_num_atoms,
        }
        if self.log_forces and self._val_f_true is not None:
            payload["force_pred"] = torch.stack(self._val_f_pred)
            payload["force_true"] = self._val_f_true
        self._save(self.save_dir / "val_predictions.pt", payload)

    # ------------------------------------------------------------------
    # training
    # ------------------------------------------------------------------

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Reset train buffer at the start of each training epoch."""
        if self.log_train:
            self._train_buf = {}

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Accumulate predictions from each training batch."""
        if self.log_train:
            self._absorb(pl_module, self._train_buf)

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Stack and save train predictions, overwriting the file each epoch."""
        if not self.log_train or not self._train_buf:
            return
        e_pred, e_true, num_atoms, f_pred, f_true = self._stack_epoch(self._train_buf)
        self._train_e_pred.append(e_pred)
        if self._train_e_true is None:
            self._train_e_true = e_true
            self._train_num_atoms = num_atoms
        if self.log_forces and f_pred is not None and f_true is not None:
            self._train_f_pred.append(f_pred)
            if self._train_f_true is None:
                self._train_f_true = f_true

        assert self._train_e_true is not None
        assert self._train_num_atoms is not None
        payload: dict[str, torch.Tensor] = {
            "energy_pred": torch.stack(self._train_e_pred),
            "energy_true": self._train_e_true,
            "num_atoms": self._train_num_atoms,
        }
        if self.log_forces and self._train_f_true is not None:
            payload["force_pred"] = torch.stack(self._train_f_pred)
            payload["force_true"] = self._train_f_true
        self._save(self.save_dir / "train_predictions.pt", payload)

    # ------------------------------------------------------------------
    # test
    # ------------------------------------------------------------------

    def on_test_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Reset test buffer and validate single dataloader."""
        test_dls = trainer.test_dataloaders
        if isinstance(test_dls, (list, tuple)) and len(test_dls) > 1:
            raise ValueError(
                "PredictionLogger does not support multiple test dataloaders. "
                "Pass a single test dataloader to trainer.test()."
            )
        self._test_buf = {}

    def on_test_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Accumulate predictions from each test batch."""
        self._absorb(pl_module, self._test_buf)

    def on_test_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Save test predictions to disk."""
        if not self._test_buf:
            return
        e_pred, e_true, num_atoms, f_pred, f_true = self._stack_epoch(self._test_buf)
        payload: dict[str, torch.Tensor] = {
            "energy_pred": e_pred,
            "energy_true": e_true,
            "num_atoms": num_atoms,
        }
        if self.log_forces and f_pred is not None and f_true is not None:
            payload["force_pred"] = f_pred
            payload["force_true"] = f_true
        self._save(self.save_dir / "test_predictions.pt", payload)
