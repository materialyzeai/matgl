"""Lightning callbacks for MatGL training."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import lightning as pl
import torch

import matgl

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping


def add_sample_indices(dataset: Any, start: int = 0) -> None:
    """Stamp a unique global index onto every sample's graph in ``dataset``.

    The index is what :class:`PredictionLogger` uses to keep per-epoch logs sorted under a
    shuffled training dataloader: column ``i`` of the saved energy / force arrays is always
    the prediction for the configuration whose index is ``i``.

    For PYG, the index is stored as ``data.sample_idx`` (a ``(1,)`` long tensor) on each
    underlying ``torch_geometric.data.Data`` graph. ``Batch.from_data_list`` then collates it
    automatically into a ``(B,)`` tensor on the batched ``Batch``.

    For DGL, the index is replicated per-atom into ``g.ndata["sample_idx"]``. ``dgl.batch``
    concatenates ``ndata`` so the per-graph index can be recovered downstream from the batch
    boundaries given by ``g.batch_num_nodes()``.

    Works with both raw ``MGLDataset`` instances and ``torch.utils.data.Subset`` /
    ``dgl.data.utils.Subset`` returned by :func:`matgl.graph.split_dataset`. Mutation is
    in-place: indices are written onto the shared underlying graph objects, so call this
    after splitting and only on the subset(s) you want logged.

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
    """Capture per-epoch energy and force predictions, labels, and errors.

    Plug into a ``lightning.Trainer`` via ``callbacks=[PredictionLogger(...)]`` while training
    a :class:`matgl.utils.training.PotentialLightningModule`. Default behaviour logs the
    **training** set; pass ``log_validation=True`` to also (or instead) log the validation set.

    The dataset(s) being logged must be stamped with global indices via
    :func:`add_sample_indices` before training so that the ``(n_epochs, n_samples)`` log
    columns align even though the training dataloader shuffles. Without indices the callback
    raises at the first batch end.

    After every (non-sanity-check) epoch the callback accumulates:

    - ``{train,val}_energy_preds``: ``(n_epochs, n_samples)`` total energies per supercell.
    - ``{train,val}_energy_labels``: ``(n_samples,)`` ground-truth total energies (recorded once).
    - ``{train,val}_energy_errors``: ``preds - labels``.
    - ``{train,val}_force_preds``: ``(n_epochs, n_atoms, 3)`` per-atom forces.
    - ``{train,val}_force_labels``: ``(n_atoms, 3)`` ground-truth forces (recorded once).
    - ``{train,val}_force_errors``: ``preds - labels``.

    Args:
        save_path: Optional path to persist the cumulative log to as a ``torch.save`` payload.
            Rewritten at every epoch end so it survives a crash. ``None`` keeps the log in
            memory only, accessed via :attr:`predictions`.
        log_train: Log the training set (default).
        log_validation: Log the validation set in addition to / instead of training.
    """

    def __init__(
        self,
        save_path: str | Path | None = None,
        log_train: bool = True,
        log_validation: bool = False,
    ) -> None:
        """See class docstring."""
        super().__init__()
        if not log_train and not log_validation:
            raise ValueError("PredictionLogger requires at least one of log_train, log_validation.")
        self.save_path: Path | None = Path(save_path) if save_path is not None else None
        self.log_train = log_train
        self.log_validation = log_validation
        # Per-epoch (current epoch) collected predictions, keyed by sample idx.
        self._epoch_train: dict[int, dict[str, torch.Tensor]] = {}
        self._epoch_val: dict[int, dict[str, torch.Tensor]] = {}
        # Per-epoch stacked tensors accumulated over all completed epochs.
        self._train_e_preds: list[torch.Tensor] = []
        self._train_f_preds: list[torch.Tensor] = []
        self._val_e_preds: list[torch.Tensor] = []
        self._val_f_preds: list[torch.Tensor] = []
        # Ground truth, recorded once.
        self._train_e_labels: torch.Tensor | None = None
        self._train_f_labels: torch.Tensor | None = None
        self._val_e_labels: torch.Tensor | None = None
        self._val_f_labels: torch.Tensor | None = None

    # --- training hooks -------------------------------------------------------------------

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Reset the per-epoch training buffer."""
        if self.log_train:
            self._epoch_train = {}

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Mapping[str, Any] | torch.Tensor | None,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Capture per-sample preds for this training batch."""
        if not self.log_train:
            return
        self._absorb(outputs, target=self._epoch_train)

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Stack per-sample preds in idx order and append to running per-epoch lists."""
        if not self.log_train or not self._epoch_train:
            return
        e_pred, f_pred, e_label, f_label = self._stack_epoch(self._epoch_train)
        self._train_e_preds.append(e_pred)
        self._train_f_preds.append(f_pred)
        if self._train_e_labels is None:
            self._train_e_labels = e_label
            self._train_f_labels = f_label
        if self.save_path is not None:
            self._save(self.save_path)

    # --- validation hooks -----------------------------------------------------------------

    def on_validation_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Reset the per-epoch validation buffer."""
        if self.log_validation and not trainer.sanity_checking:
            self._epoch_val = {}

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Mapping[str, Any] | torch.Tensor | None,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Capture per-sample preds for this validation batch."""
        if not self.log_validation or trainer.sanity_checking:
            return
        self._absorb(outputs, target=self._epoch_val)

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Stack per-sample validation preds and append to running per-epoch lists."""
        if not self.log_validation or trainer.sanity_checking or not self._epoch_val:
            return
        e_pred, f_pred, e_label, f_label = self._stack_epoch(self._epoch_val)
        self._val_e_preds.append(e_pred)
        self._val_f_preds.append(f_pred)
        if self._val_e_labels is None:
            self._val_e_labels = e_label
            self._val_f_labels = f_label
        if self.save_path is not None:
            self._save(self.save_path)

    # --- helpers --------------------------------------------------------------------------

    @staticmethod
    def _absorb(outputs: Any, target: dict[int, dict[str, torch.Tensor]]) -> None:
        if not isinstance(outputs, dict) or "preds" not in outputs or "labels" not in outputs:
            raise RuntimeError(
                "PredictionLogger requires a LightningModule whose training_step / "
                "validation_step returns a dict with 'preds' and 'labels' keys "
                "(matgl PotentialLightningModule does this)."
            )
        indices = outputs.get("indices")
        num_atoms = outputs.get("num_atoms")
        if indices is None or num_atoms is None:
            raise RuntimeError(
                "PredictionLogger could not find per-sample indices on the batch. Call "
                "`matgl.utils.callbacks.add_sample_indices(dataset)` on the dataset (or its "
                "subset) you are logging before constructing the dataloader."
            )
        e_pred = outputs["preds"][0].detach().cpu()
        f_pred = outputs["preds"][1].detach().cpu()
        e_label = outputs["labels"][0].detach().cpu()
        f_label = outputs["labels"][1].detach().cpu()
        idx_list = indices.detach().cpu().tolist()
        n_atoms_list = num_atoms.detach().cpu().tolist()
        offset = 0
        for i, (idx, n) in enumerate(zip(idx_list, n_atoms_list, strict=False)):
            target[int(idx)] = {
                "e_pred": e_pred[i].reshape(()),
                "f_pred": f_pred[offset : offset + n],
                "e_label": e_label[i].reshape(()),
                "f_label": f_label[offset : offset + n],
            }
            offset += n

    @staticmethod
    def _stack_epoch(
        buf: dict[int, dict[str, torch.Tensor]],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        sorted_idx = sorted(buf.keys())
        e_pred = torch.stack([buf[i]["e_pred"] for i in sorted_idx])
        f_pred = torch.cat([buf[i]["f_pred"] for i in sorted_idx], dim=0)
        e_label = torch.stack([buf[i]["e_label"] for i in sorted_idx])
        f_label = torch.cat([buf[i]["f_label"] for i in sorted_idx], dim=0)
        return e_pred, f_pred, e_label, f_label

    @property
    def predictions(self) -> dict[str, torch.Tensor]:
        """Return the cumulative prediction log as a dict of tensors.

        Keys are prefixed with ``train_`` and/or ``val_`` depending on which sets were logged.
        Empty dict before the first epoch completes.
        """
        out: dict[str, torch.Tensor] = {}
        out.update(
            self._collect(
                "train",
                self._train_e_preds,
                self._train_f_preds,
                self._train_e_labels,
                self._train_f_labels,
            )
        )
        out.update(
            self._collect(
                "val",
                self._val_e_preds,
                self._val_f_preds,
                self._val_e_labels,
                self._val_f_labels,
            )
        )
        return out

    @staticmethod
    def _collect(
        prefix: str,
        e_preds: Iterable[torch.Tensor],
        f_preds: Iterable[torch.Tensor],
        e_labels: torch.Tensor | None,
        f_labels: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        e_preds_list = list(e_preds)
        if not e_preds_list or e_labels is None or f_labels is None:
            return {}
        e_stack = torch.stack(e_preds_list, dim=0)
        f_stack = torch.stack(list(f_preds), dim=0)
        return {
            f"{prefix}_energy_preds": e_stack,
            f"{prefix}_energy_labels": e_labels,
            f"{prefix}_energy_errors": e_stack - e_labels.unsqueeze(0),
            f"{prefix}_force_preds": f_stack,
            f"{prefix}_force_labels": f_labels,
            f"{prefix}_force_errors": f_stack - f_labels.unsqueeze(0),
        }

    def _save(self, path: Path) -> None:
        payload = self.predictions
        if not payload:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, path)
