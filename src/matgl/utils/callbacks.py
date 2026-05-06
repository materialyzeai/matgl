"""Lightning callbacks for MatGL training."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import lightning as pl
import torch

if TYPE_CHECKING:
    from collections.abc import Mapping


class PredictionLogger(pl.Callback):
    """Capture per-epoch validation energy and force predictions and labels.

    Plug into a ``lightning.Trainer`` via ``callbacks=[PredictionLogger(...)]`` while training
    a :class:`matgl.utils.training.PotentialLightningModule`. After every (non-sanity-check)
    validation epoch, this callback accumulates:

    - ``energy_preds``: ``(n_epochs, n_supercells)`` total energies per supercell.
    - ``energy_labels``: ``(n_supercells,)`` ground-truth total energies (recorded once).
    - ``energy_errors``: ``energy_preds - energy_labels``.
    - ``force_preds``: ``(n_epochs, n_atoms, 3)`` per-atom forces.
    - ``force_labels``: ``(n_atoms, 3)`` ground-truth per-atom forces (recorded once).
    - ``force_errors``: ``force_preds - force_labels``.

    The validation dataloader must not shuffle so that the per-supercell axis aligns across
    epochs (matgl's :func:`matgl.graph.MGLDataLoader` already creates ``val_loader`` with
    ``shuffle=False``).

    Args:
        save_path: Optional path to persist the cumulative log to as a ``torch.save`` payload.
            The file is rewritten at every validation-epoch end so it survives a crash.
            When ``None`` the log is held in memory only and accessed via :meth:`log`.
    """

    def __init__(self, save_path: str | Path | None = None) -> None:
        """See class docstring."""
        super().__init__()
        self.save_path: Path | None = Path(save_path) if save_path is not None else None
        self._batch_e_pred: list[torch.Tensor] = []
        self._batch_f_pred: list[torch.Tensor] = []
        self._batch_e_label: list[torch.Tensor] = []
        self._batch_f_label: list[torch.Tensor] = []
        self._all_e_preds: list[torch.Tensor] = []
        self._all_f_preds: list[torch.Tensor] = []
        self._e_labels: torch.Tensor | None = None
        self._f_labels: torch.Tensor | None = None

    def on_validation_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Reset the per-epoch buffers."""
        if trainer.sanity_checking:
            return
        self._batch_e_pred = []
        self._batch_f_pred = []
        self._batch_e_label = []
        self._batch_f_label = []

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Mapping[str, Any] | torch.Tensor | None,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Capture this batch's preds and labels from ``validation_step``'s return dict."""
        if trainer.sanity_checking:
            return
        if not isinstance(outputs, dict) or "preds" not in outputs or "labels" not in outputs:
            raise RuntimeError(
                "PredictionLogger requires a LightningModule whose validation_step returns "
                "a dict with 'preds' and 'labels' keys. Use matgl PotentialLightningModule."
            )
        preds = outputs["preds"]
        labels = outputs["labels"]
        self._batch_e_pred.append(preds[0].detach().cpu().flatten())
        self._batch_e_label.append(labels[0].detach().cpu().flatten())
        self._batch_f_pred.append(preds[1].detach().cpu())
        self._batch_f_label.append(labels[1].detach().cpu())

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Stack per-batch tensors into per-epoch tensors and (optionally) persist the log."""
        if trainer.sanity_checking:
            return
        if not self._batch_e_pred:
            return
        self._all_e_preds.append(torch.cat(self._batch_e_pred))
        self._all_f_preds.append(torch.cat(self._batch_f_pred, dim=0))
        if self._e_labels is None:
            self._e_labels = torch.cat(self._batch_e_label)
            self._f_labels = torch.cat(self._batch_f_label, dim=0)
        if self.save_path is not None:
            self._save(self.save_path)

    @property
    def predictions(self) -> dict[str, torch.Tensor]:
        """Return the cumulative prediction log as a dict of tensors.

        Returns:
            Dict with keys ``energy_preds`` ``(n_epochs, n_supercells)``, ``energy_labels``
            ``(n_supercells,)``, ``energy_errors``, ``force_preds`` ``(n_epochs, n_atoms, 3)``,
            ``force_labels`` ``(n_atoms, 3)`` and ``force_errors``. Empty dict before the first
            validation epoch completes.
        """
        if not self._all_e_preds or self._e_labels is None or self._f_labels is None:
            return {}
        energy_preds = torch.stack(self._all_e_preds, dim=0)
        force_preds = torch.stack(self._all_f_preds, dim=0)
        return {
            "energy_preds": energy_preds,
            "energy_labels": self._e_labels,
            "energy_errors": energy_preds - self._e_labels.unsqueeze(0),
            "force_preds": force_preds,
            "force_labels": self._f_labels,
            "force_errors": force_preds - self._f_labels.unsqueeze(0),
        }

    def _save(self, path: Path) -> None:
        payload = self.predictions
        if not payload:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, path)
