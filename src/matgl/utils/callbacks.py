"""Lightning callbacks for MatGL training."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import lightning as pl
import torch

import matgl

if TYPE_CHECKING:
    from collections.abc import Mapping


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
    """Capture per-epoch predictions, labels, and errors during training.

    Plug into a ``lightning.Trainer`` via ``callbacks=[PredictionLogger(...)]`` while training
    a :class:`matgl.utils.training.PotentialLightningModule`. Default behaviour logs the
    **training** set; pass ``log_validation=True`` to also (or instead) log the validation set.

    Energies and forces are always logged. Stresses and per-atom charges are logged
    automatically when the wrapped potential computes them (i.e. when
    ``model.calc_stresses`` / ``model.calc_charge`` is true) — pass ``log_stress=False`` or
    ``log_charge=False`` to opt out.

    The dataset(s) being logged must be stamped with global indices via
    :func:`add_sample_indices` before training so that the ``(n_epochs, n_samples)`` log
    columns align even though the training dataloader shuffles. Without indices the callback
    raises at the first batch end.

    After every (non-sanity-check) epoch the callback accumulates:

    - ``{train,val}_energy_preds``: ``(n_epochs, n_samples)`` total energies per supercell.
    - ``{train,val}_energy_labels``: ``(n_samples,)`` ground-truth total energies.
    - ``{train,val}_energy_errors``: ``preds - labels``.
    - ``{train,val}_force_preds``: ``(n_epochs, n_atoms, 3)`` per-atom forces.
    - ``{train,val}_force_labels``: ``(n_atoms, 3)`` ground-truth forces.
    - ``{train,val}_force_errors``: ``preds - labels``.
    - ``{train,val}_stress_preds`` (if logged): ``(n_epochs, n_samples, 3, 3)`` per-supercell stresses.
    - ``{train,val}_stress_labels`` / ``..._errors`` analogously.
    - ``{train,val}_charge_preds`` (if logged): ``(n_epochs, n_atoms)`` per-atom charges.
    - ``{train,val}_charge_labels`` / ``..._errors`` analogously.

    Args:
        save_path: Optional path to persist the cumulative log to as a ``torch.save`` payload.
            Rewritten at every epoch end so it survives a crash. ``None`` keeps the log in
            memory only, accessed via :attr:`predictions`.
        log_train: Log the training set (default).
        log_validation: Log the validation set in addition to / instead of training.
        log_stress: Log stresses when ``model.calc_stresses`` is true (default).
        log_charge: Log per-atom charges when ``model.calc_charge`` is true (default).
    """

    _METRICS = ("e", "f", "s", "q")

    def __init__(
        self,
        save_path: str | Path | None = None,
        log_train: bool = True,
        log_validation: bool = False,
        log_stress: bool = True,
        log_charge: bool = True,
    ) -> None:
        """See class docstring."""
        super().__init__()
        if not log_train and not log_validation:
            raise ValueError("PredictionLogger requires at least one of log_train, log_validation.")
        self.save_path: Path | None = Path(save_path) if save_path is not None else None
        self.log_train = log_train
        self.log_validation = log_validation
        self.log_stress = log_stress
        self.log_charge = log_charge
        # Per-epoch (current epoch) collected predictions, keyed by sample idx.
        self._epoch_train: dict[int, dict[str, torch.Tensor]] = {}
        self._epoch_val: dict[int, dict[str, torch.Tensor]] = {}
        # Per-epoch stacked tensors accumulated over all completed epochs.
        self._train_preds: dict[str, list[torch.Tensor]] = {m: [] for m in self._METRICS}
        self._val_preds: dict[str, list[torch.Tensor]] = {m: [] for m in self._METRICS}
        # Ground truth, recorded once.
        self._train_labels: dict[str, torch.Tensor | None] = dict.fromkeys(self._METRICS)
        self._val_labels: dict[str, torch.Tensor | None] = dict.fromkeys(self._METRICS)

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
        self._absorb(outputs, target=self._epoch_train, pl_module=pl_module)

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Stack per-sample preds in idx order and append to running per-epoch lists."""
        if not self.log_train or not self._epoch_train:
            return
        self._absorb_epoch(self._epoch_train, self._train_preds, self._train_labels)
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
        self._absorb(outputs, target=self._epoch_val, pl_module=pl_module)

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Stack per-sample validation preds and append to running per-epoch lists."""
        if not self.log_validation or trainer.sanity_checking or not self._epoch_val:
            return
        self._absorb_epoch(self._epoch_val, self._val_preds, self._val_labels)
        if self.save_path is not None:
            self._save(self.save_path)

    # --- helpers --------------------------------------------------------------------------

    def _absorb(
        self,
        outputs: Any,
        target: dict[int, dict[str, torch.Tensor]],
        pl_module: pl.LightningModule,
    ) -> None:
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
        preds = outputs["preds"]
        labels = outputs["labels"]
        model = getattr(pl_module, "model", None)
        log_stress = self.log_stress and getattr(model, "calc_stresses", False)
        log_charge = self.log_charge and getattr(model, "calc_charge", False) and len(preds) >= 4

        e_pred = preds[0].detach().cpu()
        f_pred = preds[1].detach().cpu()
        e_label = labels[0].detach().cpu()
        f_label = labels[1].detach().cpu()
        # Stresses come back from collate_fn_pes / Potential as a flat (B*3, 3) stack —
        # reshape into (B, 3, 3) so we can index per-sample.
        if log_stress:
            s_pred = preds[2].detach().cpu().reshape(-1, 3, 3)
            s_label = labels[2].detach().cpu().reshape(-1, 3, 3)
        if log_charge:
            q_pred = preds[3].detach().cpu()
            q_label = labels[3].detach().cpu()

        idx_list = indices.detach().cpu().tolist()
        n_atoms_list = num_atoms.detach().cpu().tolist()
        offset = 0
        for i, (idx, n) in enumerate(zip(idx_list, n_atoms_list, strict=False)):
            entry: dict[str, torch.Tensor] = {
                "e_pred": e_pred[i].reshape(()),
                "f_pred": f_pred[offset : offset + n],
                "e_label": e_label[i].reshape(()),
                "f_label": f_label[offset : offset + n],
            }
            if log_stress:
                entry["s_pred"] = s_pred[i]
                entry["s_label"] = s_label[i]
            if log_charge:
                entry["q_pred"] = q_pred[offset : offset + n]
                entry["q_label"] = q_label[offset : offset + n]
            target[int(idx)] = entry
            offset += n

    @staticmethod
    def _stack_epoch(buf: dict[int, dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        sorted_idx = sorted(buf.keys())
        sample = buf[sorted_idx[0]]
        out: dict[str, torch.Tensor] = {
            "e_pred": torch.stack([buf[i]["e_pred"] for i in sorted_idx]),
            "f_pred": torch.cat([buf[i]["f_pred"] for i in sorted_idx], dim=0),
            "e_label": torch.stack([buf[i]["e_label"] for i in sorted_idx]),
            "f_label": torch.cat([buf[i]["f_label"] for i in sorted_idx], dim=0),
        }
        if "s_pred" in sample:
            out["s_pred"] = torch.stack([buf[i]["s_pred"] for i in sorted_idx])
            out["s_label"] = torch.stack([buf[i]["s_label"] for i in sorted_idx])
        if "q_pred" in sample:
            out["q_pred"] = torch.cat([buf[i]["q_pred"] for i in sorted_idx], dim=0)
            out["q_label"] = torch.cat([buf[i]["q_label"] for i in sorted_idx], dim=0)
        return out

    def _absorb_epoch(
        self,
        epoch_buf: dict[int, dict[str, torch.Tensor]],
        preds_store: dict[str, list[torch.Tensor]],
        labels_store: dict[str, torch.Tensor | None],
    ) -> None:
        stacked = self._stack_epoch(epoch_buf)
        for m in self._METRICS:
            pred_key = f"{m}_pred"
            label_key = f"{m}_label"
            if pred_key not in stacked:
                continue
            preds_store[m].append(stacked[pred_key])
            if labels_store[m] is None:
                labels_store[m] = stacked[label_key]

    @property
    def predictions(self) -> dict[str, torch.Tensor]:
        """Return the cumulative prediction log as a dict of tensors.

        Keys are prefixed with ``train_`` and/or ``val_`` depending on which sets were logged.
        Empty dict before the first epoch completes.
        """
        out: dict[str, torch.Tensor] = {}
        out.update(self._collect("train", self._train_preds, self._train_labels))
        out.update(self._collect("val", self._val_preds, self._val_labels))
        return out

    @staticmethod
    def _collect(
        prefix: str,
        preds_store: dict[str, list[torch.Tensor]],
        labels_store: dict[str, torch.Tensor | None],
    ) -> dict[str, torch.Tensor]:
        if not preds_store["e"] or labels_store["e"] is None:
            return {}
        names = {"e": "energy", "f": "force", "s": "stress", "q": "charge"}
        out: dict[str, torch.Tensor] = {}
        for short, long in names.items():
            preds_list = preds_store[short]
            label = labels_store[short]
            if not preds_list or label is None:
                continue
            stacked = torch.stack(preds_list, dim=0)
            out[f"{prefix}_{long}_preds"] = stacked
            out[f"{prefix}_{long}_labels"] = label
            out[f"{prefix}_{long}_errors"] = stacked - label.unsqueeze(0)
        return out

    def _save(self, path: Path) -> None:
        payload = self.predictions
        if not payload:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, path)
