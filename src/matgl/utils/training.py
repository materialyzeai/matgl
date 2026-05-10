"""Utils for training MatGL models.

This module hosts the Lightning training scaffolding used by both DGL and PyG
backends. The graph-attribute access pattern differs between the two frameworks
(``g.edata`` / ``batch_num_nodes()`` for DGL vs ``g.pos`` / ``g.batch`` for PyG),
so a small handful of methods branch on ``matgl.config.BACKEND``. Everything else
(loss, optimizer, scheduler, logging, the public class layout) is shared.
"""

from __future__ import annotations

import math
import re
import tarfile
import tempfile
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

import lightning as pl
import numpy as np
import torch
import torch.nn.functional as F
import torchmetrics
from huggingface_hub import hf_hub_download
from monty.serialization import loadfn
from pymatgen.core import Structure
from torch import nn

from matgl.config import BACKEND, MATGL_CACHE

if BACKEND == "DGL":
    from matgl.apps._pes_dgl import Potential
else:
    from matgl.apps._pes_pyg import Potential  # type: ignore[assignment]

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from numpy.typing import ArrayLike
    from torch.optim import Optimizer
    from torch.optim.lr_scheduler import LRScheduler
    from torch.utils.data import DataLoader

    from matgl.graph.data import MGLDataset


class MatglLightningModuleMixin:
    """Mix-in class implementing common functions for training."""

    def training_step(self, batch: tuple, batch_idx: int) -> Any:
        """Training step.

        Args:
            batch: Data batch.
            batch_idx: Batch index.

        Returns:
           Total loss.
        """
        results, batch_size = self.step(batch)  # type: ignore
        self.log_dict(  # type: ignore
            {f"train_{key}": val for key, val in results.items()},
            batch_size=batch_size,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
            sync_dist=self.sync_dist,  # type: ignore
        )

        return results["Total_Loss"]

    def on_train_epoch_end(self) -> None:
        """Step scheduler every epoch."""
        sch = self.lr_schedulers()  # type: ignore[attr-defined]
        sch.step()

    def validation_step(self, batch: tuple, batch_idx: int) -> Any:
        """Validation step.

        Args:
            batch: Data batch.
            batch_idx: Batch index.
        """
        results, batch_size = self.step(batch)  # type: ignore
        self.log_dict(  # type: ignore
            {f"val_{key}": val for key, val in results.items()},
            batch_size=batch_size,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
            sync_dist=self.sync_dist,  # type: ignore
        )
        return results["Total_Loss"]

    def test_step(self, batch: tuple, batch_idx: int) -> dict[str, Any]:
        """Test step.

        Args:
            batch: Data batch.
            batch_idx: Batch index.
        """
        # Grad enabling is the responsibility of ``step``: the PES path
        # (PotentialLightningModule.step) toggles it on for autograd-based
        # force/stress computation, and the non-PES path (ModelLightningModule)
        # legitimately runs under Lightning's default eval mode.
        results, batch_size = self.step(batch)  # type: ignore
        self.log_dict(  # type: ignore
            {f"test_{key}": val for key, val in results.items()},
            batch_size=batch_size,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
            sync_dist=self.sync_dist,  # type: ignore
        )
        return results

    def configure_optimizers(self) -> tuple[list[torch.optim.Optimizer], list[torch.optim.lr_scheduler.LRScheduler]]:
        """Configure optimizers."""
        if self.optimizer is None:  # type: ignore[attr-defined]
            optimizer = torch.optim.Adam(
                self.parameters(),  # type: ignore[attr-defined]
                lr=self.lr,  # type: ignore[attr-defined]
                eps=1e-8,
            )
        else:
            optimizer = self.optimizer  # type: ignore[attr-defined]
        if self.scheduler is None:  # type: ignore[attr-defined]
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.decay_steps,  # type: ignore[attr-defined]
                eta_min=self.lr * self.decay_alpha,  # type: ignore[attr-defined]
            )
        else:
            scheduler = self.scheduler  # type: ignore[attr-defined]
        return [
            optimizer,
        ], [
            scheduler,
        ]

    def on_test_model_eval(self, *args: Any, **kwargs: Any) -> None:
        """Executed on model testing.

        Args:
            *args: Pass-through
            **kwargs: Pass-through.
        """
        super().on_test_model_eval(*args, **kwargs)  # type: ignore[misc]

    def predict_step(self, batch: tuple, batch_idx: int, dataloader_idx: int = 0) -> Any:
        """Prediction step.

        Args:
            batch: Data batch.
            batch_idx: Batch index.
            dataloader_idx: Data loader index.

        Returns:
            Prediction
        """
        # See note in ``test_step``: ``step`` enables grad itself when needed
        # (Potential autograd); non-PES models run under Lightning eval mode.
        return self.step(batch)  # type: ignore[attr-defined]


class ModelLightningModule(MatglLightningModuleMixin, pl.LightningModule):
    """A PyTorch.LightningModule for training MatGL structure-wise property models."""

    def __init__(
        self,
        model,
        include_line_graph: bool = False,
        data_mean: float = 0.0,
        data_std: float = 1.0,
        loss: str = "mse_loss",
        loss_params: dict | None = None,
        optimizer: Optimizer | None = None,
        scheduler: LRScheduler | None = None,
        lr: float = 0.001,
        decay_steps: int = 1000,
        decay_alpha: float = 0.01,
        sync_dist: bool = False,
        **kwargs,
    ):
        """Init ModelLightningModule with key parameters.

        Args:
            model: Which type of the model for training
            include_line_graph: whether to include line graphs
            data_mean: average of training data
            data_std: standard deviation of training data
            loss: loss function used for training
            loss_params: parameters for loss function
            optimizer: optimizer for training
            scheduler: scheduler for training
            lr: learning rate for training
            decay_steps: number of steps for decaying learning rate
            decay_alpha: parameter determines the minimum learning rate.
            sync_dist: whether sync logging across all GPU workers or not
            **kwargs: Passthrough to parent init.
        """
        super().__init__(**kwargs)

        self.model = model
        self.include_line_graph = include_line_graph
        self.mae = torchmetrics.MeanAbsoluteError()
        self.rmse = torchmetrics.MeanSquaredError(squared=False)
        self.data_mean = data_mean
        self.data_std = data_std
        self.lr = lr
        self.decay_steps = decay_steps
        self.decay_alpha = decay_alpha
        if loss == "mse_loss":
            self.loss = F.mse_loss
        elif loss == "huber_loss":
            self.loss = F.huber_loss  # type:ignore[assignment]
        elif loss == "smooth_l1_loss":
            self.loss = F.smooth_l1_loss  # type:ignore[assignment]
        else:
            self.loss = F.l1_loss
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.sync_dist = sync_dist
        self.loss_params = loss_params if loss_params is not None else {}
        self.save_hyperparameters(ignore=["model"])

    def forward(
        self,
        g: Any,
        lat: torch.Tensor | None = None,
        l_g: Any = None,
        state_attr: torch.Tensor | None = None,
    ):
        """Run the wrapped model.

        Attaches per-node ``pos`` and per-edge ``pbc_offshift`` tensors derived from
        ``frac_coords`` / ``pbc_offset`` and the supplied lattice(s), then delegates
        to the wrapped model.

        Args:
            g: Backend graph (DGL ``DGLGraph`` or PyG ``Data``/``Batch``).
            lat: Lattice tensor. ``(3, 3)`` for a single graph or ``(B, 3, 3)`` when batched.
            l_g: Optional line graph.
            state_attr: Optional state attribute.

        Returns:
            Model prediction.
        """
        if BACKEND == "DGL":
            g.edata["lattice"] = torch.repeat_interleave(lat, g.batch_num_edges(), dim=0)  # type:ignore[arg-type]
            g.edata["pbc_offshift"] = (g.edata["pbc_offset"].unsqueeze(dim=-1) * g.edata["lattice"]).sum(dim=1)
            g.ndata["pos"] = (
                g.ndata["frac_coords"].unsqueeze(dim=-1) * torch.repeat_interleave(lat, g.batch_num_nodes(), dim=0)  # type:ignore[arg-type]
            ).sum(dim=1)
        elif lat is not None:
            if lat.dim() == 2:
                lat = lat.unsqueeze(0)
            batch = getattr(g, "batch", None)
            if batch is None:
                batch = torch.zeros(g.num_nodes, dtype=torch.long, device=g.frac_coords.device)
            node_lat = lat[batch]
            g.pos = (g.frac_coords.unsqueeze(dim=-1) * node_lat).sum(dim=1)
            edge_lat = lat[batch[g.edge_index[0]]]
            g.pbc_offshift = (g.pbc_offset.unsqueeze(dim=-1) * edge_lat).sum(dim=1)
        if self.include_line_graph:
            return self.model(g=g, l_g=l_g, state_attr=state_attr)
        return self.model(g, state_attr=state_attr)

    def step(self, batch: tuple) -> tuple[dict[str, Any], int]:
        """Run a single training/validation step.

        Args:
            batch: Batch of training data.

        Returns:
            results, batch_size
        """
        if self.include_line_graph:
            g, lat, l_g, state_attr, labels = batch
            preds = self(g=g, lat=lat, l_g=l_g, state_attr=state_attr)
        else:
            g, lat, state_attr, labels = batch
            preds = self(g=g, lat=lat, state_attr=state_attr)
        results = self.loss_fn(loss=self.loss, preds=preds, labels=labels)  # type: ignore
        batch_size = preds.numel()
        return results, batch_size

    def loss_fn(self, loss: nn.Module, labels: torch.Tensor, preds: torch.Tensor) -> dict[str, Any]:
        """Compute training loss and metrics.

        Args:
            loss: Loss function.
            labels: Labels to compute the loss.
            preds: Predictions.

        Returns:
            {"Total_Loss": total_loss, "MAE": mae, "RMSE": rmse}
        """
        scaled_pred = torch.reshape(preds * self.data_std + self.data_mean, labels.size())
        total_loss = loss(labels, scaled_pred, **self.loss_params)
        mae = self.mae(labels, scaled_pred)
        rmse = self.rmse(labels, scaled_pred)
        return {"Total_Loss": total_loss, "MAE": mae, "RMSE": rmse}


class PotentialLightningModule(MatglLightningModuleMixin, pl.LightningModule):
    """A PyTorch.LightningModule for training MatGL potentials.

    This is slightly different from the ModelLightningModel due to the need to account for energy, forces and stress
    losses.
    """

    def __init__(
        self,
        model,
        element_refs: np.ndarray | None = None,
        include_line_graph: bool = False,
        energy_weight: float = 1.0,
        force_weight: float = 1.0,
        stress_weight: float = 0.0,
        magmom_weight: float = 0.0,
        charge_weight: float = 0.0,
        data_mean: float = 0.0,
        data_std: float = 1.0,
        loss: str = "mse_loss",
        loss_params: dict | None = None,
        optimizer: Optimizer | None = None,
        scheduler: LRScheduler | None = None,
        lr: float = 0.001,
        decay_steps: int = 1000,
        decay_alpha: float = 0.01,
        sync_dist: bool = False,
        allow_missing_labels: bool = False,
        magmom_target: Literal["absolute", "symbreak"] | None = "absolute",
        **kwargs,
    ):
        """Init PotentialLightningModule with key parameters.

        Args:
            model: Which type of the model for training
            element_refs: element offset for PES
            include_line_graph: whether to include line graphs
            energy_weight: relative importance of energy
            force_weight: relative importance of force
            stress_weight: relative importance of stress
            magmom_weight: relative importance of additional magmom predictions.
            charge_weight: relative importance of additional charge predictions.
            data_mean: average of training data
            data_std: standard deviation of training data
            loss: loss function used for training
            loss_params: parameters for loss function
            optimizer: optimizer for training
            scheduler: scheduler for training
            lr: learning rate for training
            decay_steps: number of steps for decaying learning rate
            decay_alpha: parameter determines the minimum learning rate.
            sync_dist: whether sync logging across all GPU workers or not
            allow_missing_labels: Whether to allow missing labels or not.
                These should be present in the dataset as torch.nans and will be skipped in computing the loss.
            magmom_target: Whether to predict the absolute site-wise value of magmoms or adapt the loss function
                to predict the signed value breaking symmetry. If None given the loss function will be adapted.
            **kwargs: Passthrough to parent init.
        """
        assert energy_weight >= 0, f"energy_weight has to be >=0. Got {energy_weight}!"
        assert force_weight >= 0, f"force_weight has to be >=0. Got {force_weight}!"
        assert stress_weight >= 0, f"stress_weight has to be >=0. Got {stress_weight}!"
        assert magmom_weight >= 0, f"magmom_weight has to be >=0. Got {magmom_weight}!"
        assert charge_weight >= 0, f"charge_weight has to be >=0. Got {charge_weight}!"

        super().__init__(**kwargs)

        self.mae = torchmetrics.MeanAbsoluteError()
        self.rmse = torchmetrics.MeanSquaredError(squared=False)
        self.register_buffer("data_mean", torch.tensor(data_mean))
        self.register_buffer("data_std", torch.tensor(data_std))

        self.energy_weight = energy_weight
        self.force_weight = force_weight
        self.stress_weight = stress_weight
        self.magmom_weight = magmom_weight
        self.charge_weight = charge_weight
        self.lr = lr
        self.decay_steps = decay_steps
        self.decay_alpha = decay_alpha
        self.include_line_graph = include_line_graph

        self.model = Potential(
            model=model,
            element_refs=element_refs,
            calc_stresses=stress_weight != 0,
            calc_magmom=magmom_weight != 0,
            calc_charge=charge_weight != 0,
            data_std=torch.as_tensor(self.data_std),  # type: ignore[arg-type]
            data_mean=torch.as_tensor(self.data_mean),  # type: ignore[arg-type]
        )
        if loss == "mse_loss":
            self.loss = F.mse_loss
        elif loss == "huber_loss":
            self.loss = F.huber_loss  # type:ignore[assignment]
        elif loss == "smooth_l1_loss":
            self.loss = F.smooth_l1_loss
        else:
            self.loss = F.l1_loss
        self.loss_params = loss_params if loss_params is not None else {}
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.sync_dist = sync_dist
        self.allow_missing_labels = allow_missing_labels
        self.magmom_target = magmom_target
        self._last_preds: tuple[torch.Tensor, ...] | None = None
        self._last_labels: tuple[torch.Tensor, ...] | None = None
        self._last_indices: torch.Tensor | None = None
        self._last_num_atoms: torch.Tensor | None = None
        self.save_hyperparameters(ignore=["model"])

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Add missing keys to the checkpoint state dict.

        Hacky workaround for state-dict drift when model fields are added.
        """
        for key in self.state_dict():
            if key not in checkpoint["state_dict"]:
                checkpoint["state_dict"][key] = self.state_dict()[key]

    def forward(
        self,
        g: Any,
        lat: torch.Tensor,
        l_g: Any = None,
        state_attr: torch.Tensor | None = None,
    ) -> tuple:
        """Run the wrapped potential model.

        Args:
            g: Backend graph (DGL ``DGLGraph`` or PyG ``Data``/``Batch``).
            lat: Lattice tensor.
            l_g: Optional line graph.
            state_attr: Optional state attribute.

        Returns:
            energy, force, stress, hessian and optional site_wise
        """
        if self.include_line_graph:
            if self.model.calc_magmom:
                e, f, s, h, m = self.model(g=g, lat=lat, l_g=l_g, state_attr=state_attr)
                return e, f, s, h, m
            e, f, s, h = self.model(g=g, lat=lat, l_g=l_g, state_attr=state_attr)
            return e, f, s, h
        if self.model.calc_charge:
            e, f, s, h, q = self.model(g=g, lat=lat, l_g=l_g, state_attr=state_attr)
            return e, f, s, h, q
        if self.model.calc_magmom:
            e, f, s, h, m = self.model(g=g, lat=lat, state_attr=state_attr)
            return e, f, s, h, m
        e, f, s, h = self.model(g=g, lat=lat, state_attr=state_attr)
        return e, f, s, h

    def step(self, batch: tuple) -> tuple[dict[str, Any], int]:
        """Run a single training/validation step.

        Args:
            batch: Batch of training data.

        Returns:
            results, batch_size
        """
        preds: tuple
        labels: tuple

        torch.set_grad_enabled(True)
        # Batch shape is fully determined by ``include_line_graph``:
        #   line graph: (g, lat, l_g, state_attr, energies, forces, stresses, [extra])
        #   otherwise:  (g, lat,      state_attr, energies, forces, stresses, [extra])
        # where the trailing optional ``extra`` is magmoms (calc_magmom) or
        # charges (calc_charge); ``forward`` mirrors the optional with an
        # extra return slot. ``preds`` is just the forward output with the
        # hessian (index 3) dropped, then a ``squeeze`` on the charge slot
        # to match the legacy shape contract.
        if self.include_line_graph:
            g, lat, l_g, state_attr, *targets = batch
            out = self(g=g, lat=lat, state_attr=state_attr, l_g=l_g)
        else:
            g, lat, state_attr, *targets = batch
            out = self(g=g, lat=lat, state_attr=state_attr)

        preds = (out[0], out[1], out[2], *out[4:])
        labels = tuple(targets)
        if self.model.calc_charge:
            preds = (preds[0], preds[1], preds[2], preds[3].squeeze())
            labels = (labels[0], labels[1], labels[2], labels[3].squeeze())

        num_atoms = g.batch_num_nodes() if BACKEND == "DGL" else torch.bincount(g.batch)
        results = self.loss_fn(
            loss=self.loss,  # type: ignore
            preds=preds,
            labels=labels,
            num_atoms=num_atoms,
        )
        batch_size = preds[0].numel()

        self._last_preds = preds
        self._last_labels = labels
        self._last_num_atoms = num_atoms
        if BACKEND == "DGL":
            if "sample_idx" in g.ndata:
                offsets = torch.cumsum(num_atoms, dim=0) - num_atoms
                self._last_indices = g.ndata["sample_idx"][offsets].to(torch.long)
            else:
                self._last_indices = None
        else:
            self._last_indices = getattr(g, "sample_idx", None)

        return results, batch_size

    def training_step(self, batch: tuple, batch_idx: int) -> dict[str, Any]:
        """Training step that exposes per-sample preds and labels for callbacks.

        Args:
            batch: Data batch.
            batch_idx: Batch index.

        Returns:
            Dict with ``loss`` (used by Lightning for backprop) plus the raw ``preds``,
            ``labels`` tuples and per-sample ``indices`` / ``num_atoms`` so that callbacks
            such as :class:`matgl.utils.callbacks.PredictionLogger` can place predictions in
            a stable per-sample order across shuffled epochs.
        """
        loss = super().training_step(batch, batch_idx)
        return {
            "loss": loss,
            "preds": self._last_preds,
            "labels": self._last_labels,
            "indices": self._last_indices,
            "num_atoms": self._last_num_atoms,
        }

    def validation_step(self, batch: tuple, batch_idx: int) -> dict[str, Any]:
        """Validation step that exposes per-sample preds and labels for callbacks.

        Args:
            batch: Data batch.
            batch_idx: Batch index.

        Returns:
            Dict with ``loss`` plus the raw ``preds``, ``labels`` tuples and per-sample
            ``indices`` / ``num_atoms`` (the latter only present when the dataset has been
            stamped with :func:`matgl.utils.callbacks.add_sample_indices`).
        """
        loss = super().validation_step(batch, batch_idx)
        return {
            "loss": loss,
            "preds": self._last_preds,
            "labels": self._last_labels,
            "indices": self._last_indices,
            "num_atoms": self._last_num_atoms,
        }

    def loss_fn(
        self,
        loss: nn.Module,
        labels: tuple,
        preds: tuple,
        num_atoms: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        """Compute losses for EFS.

        Args:
            loss: Loss function.
            labels: Labels.
            preds: Predictions
            num_atoms: Number of atoms.

        Returns::

            {
                "Total_Loss": total_loss,
                "Energy_MAE": e_mae,
                "Force_MAE": f_mae,
                "Stress_MAE": s_mae,
                "Magmom_MAE": m_mae,
                "Charge_MAE": q_mae,
                "Energy_RMSE": e_rmse,
                "Force_RMSE": f_rmse,
                "Stress_RMSE": s_rmse,
                "Magmom_RMSE": m_rmse,
                "Charge_RMSE": q_rmse
            }

        """
        # labels and preds are (energy, force, stress, (optional) site_wise)
        if num_atoms is None:
            num_atoms = torch.ones_like(preds[0])
        if self.allow_missing_labels:
            valid_labels, valid_preds = [], []
            for index, label in enumerate(labels):
                valid_value_indices = ~torch.isnan(label)
                valid_labels.append(label[valid_value_indices])
                if index == 0:
                    valid_num_atoms = num_atoms[valid_value_indices]
                    pred = preds[index].view(1) if preds[index].shape == torch.Size([]) else preds[index]
                else:
                    pred = preds[index]
                valid_preds.append(pred[valid_value_indices])
        else:
            valid_labels, valid_preds = list(labels), list(preds)
            valid_num_atoms = num_atoms

        # Per-atom energies are reused three times (loss, MAE, RMSE) — hoist
        # the divisions out of the metric calls so each tensor is materialised
        # once per loss_fn invocation.
        e_label_per_atom = valid_labels[0] / valid_num_atoms
        e_pred_per_atom = valid_preds[0] / valid_num_atoms

        e_loss = self.loss(e_label_per_atom, e_pred_per_atom, **self.loss_params)
        f_loss = self.loss(valid_labels[1], valid_preds[1], **self.loss_params)

        e_mae = self.mae(e_label_per_atom, e_pred_per_atom)
        f_mae = self.mae(valid_labels[1], valid_preds[1])

        e_rmse = self.rmse(e_label_per_atom, e_pred_per_atom)
        f_rmse = self.rmse(valid_labels[1], valid_preds[1])

        s_mae = torch.zeros(1)
        s_rmse = torch.zeros(1)

        m_mae = torch.zeros(1)
        m_rmse = torch.zeros(1)

        q_mae = torch.zeros(1)
        q_rmse = torch.zeros(1)

        total_loss = self.energy_weight * e_loss + self.force_weight * f_loss

        if self.model.calc_stresses:
            s_loss = loss(valid_labels[2], valid_preds[2], **self.loss_params)
            s_mae = self.mae(valid_labels[2], valid_preds[2])
            s_rmse = self.rmse(valid_labels[2], valid_preds[2])
            total_loss = total_loss + self.stress_weight * s_loss

        if self.model.calc_magmom and labels[3].numel() > 0:
            if self.magmom_target == "symbreak":
                # Each metric was being recomputed twice for the +/- predictions; cache
                # the four tensors and pick the per-element minimum at the end.
                neg_pred = -valid_preds[3]
                m_loss = torch.min(
                    loss(valid_labels[3], valid_preds[3], **self.loss_params),
                    loss(valid_labels[3], neg_pred, **self.loss_params),
                )
                m_mae = torch.min(self.mae(valid_labels[3], valid_preds[3]), self.mae(valid_labels[3], neg_pred))
                m_rmse = torch.min(self.rmse(valid_labels[3], valid_preds[3]), self.rmse(valid_labels[3], neg_pred))
            else:
                labels_3 = torch.abs(valid_labels[3]) if self.magmom_target == "absolute" else valid_labels[3]
                m_loss = loss(labels_3, valid_preds[3], **self.loss_params)
                m_mae = self.mae(labels_3, valid_preds[3])
                m_rmse = self.rmse(labels_3, valid_preds[3])
            total_loss = total_loss + self.magmom_weight * m_loss

        if self.model.calc_charge:
            q_loss = loss(labels[3], preds[3])
            q_mae = self.mae(labels[3], preds[3])
            q_rmse = self.rmse(labels[3], preds[3])
            total_loss = total_loss + self.charge_weight * q_loss

        return {
            "Total_Loss": total_loss,
            "Energy_MAE": e_mae,
            "Force_MAE": f_mae,
            "Stress_MAE": s_mae,
            "Magmom_MAE": m_mae,
            "Charge_MAE": q_mae,
            "Energy_RMSE": e_rmse,
            "Force_RMSE": f_rmse,
            "Stress_RMSE": s_rmse,
            "Magmom_RMSE": m_rmse,
            "Charge_RMSE": q_rmse,
        }


def fit_element_refs(
    structures: Iterable[Structure],
    energies: ArrayLike,
    element_types: Sequence[str],
    *,
    rcond: float | None = None,
) -> np.ndarray:
    r"""Fit per-element energy offsets via linear regression.

    Solves the least-squares problem

    .. math::

        E_i \approx \sum_{Z \in S} \mu_Z \, N_{i,Z}

    where :math:`E_i` is the total energy of structure :math:`i`,
    :math:`N_{i,Z}` is the count of element :math:`Z` in that structure,
    and :math:`\mu_Z` is the per-element offset returned in the same
    order as ``element_types``. The result is shaped to drop straight
    into :class:`PotentialLightningModule` or
    :class:`matgl.apps.pes.Potential` as ``element_refs``.

    Subtracting these offsets from the targets removes the (usually
    dominant) constant-per-element contribution from the loss so the
    model only has to learn the relative-energy surface. This stabilises
    training when the absolute energy scale (~tens of eV per atom) is
    large compared to the residual variation across a chemically
    homogeneous training set.

    Args:
        structures: Iterable of pymatgen ``Structure`` (or ``Molecule``)
            objects. Composition is read via ``site.specie.symbol``.
        energies: Total potential energies, one per structure, in any
            unit consistent with downstream training (usually eV).
        element_types: Element ordering used by the model — typically
            the value of ``model.element_types`` or what
            ``matgl.ext.pymatgen.get_element_list`` returns. The output
            offset vector is in this order.
        rcond: Forwarded to ``numpy.linalg.lstsq``. ``None`` (default)
            uses NumPy's current default cutoff for small singular
            values; pass ``-1`` to retain old behaviour, or a float to
            override.

    Returns:
        ``np.ndarray`` of shape ``(len(element_types),)`` with the fitted
        per-element offsets, dtype ``float64``.

    Raises:
        ValueError: If ``structures`` and ``energies`` have different
            lengths, or if a structure contains an element not listed in
            ``element_types``.

    Note:
        For inputs already in graph form (e.g. an
        :class:`~matgl.graph._data_pyg.MGLDataset` of PyG ``Data``
        objects), :meth:`matgl.layers.AtomRef.fit` provides the same
        regression directly on the layer.

    Example:
        >>> from matgl.ext.pymatgen import get_element_list
        >>> elements = get_element_list(structures)
        >>> refs = fit_element_refs(structures, energies, elements)
        >>> module = PotentialLightningModule(model=model, element_refs=refs)
    """
    element_types = tuple(element_types)
    if not element_types:
        raise ValueError("element_types must be non-empty.")

    z_to_col = {sym: i for i, sym in enumerate(element_types)}
    structures_list = list(structures)
    energies_arr = np.asarray(energies, dtype=np.float64).reshape(-1)

    if len(structures_list) != energies_arr.shape[0]:
        raise ValueError(
            f"len(structures)={len(structures_list)} does not match len(energies)={energies_arr.shape[0]}."
        )
    if not structures_list:
        raise ValueError("structures must be non-empty.")

    counts = np.zeros((len(structures_list), len(element_types)), dtype=np.float64)
    for i, struct in enumerate(structures_list):
        for site in struct:
            sym = site.specie.symbol
            col = z_to_col.get(sym)
            if col is None:
                raise ValueError(
                    f"Structure {i} contains element {sym!r} which is not in element_types={element_types}."
                )
            counts[i, col] += 1.0

    refs, *_ = np.linalg.lstsq(counts, energies_arr, rcond=rcond)
    return refs


def xavier_init(model: nn.Module, gain: float = 1.0, distribution: Literal["uniform", "normal"] = "uniform") -> None:
    """Xavier initialization scheme for the model.

    Args:
        model (nn.Module): The model to be Xavier-initialized.
        gain (float): Gain factor. Defaults to 1.0.
        distribution (Literal["uniform", "normal"], optional): Distribution to use. Defaults to "uniform".
    """
    if distribution == "uniform":
        init_fn = nn.init.xavier_uniform_
    elif distribution == "normal":
        init_fn = nn.init.xavier_normal_
    else:
        raise ValueError(f"Invalid distribution: {distribution}")

    for name, param in model.named_parameters():
        if name.endswith(".bias"):
            param.data.fill_(0)
        elif param.dim() < 2:  # torch.nn.xavier only supports >= 2 dim tensors
            bound = gain * math.sqrt(6) / math.sqrt(2 * param.shape[0])
            if distribution == "uniform":
                param.data.uniform_(-bound, bound)
            else:
                param.data.normal_(0, bound**2)
        else:
            init_fn(param.data, gain=gain)


# ---------------------------------------------------------------------------
# MatGLPotentialTrainer — high-level "configure once, fit when asked" wrapper.
# ---------------------------------------------------------------------------

HF_MATPES_REPO_ID = "materialyze/matpes"
VALID_SPLITS: tuple[str, ...] = ("train", "test", "valid")

# MatPES versions that ship canonical {train, test, valid} split files
# alongside the monolithic JSON. Update as new releases land.
_MATPES_VERSIONS_WITH_CANONICAL_SPLITS: frozenset[str] = frozenset({"r2SCAN-2025.2", "PBE-2025.2"})

# Filename patterns that map to the (train, test, valid) trio inside an extxyz
# tarball. We accept either ``-`` or ``_`` separators around the tag, and treat
# ``val`` as a synonym for ``valid``.
_EXTXYZ_SPLIT_PATTERNS: dict[str, re.Pattern[str]] = {
    "train": re.compile(r"[-_]train\.(?:ext)?xyz$", re.IGNORECASE),
    "test": re.compile(r"[-_]test\.(?:ext)?xyz$", re.IGNORECASE),
    "valid": re.compile(r"[-_]val(?:id)?\.(?:ext)?xyz$", re.IGNORECASE),
}


def _matpes_parse_version(version: str) -> tuple[str, str]:
    """Split a MatPES version into ``(functional_upper, version_tag)``.

    Examples:
        ``"r2SCAN-2025.2"`` -> ``("R2SCAN", "2025.2")``
        ``"PBE-2025.1"``    -> ``("PBE", "2025.1")``
    """
    if "-" not in version:
        raise ValueError(
            f"Invalid MatPES version {version!r}: expected '<functional>-<tag>' (e.g. 'r2SCAN-2025.2', 'PBE-2025.2')."
        )
    functional, _, tag = version.partition("-")
    if not functional or not tag:
        raise ValueError(f"Invalid MatPES version {version!r}: both functional and tag must be non-empty.")
    return functional.upper(), tag


def _matpes_dataset_filename(version: str, split: str | None = None) -> str:
    """Return the on-disk MatPES filename for a given version (and optional split)."""
    if split is not None and split not in VALID_SPLITS:
        raise ValueError(f"Invalid split {split!r}; expected one of {VALID_SPLITS} or None.")
    functional, tag = _matpes_parse_version(version)
    suffix = f"-{split}" if split else ""
    return f"MatPES-{functional}-{tag}{suffix}.json"


def _matpes_atomrefs_filename(version: str) -> str:
    """Return the MatPES atomrefs filename (functional-specific only)."""
    functional, _ = _matpes_parse_version(version)
    return f"MatPES-{functional}-atoms.json"


def _matpes_samples_to_lists(
    samples: Iterable[Mapping],
) -> tuple[list[Structure], list[float], list, list]:
    """Walk a MatPES sample list and return parallel (structures, energies, forces, stresses) lists."""
    structures: list[Structure] = []
    energies: list[float] = []
    forces: list = []
    stresses: list = []
    for raw in samples:
        struct = raw["structure"]
        if not isinstance(struct, Structure):
            struct = Structure.from_dict(struct)
        structures.append(struct)
        energies.append(float(raw["energy"]))
        forces.append(np.asarray(raw["forces"], dtype="float64").tolist())
        stresses.append(np.asarray(raw["stress"], dtype="float64").tolist())
    return structures, energies, forces, stresses


def _resolve_cache_dir(cache_dir: str | Path | None) -> str:
    return str(cache_dir) if cache_dir is not None else str(MATGL_CACHE)


def _classify_extxyz_split(filename: str) -> str | None:
    """Return ``"train"`` / ``"test"`` / ``"valid"`` for a filename, or ``None`` if none match."""
    for tag, pattern in _EXTXYZ_SPLIT_PATTERNS.items():
        if pattern.search(filename):
            return tag
    return None


def _list_extxyz_files_in_tar(tar_path: Path) -> list[str]:
    """Return the names of ``.extxyz`` / ``.xyz`` members inside a tarball."""
    with tarfile.open(tar_path, "r:*") as tar:
        return [m.name for m in tar.getmembers() if m.isfile() and m.name.lower().endswith((".extxyz", ".xyz"))]


def _read_extxyz_file(path: str | Path) -> tuple[list[Structure], dict[str, list]]:
    """Read an ``.extxyz`` / ``.xyz`` file with ASE and return (structures, labels).

    The ``labels`` dict has keys ``"energies"``, ``"forces"`` always; ``"stresses"``
    only when every frame in the file exposes a stress (otherwise the key is dropped
    so :func:`MGLDataLoader` auto-detects ``include_stress=False``).
    """
    from ase.calculators.calculator import PropertyNotImplementedError
    from ase.io import read as ase_read
    from pymatgen.io.ase import AseAtomsAdaptor

    frames = ase_read(str(path), index=":")
    if not isinstance(frames, list):
        frames = [frames]

    structures: list[Structure] = []
    energies: list[float] = []
    forces: list = []
    stresses: list = []
    has_stress = True
    for atoms in frames:
        structures.append(AseAtomsAdaptor.get_structure(atoms))
        energies.append(float(atoms.get_potential_energy()))
        forces.append(np.asarray(atoms.get_forces(), dtype="float64").tolist())
        if has_stress:
            try:
                stresses.append(np.asarray(atoms.get_stress(voigt=False), dtype="float64").tolist())
            except (PropertyNotImplementedError, KeyError):
                has_stress = False
                stresses = []

    labels: dict[str, list] = {"energies": energies, "forces": forces}
    if has_stress and stresses:
        labels["stresses"] = stresses
    return structures, labels


def _read_extxyz_source(source: str | Path) -> dict[str, tuple[list[Structure], dict[str, list]]]:
    """Read a single extxyz / xyz file or every extxyz file inside a tarball.

    Returns a mapping ``{member_name: (structures, labels)}``. For a non-tar
    source the single member is keyed by the file's basename.
    """
    src = Path(source)
    if not src.exists():
        raise FileNotFoundError(f"extxyz source not found: {src}")

    suffixes = "".join(src.suffixes).lower()
    if suffixes.endswith((".tar.gz", ".tgz", ".tar")):
        members = _list_extxyz_files_in_tar(src)
        if not members:
            raise ValueError(f"No .extxyz / .xyz members found inside {src}")
        out: dict[str, tuple[list[Structure], dict[str, list]]] = {}
        with tempfile.TemporaryDirectory() as tmpdir:
            with tarfile.open(src, "r:*") as tar:
                tar.extractall(tmpdir)
            for member in members:
                out[member] = _read_extxyz_file(Path(tmpdir) / member)
        return out

    if suffixes.endswith((".extxyz", ".xyz")):
        return {src.name: _read_extxyz_file(src)}

    raise ValueError(
        f"Unsupported extxyz source {src.name!r}: expected one of '*.extxyz', '*.xyz', '*.tar.gz', '*.tgz', '*.tar'."
    )


def _build_pes_dataset(
    structures: Sequence[Structure],
    labels: Mapping[str, list],
    *,
    cutoff: float,
    element_types: tuple[str, ...] | None,
    save_cache: bool,
    root: str | None,
) -> MGLDataset:
    """Construct an ``MGLDataset`` from parallel structure / label lists."""
    if BACKEND != "PYG":
        raise RuntimeError(
            "MatGLPotentialTrainer dataset loaders require the PyG backend. "
            "Set MATGL_BACKEND=PYG before importing matgl."
        )
    # Lazy imports to avoid circulars (``matgl.utils.training`` is foundational).
    from matgl.ext.pymatgen import Structure2Graph, get_element_list
    from matgl.graph.data import MGLDataset

    if element_types is None:
        element_types = get_element_list(list(structures))
    converter = Structure2Graph(element_types=element_types, cutoff=cutoff)
    ds_kwargs: dict = {
        "structures": list(structures),
        "converter": converter,
        "labels": dict(labels),
        "save_cache": save_cache,
    }
    if root is not None:
        ds_kwargs["root"] = root
    dataset = MGLDataset(**ds_kwargs)
    dataset.element_types = element_types  # type: ignore[attr-defined]
    return dataset


def _resolve_to_local_path(spec: Any, *, cache_dir: str | Path | None = None) -> Path:
    """Resolve a HF dataset spec or local path to a concrete file path.

    Accepted shapes:

    - ``str`` / ``Path``: treated as a local file path.
    - ``tuple[str, str]``: ``(repo_id, filename)``; downloaded via ``hf_hub_download``.
    - ``Mapping`` with at least ``"repo_id"`` and ``"filename"`` keys: forwarded
      verbatim to ``hf_hub_download`` (so ``revision``, ``token``, ``cache_dir``,
      ``force_download``, etc. flow through).
    """
    if isinstance(spec, (str, Path)):
        return Path(spec)
    if isinstance(spec, tuple):
        if len(spec) != 2 or not all(isinstance(x, str) for x in spec):
            raise ValueError(f"Tuple HF spec must be (repo_id: str, filename: str), got {spec!r}.")
        repo_id, filename = spec
        return Path(hf_hub_download(repo_id=repo_id, filename=filename, cache_dir=_resolve_cache_dir(cache_dir)))
    if isinstance(spec, Mapping):
        kwargs = dict(spec)
        if "repo_id" not in kwargs or "filename" not in kwargs:
            raise ValueError(f"Mapping HF spec must contain 'repo_id' and 'filename'; got keys {sorted(kwargs)}.")
        if "cache_dir" not in kwargs:
            kwargs["cache_dir"] = _resolve_cache_dir(cache_dir)
        return Path(hf_hub_download(**kwargs))
    raise ValueError(f"Cannot resolve {spec!r} as a HF spec (tuple/dict) or local path (str/Path).")


def _detect_data_format(name: str) -> str:
    """Infer ``"matpes"`` or ``"extxyz"`` from a filename suffix."""
    lower = name.lower()
    if lower.endswith((".extxyz", ".xyz", ".tar.gz", ".tgz", ".tar")):
        return "extxyz"
    if lower.endswith((".json.gz", ".json")):
        return "matpes"
    raise ValueError(
        f"Cannot auto-detect data format from {name!r}. Supported: "
        f"'.json'/'.json.gz' (matpes), '.extxyz'/'.xyz'/'.tar.gz'/'.tgz'/'.tar' (extxyz). "
        f"Pass format='matpes' or format='extxyz' explicitly."
    )


def _read_matpes_dataset_local(
    local_path: str | Path,
    *,
    cutoff: float = 5.0,
    element_types: tuple[str, ...] | None = None,
    save_cache: bool = True,
    root: str | None = None,
) -> MGLDataset:
    """Read a (locally cached) MatPES JSON file and build an ``MGLDataset``."""
    payload = loadfn(str(local_path))
    structures, energies, forces, stresses = _matpes_samples_to_lists(payload["samples"])
    return _build_pes_dataset(
        structures,
        {"energies": energies, "forces": forces, "stresses": stresses},
        cutoff=cutoff,
        element_types=element_types,
        save_cache=save_cache,
        root=root,
    )


def _read_extxyz_dataset_local(
    local_path: str | Path,
    *,
    cutoff: float = 5.0,
    element_types: tuple[str, ...] | None = None,
    save_cache: bool = True,
    root: str | None = None,
) -> MGLDataset:
    """Read a (locally cached) extxyz / xyz / tarball and build a single ``MGLDataset``.

    Tarballs with multiple members have all frames concatenated.
    """
    members = _read_extxyz_source(Path(local_path))
    all_structures: list[Structure] = []
    all_energies: list[float] = []
    all_forces: list = []
    all_stresses: list = []
    has_stress_everywhere = True
    for structures, labels in members.values():
        all_structures.extend(structures)
        all_energies.extend(labels["energies"])
        all_forces.extend(labels["forces"])
        if "stresses" in labels:
            all_stresses.extend(labels["stresses"])
        else:
            has_stress_everywhere = False
    merged: dict[str, list] = {"energies": all_energies, "forces": all_forces}
    if has_stress_everywhere and all_stresses:
        merged["stresses"] = all_stresses
    return _build_pes_dataset(
        all_structures,
        merged,
        cutoff=cutoff,
        element_types=element_types,
        save_cache=save_cache,
        root=root,
    )


def _dataset_has_stresses(dataset: Any) -> bool:
    """Return True iff the dataset (single or splits dict) carries a ``"stresses"`` label.

    Falls back to True (the safe default — keep stress in the loss) for shapes we
    can't introspect, so the warning path only triggers when we're certain stresses
    are absent.
    """
    from matgl.graph.data import MGLDataset

    if isinstance(dataset, MGLDataset):
        return "stresses" in getattr(dataset, "labels", {})
    if isinstance(dataset, Mapping):
        for value in dataset.values():
            if isinstance(value, MGLDataset):
                return "stresses" in getattr(value, "labels", {})
            return True  # unknown shape inside the dict — don't second-guess
    return True


def _atomrefs_array_from_payload(
    payload: Mapping[str, Any],
    model_element_types: tuple[str, ...] | None,
    *,
    source_for_error: str = "atomrefs payload",
) -> np.ndarray:
    """Convert a ``{"element_types": [...], "refs": [...]}`` mapping into a numpy array.

    When ``model_element_types`` is supplied the array is reordered so
    ``out[i]`` is the offset for ``model_element_types[i]``.
    """
    if "element_types" not in payload or "refs" not in payload:
        raise KeyError(f"atomrefs payload must contain 'element_types' and 'refs' keys; got {sorted(payload.keys())}.")
    file_elements: list[str] = list(payload["element_types"])
    refs = np.asarray(payload["refs"], dtype="float64")
    if model_element_types is None:
        return refs
    index_of = {sym: i for i, sym in enumerate(file_elements)}
    missing = [sym for sym in model_element_types if sym not in index_of]
    if missing:
        raise KeyError(f"Element(s) {missing} not present in {source_for_error}. File covers: {file_elements}.")
    return np.asarray([refs[index_of[sym]] for sym in model_element_types], dtype="float64")


class MatGLPotentialTrainer:
    r"""Configure-once / fit-when-asked trainer for matgl ``Potential`` training.

    ``__init__`` stores hyperparameters but does not download data, build a
    dataset, or instantiate Lightning. The first network / disk activity
    happens inside :meth:`fit`.

    Dataset-loading utilities are :class:`staticmethod`\ s so callers can grab
    raw data without configuring a trainer:

    >>> ds = MatGLPotentialTrainer.load_matpes_dataset(version="r2SCAN-2025.2", split="test")
    >>> # or for non-MatPES sources:
    >>> ds = MatGLPotentialTrainer.load_extxyz_dataset(
    ...     repo_id="materialyze/mlip-lr-benchmarks", filename="cp_dimer.tar.gz"
    ... )

    A typical end-to-end MatPES call (HF download driven by ``fit``):

    >>> trainer = MatGLPotentialTrainer(model, accelerator="gpu")
    >>> potential = trainer.fit(
    ...     dataset=("materialyze/matpes", "MatPES-R2SCAN-2025.2.json"),
    ...     atomrefs=("materialyze/matpes", "MatPES-R2SCAN-atoms.json"),
    ...     format="auto",   # infers 'matpes' from .json
    ... )
    >>> trainer.save("./MatPES-TensorNet")

    The trainer is PyG-only; DGL is being deprecated. The class itself
    imports cleanly under DGL but its loader methods and ``fit`` raise
    informatively when called.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        # Loss term weights.
        energy_weight: float = 1.0,
        force_weight: float = 1.0,
        stress_weight: float = 0.1,
        loss: str = "huber_loss",
        loss_params: dict | None = None,
        # Optimizer / scheduler defaults.
        lr: float = 1e-3,
        decay_steps: int = 1000,
        decay_alpha: float = 0.01,
        # DataLoader defaults.
        batch_size: int = 32,
        # pl.Trainer placement controls.
        max_epochs: int = 100,
        accelerator: str = "auto",
        devices: int | str = "auto",
        seed: int = 42,
        # Pass-through escape hatches.
        trainer_kwargs: dict | None = None,
        loader_kwargs: dict | None = None,
    ) -> None:
        """Initialise the trainer.

        Args:
            model: The graph model to wrap (e.g. ``TensorNet(...)``,
                ``GRACE(...)``, ``M3GNet(...)``). Must already be configured
                with the ``element_types`` and ``cutoff`` you want to train.
            energy_weight: Energy loss weight.
            force_weight: Force loss weight.
            stress_weight: Stress loss weight. Set to ``0`` for datasets
                without stress labels (e.g. cluster / dimer extxyz files).
            loss: One of ``"mse_loss"``, ``"huber_loss"`` (default; robust),
                ``"smooth_l1_loss"``, or ``"l1_loss"``.
            loss_params: Optional kwargs forwarded to the loss function (e.g.
                ``{"delta": 0.1}`` for Huber).
            lr: Initial learning rate.
            decay_steps: ``CosineAnnealingLR`` ``T_max``.
            decay_alpha: Minimum-LR multiplier (``eta_min = lr * decay_alpha``).
            batch_size: Per-loader batch size.
            max_epochs: ``pl.Trainer`` max epochs.
            accelerator: ``pl.Trainer`` accelerator. Accepts any value
                Lightning accepts: ``"auto"`` (default), ``"cpu"``, ``"gpu"``,
                ``"cuda"``, ``"mps"``, ``"tpu"``.
            devices: ``pl.Trainer`` device count or selector (e.g. ``1``,
                ``"auto"``, ``[0, 1]``).
            seed: Forwarded to ``pl.seed_everything(workers=True)`` at fit time.
            trainer_kwargs: Extra ``pl.Trainer`` kwargs (e.g. ``callbacks``,
                ``logger``).
            loader_kwargs: Extra kwargs forwarded to :class:`MGLDataLoader` /
                :func:`split_dataset` via :meth:`_build_dataloaders`.
                Recognised split-only keys: ``frac_list``, ``shuffle``,
                ``random_state``.
        """
        self.model = model

        self.energy_weight = energy_weight
        self.force_weight = force_weight
        self.stress_weight = stress_weight
        self.loss = loss
        self.loss_params = loss_params

        self.lr = lr
        self.decay_steps = decay_steps
        self.decay_alpha = decay_alpha

        self.batch_size = batch_size

        self.max_epochs = max_epochs
        self.accelerator = accelerator
        self.devices = devices
        self.seed = seed

        self.trainer_kwargs = dict(trainer_kwargs or {})
        self.loader_kwargs = dict(loader_kwargs or {})

        # Populated by ``fit``; ``None`` until then.
        self.dataset: MGLDataset | Mapping[str, MGLDataset] | None = None
        self.loaders: dict[str, DataLoader] | None = None
        self.lit_module: PotentialLightningModule | None = None
        self.trainer: pl.Trainer | None = None
        self.potential: Potential | None = None
        self.atomrefs: np.ndarray | None = None

    # ------------------------------------------------------------------
    # MatPES dataset loaders (HF: materialyze/matpes).
    # ------------------------------------------------------------------

    @staticmethod
    def load_matpes_dataset(
        version: str = "r2SCAN-2025.2",
        *,
        split: str | None = None,
        cutoff: float = 5.0,
        element_types: tuple[str, ...] | None = None,
        repo_id: str = HF_MATPES_REPO_ID,
        revision: str | None = None,
        token: str | None = None,
        cache_dir: str | Path | None = None,
        save_cache: bool = True,
        root: str | None = None,
    ) -> MGLDataset:
        """Download a MatPES JSON file from HF and build an ``MGLDataset``.

        Args:
            version: MatPES version, e.g. ``"r2SCAN-2025.2"`` (case-insensitive).
            split: ``None`` for the monolithic file, or one of ``"train"`` /
                ``"test"`` / ``"valid"`` for the canonical splits (v2025.2+).
            cutoff: Neighbour cutoff (Å) handed to ``Structure2Graph``.
            element_types: Optional explicit ordering; auto-derived when None.
            repo_id: HF Hub repo id; override only for staging or forks.
            revision: Optional branch / tag / commit for ``hf_hub_download``.
            token: Optional HF auth token.
            cache_dir: HF Hub download cache dir; defaults to ``MATGL_CACHE``.
            save_cache: Whether ``MGLDataset`` persists its processed cache.
            root: ``MGLDataset`` root directory; default lets it pick.

        Returns:
            An ``MGLDataset`` ready to drop into ``MGLDataLoader``. The
            monolithic files are 1.6-2.4 GB; for memory-constrained
            environments use ``split="train"`` / ``"test"`` / ``"valid"``.
        """
        filename = _matpes_dataset_filename(version, split)
        local_path = _resolve_to_local_path(
            {"repo_id": repo_id, "filename": filename, "revision": revision, "token": token},
            cache_dir=cache_dir,
        )
        return _read_matpes_dataset_local(
            local_path,
            cutoff=cutoff,
            element_types=element_types,
            save_cache=save_cache,
            root=root,
        )

    @staticmethod
    def load_matpes_splits(
        version: str = "r2SCAN-2025.2",
        *,
        cutoff: float = 5.0,
        element_types: tuple[str, ...] | None = None,
        repo_id: str = HF_MATPES_REPO_ID,
        revision: str | None = None,
        token: str | None = None,
        cache_dir: str | Path | None = None,
        save_cache: bool = True,
    ) -> dict[str, MGLDataset]:
        """Download the canonical ``{train, test, valid}`` MatPES split trio.

        ``element_types`` is computed from the union of all three split's
        structures (when not explicitly supplied) so the three datasets share
        a single ordering compatible with one model.
        """
        if version not in _MATPES_VERSIONS_WITH_CANONICAL_SPLITS:
            raise ValueError(
                f"MatPES version {version!r} does not have canonical splits on the Hub. "
                f"Use load_matpes_dataset(version, split=None) and split locally, or pick "
                f"one of {sorted(_MATPES_VERSIONS_WITH_CANONICAL_SPLITS)}."
            )

        if element_types is None:
            all_structures: list[Structure] = []
            for sp in VALID_SPLITS:
                local_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=_matpes_dataset_filename(version, sp),
                    revision=revision,
                    token=token,
                    cache_dir=_resolve_cache_dir(cache_dir),
                )
                payload = loadfn(str(local_path))
                structs, _, _, _ = _matpes_samples_to_lists(payload["samples"])
                all_structures.extend(structs)
            from matgl.ext.pymatgen import get_element_list

            element_types = get_element_list(all_structures)

        return {
            sp: MatGLPotentialTrainer.load_matpes_dataset(
                version=version,
                split=sp,
                cutoff=cutoff,
                element_types=element_types,
                repo_id=repo_id,
                revision=revision,
                token=token,
                cache_dir=cache_dir,
                save_cache=save_cache,
            )
            for sp in VALID_SPLITS
        }

    @staticmethod
    def load_matpes_element_refs(
        version: str = "r2SCAN-2025.2",
        *,
        repo_id: str = HF_MATPES_REPO_ID,
        revision: str | None = None,
        token: str | None = None,
        cache_dir: str | Path | None = None,
        element_types: tuple[str, ...] | None = None,
    ) -> np.ndarray:
        """Download per-element energy offsets shipped alongside MatPES.

        File schema: ``{"element_types": [...], "refs": [...]}``. When
        ``element_types`` is supplied, the returned vector is reordered so
        ``refs[i]`` is the offset for ``element_types[i]``.
        """
        filename = _matpes_atomrefs_filename(version)
        local_path = _resolve_to_local_path(
            {"repo_id": repo_id, "filename": filename, "revision": revision, "token": token},
            cache_dir=cache_dir,
        )
        return _atomrefs_array_from_payload(
            loadfn(str(local_path)),
            element_types,
            source_for_error=filename,
        )

    # ------------------------------------------------------------------
    # extxyz dataset loaders (any HF / local source).
    # ------------------------------------------------------------------

    @staticmethod
    def load_extxyz_dataset(
        path: str | Path | None = None,
        *,
        repo_id: str | None = None,
        filename: str | None = None,
        revision: str | None = None,
        token: str | None = None,
        cache_dir: str | Path | None = None,
        cutoff: float = 5.0,
        element_types: tuple[str, ...] | None = None,
        save_cache: bool = True,
        root: str | None = None,
    ) -> MGLDataset:
        """Load an extxyz-format dataset into a single ``MGLDataset``.

        Provide either ``path`` (a local ``.extxyz`` / ``.xyz`` /
        ``.tar.gz`` / ``.tgz`` / ``.tar``), or both ``repo_id`` and
        ``filename`` to fetch from the Hugging Face Hub. Tarballs containing
        multiple extxyz files have all frames concatenated; use
        :meth:`load_extxyz_splits` to recover canonical train/test/valid
        splits when filenames carry the appropriate suffixes.

        ``stresses`` are included in the labels only when every frame in the
        source exposes a stress tensor; cluster / dimer datasets like
        ``cp_dimer.tar.gz`` therefore yield a forces-only dataset, and the
        :class:`MGLDataLoader` auto-detect picks ``include_stress=False``.

        Args:
            path: Local file path. Mutually exclusive with ``repo_id`` /
                ``filename``.
            repo_id: HF Hub repo id (e.g. ``"materialyze/mlip-lr-benchmarks"``).
            filename: HF Hub filename (e.g. ``"cp_dimer.tar.gz"``).
            revision: Optional HF branch / tag / commit.
            token: Optional HF auth token.
            cache_dir: HF Hub cache dir; defaults to ``MATGL_CACHE``.
            cutoff: Neighbour cutoff (Å) handed to ``Structure2Graph``.
            element_types: Optional explicit ordering; auto-derived when None.
            save_cache: Whether ``MGLDataset`` persists its processed cache.
            root: ``MGLDataset`` root directory; default lets it pick.
        """
        local_path = MatGLPotentialTrainer._resolve_local_path(
            path, repo_id=repo_id, filename=filename, revision=revision, token=token, cache_dir=cache_dir
        )
        return _read_extxyz_dataset_local(
            local_path,
            cutoff=cutoff,
            element_types=element_types,
            save_cache=save_cache,
            root=root,
        )

    @staticmethod
    def load_extxyz_splits(
        path: str | Path | None = None,
        *,
        repo_id: str | None = None,
        filename: str | None = None,
        revision: str | None = None,
        token: str | None = None,
        cache_dir: str | Path | None = None,
        cutoff: float = 5.0,
        element_types: tuple[str, ...] | None = None,
        save_cache: bool = True,
    ) -> dict[str, MGLDataset]:
        """Load split extxyz files from a tarball into a ``{train, test, valid}`` dict.

        Files are matched by filename: a member ending in ``-train.extxyz`` /
        ``_train.xyz`` (or analogous patterns for ``test`` / ``valid`` /
        ``val``) is bound to the corresponding split. Members that don't match
        any pattern are ignored. Splits without a matching file are absent
        from the returned mapping.

        ``element_types`` is computed across the union of all matched splits
        when not supplied, so the three datasets share an ordering.
        """
        local_path = MatGLPotentialTrainer._resolve_local_path(
            path, repo_id=repo_id, filename=filename, revision=revision, token=token, cache_dir=cache_dir
        )
        members = _read_extxyz_source(local_path)

        # Bucket members by detected split.
        bucketed: dict[str, list[tuple[list[Structure], dict[str, list]]]] = {sp: [] for sp in VALID_SPLITS}
        for member_name, payload in members.items():
            tag = _classify_extxyz_split(member_name)
            if tag is not None:
                bucketed[tag].append(payload)

        present = {sp: items for sp, items in bucketed.items() if items}
        if not present:
            raise ValueError(
                f"No files inside {local_path} matched a known split pattern "
                f"(*-train, *-test, *-valid|val). Use load_extxyz_dataset for a single "
                f"concatenated dataset."
            )

        if element_types is None:
            from matgl.ext.pymatgen import get_element_list

            all_structures: list[Structure] = []
            for items in present.values():
                for split_structures, _ in items:
                    all_structures.extend(split_structures)
            element_types = get_element_list(all_structures)

        out: dict[str, MGLDataset] = {}
        for sp, items in present.items():
            sp_structures: list[Structure] = []
            energies: list[float] = []
            forces: list = []
            stresses: list = []
            has_stress = True
            for st, lb in items:
                sp_structures.extend(st)
                energies.extend(lb["energies"])
                forces.extend(lb["forces"])
                if "stresses" in lb:
                    stresses.extend(lb["stresses"])
                else:
                    has_stress = False
                    stresses = []
            labels: dict[str, list] = {"energies": energies, "forces": forces}
            if has_stress and stresses:
                labels["stresses"] = stresses
            out[sp] = _build_pes_dataset(
                sp_structures,
                labels,
                cutoff=cutoff,
                element_types=element_types,
                save_cache=save_cache,
                root=None,
            )
        return out

    @staticmethod
    def _resolve_local_path(
        path: str | Path | None,
        *,
        repo_id: str | None,
        filename: str | None,
        revision: str | None,
        token: str | None,
        cache_dir: str | Path | None,
    ) -> Path:
        """Resolve the user-supplied source to a concrete local path."""
        if path is not None and (repo_id is not None or filename is not None):
            raise ValueError("Pass either 'path' (local) or 'repo_id'+'filename' (HF), not both.")
        if path is not None:
            return Path(path)
        if repo_id is None or filename is None:
            raise ValueError("Provide 'path', or both 'repo_id' and 'filename' for HF download.")
        return Path(
            hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                revision=revision,
                token=token,
                cache_dir=_resolve_cache_dir(cache_dir),
            )
        )

    # ------------------------------------------------------------------
    # Internal helpers.
    # ------------------------------------------------------------------

    def _build_dataloaders(
        self,
        dataset: MGLDataset | Mapping[str, MGLDataset],
    ) -> tuple[DataLoader, DataLoader, DataLoader]:
        """Build the ``(train, val, test)`` triple from a single dataset or a splits mapping."""
        if BACKEND != "PYG":
            raise RuntimeError("MatGLPotentialTrainer requires the PyG backend.")
        from matgl.graph.data import MGLDataLoader, MGLDataset, split_dataset

        loader_kwargs = dict(self.loader_kwargs)
        # ``frac_list``, ``shuffle``, ``random_state`` are split-only knobs;
        # peel them out so they don't get forwarded to MGLDataLoader.
        frac_list = tuple(loader_kwargs.pop("frac_list", (0.9, 0.05, 0.05)))
        shuffle = loader_kwargs.pop("shuffle", True)
        random_state = loader_kwargs.pop("random_state", self.seed)

        if isinstance(dataset, MGLDataset):
            train_data, val_data, test_data = split_dataset(
                dataset,
                frac_list=list(frac_list),
                shuffle=shuffle,
                random_state=random_state,
            )
            return MGLDataLoader(
                train_data=train_data,
                val_data=val_data,
                test_data=test_data,
                batch_size=self.batch_size,
                **loader_kwargs,
            )

        splits = cast("Mapping[str, MGLDataset]", dataset)
        try:
            train_ds = splits["train"]
            val_ds = splits["valid"]
            test_ds = splits["test"]
        except KeyError as err:
            raise KeyError(
                f"Canonical-splits mapping must contain 'train', 'valid', and 'test' keys; got {sorted(splits.keys())}."
            ) from err
        return MGLDataLoader(
            train_data=train_ds,
            val_data=val_ds,
            test_data=test_ds,
            batch_size=self.batch_size,
            **loader_kwargs,
        )

    # ------------------------------------------------------------------
    # Dataset / atomrefs resolution (used by ``fit``).
    # ------------------------------------------------------------------

    def _resolve_dataset(
        self,
        dataset: Any,
        *,
        format: Literal["auto", "matpes", "extxyz"] = "auto",
    ) -> MGLDataset | Mapping[str, MGLDataset]:
        """Coerce a user-supplied ``dataset`` argument into a usable form.

        Pre-built ``MGLDataset`` and pre-built canonical-splits mappings are
        passed through. HF specs (``(repo_id, filename)`` tuples or mappings
        with ``"repo_id"`` + ``"filename"``) and local paths are downloaded
        (if needed) and parsed by the loader picked from ``format``. ``"auto"``
        infers from the filename suffix (``.json`` / ``.json.gz`` -> matpes;
        ``.extxyz`` / ``.xyz`` / ``.tar.gz`` / ``.tgz`` / ``.tar`` -> extxyz).
        """
        from matgl.graph.data import MGLDataset

        if isinstance(dataset, MGLDataset):
            return dataset
        # A mapping with "repo_id" is an HF spec, not a splits dict.
        if isinstance(dataset, Mapping) and "repo_id" not in dataset:
            return cast("Mapping[str, MGLDataset]", dataset)

        local_path = _resolve_to_local_path(dataset)
        fmt: str = format
        if fmt == "auto":
            fmt = _detect_data_format(local_path.name)
        # ``save_cache=False`` here so an in-fit dataset doesn't pollute the
        # shared default ``./MGLDataset/`` cache between fits or between
        # otherwise-isolated tests. Users wanting on-disk caching should
        # pre-build via :meth:`load_matpes_dataset` / :meth:`load_extxyz_dataset`.
        if fmt == "matpes":
            return _read_matpes_dataset_local(local_path, save_cache=False)
        if fmt == "extxyz":
            return _read_extxyz_dataset_local(local_path, save_cache=False)
        raise ValueError(f"Unknown dataset format {fmt!r}; expected 'auto', 'matpes', or 'extxyz'.")

    def _resolve_atomrefs(self, atomrefs: Any) -> np.ndarray | None:
        """Coerce a user-supplied ``atomrefs`` argument into a numpy array (or ``None``).

        Accepted shapes:

        - ``None``: no offsets.
        - ``np.ndarray``: used as-is (assumed to be in ``model.element_types`` order).
        - ``AtomRef`` instance: ``property_offset`` extracted as a numpy array.
        - ``Mapping`` with ``"element_types"`` + ``"refs"``: parsed as an in-memory
          payload and reordered to ``model.element_types``.
        - HF spec (``tuple[str, str]`` or ``Mapping`` with ``"repo_id"`` +
          ``"filename"``): downloaded and parsed as a JSON payload.
        - ``str`` / ``Path``: local JSON file with the same payload schema.
        """
        if atomrefs is None:
            return None
        if isinstance(atomrefs, np.ndarray):
            return atomrefs.astype("float64", copy=False)
        # Lazy import to avoid pulling layers at module import time.
        from matgl.layers._atom_ref_pyg import AtomRef as _AtomRef

        if isinstance(atomrefs, _AtomRef):
            return atomrefs.property_offset.detach().cpu().numpy().astype("float64")

        model_element_types: tuple[str, ...] | None = getattr(self.model, "element_types", None)

        if isinstance(atomrefs, Mapping) and "element_types" in atomrefs and "refs" in atomrefs:
            return _atomrefs_array_from_payload(
                atomrefs, model_element_types, source_for_error="in-memory atomrefs dict"
            )

        # Otherwise treat as a HF spec or local-path; download / read JSON.
        local_path = _resolve_to_local_path(atomrefs)
        return _atomrefs_array_from_payload(
            loadfn(str(local_path)), model_element_types, source_for_error=str(local_path.name)
        )

    # ------------------------------------------------------------------
    # Public training entry point.
    # ------------------------------------------------------------------

    def fit(
        self,
        dataset: Any,
        *,
        format: Literal["auto", "matpes", "extxyz"] = "auto",
        atomrefs: Any = None,
        save_path: str | Path | None = None,
        push_to_hub: str | None = None,
    ) -> Potential:
        """Run training end-to-end.

        Args:
            dataset: A HF dataset definition or a pre-built dataset. Accepted
                shapes:

                - ``(repo_id, filename)`` tuple: downloaded from HF and parsed
                  according to ``format``.
                - ``Mapping`` with ``"repo_id"`` + ``"filename"`` (and optional
                  ``"revision"`` / ``"token"`` / ``"cache_dir"``): same as
                  above with extra ``hf_hub_download`` kwargs.
                - ``str`` / ``Path`` to a local file: parsed according to
                  ``format``.
                - Pre-built :class:`MGLDataset` (used directly).
                - Pre-built canonical-splits ``Mapping[str, MGLDataset]``
                  with keys ``"train"`` / ``"valid"`` / ``"test"``.
            format: One of ``"auto"`` (infer from filename suffix; the default),
                ``"matpes"`` (force the MatPES JSON loader), or ``"extxyz"``
                (force the extxyz / tarball loader). Ignored when ``dataset``
                is already an :class:`MGLDataset` or splits mapping.
            atomrefs: Optional per-element energy offsets. Accepted shapes:

                - ``None``: no offsets (default).
                - ``np.ndarray``: already in ``model.element_types`` order.
                - :class:`AtomRef` instance: the layer's ``property_offset``
                  is extracted.
                - ``Mapping`` with ``"element_types"`` + ``"refs"``: in-memory
                  atomrefs payload; reordered to ``model.element_types``.
                - HF spec (tuple or mapping with ``"repo_id"`` + ``"filename"``)
                  or local path: downloaded / read as a JSON payload of the
                  same shape and reordered.
            save_path: If given, ``potential.save(save_path)`` after training.
            push_to_hub: If given, ``potential.push_to_hub(push_to_hub)``
                after training.

        Returns:
            The trained :class:`~matgl.apps.pes.Potential`. Also reachable as
            ``self.potential``; auxiliary state (``self.lit_module``,
            ``self.trainer``, ``self.loaders``, ``self.dataset``,
            ``self.atomrefs``) is updated.
        """
        if BACKEND != "PYG":
            raise RuntimeError("MatGLPotentialTrainer.fit requires the PyG backend.")
        pl.seed_everything(self.seed, workers=True)

        resolved_dataset = self._resolve_dataset(dataset, format=format)
        resolved_atomrefs = self._resolve_atomrefs(atomrefs)

        self.dataset = resolved_dataset
        self.atomrefs = resolved_atomrefs

        # If the dataset has no stress labels (e.g. cluster / dimer extxyz),
        # the stress loss term can't be computed; auto-disable it for this fit
        # so the trainer-level default (``stress_weight=0.1``) doesn't blow up
        # with a shape mismatch deep inside ``collate_fn_pes`` / torchmetrics.
        effective_stress_weight = self.stress_weight
        if effective_stress_weight > 0 and not _dataset_has_stresses(resolved_dataset):
            warnings.warn(
                "Dataset has no stress labels; disabling stress loss term "
                f"(stress_weight={self.stress_weight} -> 0) for this fit.",
                stacklevel=2,
            )
            effective_stress_weight = 0.0

        train_loader, val_loader, test_loader = self._build_dataloaders(resolved_dataset)
        self.loaders = {"train": train_loader, "val": val_loader, "test": test_loader}

        self.lit_module = PotentialLightningModule(
            model=self.model,
            element_refs=resolved_atomrefs,
            energy_weight=self.energy_weight,
            force_weight=self.force_weight,
            stress_weight=effective_stress_weight,
            loss=self.loss,
            loss_params=self.loss_params,
            lr=self.lr,
            decay_steps=self.decay_steps,
            decay_alpha=self.decay_alpha,
        )
        self.trainer = pl.Trainer(
            max_epochs=self.max_epochs,
            accelerator=self.accelerator,
            devices=self.devices,
            inference_mode=False,
            **self.trainer_kwargs,
        )
        self.trainer.fit(model=self.lit_module, train_dataloaders=train_loader, val_dataloaders=val_loader)
        self.trainer.test(self.lit_module, dataloaders=test_loader)

        self.potential = cast("Potential", self.lit_module.model)
        if save_path is not None:
            self.save(save_path)
        if push_to_hub is not None:
            self.push_to_hub(push_to_hub)
        return self.potential

    # ------------------------------------------------------------------
    # Convenience persistence helpers (post-fit).
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Save the trained ``Potential`` (delegates to :meth:`IOMixIn.save`)."""
        if self.potential is None:
            raise RuntimeError("MatGLPotentialTrainer.save called before fit; nothing to save yet.")
        self.potential.save(str(path))

    def push_to_hub(self, repo_id: str, **kwargs) -> str:
        """Push the trained ``Potential`` to the Hub (delegates to :meth:`IOMixIn.push_to_hub`)."""
        if self.potential is None:
            raise RuntimeError("MatGLPotentialTrainer.push_to_hub called before fit; nothing to push yet.")
        return cast("str", self.potential.push_to_hub(repo_id, **kwargs))
