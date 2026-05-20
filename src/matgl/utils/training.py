"""Utils for training MatGL models."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Literal, cast

import lightning as pl
import numpy as np
import torch
import torch.nn.functional as F
import torchmetrics
from huggingface_hub import _CACHED_NO_EXIST, hf_hub_download, try_to_load_from_cache
from monty.serialization import loadfn
from pymatgen.core import Structure
from torch import nn

from matgl.apps._pes import Potential
from matgl.config import MATGL_CACHE

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence
    from pathlib import Path

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

    def optimizer_zero_grad(
        self,
        epoch: int,
        batch_idx: int,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        """Zero gradients with ``set_to_none=True`` to free memory and skip the zero kernel.

        Lightning's default already forwards to PyTorch's default
        (``set_to_none=True`` since PyTorch 1.7), but we override here to make the
        intent explicit and survive any upstream default change.
        """
        optimizer.zero_grad(set_to_none=True)

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
            g: PyG ``Data``/``Batch`` graph.
            lat: Lattice tensor. ``(3, 3)`` for a single graph or ``(B, 3, 3)`` when batched.
            l_g: Optional line graph.
            state_attr: Optional state attribute.

        Returns:
            Model prediction.
        """
        if lat is not None:
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
        data_mean: float | torch.Tensor = 0.0,
        data_std: float | torch.Tensor = 1.0,
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
            g: PyG ``Data``/``Batch`` graph.
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

        num_atoms = torch.bincount(g.batch)
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
        :class:`~matgl.graph._data.MGLDataset` of PyG ``Data``
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
# MGLDatasetLoader / MGLPotentialTrainer — dataset factory + training wrapper.
# ---------------------------------------------------------------------------

HF_MATPES_REPO_ID = "materialyze/matpes"

# Stress-unit conversion to matgl's internal unit (**GPa**, compressive =
# negative — see the "Model Training" section of the README). ``1 GPa = 10
# kbar = 1/160.21766208 eV/Å³``. The ``"kbar"`` factor is ``-0.1``, which both
# rescales (kbar → GPa, /10) **and** flips the sign from VASP's
# compressive-positive convention to matgl's compressive-negative one — i.e.
# applying ``stress_unit="kbar"`` is exactly the README's "multiply VASP
# stresses by -0.1" recipe in one step. MatPES JSONs ship raw VASP stress in
# kbar, hence the default; pass ``stress_unit="GPa"`` to skip the conversion
# or ``"eV/A3"`` if your file is already in eV/Å³ (magnitude only — supply the
# sign yourself for ``eV/A3``).
StressUnit = Literal["kbar", "GPa", "eV/A3"]
_STRESS_UNIT_TO_GPA: dict[str, float] = {
    "GPa": 1.0,
    "kbar": -0.1,
    "eV/A3": 160.21766208,
}


# Per-atom MatPES partial-charge sources. DDEC6 net atomic charges live under
# ``ddec6.partial_charges`` (Chargemol output) and are the matgl default; the
# Bader analysis ships in the top-level ``bader_charges`` field as electron
# counts (e); CM5 charges under the top-level ``cm5_partial_charges``. Any of
# the three can be ``None`` for a given sample if the upstream calculation
# failed.
ChargesSource = Literal["bader", "ddec6", "cm5"]
MagmomsSource = Literal["bader", "ddec6"]


def _extract_bader_charges(raw: Mapping) -> list | None:
    return raw.get("bader_charges")


def _extract_ddec6_charges(raw: Mapping) -> list | None:
    ddec6 = raw.get("ddec6")
    return None if ddec6 is None else ddec6.get("partial_charges")


def _extract_cm5_charges(raw: Mapping) -> list | None:
    return raw.get("cm5_partial_charges")


def _extract_bader_magmoms(raw: Mapping) -> list | None:
    return raw.get("bader_magmoms")


def _extract_ddec6_magmoms(raw: Mapping) -> list | None:
    ddec6 = raw.get("ddec6")
    return None if ddec6 is None else ddec6.get("spin_moments")


_CHARGES_EXTRACTORS: dict[str, tuple[str, Any]] = {
    "bader": ("bader_charges", _extract_bader_charges),
    "ddec6": ("ddec6.partial_charges", _extract_ddec6_charges),
    "cm5": ("cm5_partial_charges", _extract_cm5_charges),
}

_MAGMOMS_EXTRACTORS: dict[str, tuple[str, Any]] = {
    "bader": ("bader_magmoms", _extract_bader_magmoms),
    "ddec6": ("ddec6.spin_moments", _extract_ddec6_magmoms),
}


def _resolve_optional_source(
    flag: bool | str,
    label: str,
    extractors: Mapping[str, tuple[str, Any]],
    default_source: str,
) -> tuple[str, Any] | None:
    """Map an ``include_*`` flag to (display_name, extractor) or None when disabled.

    ``True`` -> ``default_source``; a string is looked up in ``extractors``;
    ``False`` returns ``None`` (skip extraction).
    """
    if flag is False:
        return None
    src = default_source if flag is True else str(flag).lower()
    if src not in extractors:
        raise ValueError(
            f"Unknown {label} source {src!r}. Available: {sorted(extractors.keys())}.",
        )
    return extractors[src]


def _matpes_samples_to_lists(
    samples: Iterable[Mapping],
    stress_unit: StressUnit = "kbar",
    *,
    include_charges: bool | ChargesSource = False,
    include_magmoms: bool | MagmomsSource = False,
) -> tuple[list[Structure], dict[str, list]]:
    """Walk a MatPES sample list and return parallel structures + labels dict.

    Always extracts ``energies``, ``forces``, and ``stresses``. Per-atom optional
    partial-charge / magmom fields are extracted when the corresponding
    ``include_*`` flag is truthy; samples whose requested optional field is
    ``None`` (i.e. the upstream calculation failed) are dropped from every
    parallel list, so the returned lists stay aligned.

    Args:
        samples: Iterable of MatPES sample dicts.
        stress_unit: On-disk stress unit; see :meth:`MGLDatasetLoader.from_json`.
        include_charges: If truthy, extract per-atom partial charges under the
            label key ``"charges"`` and drop samples with missing values.
            ``True`` selects DDEC6 net atomic charges
            (``ddec6.partial_charges``, the matgl default for QET-style
            training); pass ``"bader"`` for Bader electron counts
            (``bader_charges``) or ``"cm5"`` for CM5 charges
            (``cm5_partial_charges``).
        include_magmoms: If truthy, extract per-atom magnetic moments under
            ``"magmoms"`` and drop samples with missing values. ``True``
            selects DDEC6 spin moments (``ddec6.spin_moments``); pass
            ``"bader"`` for Bader magmoms (``bader_magmoms``).

    Returns:
        ``(structures, labels)`` where ``labels`` has keys ``"energies"``,
        ``"forces"``, ``"stresses"``, and optionally ``"charges"`` / ``"magmoms"``.
    """
    # (display_name, extractor_callable) tuples, one per enabled optional.
    enabled: list[tuple[str, str, Any]] = []
    charges_resolved = _resolve_optional_source(include_charges, "charges", _CHARGES_EXTRACTORS, "ddec6")
    if charges_resolved is not None:
        enabled.append(("charges", *charges_resolved))
    magmoms_resolved = _resolve_optional_source(include_magmoms, "magmoms", _MAGMOMS_EXTRACTORS, "ddec6")
    if magmoms_resolved is not None:
        enabled.append(("magmoms", *magmoms_resolved))

    structures: list[Structure] = []
    energies: list[float] = []
    forces: list = []
    stresses: list = []
    optional_lists: dict[str, list] = {label: [] for label, _, _ in enabled}
    factor = _STRESS_UNIT_TO_GPA[stress_unit]
    dropped = 0

    for raw in samples:
        # Skip samples missing any requested optional field. We check before
        # appending anything to keep the parallel lists in lockstep.
        skip = False
        optional_values: dict[str, list] = {}
        for label, _src_name, extractor in enabled:
            value = extractor(raw)
            if value is None:
                skip = True
                break
            optional_values[label] = np.asarray(value, dtype="float64").tolist()
        if skip:
            dropped += 1
            continue

        struct = raw["structure"]
        if not isinstance(struct, Structure):
            struct = Structure.from_dict(struct)
        structures.append(struct)
        energies.append(float(raw["energy"]))
        forces.append(np.asarray(raw["forces"], dtype="float64").tolist())
        stresses.append((np.asarray(raw["stress"], dtype="float64") * factor).tolist())
        for label, val in optional_values.items():
            optional_lists[label].append(val)

    labels: dict[str, list] = {"energies": energies, "forces": forces, "stresses": stresses, **optional_lists}
    if dropped and enabled:
        import warnings

        missing = ", ".join(f"{label}({src})" for label, src, _ in enabled)
        warnings.warn(
            f"Dropped {dropped} MatPES samples missing one of: {missing}.",
            UserWarning,
            stacklevel=2,
        )
    return structures, labels


def _hf_download_cached_first(
    *,
    repo_id: str,
    filename: str,
    repo_type: str,
    revision: str | None,
    token: str | None,
    cache_dir: str | Path | None = MATGL_CACHE,
) -> str:
    """Return the local path for an HF Hub file, using the cache when available.

    ``hf_hub_download`` already caches blobs but still makes an HTTP HEAD
    request to validate the etag on every call, which slows repeated loads
    and fails entirely when offline. ``try_to_load_from_cache`` resolves the
    cached snapshot path with no network access, so we use it as a fast path
    and only fall through to ``hf_hub_download`` on a miss.
    """
    cached = try_to_load_from_cache(
        repo_id=repo_id,
        filename=filename,
        repo_type=repo_type,
        cache_dir=cache_dir,
        revision=revision,
    )
    # ``cached`` is a path string when the file is in cache, ``None`` when not
    # cached, and the ``_CACHED_NO_EXIST`` sentinel when a previous lookup
    # cached a known-missing status.
    if isinstance(cached, str) and cached is not _CACHED_NO_EXIST:
        return cached
    return hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type=repo_type,
        revision=revision,
        token=token,
        cache_dir=cache_dir,
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


class MGLDatasetLoader:
    """Factory for building :class:`MGLDataset` objects from external sources.

    Hoists the HF Hub auth / cache configuration to one place so successive
    downloads (e.g. dataset + atomrefs) can share it without repeating
    ``repo_id`` / ``revision`` / ``token`` / ``cache_dir`` per call::

        loader = MGLDatasetLoader()  # defaults to ``materialyze/matpes``
        ds = loader.matpes_dataset(version="r2SCAN-2025.2")
        refs = loader.matpes_element_refs(
            version="r2SCAN-2025.2", element_types=ds.element_types
        )

    Override the HF source for staging or forks::

        loader = MGLDatasetLoader(
            repo_id="my-org/matpes-fork", token="hf_...", revision="dev"
        )

    Already have a MatPES (or MatPES-shaped) JSON on disk? Skip the HF round
    trip and load it directly. :meth:`from_json` is a ``@staticmethod`` — it
    needs no loader state and can be called on the class itself::

        ds = MGLDatasetLoader.from_json("/path/to/MatPES-r2SCAN-2025.2.json")

    (The instance form ``loader.from_json(...)`` works too, for callers that
    already hold a loader.)

    Splitting + ``MGLDataLoader`` wrapping is the trainer's job — hand the
    returned ``MGLDataset`` straight to ``MGLPotentialTrainer.fit(...)`` and
    let it apply ``frac_list`` / ``shuffle`` / ``random_state`` from its own
    ``loader_kwargs``.

    See :meth:`from_json` for the expected schema (the same one the live
    MatPES JSONs ship in: ``structure`` + ``energy`` + ``forces`` + ``stress``
    per record).

    This class only constructs datasets; training itself is handled by
    :class:`MGLPotentialTrainer`.
    """

    def __init__(
        self,
        *,
        repo_id: str = HF_MATPES_REPO_ID,
        revision: str | None = None,
        token: str | None = None,
        cache_dir: str | Path | None = MATGL_CACHE,
    ) -> None:
        """Initialise the loader with shared HF Hub config.

        Args:
            repo_id: Default HF Hub dataset repo id used by the ``matpes_*``
                helpers. Override per-call by passing ``repo_id=...``.
            revision: Optional branch / tag / commit forwarded to
                ``hf_hub_download``.
            token: Optional HF auth token for private repos.
            cache_dir: HF Hub download cache directory; defaults to
                ``MATGL_CACHE``.
        """
        self.repo_id = repo_id
        self.revision = revision
        self.token = token
        self.cache_dir = cache_dir

    def matpes_dataset(
        self,
        version: str = "R2SCAN-2025.2",
        *,
        cutoff: float = 5.0,
        element_types: tuple[str, ...] | None = None,
        repo_id: str | None = None,
        save_cache: bool = True,
        root: str | None = None,
        stress_unit: StressUnit = "kbar",
        include_charges: bool | ChargesSource = False,
        include_magmoms: bool | MagmomsSource = False,
    ) -> MGLDataset:
        """Download a MatPES JSON file from HF and build an ``MGLDataset``.

        Args:
            version: MatPES version, e.g. ``"r2SCAN-2025.2"`` (case-insensitive).
            cutoff: Neighbour cutoff (Å) handed to ``Structure2Graph``.
            element_types: Optional explicit ordering; auto-derived when None.
            repo_id: Per-call override of the loader's default ``repo_id``.
            save_cache: Whether ``MGLDataset`` persists its processed cache.
            root: ``MGLDataset`` root directory; default lets it pick.
            stress_unit: Unit of the on-disk ``stress`` field. Defaults to
                ``"kbar"`` (raw VASP convention, compressive = positive — the
                standard for MatPES JSONs). matgl's internal unit is **GPa**
                with compressive = negative (README, "Model Training"), so the
                default ``"kbar"`` factor is ``-0.1`` and applies the
                "multiply VASP stress by -0.1" recipe in one step. Pass
                ``"GPa"`` if the file is already in matgl convention, or
                ``"eV/A3"`` for an eV/Å³ source (magnitude only — supply the
                correct sign yourself).
            include_charges: If truthy, copy per-atom partial charges into the
                dataset's labels under the key ``"charges"``. ``True`` selects
                DDEC6 net atomic charges (``ddec6.partial_charges``, from
                Chargemol — the matgl default for QET-style training); pass
                ``"bader"`` for Bader electron counts (``bader_charges``) or
                ``"cm5"`` for CM5 (``cm5_partial_charges``). Samples whose
                chosen source is ``None`` (calculation failed upstream) are
                dropped — the dataset is silently shorter than the full
                MatPES file. Required for QET-style training where
                ``charge_weight > 0``.
            include_magmoms: If truthy, copy per-atom magnetic moments into
                the dataset's labels under ``"magmoms"``. ``True`` selects
                DDEC6 spin moments (``ddec6.spin_moments``); pass ``"bader"``
                for Bader magmoms (``bader_magmoms``). Same drop-on-missing
                semantics as ``include_charges``. Required for CHGNet-style
                training where ``magmom_weight > 0``.

        Returns:
            An ``MGLDataset`` ready to drop into ``MGLDataLoader``. The
            monolithic files are 1.6-2.4 GB.
        """
        functional, tag = version.split("-")
        filename = f"MatPES-{functional.upper()}-{tag}.json"
        local_path = _hf_download_cached_first(
            repo_id=repo_id or self.repo_id,
            filename=filename,
            repo_type="dataset",
            revision=self.revision,
            token=self.token,
            cache_dir=self.cache_dir,
        )
        return MGLDatasetLoader.from_json(
            local_path,
            cutoff=cutoff,
            element_types=element_types,
            save_cache=save_cache,
            root=root,
            stress_unit=stress_unit,
            include_charges=include_charges,
            include_magmoms=include_magmoms,
        )

    @staticmethod
    def from_json(
        path: str | Path,
        *,
        cutoff: float = 5.0,
        element_types: tuple[str, ...] | None = None,
        save_cache: bool = True,
        root: str | None = None,
        stress_unit: StressUnit = "kbar",
        include_charges: bool | ChargesSource = False,
        include_magmoms: bool | MagmomsSource = False,
    ) -> MGLDataset:
        """Build an ``MGLDataset`` from a local MatPES-shaped JSON file.

        The file format mirrors the live MatPES JSON dataset exactly — a flat
        JSON list of per-frame records, one record per (structure, PES data)
        pair. Custom DFT runs and MatPES forks that respect this schema drop
        in unchanged. No HF Hub round trip happens; this is the local-disk
        sibling of :meth:`matpes_dataset`.

        **Schema (per record)** — the important keys are:

        - ``structure`` (**required**): a pymatgen ``Structure`` or its
          MSONable ``as_dict()`` form — anything ``Structure.from_dict`` can
          rebuild. Carries the atomic positions, lattice, and species.
        - ``energy`` (**required**): total energy in **eV** as a scalar.
        - ``forces`` (**required**): per-atom forces in **eV/Å**, shape
          ``(N_atoms, 3)`` (list-of-lists is fine).
        - ``stress`` (**required**): stress tensor as either a ``3x3`` or
          length-6 (Voigt) array. Units are controlled by ``stress_unit``
          below; the default matches MatPES on-disk convention.
        - ``bader_charges`` / ``cm5_partial_charges`` / ``ddec6.partial_charges``
          (optional): per-atom partial charges, surfaced under the dataset
          label ``"charges"`` when ``include_charges`` is truthy.
        - ``bader_magmoms`` / ``ddec6.spin_moments`` (optional): per-atom
          magnetic moments, surfaced under ``"magmoms"`` when
          ``include_magmoms`` is truthy.

        Any other extra keys in each record (e.g. ``functional``, ``bandgap``,
        provenance metadata) are ignored.

        Splitting + ``MGLDataLoader`` wrapping is handled inside
        :class:`MGLPotentialTrainer` (see its ``frac_list`` / ``shuffle`` /
        ``random_state`` ``loader_kwargs``), so this method only returns the
        raw ``MGLDataset``; hand it straight to ``trainer.fit(dataset=...)``.

        Args:
            path: Path to the JSON file. ``.json``, ``.json.gz``, and other
                formats supported by ``monty.serialization.loadfn`` all work.
            cutoff: Neighbour cutoff (Å) handed to ``Structure2Graph``.
            element_types: Optional explicit ordering; auto-derived from the
                structures when ``None``.
            save_cache: Whether ``MGLDataset`` persists its processed cache.
            root: ``MGLDataset`` root directory; default lets it pick.
            stress_unit: Unit of the on-disk ``stress`` field. Defaults to
                ``"kbar"`` (raw VASP convention, compressive = positive — the
                standard for MatPES JSONs). matgl's internal unit is **GPa**
                with compressive = negative (README, "Model Training"), so the
                default ``"kbar"`` factor is ``-0.1`` and applies the
                "multiply VASP stress by -0.1" recipe in one step. Pass
                ``"GPa"`` if the file is already in matgl convention, or
                ``"eV/A3"`` for an eV/Å³ source (magnitude only — supply the
                correct sign yourself).
            include_charges: If truthy, copy per-atom partial charges into the
                dataset's labels under the key ``"charges"``. ``True`` selects
                DDEC6 net atomic charges (``ddec6.partial_charges``, from
                Chargemol — the matgl default for QET-style training); pass
                ``"bader"`` for Bader electron counts (``bader_charges``) or
                ``"cm5"`` for CM5 (``cm5_partial_charges``). Samples whose
                chosen source is ``None`` (calculation failed upstream) are
                dropped — the dataset is silently shorter than the input file.
            include_magmoms: If truthy, copy per-atom magnetic moments into
                the dataset's labels under ``"magmoms"``. ``True`` selects
                DDEC6 spin moments (``ddec6.spin_moments``); pass ``"bader"``
                for Bader magmoms (``bader_magmoms``). Same drop-on-missing
                semantics as ``include_charges``.

        Returns:
            An ``MGLDataset`` ready to drop into
            :class:`MGLPotentialTrainer.fit`.
        """
        structures, labels = _matpes_samples_to_lists(
            loadfn(path),
            stress_unit=stress_unit,
            include_charges=include_charges,
            include_magmoms=include_magmoms,
        )

        return _build_pes_dataset(
            structures,
            labels,
            cutoff=cutoff,
            element_types=element_types,
            save_cache=save_cache,
            root=root,
        )

    def matpes_element_refs(
        self,
        version: str = "R2SCAN-2025.2",
        *,
        repo_id: str | None = None,
        element_types: tuple[str, ...] = (),
    ) -> np.ndarray:
        """Download per-element energy offsets shipped alongside MatPES.

        File schema: a flat list of ``{"chemsys": <symbol>, "energy": <eV>}``
        records (one per element). When ``element_types`` is supplied the
        returned vector is reordered so ``refs[i]`` is the offset for
        ``element_types[i]``; the default empty tuple returns a length-0 array.

        Args:
            version: MatPES version, e.g. ``"r2SCAN-2025.2"``.
            repo_id: Per-call override of the loader's default ``repo_id``.
            element_types: Element ordering for the returned vector. The
                output is ``np.asarray([refs[sym] for sym in element_types])``;
                pass ``model.element_types`` to align with a downstream
                ``Potential``.

        Returns:
            ``np.ndarray`` of shape ``(len(element_types),)``, dtype
            ``float64``, with offsets in the supplied element order.
        """
        functional, _ = version.split("-")
        filename = f"MatPES-{functional.upper()}-atoms.json"
        local_path = _hf_download_cached_first(
            repo_id=repo_id or self.repo_id,
            filename=filename,
            repo_type="dataset",
            revision=self.revision,
            token=self.token,
            cache_dir=self.cache_dir,
        )
        payload = loadfn(str(local_path))
        refs = {d["chemsys"]: d["energy"] for d in payload}
        return np.asarray([refs[sym] for sym in element_types], dtype="float64")


class MGLPotentialTrainer:
    """Configure-once / fit-when-asked trainer for matgl ``Potential`` training.

    ``__init__`` stores hyperparameters but does not download data, build a
    dataset, or instantiate Lightning. The first network / disk activity
    happens inside :meth:`fit`.

    Dataset construction is delegated to :class:`MGLDatasetLoader`, which
    holds the HF Hub auth / cache configuration. A typical end-to-end MatPES
    call::

        loader = MGLDatasetLoader()
        ds = loader.matpes_dataset(version="r2SCAN-2025.2")
        refs = loader.matpes_element_refs(
            version="r2SCAN-2025.2", element_types=ds.element_types
        )
        trainer = MGLPotentialTrainer(model, accelerator="gpu")
        potential = trainer.fit(dataset=ds, atomrefs=refs, save_path="./MatPES-TensorNet")

    **Logging / callbacks.** The trainer instantiates ``pl.Trainer`` inside
    :meth:`fit` and forwards anything passed via ``trainer_kwargs`` straight
    to it — Lightning's full ``callbacks`` / ``logger`` vocabulary works
    unchanged. :class:`PotentialLightningModule` logs ``Total_Loss``,
    ``Energy_MAE``, ``Force_MAE``, ``Stress_MAE`` (plus ``Magmom_MAE`` /
    ``Charge_MAE`` when those heads are active) each prefixed with
    ``train_`` / ``val_`` / ``test_`` and the same set of ``*_RMSE`` keys
    — those are the names a ``ModelCheckpoint`` / ``EarlyStopping`` /
    logger sees::

        from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
        from lightning.pytorch.loggers import CSVLogger

        trainer = MGLPotentialTrainer(
            model,
            accelerator="gpu",
            trainer_kwargs={
                "logger": CSVLogger(save_dir="./logs", name="matpes-tensornet"),
                "callbacks": [
                    LearningRateMonitor(logging_interval="epoch"),
                    ModelCheckpoint(
                        dirpath="./logs/ckpts",
                        monitor="val_Total_Loss",
                        mode="min",
                        save_top_k=3,
                    ),
                    EarlyStopping(monitor="val_Total_Loss", patience=20, mode="min"),
                ],
            },
        )

    Swap ``CSVLogger`` for ``TensorBoardLogger`` / ``WandbLogger`` /
    ``MLFlowLogger`` as needed (or pass a list for multi-sink logging).

    For per-epoch dumping of every prediction / label / error in stable
    sample order, matgl ships :class:`matgl.utils.callbacks.PredictionLogger`
    (call :func:`matgl.utils.callbacks.add_sample_indices` on the split(s)
    you want logged before ``fit``). For ad-hoc instrumentation, drop a
    custom ``pl.Callback`` subclass into ``callbacks``;
    :meth:`PotentialLightningModule.training_step` /
    :meth:`~PotentialLightningModule.validation_step` return
    ``{"loss", "preds", "labels", "indices", "num_atoms"}`` as ``outputs``::

        import lightning.pytorch as pl

        class LogForceRMS(pl.Callback):
            def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
                force_pred, force_label = outputs["preds"][1], outputs["labels"][1]
                pl_module.log(
                    "train_force_rms",
                    (force_pred - force_label).pow(2).mean().sqrt(),
                    on_step=True, on_epoch=False,
                )

        trainer = MGLPotentialTrainer(
            model, trainer_kwargs={"callbacks": [LogForceRMS()]}
        )

    """

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        # Loss term weights.
        energy_weight: float = 1.0,
        force_weight: float = 1.0,
        stress_weight: float = 0.1,
        magmom_weight: float = 0.0,
        charge_weight: float = 0.0,
        loss: str = "huber_loss",
        loss_params: dict | None = None,
        # Target normalisation (forwarded to PotentialLightningModule).
        data_mean: float | torch.Tensor = 0.0,
        data_std: float | torch.Tensor = 1.0,
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
                Stress labels are expected in **GPa** with compressive =
                negative (matgl convention; see the README "Model Training"
                section). The MatPES loaders apply the kbar → GPa /
                sign-flip conversion automatically.
            magmom_weight: Site-wise magmom loss weight. Set ``> 0`` to train
                on ``"magmoms"`` labels (e.g. CHGNet-style fits); ``0``
                (default) disables the head and the loss term.
            charge_weight: Site-wise charge loss weight. Set ``> 0`` to train
                on ``"charges"`` labels (e.g. QET-style fits); ``0`` (default)
                disables the head and the loss term.
            loss: One of ``"mse_loss"``, ``"huber_loss"`` (default; robust),
                ``"smooth_l1_loss"``, or ``"l1_loss"``.
            loss_params: Optional kwargs forwarded to the loss function (e.g.
                ``{"delta": 0.1}`` for Huber).
            data_mean: Per-atom energy mean used by the inner ``Potential`` to
                rescale predictions before the loss is computed. Forwarded to
                :class:`PotentialLightningModule`. Default ``0.0``.
            data_std: Per-atom energy / force scale used by the inner
                ``Potential`` to rescale predictions before the loss is
                computed. Typical recipe for foundation-potential training:
                pass the RMS of all force components in the training set so
                forces and energies share an order of magnitude. Default
                ``1.0``.
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
            trainer_kwargs: Extra ``pl.Trainer`` kwargs forwarded verbatim;
                this is the entry point for Lightning ``callbacks`` and
                ``logger``. See the *Logging / callbacks* recipe in the class
                docstring for the canonical
                ``CSVLogger`` + ``ModelCheckpoint`` + ``EarlyStopping`` setup,
                the matgl-native :class:`matgl.utils.callbacks.PredictionLogger`,
                and the metric names (``train_*`` / ``val_*`` / ``test_*``)
                available to monitor.
            loader_kwargs: Extra kwargs forwarded to :class:`MGLDataLoader` /
                :func:`split_dataset` via :meth:`_build_dataloaders`.
                Recognised split-only keys: ``frac_list``, ``shuffle``,
                ``random_state``.
        """
        self.model = model

        self.energy_weight = energy_weight
        self.force_weight = force_weight
        self.stress_weight = stress_weight
        self.magmom_weight = magmom_weight
        self.charge_weight = charge_weight
        self.loss = loss
        self.loss_params = loss_params

        self.data_mean = data_mean
        self.data_std = data_std

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

    def _build_dataloaders(
        self,
        dataset: MGLDataset | Mapping[str, MGLDataset],
    ) -> tuple[DataLoader, DataLoader, DataLoader]:
        """Build the ``(train, val, test)`` triple from a single dataset or a splits mapping."""
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
            return MGLDataLoader(  # type: ignore[return-value]
                train_data=train_data,  # type: ignore[arg-type]
                val_data=val_data,  # type: ignore[arg-type]
                test_data=test_data,  # type: ignore[arg-type]
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
        return MGLDataLoader(  # type: ignore[return-value]
            train_data=train_ds,
            val_data=val_ds,
            test_data=test_ds,
            batch_size=self.batch_size,
            **loader_kwargs,
        )

    def fit(
        self,
        dataset: MGLDataset | Mapping[str, MGLDataset],
        *,
        atomrefs: np.ndarray | Any = None,
        save_path: str | Path | None = None,
        ckpt_path: str | Path | None = None,
    ) -> Potential:
        """Run training end-to-end on a pre-built dataset.

        Args:
            dataset: An :class:`MGLDataset` (random split inside
                :meth:`_build_dataloaders`) or a canonical-splits mapping with
                keys ``"train"`` / ``"valid"`` / ``"test"``. Use
                :class:`MGLDatasetLoader` (e.g.
                ``MGLDatasetLoader().matpes_dataset(...)``) to build one.
            atomrefs: Optional per-element energy offsets. Either an
                ``np.ndarray`` (in ``model.element_types`` order), an
                :class:`AtomRef` instance (the layer's ``property_offset`` is
                extracted), or ``None`` (no offsets, default). Use
                ``MGLDatasetLoader().matpes_element_refs(...)`` (download from
                HF) or :func:`fit_element_refs` (fit locally) to obtain one.
            save_path: If given, ``potential.save(save_path)`` after training.
            ckpt_path: Forwarded to ``pl.Trainer.fit(ckpt_path=...)`` so a
                previous run can be resumed. Accepts an explicit Lightning
                checkpoint path or the string ``"last"`` to pick up the most
                recent ``last.ckpt`` from the configured ``ModelCheckpoint``.
                Optimizer / scheduler / epoch counter / RNG state are all
                restored. ``None`` (default) starts fresh.

        Returns:
            The trained :class:`~matgl.apps.pes.Potential`. Also reachable as
            ``self.potential``; auxiliary state (``self.lit_module``,
            ``self.trainer``, ``self.loaders``, ``self.dataset``,
            ``self.atomrefs``) is updated.

        Notes:
            Loss-term toggling follows the constructor weights: set
            ``stress_weight=0`` for datasets without stress labels (e.g.
            cluster / dimer extxyz files), and set ``magmom_weight`` /
            ``charge_weight`` ``> 0`` only when the dataset actually carries
            ``"magmoms"`` / ``"charges"`` labels.
        """
        pl.seed_everything(self.seed, workers=True)

        self.dataset = dataset
        self.atomrefs = atomrefs

        train_loader, val_loader, test_loader = self._build_dataloaders(dataset)
        self.loaders = {"train": train_loader, "val": val_loader, "test": test_loader}

        self.lit_module = PotentialLightningModule(
            model=self.model,
            element_refs=atomrefs,
            energy_weight=self.energy_weight,
            force_weight=self.force_weight,
            stress_weight=self.stress_weight,
            magmom_weight=self.magmom_weight,
            charge_weight=self.charge_weight,
            data_mean=self.data_mean,
            data_std=self.data_std,
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
        self.trainer.fit(
            model=self.lit_module,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
            ckpt_path=str(ckpt_path) if ckpt_path is not None else None,
        )
        self.trainer.test(self.lit_module, dataloaders=test_loader)

        self.potential = self.lit_module.model
        if save_path is not None:
            self.potential.save(save_path)

        return self.potential
