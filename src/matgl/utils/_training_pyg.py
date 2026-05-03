"""Utils for training MatGL models."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Literal

import lightning as pl
import torch
import torch.nn.functional as F
import torchmetrics
from torch import nn

from matgl.apps._pes_pyg import Potential

if TYPE_CHECKING:
    import numpy as np
    from torch.optim import Optimizer
    from torch.optim.lr_scheduler import LRScheduler
    from torch_geometric.data import Data


class MatglLightningModuleMixin:
    """Mix-in class implementing common functions for training."""

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
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

    def validation_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
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
        torch.set_grad_enabled(True)
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
        """
        Executed on model testing.

        Args:
            *args: Pass-through
            **kwargs: Pass-through.
        """
        super().on_test_model_eval(*args, **kwargs)  # type: ignore[misc]

    def predict_step(self, batch: tuple, batch_idx: int, dataloader_idx: int = 0) -> Any:
        """
        Prediction step.

        Args:
            batch: Data batch.
            batch_idx: Batch index.
            dataloader_idx: Data loader index.

        Returns:
            Prediction
        """
        torch.set_grad_enabled(True)
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
        """
        Init ModelLightningModule with key parameters.

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
        g: Data,
        lat: torch.Tensor | None = None,
        l_g: Data | None = None,
        state_attr: torch.Tensor | None = None,
    ):
        """Compute model predictions.

        Attaches per-node ``pos`` and per-edge ``pbc_offshift`` tensors derived from
        ``frac_coords``/``pbc_offset`` and the supplied lattice(s), then delegates to
        the wrapped model. Works for both single-graph (``lat`` shape ``(3, 3)``) and
        batched (``lat`` shape ``(B, 3, 3)``) inputs.

        Args:
            g: PyG graph (single ``Data`` or batched ``Batch``).
            lat: Lattice tensor.
            l_g: Optional line graph.
            state_attr: Optional state attribute.

        Returns:
            Model prediction.
        """
        if lat is not None:
            # Normalize lat to (B, 3, 3) regardless of whether the caller supplied a
            # single-graph (3, 3) tensor or a (B, 3, 3) batched tensor.
            if lat.dim() == 2:
                lat = lat.unsqueeze(0)
            # PyG batches assign each node a graph index via ``g.batch``; for a
            # non-batched ``Data`` this attribute is missing/None, in which case
            # every node belongs to graph 0.
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
        """Args:
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
        """Args:
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
        """
        Init PotentialLightningModule with key parameters.

        Args:
            model: Which type of the model for training
            element_refs: element offset for PES
            include_line_graph: whether to include line graphs
            energy_weight: relative importance of energy
            force_weight: relative importance of force
            stress_weight: relative importance of stress
            magmom_weight: relative importance of additional magmom predictions.
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

        super().__init__(**kwargs)

        self.mae = torchmetrics.MeanAbsoluteError()
        self.rmse = torchmetrics.MeanSquaredError(squared=False)
        self.register_buffer("data_mean", torch.tensor(data_mean))
        self.register_buffer("data_std", torch.tensor(data_std))

        self.energy_weight = energy_weight
        self.force_weight = force_weight
        self.stress_weight = stress_weight
        self.magmom_weight = magmom_weight
        self.lr = lr
        self.decay_steps = decay_steps
        self.decay_alpha = decay_alpha
        self.include_line_graph = include_line_graph

        self.model = Potential(
            model=model,
            element_refs=element_refs,
            calc_stresses=stress_weight != 0,
            calc_magmom=magmom_weight != 0,
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
        self.save_hyperparameters(ignore=["model"])

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """# noqa: D200
        hacky hacky hack to add missing keys to the state dict when changes are made.
        """
        for key in self.state_dict():
            if key not in checkpoint["state_dict"]:
                checkpoint["state_dict"][key] = self.state_dict()[key]

    def forward(
        self,
        g: Data,
        lat: torch.Tensor,
        l_g: Data | None = None,
        state_attr: torch.Tensor | None = None,
    ) -> tuple:
        """Args:
            g: dgl Graph
            lat: lattice
            l_g: Line graph
            state_attr: State attr.

        Returns:
            energy, force, stress, hessian and optional site_wise
        """
        if self.include_line_graph:
            if self.model.calc_magmom:
                e, f, s, h, m = self.model(g=g, lat=lat, l_g=l_g, state_attr=state_attr)
                return e, f, s, h, m
            e, f, s, h = self.model(g=g, lat=lat, l_g=l_g, state_attr=state_attr)
            return e, f, s, h
        else:  # noqa: RET505
            if self.model.calc_magmom:
                e, f, s, h, m = self.model(g=g, lat=lat, state_attr=state_attr)
                return e, f, s, h, m
            e, f, s, h = self.model(g=g, lat=lat, state_attr=state_attr)
            return e, f, s, h

    def step(self, batch: tuple) -> tuple[dict[str, Any], int]:
        """Args:
            batch: Batch of training data.

        Returns:
            results, batch_size
        """
        preds: tuple
        labels: tuple

        torch.set_grad_enabled(True)
        if self.include_line_graph:
            if self.model.calc_magmom:
                g, lat, l_g, state_attr, energies, forces, stresses, magmoms = batch
                e, f, s, _, m = self(g=g, lat=lat, state_attr=state_attr, l_g=l_g)
                preds = (e, f, s, m)
                labels = (energies, forces, stresses, magmoms)
            else:
                g, lat, l_g, state_attr, energies, forces, stresses = batch
                e, f, s, _ = self(g=g, lat=lat, state_attr=state_attr, l_g=l_g)
                preds = (e, f, s)
                labels = (energies, forces, stresses)
        else:
            if self.model.calc_magmom:
                g, lat, state_attr, energies, forces, stresses, magmoms = batch
                e, f, s, _, m = self(g=g, lat=lat, state_attr=state_attr)
                preds = (e, f, s, m)
                labels = (energies, forces, stresses, magmoms)
            else:
                g, lat, state_attr, energies, forces, stresses = batch
                e, f, s, _ = self(g=g, lat=lat, state_attr=state_attr)
                preds = (e, f, s)
                labels = (energies, forces, stresses)

        num_atoms = torch.bincount(g.batch)
        results = self.loss_fn(
            loss=self.loss,  # type: ignore
            preds=preds,
            labels=labels,
            num_atoms=num_atoms,
        )
        batch_size = preds[0].numel()

        return results, batch_size

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
                "Energy_RMSE": e_rmse,
                "Force_RMSE": f_rmse,
                "Stress_RMSE": s_rmse,
                "Magmom_RMSE": m_rmse
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
        # TODO: We should get rid of this dirty fix
        #        valid_labels[0] = valid_labels[0].squeeze()
        e_loss = self.loss(valid_labels[0] / valid_num_atoms, valid_preds[0] / valid_num_atoms, **self.loss_params)
        f_loss = self.loss(valid_labels[1], valid_preds[1], **self.loss_params)

        e_mae = self.mae(valid_labels[0] / valid_num_atoms, valid_preds[0] / valid_num_atoms)
        f_mae = self.mae(valid_labels[1], valid_preds[1])

        e_rmse = self.rmse(valid_labels[0] / valid_num_atoms, valid_preds[0] / valid_num_atoms)
        f_rmse = self.rmse(valid_labels[1], valid_preds[1])

        s_mae = torch.zeros(1)
        s_rmse = torch.zeros(1)

        m_mae = torch.zeros(1)
        m_rmse = torch.zeros(1)

        total_loss = self.energy_weight * e_loss + self.force_weight * f_loss

        if self.model.calc_stresses:
            s_loss = loss(valid_labels[2], valid_preds[2], **self.loss_params)
            s_mae = self.mae(valid_labels[2], valid_preds[2])
            s_rmse = self.rmse(valid_labels[2], valid_preds[2])
            total_loss = total_loss + self.stress_weight * s_loss

        if self.model.calc_magmom and labels[3].numel() > 0:
            if self.magmom_target == "symbreak":
                m_loss = torch.min(
                    loss(valid_labels[3], valid_preds[3], **self.loss_params),
                    loss(valid_labels[3], -valid_preds[3], **self.loss_params),
                )
                m_mae = torch.min(self.mae(valid_labels[3], valid_preds[3]), self.mae(valid_labels[3], -valid_preds[3]))
                m_rmse = torch.min(
                    self.rmse(valid_labels[3], valid_preds[3]), self.rmse(valid_labels[3], -valid_preds[3])
                )
            else:
                labels_3 = torch.abs(valid_labels[3]) if self.magmom_target == "absolute" else valid_labels[3]
                m_loss = loss(labels_3, valid_preds[3], **self.loss_params)
                m_mae = self.mae(labels_3, valid_preds[3])
                m_rmse = self.rmse(labels_3, valid_preds[3])

            total_loss = total_loss + self.magmom_weight * m_loss

        return {
            "Total_Loss": total_loss,
            "Energy_MAE": e_mae,
            "Force_MAE": f_mae,
            "Stress_MAE": s_mae,
            "Magmom_MAE": m_mae,
            "Energy_RMSE": e_rmse,
            "Force_RMSE": f_rmse,
            "Stress_RMSE": s_rmse,
            "Magmom_RMSE": m_rmse,
        }


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
