"""Monte Carlo Dropout wrapper for uncertainty-aware inference with MatGL models.

Enables predictive uncertainty estimation from any pretrained ``MatGLModel`` via
the MC Dropout approximation (Gal & Ghahramani, 2016): the model is set to
``eval()`` to freeze BatchNorm statistics while the readout dropout layers are
kept in ``train()`` mode, yielding a stochastic distribution over predictions.

Typical use case: iterative active learning where each candidate is ranked by an
acquisition function (e.g. UCB = mean - lambda * std) to prioritise DFT evaluation of
structures with the highest expected stability.

Reference:
    Gal, Y. & Ghahramani, Z. (2016). Dropout as a Bayesian Approximation.
    ICML. https://arxiv.org/abs/1506.02142
"""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from typing import TYPE_CHECKING

import torch
from torch import nn

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pymatgen.core import Molecule, Structure

    from matgl.graph._converters import GraphConverter
    from matgl.models._core import MatGLModel

# Readout attribute names searched by default:
#   CHGNet      → final_dropout (nn.Identity placeholder when p=0), final_layer
#   M3GNet      → final_layer (MLP when is_intensive=True; WeightedReadOut otherwise — unsupported)
#   TensorNet   → final_layer
_DEFAULT_READOUT_ATTRS: tuple[str, ...] = ("final_dropout", "final_layer")


def _inject_dropout(module: nn.Module, p: float) -> nn.Dropout | None:
    """Insert dropout on the hidden features entering the last Linear in *module*.

    Works for ``MLP`` (``MLP.layers`` is a ``ModuleList`` whose ``forward`` iterates
    linearly). Dropping the scalar output instead would make its uncertainty
    a fixed fraction of the prediction, independent of the hidden features.

    Returns the new ``Dropout`` on success, ``None`` if no suitable ``layers``
    attribute was found.
    """
    layers = getattr(module, "layers", None)
    if isinstance(layers, (nn.ModuleList, nn.Sequential)):
        for index in range(len(layers) - 1, -1, -1):
            if isinstance(layers[index], nn.Linear):
                drop = nn.Dropout(p)
                layers.insert(index, drop)
                return drop
    return None


class MCDropoutWrapper:
    """Uncertainty-aware wrapper for MatGL models using Monte Carlo Dropout.

    Copies the model and injects dropout into the copy's readout layers at construction
    time, leaving the caller's parameters, modules and training mode untouched.
    During ``predict_uncertainty``, the backbone stays deterministic (eval)
    while the readout dropout layers remain stochastic (train), giving per-structure
    predictive mean and standard deviation via N forward passes.

    Example:
        >>> from matgl.models._chgnet import CHGNet
        >>> from matgl.utils.uncertainty import MCDropoutWrapper
        >>> model = CHGNet.load()
        >>> wrapper = MCDropoutWrapper(model, dropout_p=0.1)
        >>> structures = [...]  # list of pymatgen Structure objects
        >>> mean, std = wrapper.predict_uncertainty(structures, n_passes=20)
    """

    def __init__(
        self,
        model: MatGLModel,
        dropout_p: float = 0.1,
        readout_attrs: Sequence[str] = _DEFAULT_READOUT_ATTRS,
    ) -> None:
        """Initialise MCDropoutWrapper and inject dropout into the model's readout layers.

        Args:
            model: Any pretrained ``MatGLModel`` (CHGNet, M3GNet, TensorNet, …).
                A private copy is used; subsequent changes to the original model
                are not reflected in the wrapper.
            dropout_p: Dropout probability to inject into readout layers. Default = 0.1.
            readout_attrs: Attribute names on ``model`` to search for dropout layers.
                Defaults to ``("final_dropout", "final_layer")``. Override when using
                a custom architecture with differently-named readout modules.

        Raises:
            ValueError: If no dropout layers are found within the specified readout
                attributes and none can be injected (i.e. there are no ``nn.Identity``
                placeholders to replace).
        """
        if not 0.0 < dropout_p < 1.0:
            raise ValueError(f"dropout_p must be in (0, 1), got {dropout_p}")

        # Cached activations can be non-leaf tensors, which torch cannot deepcopy.
        # They are recomputed on the first prediction and need not be copied.
        model = deepcopy(model, {id(model.feature_dict): {}})
        self.model = model
        self.dropout_p = dropout_p
        self._stochastic_modules: list[nn.Dropout] = []
        # The readout module the dropout lives in. When this module is the model's
        # terminal op (head output == model output), predict_uncertainty can cache
        # the deterministic backbone and replay only this head -- see _predict_cached.
        self._head_module: nn.Module | None = None

        for attr in readout_attrs:
            if self._stochastic_modules:
                # Already wired up dropout from an earlier attr; stop so we don't
                # accidentally mutate a downstream module (e.g. CHGNet's final_layer
                # uses indexed layers[-1] access, so appending there would corrupt it).
                break
            readout_module = getattr(model, attr, None)
            if readout_module is None:
                continue
            if isinstance(readout_module, nn.Identity):
                new_drop = nn.Dropout(p=dropout_p)
                setattr(model, attr, new_drop)
                self._stochastic_modules.append(new_drop)
                self._head_module = new_drop
            elif isinstance(readout_module, nn.Dropout):
                readout_module.p = dropout_p
                self._stochastic_modules.append(readout_module)
                self._head_module = readout_module
            else:
                found = [m for m in readout_module.modules() if isinstance(m, nn.Dropout)]
                if found:
                    for m in found:
                        m.p = dropout_p
                    self._stochastic_modules.extend(found)
                    self._head_module = readout_module
                else:
                    injected = _inject_dropout(readout_module, dropout_p)
                    if injected is not None:
                        self._stochastic_modules.append(injected)
                        self._head_module = readout_module

        if not self._stochastic_modules:
            raise ValueError(
                f"No dropout layers found or injected in {list(readout_attrs)} on "
                f"{type(model).__name__}. Initialise the model with dropout > 0 "
                "(e.g. CHGNet(final_dropout=0.1)) or supply the correct readout_attrs."
            )

    @contextmanager
    def _stochastic_mode(self):
        """Set the full model to eval then re-enable only the readout dropout layers."""
        self.model.eval()
        for m in self._stochastic_modules:
            m.train()
        try:
            yield
        finally:
            self.model.eval()

    def _to_graph(
        self,
        structure: Structure | Molecule,
        graph_converter: GraphConverter | None,
    ) -> tuple:
        """Convert a structure to a (graph, state_feats) pair.

        Uses the same conversion as ``MatGLModel.predict_structure`` so each structure
        is converted once and the graph reused across all N forward passes.
        """
        from matgl.models._core import structure_to_graph

        g, state_feats = structure_to_graph(self.model, structure, graph_converter)
        return g, state_feats

    def predict_uncertainty(
        self,
        structures: Structure | Molecule | Sequence[Structure | Molecule],
        n_passes: int = 20,
        graph_converter: GraphConverter | None = None,
        cache_backbone: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run N stochastic forward passes and return predictive mean and std.

        All structures are batched into a single PyG ``Batch`` per pass, so cost
        scales as O(n_passes) forward calls rather than O(N · n_passes). Graph
        conversion still happens exactly once per structure.

        When ``cache_backbone`` is enabled and the readout head is the model's
        terminal op (true for M3GNet/TensorNet intensive models), the deterministic
        backbone is evaluated **once** and only the cheap stochastic head is replayed
        N times, giving an ~``n_passes``x speed-up. MC Dropout only perturbs the
        readout, so the backbone output is identical across passes — the result is
        numerically equivalent to the naive loop (verified in the tests). The fast
        path is used only when a probe confirms head-output == model-output; models
        with post-dropout ops (e.g. CHGNet's graph pooling) fall back automatically.

        Args:
            structures: A single or list of pymatgen ``Structure`` / ``Molecule``.
            n_passes: Number of stochastic forward passes. Must be ≥ 2. Increase
                for tighter uncertainty estimates at the cost of compute. Default = 20.
            graph_converter: Optional graph converter. If ``None``, uses the
                model's default ``Structure2Graph`` (requires ``model.element_types``
                and ``model.cutoff`` attributes, present on all standard MatGL models).
            cache_backbone: If ``True`` (default), use the backbone-once fast path
                when it is provably exact for this model, else fall back to the naive
                per-pass loop. Set ``False`` to force the naive loop.

        Returns:
            mean: Float tensor of shape ``(N,)`` (or ``(N, ntargets)`` for
                multi-target models) with the per-structure predictive mean.
            std:  Float tensor of the same shape with the per-structure predictive
                standard deviation.

        Raises:
            ValueError: If ``n_passes < 2`` (std of a single sample is undefined).
        """
        from pymatgen.core import Molecule as _Molecule
        from pymatgen.core import Structure as _Structure
        from torch_geometric.data import Batch as _Batch

        if n_passes < 2:
            raise ValueError(f"n_passes must be ≥ 2, got {n_passes}")

        single = isinstance(structures, (_Structure, _Molecule))
        if single:
            structures = [structures]

        graphs = [self._to_graph(s, graph_converter) for s in structures]
        batch = _Batch.from_data_list([g for g, _ in graphs])
        sfs = [sf for _, sf in graphs]
        batch_sf = torch.stack(sfs) if sfs[0].numel() > 0 else None

        # Run on the model's device (enables GPU inference; no-op for CPU models).
        try:
            device = next(self.model.parameters()).device
            batch = batch.to(device)
            if batch_sf is not None:
                batch_sf = batch_sf.to(device)
        except StopIteration:  # model with no parameters — leave tensors as-is
            pass

        cached = self._predict_cached(batch, batch_sf, n_passes) if cache_backbone else None
        if cached is not None:
            mean, std = cached
        else:
            with self._stochastic_mode(), torch.no_grad():
                samples = [self.model(g=batch, state_attr=batch_sf) for _ in range(n_passes)]
            stacked = torch.stack(samples)  # (n_passes, N) or (n_passes, N, ntargets)
            mean, std = stacked.mean(dim=0), stacked.std(dim=0)

        if single:
            return mean.squeeze(0), std.squeeze(0)
        return mean, std

    def _predict_cached(self, batch, batch_sf, n_passes: int) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Backbone-once MC Dropout fast path.

        Evaluate the deterministic backbone once, then replay only the stochastic
        readout head ``n_passes`` times.

        A pre/forward hook on ``self._head_module`` captures the head's (deterministic)
        input and output during one backbone pass. The fast path is taken only if the
        head receives a single tensor argument **and** its output equals the model
        output (i.e. the head is terminal) — otherwise this returns ``None`` and the
        caller uses the naive per-pass loop. The cached input is tiled along the batch
        axis so the N stochastic head evaluations run as a single forward with
        independent dropout masks.

        Returns ``(mean, std)`` on success, or ``None`` if the fast path is not
        provably exact for this model.
        """
        head = self._head_module
        if head is None:
            return None

        cap: dict[str, torch.Tensor | None] = {}

        def pre_hook(_module, args, kwargs):
            single_tensor_arg = len(args) == 1 and not kwargs and isinstance(args[0], torch.Tensor)
            cap["in"] = args[0].detach() if single_tensor_arg else None

        def fwd_hook(_module, _args, output):
            cap["out"] = output.detach() if isinstance(output, torch.Tensor) else None

        ph = head.register_forward_pre_hook(pre_hook, with_kwargs=True)
        fh = head.register_forward_hook(fwd_hook)
        try:
            self.model.eval()  # deterministic backbone (no stochastic layers active)
            with torch.no_grad():
                model_out = self.model(g=batch, state_attr=batch_sf)
        finally:
            ph.remove()
            fh.remove()
            self.model.eval()

        head_in, head_out = cap.get("in"), cap.get("out")
        if head_in is None or head_out is None:
            return None
        # Only exact when the head is the model's terminal op. The model may apply a
        # value-preserving reshape/squeeze after the head (e.g. M3GNet squeezes the
        # trailing target dim), so compare modulo squeeze.
        if head_out.squeeze().shape != model_out.squeeze().shape or not torch.allclose(
            head_out.squeeze(), model_out.squeeze(), atol=1e-5, rtol=1e-4
        ):
            return None

        n_struct = head_in.shape[0]
        with self._stochastic_mode(), torch.no_grad():
            tiled = head_in.repeat(n_passes, *([1] * (head_in.dim() - 1)))  # (n_passes * N, ...)
            out = head(tiled)
        out = out.view(n_passes, n_struct, *out.shape[1:])
        # Reproduce the model's post-head shape (the squeeze validated above).
        return out.mean(dim=0).reshape(model_out.shape), out.std(dim=0).reshape(model_out.shape)
