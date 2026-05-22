"""Convert a PyTorch ``TensorNet`` / ``QET`` (or ``Potential``) to a JAX pytree.

The JAX side keeps weights in a nested ``dict`` keyed to mirror the PyTorch
``state_dict``. The only non-trivial transforms are:

* ``nn.Linear``  -- torch stores ``weight`` as ``(out, in)`` and computes
  ``x @ W.T``; we store the transpose ``(in, out)`` and compute ``x @ W``.
* ``nn.LayerNorm`` / ``nn.Embedding`` -- copied verbatim.
* Bessel-root buffers are not in the ``state_dict``; they are rebuilt by
  :mod:`._basis`.
"""

from __future__ import annotations

import jax.numpy as jnp

from ._basis import build_basis_config

_ACT_NAME = {
    "SiLU": "swish",
    "Tanh": "tanh",
    "Sigmoid": "sigmoid",
    "Softplus": "softplus",
    "SoftPlus2": "softplus2",
}


def _arr(t):
    return jnp.asarray(t.detach().cpu().numpy())


def _lin(sd, prefix, bias=True):
    """Convert an ``nn.Linear`` — stored transposed so JAX computes ``x @ W``."""
    p = {"w": _arr(sd[f"{prefix}.weight"].T)}
    if bias and f"{prefix}.bias" in sd:
        p["b"] = _arr(sd[f"{prefix}.bias"])
    return p


def _norm(sd, prefix):
    """Convert an ``nn.LayerNorm``."""
    return {"w": _arr(sd[f"{prefix}.weight"]), "b": _arr(sd[f"{prefix}.bias"])}


def _collect(sd, prefix, bias=True):
    """Convert every ``nn.Linear`` under ``prefix`` (an ``nn.ModuleList``/``Sequential``)."""
    idxs = sorted(
        {int(k[len(prefix) + 1 :].split(".")[0]) for k in sd if k.startswith(prefix + ".") and k.endswith(".weight")}
    )
    return [_lin(sd, f"{prefix}.{i}", bias=bias) for i in idxs]


def _distance_projs(sd, prefix):
    """Convert the embedding's three distance projections.

    The plain PyG ``TensorEmbedding`` keeps them as three separate ``Linear``
    layers (``distance_proj1/2/3``); the Warp-accelerated ``TensorEmbedding``
    fuses them into one ``Linear(rbf, 3*units)`` (``distance_proj``) whose weight
    is the row-concatenation of the three. Accept either so a Warp-enabled
    TensorNet converts to the same JAX pytree as its PyG twin.
    """
    if f"{prefix}.distance_proj.weight" in sd:  # fused Warp layout
        w, b = sd[f"{prefix}.distance_proj.weight"], sd[f"{prefix}.distance_proj.bias"]
        units = w.shape[0] // 3
        return {
            f"distance_proj{i + 1}": {
                "w": _arr(w[i * units : (i + 1) * units].T),
                "b": _arr(b[i * units : (i + 1) * units]),
            }
            for i in range(3)
        }
    return {f"distance_proj{i}": _lin(sd, f"{prefix}.distance_proj{i}") for i in (1, 2, 3)}


def build_config(model) -> dict:
    """Static architecture config (Python scalars + non-learned basis arrays)."""
    return {
        "units": int(model.units),
        "num_layers": int(model.num_layers),
        "cutoff": float(model.cutoff),
        "group": model.equivariance_invariance_group,
        "activation": _ACT_NAME[type(model.activation).__name__],
        "is_intensive": bool(model.is_intensive),
        "basis": build_basis_config(model),
        "model_type": type(model).__name__,
        # QET-only fields (harmless defaults for plain TensorNet)
        "is_hardness_envs": bool(getattr(model, "is_hardness_envs", False)),
        "include_magmom": bool(getattr(model, "include_magmom", False)),
        "total_charge": 0.0,
    }


def convert_tensornet(model) -> dict:
    """Convert a torch ``TensorNet`` to a JAX parameter pytree."""
    sd = model.state_dict()
    emb = "tensor_embedding"
    params: dict = {
        "tensor_embedding": {
            **_distance_projs(sd, emb),
            "emb": _arr(sd[f"{emb}.emb.weight"]),
            "emb2": _lin(sd, f"{emb}.emb2"),
            "linears_tensor": _collect(sd, f"{emb}.linears_tensor", bias=False),
            "linears_scalar": _collect(sd, f"{emb}.linears_scalar", bias=True),
            "init_norm": _norm(sd, f"{emb}.init_norm"),
        },
        "layers": [
            {
                "linears_scalar": _collect(sd, f"layers.{i}.linears_scalar", bias=True),
                "linears_tensor": _collect(sd, f"layers.{i}.linears_tensor", bias=False),
            }
            for i in range(int(model.num_layers))
        ],
        "out_norm": _norm(sd, "out_norm"),
        "linear": _lin(sd, "linear"),
    }
    if model.is_intensive:
        params["readout"] = {
            "mlp": _collect(sd, "readout.mlp.layers", bias=True),
            "weight": _collect(sd, "readout.weight.layers", bias=True),
        }
        params["final_layer"] = _collect(sd, "final_layer.layers", bias=True)
    else:
        params["final_layer"] = {
            "value": _collect(sd, "final_layer.gated.layers", bias=True),
            "gate": _collect(sd, "final_layer.gated.gates", bias=True),
        }
    return params


def convert_qet(model) -> dict:
    """Convert a torch ``QET`` to a JAX parameter pytree.

    Reuses :func:`convert_tensornet` for the shared feature stack (and the wider
    ``final_layer`` gated readout) and adds the charge-equilibration head.
    """
    params = convert_tensornet(model)
    sd = model.state_dict()
    params["chi_readout"] = _collect(sd, "chi_readout.layers", bias=True)
    if model.is_hardness_envs:
        params["hardness_readout"] = _collect(sd, "hardness_readout.layers", bias=True)
    else:
        params["hardness_readout"] = _arr(sd["hardness_readout"])  # per-element nn.Parameter
    params["sigma"] = _arr(sd["sigma"]) if "sigma" in sd else _arr(model.sigma)
    params["norm"] = _norm(sd, "norm")
    if model.include_magmom:
        params["magmom_readout"] = _collect(sd, "magmom_readout.layers", bias=True)
    return params


def convert_potential(potential) -> tuple[dict, dict, dict]:
    """Convert a torch ``Potential`` (wrapping TensorNet/QET) to JAX.

    Returns ``(params, cfg, extras)`` where ``extras`` carries the
    denormalisation scalars and the optional per-element reference offsets.
    """
    model = potential.model
    cfg = build_config(model)
    params = convert_qet(model) if cfg["model_type"] == "QET" else convert_tensornet(model)
    extras = {
        "data_mean": float(potential.data_mean),
        "data_std": float(potential.data_std),
        "element_refs": (_arr(potential.element_refs.property_offset) if potential.element_refs is not None else None),
    }
    return params, cfg, extras
