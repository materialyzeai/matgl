"""Training and inference timing for M3GNet, TensorNet, and GRACE (PYG backend).

Each model is initialised with hyperparameters that put its parameter count
in the 200K-300K range so the comparison is roughly compute-balanced.

This is a *timing* benchmark only — no accuracy is evaluated, no real
dataset is loaded. Synthetic random "labels" drive the loss so the
backward step is a complete autograd traversal.

Usage::

    uv run python scripts/bench/bench_train_infer.py
    uv run python scripts/bench/bench_train_infer.py --batch-size 8 --supercell 3
    uv run python scripts/bench/bench_train_infer.py --steps 30 --warmup 5

The default workload (batch=4, 2x2x2 LiFePO4 supercell ≈ 224 atoms per
batch) is sized to fit on CPU on a laptop in a few seconds per step. Bump
``--supercell`` or ``--batch-size`` for a heavier benchmark.

Numbers reported (per-step wall time in ms): median, p10, p90.
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

# Has to be set before importing torch / cuBLAS-using libs.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from pymatgen.core import Lattice, Structure
from torch import nn

import matgl

assert matgl.config.BACKEND == "PYG", (
    "This benchmark requires the PYG backend (GRACE is PYG-only). Unset MATGL_BACKEND or set it to PYG."
)

from matgl.apps.pes import Potential  # noqa: E402
from matgl.ext.pymatgen import Structure2Graph, get_element_list  # noqa: E402
from matgl.graph.data import MGLDataset, collate_fn_pes  # noqa: E402
from matgl.models import GRACE, M3GNet, TensorNet  # noqa: E402

if TYPE_CHECKING:
    from collections.abc import Callable

# ----------------------------------------------------------------------
# Param-count-targeted model factories. Each one lands in [200K, 300K].
# ----------------------------------------------------------------------


def make_m3gnet(element_types: tuple[str, ...]) -> nn.Module:
    """M3GNet with default hyperparameters (~279K params on 8-element table)."""
    return M3GNet(element_types=element_types)


def make_tensornet(element_types: tuple[str, ...]) -> nn.Module:
    """TensorNet with defaults, ``use_warp=False`` for portability (~218K params)."""
    return TensorNet(
        element_types=element_types,
        is_intensive=False,
        use_warp=False,
    )


def make_grace(element_types: tuple[str, ...]) -> nn.Module:
    """GRACE with library defaults (~213K params on 8-element table).

    The defaults were picked to sit within ~2% of TensorNet's parameter
    count for a roughly compute-balanced comparison.
    """
    return GRACE(element_types=element_types)


MODEL_FACTORIES: dict[str, Callable[[tuple[str, ...]], nn.Module]] = {
    "M3GNet": make_m3gnet,
    "TensorNet": make_tensornet,
    "GRACE": make_grace,
}


# ----------------------------------------------------------------------
# Synthetic dataset: a fixed structure tiled to a supercell, batched.
# ----------------------------------------------------------------------


def build_lifepo4(supercell: int) -> Structure:
    """A LiFePO4-like ordered structure expanded into a supercell.

    LiFePO4 is small (28 atoms unit cell) and chemically diverse enough to
    exercise the embedding tables of all three models.
    """
    lat = Lattice.from_parameters(a=10.32, b=6.0, c=4.69, alpha=90.0, beta=90.0, gamma=90.0)
    species = ["Li", "Fe", "P", "O", "O", "O", "O"]
    coords = [
        [0.0, 0.0, 0.0],
        [0.28, 0.25, 0.97],
        [0.09, 0.25, 0.42],
        [0.10, 0.25, 0.74],
        [0.46, 0.25, 0.21],
        [0.16, 0.05, 0.28],
        [0.16, 0.45, 0.28],
    ]
    s = Structure(lat, species, coords)
    s.make_supercell([supercell, supercell, supercell])
    return s


def build_batch(
    structure: Structure,
    converter: Structure2Graph,
    batch_size: int,
) -> tuple[Any, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build a single batched (Batch, lat, state_attr, e_lab, f_lab, s_lab) tuple.

    All structures in the batch are the same — we just want a representative
    fixed-shape graph so the forward pass cost is stable.
    """
    structures = [structure] * batch_size
    energies = [0.0] * batch_size
    forces = [np.zeros((len(structure), 3)).tolist() for _ in range(batch_size)]
    stresses = [np.zeros((3, 3)).tolist() for _ in range(batch_size)]

    dataset = MGLDataset(
        structures=structures,
        converter=converter,
        labels={"energies": energies, "forces": forces, "stresses": stresses},
        save_cache=False,
    )

    items = [dataset[i] for i in range(batch_size)]
    return collate_fn_pes(items, include_stress=True)
    # batch == (g, lat, state_attr, e, f, s)


# ----------------------------------------------------------------------
# Timing helpers.
# ----------------------------------------------------------------------


@dataclass
class TimingResult:
    median_ms: float
    p10_ms: float
    p90_ms: float
    n: int


def _percentile(times: list[float], q: float) -> float:
    if not times:
        return float("nan")
    times = sorted(times)
    idx = max(0, min(len(times) - 1, int(q * (len(times) - 1))))
    return times[idx]


def time_loop(fn: Callable[[], None], *, warmup: int, steps: int) -> TimingResult:
    for _ in range(warmup):
        fn()
    times: list[float] = []
    for _ in range(steps):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1e3)
    return TimingResult(
        median_ms=_percentile(times, 0.5),
        p10_ms=_percentile(times, 0.1),
        p90_ms=_percentile(times, 0.9),
        n=steps,
    )


# ----------------------------------------------------------------------
# Build the train-step / infer-step closures for a single Potential.
# ----------------------------------------------------------------------


def make_train_step(
    potential: Potential,
    batch: tuple,
    optimizer: torch.optim.Optimizer,
) -> Callable[[], None]:
    """Closure that does forward + MSE on (E, F, S) + backward + optimizer step."""
    g, lat, state_attr, e_lab, f_lab, s_lab = batch
    state_attr = state_attr if state_attr is not None else None

    def step() -> None:
        optimizer.zero_grad(set_to_none=True)
        # Each call needs to re-attach autograd-tracked tensors. Potential mutates
        # ``g.pos`` and ``g.pbc_offshift`` so we leave that to it; the same
        # ``g`` object is reused.
        e, f, s, _h = potential(g=g, lat=lat, state_attr=state_attr)
        # Synthetic squared-error loss across all three heads.
        loss = (
            torch.nn.functional.mse_loss(e.reshape(-1), e_lab.reshape(-1))
            + torch.nn.functional.mse_loss(f, f_lab)
            + torch.nn.functional.mse_loss(s.reshape(-1), s_lab.reshape(-1))
        )
        loss.backward()
        optimizer.step()

    return step


def make_infer_step(potential: Potential, batch: tuple) -> Callable[[], None]:
    """Closure that does a full forward + autograd-grad for forces/stress.

    The Potential's autograd-grad path fires regardless of whether we then
    call ``loss.backward()``; this captures the cost a user pays per
    ``ase.Atoms.get_forces()`` call.
    """
    g, lat, state_attr, _e_lab, _f_lab, _s_lab = batch

    def step() -> None:
        # Drop any cached grad on parameters to mimic a fresh Atoms call.
        for p in potential.parameters():
            p.grad = None
        e, f, s, _h = potential(g=g, lat=lat, state_attr=state_attr)
        # Force materialisation of all three heads (autograd grad already ran
        # inside ``potential.__call__``; we just need to consume the tensors
        # so the compiler/runtime can't elide them).
        _ = (e.detach().sum().item(), f.detach().abs().mean().item(), s.detach().abs().mean().item())

    return step


# ----------------------------------------------------------------------
# Driver.
# ----------------------------------------------------------------------


def count_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def run(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_default_device(args.device)
    torch.set_float32_matmul_precision("high")

    structure = build_lifepo4(args.supercell)
    element_types = get_element_list([structure])
    converter = Structure2Graph(element_types=element_types, cutoff=args.cutoff)

    n_atoms = len(structure) * args.batch_size
    print("=" * 78)
    print("matgl train/infer benchmark")
    print(f"  device       : {args.device}")
    print(f"  backend      : {matgl.config.BACKEND}")
    print(
        f"  structure    : LiFePO4 {args.supercell}x{args.supercell}x{args.supercell} ({len(structure)} atoms / cell)"
    )
    print(f"  batch_size   : {args.batch_size}  ->  {n_atoms} atoms/batch")
    print(f"  cutoff       : {args.cutoff} A")
    print(f"  warmup/steps : {args.warmup}/{args.steps}")
    print(f"  element_types: {element_types}")
    print("=" * 78)

    print()
    header = f"  {'model':<22s} {'#params':>10s} {'train ms (med)':>16s} "
    header += f"{'p10':>8s} {'p90':>8s} {'infer ms (med)':>17s} {'p10':>8s} {'p90':>8s}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    rows: list[tuple[str, int, TimingResult, TimingResult]] = []

    # Variants: ("eager", no compile) and optionally ("compiled", torch.compile).
    # torch.compile around Potential's autograd-grad path can be flaky; on any
    # compile/runtime failure we fall back to printing FAILED for that row so
    # the rest of the table still runs.
    variants: list[str] = ["eager"]
    if args.compile:
        variants.append("compiled")

    for name, factory in MODEL_FACTORIES.items():
        for variant in variants:
            torch.manual_seed(args.seed)
            model = factory(element_types).to(matgl.float_th)
            potential = Potential(model=model, calc_stresses=True).to(matgl.float_th)
            n_params = count_params(potential)
            if variant == "eager" and not (args.min_params <= n_params <= args.max_params):
                print(
                    f"  WARNING: {name} has {n_params:,} params, outside the "
                    f"[{args.min_params:,}, {args.max_params:,}] target window."
                )

            # Build the batch once; reusing the same graph keeps shape-dependent
            # autograd graphs cached and lets us focus on raw forward/backward cost.
            batch = build_batch(structure, converter, args.batch_size)

            optimizer = torch.optim.Adam(potential.parameters(), lr=1e-3)
            run_module: nn.Module = potential
            if variant == "compiled":
                # Compile only the inner graph model. We can't safely compile
                # Potential itself because dynamo doesn't trace cleanly through
                # the autograd-grad call used to compute forces/stresses
                # ("tensor not used in graph" / fake-tensor errors). Compiling
                # the model alone covers the bulk of forward FLOPs and leaves
                # the grad call eager.
                potential.model = torch.compile(potential.model, fullgraph=False, dynamic=False)

            label = f"{name} ({variant})" if args.compile else name

            # Time train and infer independently; either may fail under
            # torch.compile (M3GNet has data-dependent shapes Dynamo can't
            # trace; train double-backward isn't supported by AOTAutograd).
            train_t: TimingResult | None = None
            train_err: str | None = None
            infer_t: TimingResult | None = None
            infer_err: str | None = None

            try:
                train_step = make_train_step(run_module, batch, optimizer)
                train_t = time_loop(train_step, warmup=args.warmup, steps=args.steps)
            except Exception as exc:
                train_err = f"{type(exc).__name__}: {str(exc).splitlines()[0][:80]}"

            potential.eval()
            try:
                infer_step = make_infer_step(run_module, batch)
                infer_t = time_loop(infer_step, warmup=args.warmup, steps=args.steps)
            except Exception as exc:
                infer_err = f"{type(exc).__name__}: {str(exc).splitlines()[0][:80]}"
            potential.train()

            def fmt(t: TimingResult | None) -> str:
                if t is None:
                    return f"{'FAIL':>13s}  {'':>7s} {'':>7s}"
                return f"{t.median_ms:>13.2f}  {t.p10_ms:>7.2f} {t.p90_ms:>7.2f}"

            print(f"  {label:<22s} {n_params:>10,d}   {fmt(train_t)}   {fmt(infer_t)}")
            if variant == "compiled":
                if train_err:
                    print(f"      train FAIL: {train_err}")
                if infer_err:
                    print(f"      infer FAIL: {infer_err}")
            if train_t is not None and infer_t is not None:
                rows.append((label, n_params, train_t, infer_t))

            # Free parameters before next variant / model.
            del potential, model, optimizer, run_module, batch

    print()
    print("Notes:")
    print("  - 'train ms' = forward + MSE(E,F,S) + backward + optimizer.step")
    print("  - 'infer ms' = forward + autograd-grad for forces/stress (no loss.backward)")
    print("  - The same batch is reused every step; no DataLoader / disk I/O.")
    print("  - Run on PYG backend; GRACE is PYG-only so DGL twin is omitted.")
    if args.compile:
        print("  - 'compiled' rows compile only the inner graph model (potential.model)")
        print("    via torch.compile(fullgraph=False). Warmup absorbs compile time.")
        print("  - Compiled-train commonly fails because force/stress losses require")
        print("    double-backward, which AOTAutograd does not support.")
        print("  - M3GNet does not compile (data-dependent shape in the 3-body index path).")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--supercell", type=int, default=2, help="LiFePO4 supercell multiplier (per axis)")
    p.add_argument("--cutoff", type=float, default=5.0)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--steps", type=int, default=15)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--min-params", type=int, default=200_000)
    p.add_argument("--max-params", type=int, default=300_000)
    p.add_argument(
        "--compile",
        action="store_true",
        help="Also time torch.compile(potential) for each model (rows added after the eager rows).",
    )
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
