"""Benchmark: JAX/XLA vs eager-PyTorch (vs warp/torch.compile) for TensorNet/QET inference.

Measures the per-step wall time of a full energy + forces + stress evaluation --
the inner loop of an ASE MD / relaxation -- across system sizes.

    python dev/jax_benchmark.py
    python dev/jax_benchmark.py --model qet --torch-compile --xlarge
    python dev/jax_benchmark.py --pretrained TensorNet-PES-MatPES-r2SCAN-2025.2

    # CI matrix mode: time a single backend on GPU and emit JSON for aggregation.
    python dev/jax_benchmark.py --variant warp --device cuda --json out.json

Requires the optional ``jax`` extra (``pip install matgl[jax]``) for the JAX
variant and ``matgl[ops]`` (``nvalchemi-toolkit-ops``) for the warp variant.
XLA compile latency is reported separately from steady-state throughput.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from ase.calculators.calculator import all_changes
from pymatgen.core import Lattice, Structure
from pymatgen.io.ase import AseAtomsAdaptor

from matgl.apps.pes import Potential
from matgl.config import DEFAULT_ELEMENTS
from matgl.ext.ase import PESCalculator
from matgl.models import QET, TensorNet

PROPS = ["energy", "forces", "stress"]

_SI_CONV = Structure.from_spacegroup("Fd-3m", Lattice.cubic(5.43), ["Si"], [[0.0, 0.0, 0.0]])  # 8 atoms


def make_structures(include_xlarge: bool = False) -> dict[str, Structure]:
    """Return ``name -> Structure`` covering a range of system sizes."""
    out: dict[str, Structure] = {
        "tiny-2": Structure(Lattice.cubic(3.0), ["Si", "Si"], [[0, 0, 0], [0.5, 0.5, 0.5]]),
    }
    for label, mult in [("small-64", 2), ("medium-216", 3), ("large-512", 4)]:
        s = _SI_CONV.copy()
        s.make_supercell([mult, mult, mult])
        out[label] = s
    if include_xlarge:
        s = _SI_CONV.copy()
        s.make_supercell([5, 5, 5])
        out["xlarge-1000"] = s
    return out


def build_potential(
    model_type: str = "tensornet", seed: int = 0, use_warp: bool = False, **compile_kwargs
) -> Potential:
    """A TensorNet/QET (SphericalBessel-smooth) wrapped in a Potential."""
    torch.manual_seed(seed)
    common = {
        "element_types": DEFAULT_ELEMENTS,
        "units": 64,
        "nblocks": 2,
        "cutoff": 5.0,
        "use_warp": use_warp,
        "rbf_type": "SphericalBessel",
        "use_smooth": True,
        "max_n": 8,
        "max_l": 3,
    }
    model = QET(**common) if model_type == "qet" else TensorNet(**common, is_intensive=False)
    model.eval()
    pot = Potential(model=model, calc_forces=True, calc_stresses=True, **compile_kwargs)
    pot.eval()
    return pot


def _cuda_sync(device: str) -> None:
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()


def time_calc(calc, atoms, n_warmup: int, n_iter: int, device: str = "cpu") -> tuple[float, float]:
    """Return ``(first_call_seconds, steady_state_seconds_per_step)``."""
    _cuda_sync(device)
    t0 = time.perf_counter()
    calc.calculate(atoms, PROPS, all_changes)
    _cuda_sync(device)
    first = time.perf_counter() - t0
    for _ in range(max(0, n_warmup - 1)):
        calc.calculate(atoms, PROPS, all_changes)
    _cuda_sync(device)
    t0 = time.perf_counter()
    for _ in range(n_iter):
        calc.calculate(atoms, PROPS, all_changes)
    _cuda_sync(device)
    return first, (time.perf_counter() - t0) / n_iter


def _resolve_pot(args, *, use_warp: bool, device: str) -> Potential:
    """Build (or load) the potential, move to device, optionally enable warp."""
    if args.pretrained:
        import matgl

        pot = matgl.load_model(args.pretrained)
        # Pretrained TensorNet/QET checkpoints expose ``_use_warp``; if the
        # warp variant was requested against a non-warp checkpoint, bail out
        # rather than silently producing comparable-looking but wrong numbers.
        if use_warp and hasattr(pot.model, "_use_warp") and not pot.model._use_warp:
            raise SystemExit(
                "--variant=warp with --pretrained requires a checkpoint whose layers were "
                "built with use_warp=True. Re-load via build_potential() instead."
            )
        pot.eval()
    else:
        pot = build_potential(args.model, use_warp=use_warp)
    pot.to(device)
    return pot


def main() -> None:
    """Run the variant benchmark(s) and print a summary table."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["tensornet", "qet"], default="tensornet")
    parser.add_argument(
        "--variant",
        choices=["eager", "warp", "jax", "all"],
        default="all",
        help="Backend(s) to time. 'all' runs every variant available; the others run only that one"
        " and emit a single-column table (CI/matrix mode).",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cpu",
        help="Torch device for the eager and warp variants. JAX uses its own device picker.",
    )
    parser.add_argument("--torch-compile", action="store_true", help="also benchmark torch.compile")
    parser.add_argument("--xlarge", action="store_true", help="include the ~1000-atom cell")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument(
        "--pretrained",
        default=None,
        help="load a pretrained model by name (e.g. TensorNet-PES-MatPES-r2SCAN-2025.2) instead of random weights",
    )
    parser.add_argument(
        "--json",
        default=None,
        help="If set, write the per-system timings to this JSON path for downstream aggregation.",
    )
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device=cuda requested but torch.cuda.is_available() is False")

    want = {"eager", "warp", "jax"} if args.variant == "all" else {args.variant}

    # When ``--variant all`` is used on a machine without ``nvalchemi-toolkit-ops``,
    # quietly drop the warp variant so the default invocation still produces an
    # eager-vs-JAX table. An explicit ``--variant warp`` still fails loudly.
    if args.variant == "all":
        try:
            from matgl.models._tensornet import _warp_available
        except Exception:
            _warp_available = False
        if not _warp_available:
            want.discard("warp")

    pot_eager = _resolve_pot(args, use_warp=False, device=args.device) if {"eager", "jax"} & want else None
    pot_warp = _resolve_pot(args, use_warp=True, device=args.device) if "warp" in want else None
    label = args.pretrained or f"{args.model} (random weights)"

    info = {
        "model": label,
        "device": args.device,
        "variant": args.variant,
        "torch_threads": torch.get_num_threads(),
    }
    if "jax" in want:
        import jax

        info["jax_devices"] = [str(d) for d in jax.devices()]
    print(
        f"model: {label}   variant: {args.variant}   device: {args.device}   "
        f"torch threads: {torch.get_num_threads()}"
    )
    if "jax_devices" in info:
        print(f"jax devices: {info['jax_devices']}")

    rows = []
    for name, struct in make_structures(include_xlarge=args.xlarge).items():
        atoms = AseAtomsAdaptor.get_atoms(struct)
        n_atoms = len(atoms)
        row: dict = {"system": name, "atoms": n_atoms}

        # graph-build cost is shared by every backend; always measure it so the
        # JSON output is self-describing for the aggregator.
        # Use whichever potential we have to source an Atoms2Graph; both share the
        # same converter when element_types/cutoff match.
        ref_calc = PESCalculator(pot_eager or pot_warp, stress_unit="GPa")
        n_edges = int(ref_calc._atoms2graph.get_graph(atoms)[0].edge_index.shape[1])
        for _ in range(args.warmup):
            ref_calc._atoms2graph.get_graph(atoms)
        t0 = time.perf_counter()
        for _ in range(args.iters):
            ref_calc._atoms2graph.get_graph(atoms)
        row["edges"] = n_edges
        row["graph_ms"] = (time.perf_counter() - t0) / args.iters * 1e3

        if "eager" in want:
            eager = PESCalculator(pot_eager, stress_unit="GPa")
            _, t = time_calc(eager, atoms, args.warmup, args.iters, device=args.device)
            row["eager_ms"] = t * 1e3
            row["eager_energy"] = float(eager.results["energy"])
            row["eager_forces"] = eager.results["forces"].tolist()
        if "warp" in want:
            warp_calc = PESCalculator(pot_warp, stress_unit="GPa")
            _, t = time_calc(warp_calc, atoms, args.warmup, args.iters, device=args.device)
            row["warp_ms"] = t * 1e3
            row["warp_energy"] = float(warp_calc.results["energy"])
            row["warp_forces"] = warp_calc.results["forces"].tolist()
        if "jax" in want:
            from matgl.ext.jax import JAXPESCalculator

            jax_calc = JAXPESCalculator(pot_eager, stress_unit="GPa")
            tc, t = time_calc(jax_calc, atoms, args.warmup, args.iters)
            row["jax_ms"] = t * 1e3
            row["jax_compile_s"] = tc
            row["jax_energy"] = float(jax_calc.results["energy"])
            row["jax_forces"] = jax_calc.results["forces"].tolist()

        if args.torch_compile and not args.pretrained and "eager" in want:
            cpot = build_potential(args.model, compile_model=True).to(args.device)
            ccalc = PESCalculator(cpot, stress_unit="GPa")
            _, c_ms = time_calc(ccalc, atoms, max(args.warmup, 3), args.iters, device=args.device)
            row["tcompile_ms"] = c_ms * 1e3

        # derived comparisons when we have both columns available
        if {"eager", "jax"} <= want:
            gb = row["graph_ms"] / 1e3
            row["speedup_jax_vs_eager"] = (row["eager_ms"] / 1e3) / (row["jax_ms"] / 1e3)
            row["model_speedup_jax_vs_eager"] = max(row["eager_ms"] / 1e3 - gb, 1e-12) / max(
                row["jax_ms"] / 1e3 - gb, 1e-12
            )
            row["Ediff_jax"] = abs(row["eager_energy"] - row["jax_energy"])
            row["Fdiff_jax"] = float(
                np.abs(np.asarray(row["eager_forces"]) - np.asarray(row["jax_forces"])).max()
            )
        if {"eager", "warp"} <= want:
            row["speedup_warp_vs_eager"] = (row["eager_ms"] / 1e3) / (row["warp_ms"] / 1e3)
            row["Ediff_warp"] = abs(row["eager_energy"] - row["warp_energy"])
            row["Fdiff_warp"] = float(
                np.abs(np.asarray(row["eager_forces"]) - np.asarray(row["warp_forces"])).max()
            )
        rows.append(row)

    _print_table(rows, want, include_tcompile=bool(args.torch_compile and "eager" in want))

    if args.json:
        # strip the raw force arrays before serialising so the artifact stays small
        slim = []
        for r in rows:
            sr = {k: v for k, v in r.items() if not k.endswith("_forces")}
            slim.append(sr)
        Path(args.json).write_text(json.dumps({"info": info, "rows": slim}, indent=2))
        print(f"wrote {args.json}")


def _print_table(rows: list[dict], want: set[str], include_tcompile: bool) -> None:
    # (key, width, header, fmt) -- fmt is applied via format() on the value
    cols: list[tuple[str, int, str, str]] = [
        ("system", 13, "system", "<13s"),
        ("atoms", 6, "atoms", ">6d"),
        ("edges", 7, "edges", ">7d"),
        ("graph_ms", 10, "graph ms", ">10.2f"),
    ]
    if "eager" in want:
        cols.append(("eager_ms", 10, "eager ms", ">10.2f"))
    if "warp" in want:
        cols.append(("warp_ms", 10, "warp ms", ">10.2f"))
    if "jax" in want:
        cols.append(("jax_ms", 9, "jax ms", ">9.2f"))
        cols.append(("jax_compile_s", 10, "compile s", ">10.2f"))
    if include_tcompile:
        cols.append(("tcompile_ms", 13, "tcompile ms", ">13.2f"))
    if {"eager", "jax"} <= want:
        cols.append(("speedup_jax_vs_eager", 9, "jax x", ">9.2f"))
    if {"eager", "warp"} <= want:
        cols.append(("speedup_warp_vs_eager", 9, "warp x", ">9.2f"))

    hdr = "".join(f"{label:>{width}s}" for _, width, label, _ in cols)
    print("\n" + hdr)
    print("-" * len(hdr))
    for r in rows:
        line = ""
        for key, width, _label, fmt in cols:
            v = r.get(key)
            line += " " * width if v is None else format(v, fmt)
        print(line)
    print()
    print("ms/step = one energy+forces+stress eval (lower is better).")
    if {"eager", "jax"} <= want:
        maxe = max(r.get("Ediff_jax", 0.0) for r in rows)
        maxf = max(r.get("Fdiff_jax", 0.0) for r in rows)
        print(f"parity jax vs eager (float32): max |dE| {maxe:.2e} eV, max |dF| {maxf:.2e} eV/A")
    if {"eager", "warp"} <= want:
        maxe = max(r.get("Ediff_warp", 0.0) for r in rows)
        maxf = max(r.get("Fdiff_warp", 0.0) for r in rows)
        print(f"parity warp vs eager:          max |dE| {maxe:.2e} eV, max |dF| {maxf:.2e} eV/A")


if __name__ == "__main__":
    main()
