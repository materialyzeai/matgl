"""Benchmark: JAX/XLA vs eager-PyTorch (vs torch.compile) for TensorNet inference.

Measures the per-step wall time of a full energy + forces + stress evaluation --
the inner loop of an ASE MD / relaxation -- across system sizes.

    .venv/bin/python jax_prototype/benchmarks/bench.py
    .venv/bin/python jax_prototype/benchmarks/bench.py --torch-compile --xlarge

XLA compile latency is reported separately from steady-state throughput. All
backends share one set of (random) TensorNet weights, so the comparison is the
graph-kernel/dispatch overhead, not the model.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
from ase.calculators.calculator import all_changes
from matgl.apps.pes import Potential
from matgl.config import DEFAULT_ELEMENTS
from matgl.ext.ase import PESCalculator
from matgl.models import QET, TensorNet
from pymatgen.io.ase import AseAtomsAdaptor

sys.path.insert(0, str(Path(__file__).parents[1]))
sys.path.insert(0, str(Path(__file__).parent))
from structures import make_structures

from matgl_jax import JAXPESCalculator

PROPS = ["energy", "forces", "stress"]


def build_potential(model_type: str = "tensornet", seed: int = 0, **compile_kwargs) -> Potential:
    """A TensorNet/QET (SphericalBessel-smooth) wrapped in a Potential."""
    torch.manual_seed(seed)
    common = {
        "element_types": DEFAULT_ELEMENTS,
        "units": 64,
        "nblocks": 2,
        "cutoff": 5.0,
        "use_warp": False,
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


def time_calc(calc, atoms, n_warmup: int, n_iter: int) -> tuple[float, float]:
    """Return ``(first_call_seconds, steady_state_seconds_per_step)``."""
    t0 = time.perf_counter()
    calc.calculate(atoms, PROPS, all_changes)
    first = time.perf_counter() - t0
    for _ in range(max(0, n_warmup - 1)):
        calc.calculate(atoms, PROPS, all_changes)
    t0 = time.perf_counter()
    for _ in range(n_iter):
        calc.calculate(atoms, PROPS, all_changes)
    return first, (time.perf_counter() - t0) / n_iter


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["tensornet", "qet"], default="tensornet")
    parser.add_argument("--torch-compile", action="store_true", help="also benchmark torch.compile")
    parser.add_argument("--xlarge", action="store_true", help="include the ~1000-atom cell")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    args = parser.parse_args()

    import jax

    print(f"model: {args.model}   jax devices: {jax.devices()}   torch threads: {torch.get_num_threads()}")
    pot = build_potential(args.model)

    rows = []
    for name, struct in make_structures(include_xlarge=args.xlarge).items():
        atoms = AseAtomsAdaptor.get_atoms(struct)
        n_atoms = len(atoms)

        eager = PESCalculator(pot, stress_unit="GPa")
        _, eager_ms = time_calc(eager, atoms, args.warmup, args.iters)
        n_edges = int(eager._atoms2graph.get_graph(atoms)[0].edge_index.shape[1])

        # graph build is CPU-bound (pymatgen neighbour list) and shared by every
        # backend — time it on its own so the model speedup can be isolated.
        for _ in range(args.warmup):
            eager._atoms2graph.get_graph(atoms)
        t0 = time.perf_counter()
        for _ in range(args.iters):
            eager._atoms2graph.get_graph(atoms)
        graph_ms = (time.perf_counter() - t0) / args.iters * 1e3

        jax_calc = JAXPESCalculator(pot, stress_unit="GPa")
        jax_compile, jax_ms = time_calc(jax_calc, atoms, args.warmup, args.iters)

        # correctness guard: JAX must agree with eager within float32 noise
        ediff = abs(eager.results["energy"] - jax_calc.results["energy"])

        # subtract the shared graph-build cost to isolate the model speedup
        gb = graph_ms / 1e3
        model_speedup = (eager_ms - gb) / max(jax_ms - gb, 1e-9)
        row = {
            "system": name,
            "atoms": n_atoms,
            "edges": n_edges,
            "graph_ms": graph_ms,
            "eager_ms": eager_ms * 1e3,
            "jax_ms": jax_ms * 1e3,
            "jax_compile_s": jax_compile,
            "speedup": eager_ms / jax_ms,
            "model_speedup": model_speedup,
            "Ediff": ediff,
        }
        if args.torch_compile:
            cpot = build_potential(args.model, compile_model=True)
            ccalc = PESCalculator(cpot, stress_unit="GPa")
            _, c_ms = time_calc(ccalc, atoms, max(args.warmup, 3), args.iters)
            row["tcompile_ms"] = c_ms * 1e3
        rows.append(row)

    hdr = f"{'system':13s}{'atoms':>6s}{'edges':>7s}{'graph ms':>10s}{'eager ms':>10s}{'jax ms':>9s}"
    hdr += f"{'compile s':>10s}{'speedup':>9s}{'model spd':>10s}"
    if args.torch_compile:
        hdr += f"{'tcompile ms':>13s}"
    print("\n" + hdr)
    print("-" * len(hdr))
    for r in rows:
        line = f"{r['system']:13s}{r['atoms']:>6d}{r['edges']:>7d}{r['graph_ms']:>10.2f}"
        line += f"{r['eager_ms']:>10.2f}{r['jax_ms']:>9.2f}{r['jax_compile_s']:>10.2f}"
        line += f"{r['speedup']:>8.2f}x{r['model_speedup']:>9.2f}x"
        if "tcompile_ms" in r:
            line += f"{r['tcompile_ms']:>13.2f}"
        print(line)
    print()
    print("ms/step = one energy+forces+stress eval (lower is better).")
    print("speedup = end-to-end vs eager;  model spd = vs eager after subtracting graph build.")
    maxe = max(r["Ediff"] for r in rows)
    print(f"max |E_eager - E_jax| across systems: {maxe:.2e} eV  (float32 parity check)")


if __name__ == "__main__":
    main()
