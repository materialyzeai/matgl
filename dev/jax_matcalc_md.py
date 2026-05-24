"""Benchmark: MatCalc ``MDCalc`` 1000-step NVT MD of LiFePO4 — JAX vs eager PyTorch.

Runs the same MD through MatCalc's ``MDCalc`` PropCalc with matgl's standard eager
``PESCalculator`` and with the JAX-accelerated ``JAXPESCalculator``
(``matgl.ext.jax``), and reports the wall-time difference.

The JAX path is timed twice: "cold" (the first call pays a one-time XLA
compilation) and "warm" (steady state, compilation reused). The warm number is
the per-step cost a longer trajectory converges to.

    python dev/jax_matcalc_md.py

Requires: ``pip install matgl[jax] matcalc``.
"""

from __future__ import annotations

import time
import warnings

warnings.filterwarnings("ignore")

import matgl  # noqa: E402
from matcalc import MDCalc  # noqa: E402
from matgl.ext.ase import PESCalculator  # noqa: E402
from matgl.ext.jax import JAXPESCalculator  # noqa: E402

STEPS = 1000
MODEL = "QET-PES-MatPES-r2SCAN-2025.2"


def get_lifepo4():
    """Return the pymatgen LiFePO4 (olivine) test structure."""
    try:
        from pymatgen.util.testing import MatSciTest

        return MatSciTest().get_structure("LiFePO4")
    except ImportError:
        from pymatgen.util.testing import PymatgenTest

        return PymatgenTest().get_structure("LiFePO4")


def run_md(calculator, structure) -> float:
    """Run the NVT MD via MatCalc MDCalc; return the wall time in seconds."""
    md = MDCalc(
        calculator=calculator,
        ensemble="nvt",
        temperature=300,
        timestep=1.0,
        steps=STEPS,
        relax_structure=False,  # time only the MD steps
        logfile=None,
    )
    t0 = time.perf_counter()
    md.calc(structure)
    return time.perf_counter() - t0


def main() -> None:
    """Run the eager-vs-JAX MD benchmark and print the comparison."""
    structure = get_lifepo4()
    potential = matgl.load_model(MODEL)
    potential.eval()
    print(
        f"MatCalc MDCalc — {STEPS}-step NVT MD of {structure.composition.reduced_formula} "
        f"({len(structure)} atoms), model {MODEL}\n"
    )

    t_eager = run_md(PESCalculator(potential, stress_unit="eV/A3"), structure)

    # One JAXPESCalculator instance: the jitted fn (and its XLA compilation) is
    # cached on it, so the second run reuses the compiled program.
    jax_calc = JAXPESCalculator(potential, stress_unit="eV/A3")
    t_jax_cold = run_md(jax_calc, structure)
    t_jax_warm = run_md(jax_calc, structure)

    def line(label, t):
        print(f"  {label:28s} {t:8.2f} s   ({t / STEPS * 1e3:7.1f} ms/step)")

    line("eager PyTorch", t_eager)
    line("JAX/XLA  (cold, 1st run)", t_jax_cold)
    line("JAX/XLA  (warm, compiled)", t_jax_warm)
    print(
        f"\n  XLA compile (one-time):   ~{t_jax_cold - t_jax_warm:.1f} s"
        f"\n  speedup cold (1st run):   {t_eager / t_jax_cold:.2f}x"
        f"\n  speedup warm (steady):    {t_eager / t_jax_warm:.2f}x"
    )


if __name__ == "__main__":
    main()
