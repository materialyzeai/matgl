"""Validate LAMMPS hydrostatic pressure against a finite energy derivative."""

from __future__ import annotations

import argparse
import math
from pathlib import Path


EV_PER_ANGSTROM3_TO_BAR = 1.602176634e6


def read_thermo(path: Path) -> dict[str, float]:
    """Return the last thermo row containing PotEng and pressure columns."""
    header: list[str] | None = None
    result: dict[str, float] | None = None

    for line in path.read_text().splitlines():
        fields = line.split()
        if fields and fields[0] == "Step" and "PotEng" in fields and "Press" in fields:
            header = fields
            continue
        if header is None or len(fields) != len(header):
            continue
        try:
            values = [float(value) for value in fields]
        except ValueError:
            continue
        result = dict(zip(header, values))

    if result is None:
        raise ValueError(f"Could not find a pressure thermo row in {path}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("minus", type=Path, help="LAMMPS log at scale 1-epsilon")
    parser.add_argument("zero", type=Path, help="LAMMPS log at scale 1")
    parser.add_argument("plus", type=Path, help="LAMMPS log at scale 1+epsilon")
    parser.add_argument("--epsilon", type=float, default=0.01)
    parser.add_argument("--relative-tolerance", type=float, default=0.05)
    parser.add_argument("--absolute-tolerance-bar", type=float, default=10.0)
    args = parser.parse_args()

    minus = read_thermo(args.minus)
    zero = read_thermo(args.zero)
    plus = read_thermo(args.plus)

    # lambda uniformly scales all three cell vectors and fractional atomic
    # coordinates.  At lambda=1, dE/dlambda is the trace of dE/dstrain.
    # LAMMPS pressure is -trace(dE/dstrain)/(3V).
    energy_derivative = (plus["PotEng"] - minus["PotEng"]) / (2 * args.epsilon)
    finite_difference_pressure = (
        -energy_derivative / (3 * zero["Volume"]) * EV_PER_ANGSTROM3_TO_BAR
    )
    diagonal_pressure = (zero["Pxx"] + zero["Pyy"] + zero["Pzz"]) / 3

    print(f"E(1-epsilon) = {minus['PotEng']:.12g} eV")
    print(f"E(1+epsilon) = {plus['PotEng']:.12g} eV")
    print(f"dE/dlambda   = {energy_derivative:.12g} eV")
    print(f"FD pressure  = {finite_difference_pressure:.12g} bar")
    print(f"LAMMPS press = {zero['Press']:.12g} bar")

    if not math.isclose(
        zero["Press"],
        diagonal_pressure,
        rel_tol=1e-10,
        abs_tol=1e-6,
    ):
        raise AssertionError(
            f"Press ({zero['Press']}) does not equal mean diagonal pressure "
            f"({diagonal_pressure})"
        )

    if not math.isclose(
        zero["Press"],
        finite_difference_pressure,
        rel_tol=args.relative_tolerance,
        abs_tol=args.absolute_tolerance_bar,
    ):
        raise AssertionError(
            "LAMMPS pressure does not match the finite-difference energy "
            f"derivative: {zero['Press']} vs {finite_difference_pressure} bar"
        )


if __name__ == "__main__":
    main()
