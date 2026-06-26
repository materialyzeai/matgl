"""Command line interface for matgl."""

from __future__ import annotations

import argparse
import logging
import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from pymatgen.core.structure import Structure
from pymatgen.io.ase import AseAtomsAdaptor

import matgl
from matgl.ext.ase import MolecularDynamics, Relaxer

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence

    from pymatgen.core.sites import PeriodicSite

    from matgl.apps.pes import Potential

warnings.filterwarnings("ignore", category=UserWarning, module="ase")
logger = logging.getLogger("MGL")


def _configure_logging(verbose: bool) -> None:
    """Set up logging configuration once per command execution."""
    if verbose and not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO)


def _load_potential(model_name: str) -> Potential:
    """Load a MatGL model and emit a consistent log message."""
    logger.info("Loading model...")
    return matgl.load_model(model_name)


def _format_lattice_delta(old_lattice: object, new_lattice: object) -> Iterable[str]:
    """Yield formatted lattice-parameter comparisons."""
    for param in ("a", "b", "c", "alpha", "beta", "gamma"):
        yield f"{param}: {getattr(old_lattice, param):.3f} -> {getattr(new_lattice, param):.3f}"


def _format_site_delta(formatter: Callable[[np.ndarray], str], old_site: PeriodicSite, new_site: PeriodicSite) -> str:
    """Return a formatted per-site fractional-coordinate change."""
    return f"{old_site.species}: {formatter(old_site.frac_coords)} -> {formatter(new_site.frac_coords)}"


def relax_structure(args: argparse.Namespace) -> int:
    """Relax one or more crystal structures using a pretrained potential.

    Args:
        args: Parsed CLI arguments carrying `infile`, `model`, and output options.

    Returns:
        Exit status code where ``0`` indicates success.

    Side Effects:
        Writes relaxed structures to disk or prints lattice/site comparisons.
    """
    _configure_logging(args.verbose)

    for fn in args.infile:
        structure = Structure.from_file(fn)

        logger.info("Initial structure\n%s", structure)
        potential = _load_potential(args.model)
        logger.info("Relaxing...")
        relaxer = Relaxer(potential=potential)
        relax_results = relaxer.relax(structure, fmax=0.01)
        final_structure = relax_results["final_structure"]

        if args.suffix:
            p = Path(fn)
            outfn = f"{p.with_suffix('')}{args.suffix}{p.suffix}"
            final_structure.to(filename=outfn)
            print(f"Structure written to {outfn}!")
        elif args.outfile is not None:
            final_structure.to(filename=args.outfile)
            print(f"Structure written to {args.outfile}!")
        else:
            print("Lattice parameters")
            for line in _format_lattice_delta(structure.lattice, final_structure.lattice):
                print(line)
            print("Sites (Fractional coordinates)")

            def fmt_fcoords(fc: np.ndarray) -> str:
                return np.array2string(fc, formatter={"float_kind": lambda x: f"{x:.5f}"})

            for old_site, new_site in zip(structure, final_structure, strict=False):
                print(_format_site_delta(fmt_fcoords, old_site, new_site))

    return 0


def _resolve_state_attributes(state_attr: Sequence[str | int] | None, expected_count: int) -> Sequence[int]:
    """Coerce state attributes to integers and validate lengths."""
    if state_attr is None:
        raise ValueError("State attributes must be supplied for this model.")
    if len(state_attr) != expected_count:
        raise ValueError("Number of state attributes must match the number of input files.")
    return [int(s) for s in state_attr]


def predict_structure(args: argparse.Namespace) -> None:
    """Predict scalar properties for structures or Materials Project IDs.

    Args:
        args: Parsed CLI arguments with `model`, `infile`, or `mpids` selections.

    Side Effects:
        Prints prediction results to stdout.
    """
    model = _load_potential(args.model)
    if args.infile:
        if args.model == "MEGNet-MP-2019.4.1-BandGap-mfi":
            state_dict = ["PBE", "GLLB-SC", "HSE", "SCAN"]
            attrs = _resolve_state_attributes(args.state_attr, len(args.infile))
            for file_path, state in zip(args.infile, attrs, strict=False):
                structure = Structure.from_file(file_path)
                value = model.predict_structure(structure, torch.tensor(state))  # type:ignore[operator]
                print(f"{args.model} prediction for {file_path} with {state_dict[state]} bandgap: {value} eV.")
        else:
            for file_path in args.infile:
                structure = Structure.from_file(file_path)
                value = model.predict_structure(structure)  # type:ignore[operator]
                print(f"{args.model} prediction for {file_path}: {value} eV/atom.")
    if args.mpids:
        # Lazy import: ``MPRester`` lives in the full ``pymatgen`` package which is
        # an optional dep (only ``pymatgen-core`` is required at install time).
        from pymatgen.ext.matproj import MPRester

        mpr = MPRester()
        for material_id in args.mpids:
            structure = mpr.get_structure_by_material_id(material_id)
            value = model.predict_structure(structure)  # type:ignore[operator]
            print(f"{args.model} prediction for {material_id} ({structure.composition.reduced_formula}): {value}.")


def molecular_dynamics(args: argparse.Namespace) -> int:
    """Run molecular dynamics trajectories with MatGL potentials.

    Args:
        args: Parsed CLI arguments containing MD configuration.

    Returns:
        Exit status code where ``0`` indicates success.

    Side Effects:
        Writes trajectory and log files to the current working directory.
    """
    for file in args.infile:
        name = file.split(".")[0]
        structure = Structure.from_file(file)
        adaptor = AseAtomsAdaptor()
        atoms = adaptor.get_atoms(structure)

        logger.info("Initial structure\n%s", structure)
        potential = _load_potential(args.model)
        logger.info("Running MD...")
        MaxwellBoltzmannDistribution(atoms, temperature_K=args.temp)
        md = MolecularDynamics(
            atoms,
            potential=potential,
            ensemble=args.ensemble,
            pressure=args.pressure,
            timestep=args.stepsize,
            trajectory=name + ".traj",
            logfile=name + ".log",
            temperature=args.temp,
            taut=args.taut,
            taup=args.taup,
            friction=args.friction,
            andersen_prob=args.andersen_prob,
            ttime=args.ttime,
            pfactor=args.pfactor,
            external_stress=args.external_stress,
            compressibility_au=args.compressibility_au,
            loginterval=args.loginterval,
            append_trajectory=args.append_trajectory,
            mask=args.mask,
        )
        md.run(args.nsteps)
    return 0


def clear_cache(args: argparse.Namespace) -> None:
    """Clear cache command.

    Args:
        args: Parsed CLI arguments, honoring the `--yes` confirmation override.
    """
    matgl.clear_cache(not args.yes)


def create_lammps_model(args: argparse.Namespace) -> int:
    """Export a MatGL Potential as a LAMMPS-loadable TorchScript artifact.

    Loads the named/local model, wraps it in :class:`LAMMPSMatGLModel`, runs
    ``torch.jit.script``, and writes the result to ``--outfile``. The artifact
    is consumed by the ``pair_matgl`` and ``pair_matgl/kokkos`` LAMMPS pair
    styles via ``torch::jit::load``.

    Args:
        args: Parsed CLI arguments — ``model``, ``outfile``, ``dtype``,
            ``device``, ``no_script``.

    Returns:
        ``0`` on success, ``1`` if the underlying potential is unsupported.
    """
    # Lazy import keeps the CLI responsive when this subcommand isn't used and
    # avoids dragging the export-only deps onto the import path.
    from matgl.ext._lammps import LAMMPSMatGLModel

    dtype_map = {"float32": torch.float32, "float64": torch.float64}
    dtype = dtype_map[args.dtype]

    logger.info("Loading model %s ...", args.model)
    potential = _load_potential(args.model)
    potential.eval()

    if args.device != "cpu":
        potential.to(args.device)

    wrapper = LAMMPSMatGLModel(potential=potential, dtype=dtype)  # type:ignore[arg-type]
    wrapper.eval()

    if args.no_script:
        torch.save(wrapper, args.outfile)
        print(f"Wrote eager wrapper (NOT TorchScript-compiled) to {args.outfile}")
    else:
        scripted = torch.jit.script(wrapper)
        scripted.save(args.outfile)
        print(f"Wrote scripted LAMMPS-MatGL artifact to {args.outfile}")

    print("  r_max     :", wrapper.r_max)
    print("  n_species :", wrapper.n_species)
    print("  dtype     :", args.dtype)
    species = list(potential.model.element_types)  # type:ignore[union-attr,arg-type]
    print("  species   :", species[: wrapper.n_species])
    return 0


def main():
    """Handle main."""
    parser = argparse.ArgumentParser(
        description="""
    This script works based on several sub-commands with their own options. To see the options for the
    sub-commands, type "mgl sub-command -h".""",
        epilog="""Author: MatGL Development Team""",
    )

    subparsers = parser.add_subparsers()

    p_relax = subparsers.add_parser("relax", help="Relax crystal structures.")

    p_relax.add_argument(
        "-i",
        "--infile",
        dest="infile",
        nargs="+",
        required=True,
        help="Input files containing structure. Any format supported by pymatgen's Structure.from_file method.",
    )

    p_relax.add_argument(
        "-m",
        "--model",
        dest="model",
        choices=[m for m in matgl.get_available_pretrained_models() if m.endswith("PES")],
        default="M3GNet-MP-2021.2.8-DIRECT-PES",
        help="Model to use.",
    )

    p_relax.add_argument(
        "-v",
        "--verbose",
        dest="verbose",
        default=False,
        action="store_true",
        help="Verbose output.",
    )

    groups = p_relax.add_mutually_exclusive_group(required=False)
    groups.add_argument(
        "-s",
        "--suffix",
        dest="suffix",
        help="Suffix to be added to input file names for relaxed structures. E.g., _relax.",
    )

    groups.add_argument(
        "-o",
        "--outfile",
        dest="outfile",
        help="Output filename.",
    )

    p_relax.set_defaults(func=relax_structure)

    p_predict = subparsers.add_parser("predict", help="Perform a prediction with pre-trained models.")

    groups = p_predict.add_mutually_exclusive_group(required=True)
    groups.add_argument(
        "-p",
        "--mpids",
        dest="mpids",
        nargs="+",
        help="Materials Project IDs. Requires mp-api to be installed and set up.",
    )

    groups.add_argument(
        "-i",
        "--infile",
        dest="infile",
        nargs="+",
        help="Input files containing structure. Any format supported by pymatgen's Structure.from_file method.",
    )

    p_predict.add_argument(
        "-s",
        "--state",
        dest="state_attr",
        nargs="+",
        help="state attributes containing label. This should be an integer.",
    )

    p_predict.add_argument(
        "-m",
        "--model",
        dest="model",
        choices=matgl.get_available_pretrained_models(),
        required=True,
        help="Model to use",
    )

    p_predict.set_defaults(func=predict_structure)

    # MD simulations
    p_md = subparsers.add_parser("md", help="Perform MD simulations with pre-trained and customized models.")

    p_md.add_argument(
        "-i",
        "--infile",
        nargs="+",
        dest="infile",
        required=True,
        help="Input files containing structure. Any format supported by pymatgen Structure.from_file method.",
    )

    p_md.add_argument(
        "-m",
        "--model",
        dest="model",
        choices=[m for m in matgl.get_available_pretrained_models() if m.endswith("PES")],
        default="M3GNet-MP-2021.2.8-DIRECT-PES",
        help="Path for loading MLIPs trained from MatGL. Default='M3GNet-MP-2021.2.8-DIRECT-PES'.",
    )

    p_md.add_argument(
        "-e",
        "--ensemble",
        dest="ensemble",
        choices=["nve", "nvt", "nvt_langevin", "nvt_andersen", "npt", "npt_berendsen", "npt_nose_hoover"],
        default="nve",
        help="Ensemble used for MD simulation. Default='nve'.",
    )

    p_md.add_argument(
        "-n",
        "--nsteps",
        dest="nsteps",
        type=int,
        default=100,
        help="Number of steps used for MD simulation. Default=100.",
    )

    p_md.add_argument(
        "--stepsize",
        dest="stepsize",
        type=float,
        default=1.0,
        help="Step size used for MD simulation. Default=1.0 fs.",
    )

    p_md.add_argument(
        "-t",
        "--temp",
        dest="temp",
        type=float,
        default=300.0,
        help="Temperature used for MD simulation. Default=300.0 in K.",
    )

    p_md.add_argument(
        "-p",
        "--pressure",
        dest="pressure",
        type=float,
        default=1.01325,
        help="Pressure used for MD simulation. Default=1.01325 in Bar.",
    )

    p_md.add_argument(
        "--taut",
        dest="taut",
        type=float,
        default=None,
        help="Time constant for Berendsen temperature coupling. Default is None.",
    )

    p_md.add_argument(
        "--taup",
        dest="taup",
        type=float,
        default=None,
        help="Time constant for Berendsen pressure coupling. Default is None.",
    )

    p_md.add_argument(
        "--andersen_prob",
        dest="andersen_prob",
        type=float,
        default=0.01,
        help="Random collision probability for nvt_andersen. Default is 0.01.",
    )

    p_md.add_argument(
        "--friction",
        dest="friction",
        type=float,
        default=0.001,
        help="Friction coefficient for nvt_langevin. Default is 0.001.",
    )

    p_md.add_argument(
        "--ttime",
        dest="ttime",
        type=float,
        default=25.0,
        help="Characteristic timescale of the thermostat in ASE internal units. Default is 25.0.",
    )

    p_md.add_argument(
        "--pfactor",
        dest="pfactor",
        type=float,
        default=75.0**2.0,
        help="A constant in the barostat differential equation. Default is 25.0 in eV/A$^{3}$.",
    )

    p_md.add_argument(
        "--external_stress",
        dest="external_stress",
        type=float,
        default=None,
        help="The external stress either 3x3 tensor, 6-vector or a scalar in eV/A$^{3}$. Default is None.",
    )

    p_md.add_argument(
        "--compressibility_au",
        dest="compressibility_au",
        type=float,
        default=None,
        help="Compressibility of the material in eV/A^{3}. Default is None.",
    )

    p_md.add_argument(
        "--loginterval",
        dest="loginterval",
        type=int,
        default=1,
        help="Write to log file every interval steps. Default is 1.",
    )

    p_md.add_argument(
        "--append_trajectory",
        dest="append_trajectory",
        type=bool,
        default=False,
        help="Whether to append to prev trajectory. Default is False.",
    )

    p_md.add_argument(
        "--mask",
        dest="mask",
        type=np.array,
        default=None,
        help="a symmetric 3x3 array indicating, which strain values may change for NPT simulations",
    )

    p_md.set_defaults(func=molecular_dynamics)

    p_clear = subparsers.add_parser("clear", help="Clear cache.")

    p_clear.add_argument(
        "-y",
        "--yes",
        dest="yes",
        action="store_true",
        help="Skip confirmation.",
    )

    p_clear.set_defaults(func=clear_cache)

    # LAMMPS export
    p_lammps = subparsers.add_parser(
        "create-lammps-model",
        help="Export a MatGL Potential as a TorchScript artifact loadable by pair_matgl[/kokkos].",
    )
    p_lammps.add_argument(
        "-m",
        "--model",
        dest="model",
        required=True,
        help="Path or name of a saved MatGL model (TensorNet PyG, extensive PES).",
    )
    p_lammps.add_argument(
        "-o",
        "--outfile",
        dest="outfile",
        required=True,
        help="Output path for the LAMMPS-loadable artifact (e.g. matgl_model.pt).",
    )
    p_lammps.add_argument(
        "--dtype",
        dest="dtype",
        choices=["float32", "float64"],
        default="float32",
        help="Wrapper buffer dtype. Match what your LAMMPS LibTorch was built with.",
    )
    p_lammps.add_argument(
        "--device",
        dest="device",
        default="cpu",
        help="Device to load weights onto before export (cpu | cuda[:N]).",
    )
    p_lammps.add_argument(
        "--no-script",
        dest="no_script",
        action="store_true",
        help="Save the eager wrapper instead of running torch.jit.script. "
        "Only useful for debugging — not loadable from LAMMPS C++.",
    )
    p_lammps.set_defaults(func=create_lammps_model)

    args = parser.parse_args()

    return args.func(args)
