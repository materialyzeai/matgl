"""Command line interface for matgl."""

from __future__ import annotations

import argparse
import json
import logging
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import torch
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from pymatgen.core.structure import Molecule, Structure
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


def _parse_json_object(value: str) -> dict:
    """Parse a JSON object supplied on the command line."""
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as err:
        raise argparse.ArgumentTypeError(f"invalid JSON: {err.msg}") from err
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError("expected a JSON object")
    return parsed


def _parse_devices(value: str) -> int | str:
    """Parse a Lightning device count while retaining selectors such as ``auto``."""
    try:
        return int(value)
    except ValueError:
        return value


def _parse_md_mask(value: str) -> np.ndarray:
    """Parse an ASE NPT mask from three diagonal or nine matrix entries."""
    tokens = value.replace(",", " ").split()
    if len(tokens) not in {3, 9}:
        raise argparse.ArgumentTypeError("mask must contain 3 or 9 values (0 or 1)")
    try:
        values = np.asarray([int(token) for token in tokens], dtype=int)
    except ValueError as err:
        raise argparse.ArgumentTypeError("mask values must be 0 or 1") from err
    if not np.isin(values, (0, 1)).all():
        raise argparse.ArgumentTypeError("mask values must be 0 or 1")
    return values if len(values) == 3 else values.reshape(3, 3)


def _is_xyz_path(path: str | Path) -> bool:
    """Return whether a path should be handled by ASE's Extended XYZ reader."""
    return Path(path).suffix.lower() in {".xyz", ".extxyz"}


def _read_geometries(path: str | Path) -> list[Structure | Molecule]:
    """Read one conventional structure or every frame of an Extended XYZ file."""
    if not _is_xyz_path(path):
        return [Structure.from_file(path)]

    from ase.io import read

    frames = read(path, index=":", format="extxyz")
    if not frames:
        raise ValueError(f"Extended XYZ file contains no frames: {path}")
    adaptor = AseAtomsAdaptor()
    geometries: list[Structure | Molecule] = []
    for atoms in frames:
        pbc = np.asarray(atoms.pbc, dtype=bool)
        if pbc.any() and not pbc.all():
            raise ValueError("Partially periodic Extended XYZ frames are not supported.")
        geometries.append(adaptor.get_structure(atoms) if pbc.all() else adaptor.get_molecule(atoms))
    return geometries


def _write_geometries(geometries: Sequence[Structure | Molecule], path: str | Path) -> None:
    """Write one conventional geometry or an Extended XYZ trajectory."""
    if _is_xyz_path(path):
        from ase.io import write

        adaptor = AseAtomsAdaptor()
        write(path, [adaptor.get_atoms(geometry) for geometry in geometries], format="extxyz")
        return
    if len(geometries) != 1:
        raise ValueError("Multiple relaxed frames require an .xyz or .extxyz output file.")
    geometries[0].to(filename=str(path))


def read_frames(path: str | Path):
    """Read input configurations as ASE atoms, preserving every Extended XYZ frame."""
    if _is_xyz_path(path):
        from ase.io import read

        frames = read(path, index=":", format="extxyz")
        if not frames:
            raise ValueError(f"Extended XYZ file contains no frames: {path}")
        return frames
    adaptor = AseAtomsAdaptor()
    return [adaptor.get_atoms(Structure.from_file(path))]


def _molecule_graph_converter(model: Any):
    """Build the molecule converter matching a loaded model's element ordering."""
    from matgl.ext.pymatgen import Molecule2Graph

    return Molecule2Graph(element_types=tuple(model.element_types), cutoff=float(model.cutoff))


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

    potential = _load_potential(args.model)
    relaxer = Relaxer(potential=potential, optimizer=args.optimizer, relax_cell=args.relax_cell)
    outfile_geometries: list[Structure | Molecule] = []

    for fn in args.infile:
        geometries = _read_geometries(fn)
        final_geometries: list[Structure | Molecule] = []
        for frame_index, geometry in enumerate(geometries):
            if isinstance(geometry, Molecule) and args.relax_cell:
                raise ValueError("Cell relaxation is not defined for nonperiodic XYZ molecules; use --no-relax-cell.")
            logger.info("Initial geometry %s[%d]\n%s", fn, frame_index, geometry)
            logger.info("Relaxing...")
            relax_results = relaxer.relax(geometry, fmax=args.f_max, steps=args.steps, verbose=args.verbose)
            final_geometry = relax_results["final_structure"]
            final_geometries.append(final_geometry)

            if not args.suffix and args.outfile is None:
                if isinstance(geometry, Structure) and isinstance(final_geometry, Structure):
                    print("Lattice parameters")
                    for line in _format_lattice_delta(geometry.lattice, final_geometry.lattice):
                        print(line)
                    print("Sites (Fractional coordinates)")
                    coordinate_name = "frac_coords"
                else:
                    print("Sites (Cartesian coordinates)")
                    coordinate_name = "coords"

                def fmt_coords(coords: np.ndarray) -> str:
                    return np.array2string(coords, formatter={"float_kind": lambda x: f"{x:.5f}"})

                for old_site, new_site in zip(geometry, final_geometry, strict=False):
                    old_coords = getattr(old_site, coordinate_name)
                    new_coords = getattr(new_site, coordinate_name)
                    print(f"{old_site.species}: {fmt_coords(old_coords)} -> {fmt_coords(new_coords)}")

        if args.suffix:
            p = Path(fn)
            outfn = f"{p.with_suffix('')}{args.suffix}{p.suffix}"
            _write_geometries(final_geometries, outfn)
            print(f"Structure written to {outfn}!")
        else:
            outfile_geometries.extend(final_geometries)

    if args.outfile is not None:
        _write_geometries(outfile_geometries, args.outfile)
        print(f"Structure written to {args.outfile}!")

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
                geometries = _read_geometries(file_path)
                for frame_index, geometry in enumerate(geometries):
                    label = file_path if len(geometries) == 1 else f"{file_path}[{frame_index}]"
                    kwargs = {}
                    if isinstance(geometry, Molecule):
                        kwargs["graph_converter"] = _molecule_graph_converter(model)
                    value = model.predict_structure(geometry, torch.tensor(state), **kwargs)  # type:ignore[operator]
                    print(f"{args.model} prediction for {label} with {state_dict[state]} bandgap: {value} eV.")
        else:
            for file_path in args.infile:
                geometries = _read_geometries(file_path)
                for frame_index, geometry in enumerate(geometries):
                    label = file_path if len(geometries) == 1 else f"{file_path}[{frame_index}]"
                    kwargs = {}
                    if isinstance(geometry, Molecule):
                        kwargs["graph_converter"] = _molecule_graph_converter(model)
                    value = model.predict_structure(geometry, **kwargs)  # type:ignore[operator]
                    print(f"{args.model} prediction for {label}: {value} eV/atom.")
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
    potential = _load_potential(args.model)
    for file in args.infile:
        frames = read_frames(file)
        base_name = str(Path(file).with_suffix(""))
        for frame_index, atoms in enumerate(frames):
            name = base_name if len(frames) == 1 else f"{base_name}_{frame_index}"
            logger.info("Initial atoms %s[%d]\n%s", file, frame_index, atoms)
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


def _potential_element_refs(potential: Potential) -> np.ndarray | None:
    """Return saved elemental offsets in model element order, when present."""
    element_refs = getattr(potential, "element_refs", None)
    if element_refs is None:
        return None
    return element_refs.property_offset.detach().cpu().numpy()


def _load_cli_pes_dataset(args: argparse.Namespace, *, cutoff: float, element_types: tuple[str, ...] | None):
    """Load JSON or Extended XYZ PES data according to CLI options."""
    from matgl.utils.training import MGLDatasetLoader

    input_format = args.input_format
    if input_format == "auto":
        input_format = "extxyz" if Path(args.infile).suffix.lower() in {".xyz", ".extxyz"} else "json"
    stress_unit = args.stress_unit
    if stress_unit == "auto":
        stress_unit = "eV/A3" if input_format == "extxyz" else "kbar"

    common = {
        "cutoff": cutoff,
        "element_types": element_types,
        "save_cache": args.cache_dataset,
        "root": args.dataset_root,
        "stress_unit": stress_unit,
        "include_charges": args.charges,
        "include_magmoms": args.magmoms,
    }
    if input_format == "extxyz":
        return MGLDatasetLoader.from_extxyz(
            args.infile,
            **common,
            energy_key=args.energy_key,
            forces_key=args.forces_key,
            stress_key=args.stress_key,
            charges_key=args.charges_key,
            magmoms_key=args.magmoms_key,
        )
    return MGLDatasetLoader.from_json(args.infile, **common)


def train_potential(args: argparse.Namespace) -> int:
    """Train or fine-tune a PyG interatomic potential from local JSON/JSONL data."""
    from matgl import models
    from matgl.utils.training import MGLPotentialTrainer

    _configure_logging(args.verbose)
    fractions = (args.train_ratio, args.valid_ratio, args.test_ratio)
    if any(value <= 0 for value in fractions) or not np.isclose(sum(fractions), 1.0):
        raise ValueError("train, validation, and test ratios must be positive and sum to 1")

    architecture_names = {"CHGNet", "GRACE", "M3GNet", "MEGNet", "QET", "SO3Net", "TensorNet"}
    model_source = args.path_load_model or args.model
    is_scratch = model_source in architecture_names

    pretrained = None
    if is_scratch:
        element_types = None
        cutoff = args.cutoff
    else:
        pretrained = _load_potential(model_source)
        pretrained_model = cast("Any", pretrained.model)
        element_types = tuple(pretrained_model.element_types)
        cutoff = float(pretrained_model.cutoff)

    dataset = _load_cli_pes_dataset(args, cutoff=cutoff, element_types=element_types)
    stress_weight = (
        args.stress_weight if args.stress_weight is not None else (0.1 if "stresses" in dataset.labels else 0.0)
    )
    magmom_weight = (
        args.magmom_weight if args.magmom_weight is not None else (0.1 if "magmoms" in dataset.labels else 0.0)
    )
    charge_weight = (
        args.charge_weight if args.charge_weight is not None else (0.1 if "charges" in dataset.labels else 0.0)
    )
    for label, weight in (("stresses", stress_weight), ("magmoms", magmom_weight), ("charges", charge_weight)):
        if weight > 0 and label not in dataset.labels:
            raise ValueError(f"{label} weight is positive, but the dataset has no {label} labels")

    if is_scratch:
        reserved = {"element_types", "cutoff", "is_intensive"}.intersection(args.model_kwargs)
        if reserved:
            names = ", ".join(sorted(reserved))
            raise ValueError(f"--model-kwargs cannot override CLI-managed option(s): {names}")
        model_class = getattr(models, model_source)
        model = model_class(
            element_types=tuple(dataset.element_types),
            cutoff=cutoff,
            is_intensive=False,
            **args.model_kwargs,
        )
        atomrefs = None
        data_mean: float | torch.Tensor = 0.0
        data_std: float | torch.Tensor = 1.0
    else:
        assert pretrained is not None
        model = pretrained.model
        atomrefs = _potential_element_refs(pretrained)
        data_mean = float(pretrained.data_mean.detach().cpu().item())
        data_std = float(pretrained.data_std.detach().cpu().item())

    trainer_kwargs = {
        "gradient_clip_val": args.gradient_clip_val,
        "accumulate_grad_batches": args.accumulate_grad_batches,
        "logger": False,
        "enable_checkpointing": False,
    }
    if args.log_dir:
        from lightning.pytorch.loggers import CSVLogger

        trainer_kwargs["logger"] = CSVLogger(save_dir=args.log_dir, name="matgl")
        trainer_kwargs["enable_checkpointing"] = True

    trainer = MGLPotentialTrainer(
        model,
        energy_weight=args.energy_weight,
        force_weight=args.force_weight,
        stress_weight=stress_weight,
        magmom_weight=magmom_weight,
        charge_weight=charge_weight,
        loss=args.loss,
        data_mean=data_mean,
        data_std=data_std,
        lr=args.lr,
        decay_steps=args.decay_steps,
        decay_alpha=args.decay_alpha,
        batch_size=args.batch_size,
        max_epochs=args.epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        seed=args.seed,
        trainer_kwargs=trainer_kwargs,
        loader_kwargs={"frac_list": fractions, "num_workers": args.num_workers},
    )
    trainer.fit(dataset, atomrefs=atomrefs, save_path=args.output, ckpt_path=args.checkpoint)
    print(f"Trained potential written to {args.output}")
    return 0


def evaluate_potential(args: argparse.Namespace) -> int:
    """Evaluate a saved PyG potential on all samples in a local PES dataset."""
    import lightning as pl

    from matgl.graph.data import MGLDataLoader
    from matgl.utils.training import PotentialLightningModule

    _configure_logging(args.verbose)
    potential = _load_potential(args.model)
    potential_model = cast("Any", potential.model)
    dataset = _load_cli_pes_dataset(
        args,
        cutoff=float(potential_model.cutoff),
        element_types=tuple(potential_model.element_types),
    )
    labels = dataset.labels
    lit_model = PotentialLightningModule(
        model=potential.model,
        element_refs=_potential_element_refs(potential),
        data_mean=float(potential.data_mean.detach().cpu().item()),
        data_std=float(potential.data_std.detach().cpu().item()),
        stress_weight=1.0 if "stresses" in labels else 0.0,
        magmom_weight=1.0 if "magmoms" in labels else 0.0,
        charge_weight=1.0 if "charges" in labels else 0.0,
    )
    _, data_loader = MGLDataLoader(
        train_data=dataset,
        val_data=dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    trainer = pl.Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        inference_mode=False,
        logger=False,
    )
    trainer.test(model=lit_model, dataloaders=data_loader)
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
    from matgl.ext.lammps import LAMMPSMatGLModel

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
    species = list(potential.model.element_types)  # type:ignore[union-attr,arg-type,attr-defined]
    print("  species   :", species[: wrapper.n_species])
    return 0


def _add_pes_input_arguments(parser: argparse.ArgumentParser) -> None:
    """Add shared JSON/Extended XYZ dataset options to a subparser."""
    parser.add_argument(
        "--input-format",
        choices=["auto", "json", "extxyz"],
        default="auto",
        help="Input format; auto detects .xyz/.extxyz and otherwise uses JSON (default: auto).",
    )
    parser.add_argument(
        "--stress-unit",
        choices=["auto", "kbar", "GPa", "eV/A3"],
        default="auto",
        help="Input stress unit (default: eV/A3 for Extended XYZ, kbar for JSON).",
    )
    parser.add_argument("--energy-key", default="energy", help="Extended XYZ energy field name.")
    parser.add_argument("--forces-key", default="forces", help="Extended XYZ forces field name.")
    parser.add_argument("--stress-key", default="stress", help="Extended XYZ stress field name.")
    parser.add_argument("--charges-key", default="charges", help="Extended XYZ per-atom charge field name.")
    parser.add_argument("--magmoms-key", default="magmoms", help="Extended XYZ per-atom magmom field name.")
    parser.add_argument("--charges", action="store_true", help="Load per-atom partial-charge labels.")
    parser.add_argument("--magmoms", action="store_true", help="Load per-atom magnetic-moment labels.")
    parser.add_argument("--cache-dataset", action="store_true", help="Persist the processed graph dataset.")
    parser.add_argument("--dataset-root", default=None, help="Processed-dataset cache directory.")


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser without performing network access."""
    parser = argparse.ArgumentParser(
        description="""
    This script works based on several sub-commands with their own options. To see the options for the
    sub-commands, type "mgl sub-command -h".""",
        epilog="""Author: MatGL Development Team""",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

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
        default="M3GNet-MP-2021.2.8-DIRECT-PES",
        help="Pretrained model name or path to a locally saved potential.",
    )

    p_relax.add_argument(
        "--optimizer",
        choices=["FIRE", "FIRE2", "BFGS", "LBFGS", "LBFGSLineSearch"],
        default="FIRE",
        help="ASE geometry optimizer (default: FIRE).",
    )
    relax_cell_group = p_relax.add_mutually_exclusive_group()
    relax_cell_group.add_argument("--relax-cell", "--relax_cell", dest="relax_cell", action="store_true")
    relax_cell_group.add_argument("--no-relax-cell", dest="relax_cell", action="store_false")
    p_relax.set_defaults(relax_cell=True)
    p_relax.add_argument(
        "--force-max",
        "--force_max",
        dest="f_max",
        type=float,
        default=0.01,
        help="Maximum force convergence threshold in eV/angstrom (default: 0.01).",
    )
    p_relax.add_argument("--steps", type=int, default=500, help="Maximum optimization steps (default: 500).")

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
        required=True,
        help="Pretrained model name or path to a locally saved model.",
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
        "--append-trajectory",
        "--append_trajectory",
        dest="append_trajectory",
        action="store_true",
        help="Whether to append to prev trajectory. Default is False.",
    )

    p_md.add_argument(
        "--mask",
        dest="mask",
        type=_parse_md_mask,
        default=None,
        help="NPT strain mask as 3 diagonal or 9 matrix values, separated by commas or spaces.",
    )

    p_md.set_defaults(func=molecular_dynamics)

    p_train = subparsers.add_parser(
        "train",
        help="Train or fine-tune a PyG interatomic potential from local MatPES-shaped JSON/JSONL data.",
    )
    p_train.add_argument("-i", "--infile", required=True, help="JSON/JSONL or Extended XYZ training data.")
    p_train.add_argument(
        "-m",
        "--model",
        default="M3GNet",
        help="Architecture name for scratch training, or a pretrained model name/local path for fine-tuning.",
    )
    p_train.add_argument(
        "-o",
        "--output",
        default="trained_model",
        help="Directory in which to save the trained potential (default: trained_model).",
    )
    p_train.add_argument(
        "--path-load-model",
        "--path_load_model",
        default=None,
        help="Compatibility option for a pretrained potential to fine-tune; overrides --model.",
    )
    p_train.add_argument("--target", choices=["pes"], default="pes", help=argparse.SUPPRESS)
    p_train.add_argument(
        "--model-kwargs",
        type=_parse_json_object,
        default={},
        help='Architecture-specific options as JSON, e.g. \'{"units": 32, "nblocks": 2}\'.',
    )
    p_train.add_argument("--cutoff", type=float, default=5.0, help="Graph cutoff for scratch training.")
    _add_pes_input_arguments(p_train)
    p_train.add_argument("--train-ratio", "--train_ratio", type=float, default=0.8)
    p_train.add_argument("--valid-ratio", "--valid_ratio", type=float, default=0.1)
    p_train.add_argument("--test-ratio", "--test_ratio", type=float, default=0.1)
    p_train.add_argument("--energy-weight", "--energy_weight", type=float, default=1.0)
    p_train.add_argument("--force-weight", "--force_weight", type=float, default=1.0)
    p_train.add_argument(
        "--stress-weight",
        "--stress_weight",
        type=float,
        default=None,
        help="Stress loss weight (default: 0.1 when labels exist, otherwise 0).",
    )
    p_train.add_argument(
        "--magmom-weight",
        "--magmom_weight",
        type=float,
        default=None,
        help="Magmom loss weight (default: 0.1 with --magmoms, otherwise 0).",
    )
    p_train.add_argument(
        "--charge-weight",
        "--charge_weight",
        type=float,
        default=None,
        help="Charge loss weight (default: 0.1 with --charges, otherwise 0).",
    )
    p_train.add_argument(
        "--loss",
        choices=["mse_loss", "huber_loss", "smooth_l1_loss", "l1_loss"],
        default="huber_loss",
    )
    p_train.add_argument("--batch-size", "--batch_size", type=int, default=32)
    p_train.add_argument("--epochs", "--number-of-epochs", "--number_of_epochs", dest="epochs", type=int, default=100)
    p_train.add_argument("--lr", type=float, default=1e-3)
    p_train.add_argument("--decay-steps", type=int, default=1000)
    p_train.add_argument("--decay-alpha", type=float, default=0.01)
    p_train.add_argument("--accelerator", default="auto")
    p_train.add_argument("--devices", type=_parse_devices, default="auto")
    p_train.add_argument("--num-workers", type=int, default=0)
    p_train.add_argument("--seed", type=int, default=42)
    p_train.add_argument("--gradient-clip-val", type=float, default=0.0)
    p_train.add_argument("--accumulate-grad-batches", type=int, default=1)
    p_train.add_argument("--checkpoint", default=None, help="Lightning checkpoint path, or 'last', to resume.")
    p_train.add_argument("--log-dir", default=None, help="Enable Lightning CSV logging in this directory.")
    p_train.add_argument("-v", "--verbose", action="store_true")
    p_train.set_defaults(func=train_potential)

    p_evaluate = subparsers.add_parser("evaluate", help="Evaluate a saved PyG potential on local PES data.")
    p_evaluate.add_argument("-i", "--infile", required=True, help="JSON/JSONL or Extended XYZ evaluation data.")
    p_evaluate.add_argument(
        "-m",
        "--model",
        "--path-load-model",
        "--path_load_model",
        required=True,
        help="Saved potential name or path.",
    )
    _add_pes_input_arguments(p_evaluate)
    p_evaluate.add_argument("--batch-size", type=int, default=32)
    p_evaluate.add_argument("--accelerator", default="auto")
    p_evaluate.add_argument("--devices", type=_parse_devices, default="auto")
    p_evaluate.add_argument("--num-workers", type=int, default=0)
    p_evaluate.add_argument("-v", "--verbose", action="store_true")
    p_evaluate.set_defaults(func=evaluate_potential)

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

    return parser


def main(argv: Sequence[str] | None = None):
    """Parse command-line arguments and dispatch the selected command."""
    args = build_parser().parse_args(argv)

    return args.func(args)
