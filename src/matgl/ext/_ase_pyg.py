"""Interfaces to the Atomic Simulation Environment package for dynamic simulations."""

from __future__ import annotations

import collections
import contextlib
import io
import logging
import math
import pickle
import sys
import warnings
from enum import Enum
from typing import TYPE_CHECKING, Literal

import ase.optimize as opt
import numpy as np
import pandas as pd
import torch
from ase import Atoms, units
from ase.calculators.calculator import Calculator, all_changes
from ase.filters import Filter, FrechetCellFilter
from ase.md import Langevin
from ase.md.andersen import Andersen
from ase.md.bussi import Bussi
from ase.md.nose_hoover_chain import IsotropicMTKNPT, NoseHooverChainNVT
from ase.md.npt import NPT
from ase.md.nptberendsen import Inhomogeneous_NPTBerendsen, NPTBerendsen
from ase.md.nvtberendsen import NVTBerendsen
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet
from ase.stress import full_3x3_to_voigt_6_stress
from pymatgen.core.structure import Molecule, Structure
from pymatgen.io.ase import AseAtomsAdaptor

from matgl.ext._alchmtk import neighbor_list_from_ase
from matgl.graph._converters_pyg import GraphConverter

if TYPE_CHECKING:
    from typing import Any

    from ase.optimize.optimize import Optimizer
    from torch_geometric.data import Data

    from matgl.apps._pes_pyg import Potential

logger = logging.getLogger(__name__)


class OPTIMIZERS(Enum):
    """An enumeration of optimizers for used in."""

    fire = opt.fire.FIRE
    bfgs = opt.bfgs.BFGS
    lbfgs = opt.lbfgs.LBFGS
    lbfgslinesearch = opt.lbfgs.LBFGSLineSearch
    mdmin = opt.mdmin.MDMin
    scipyfmincg = opt.sciopt.SciPyFminCG
    scipyfminbfgs = opt.sciopt.SciPyFminBFGS
    bfgslinesearch = opt.bfgslinesearch.BFGSLineSearch


class Atoms2Graph(GraphConverter):
    """Construct a DGL graph from ASE Atoms."""

    def __init__(
        self,
        element_types: tuple[str, ...],
        cutoff: float = 5.0,
    ):
        """Init Atoms2Graph from element types and cutoff radius.

        Args:
            element_types: List of elements present in dataset for graph conversion. This ensures all graphs are
                constructed with the same dimensionality of features.
            cutoff: Cutoff radius for graph representation
        """
        self.element_types = tuple(element_types)
        self.cutoff = cutoff
        self.max_neighbors = 0.3

    def get_graph(self, atoms: Atoms) -> tuple[Data, torch.Tensor, list | np.ndarray]:
        """Get a DGL graph from an input Atoms.

        Args:
            atoms: Atoms object.

        Returns:
            g: DGL graph
            state_attr: state features
        """
        src_id, dst_id, _, images, positions, max_neighbors = neighbor_list_from_ase(
            atoms=atoms,
            cutoff=self.cutoff,
            compute_distances=False,
            density_guess=self.max_neighbors,
        )
        self.max_neighbors = max_neighbors
        if atoms.get_pbc().any():
            lattice_matrix = torch.as_tensor(atoms.cell.array, dtype=torch.float32, device=src_id.device)
            # Convert Cartesian positions to fractional coordinates
            # The model expects fractional coords and computes: g.pos = g.frac_coords @ lattice
            frac_coords = positions @ torch.linalg.inv(lattice_matrix)
        else:
            lattice_matrix = torch.eye(3, dtype=torch.float32, device=src_id.device).unsqueeze(0)
            # For molecules (non-periodic), use Cartesian positions directly
            frac_coords = positions
        element_types = self.element_types

        g, lat, state_attr = super().get_graph_from_processed_structure(
            atoms,
            src_id,
            dst_id,
            images,
            lattice_matrix,
            element_types,
            frac_coords,
            is_atoms=True,
        )

        return g, lat, state_attr


class PESCalculator(Calculator):
    """Potential calculator for ASE."""

    implemented_properties = ["energy", "free_energy", "forces", "stress", "hessian", "magmoms"]  # noqa:RUF012

    def __init__(
        self,
        potential: Potential,
        state_attr: torch.Tensor | None = None,
        stress_unit: Literal["eV/A3", "GPa"] = "GPa",
        stress_weight: float = 1.0,
        use_voigt: bool = False,
        **kwargs,
    ):
        """
        Init PESCalculator with a Potential from matgl.

        Args:
            potential (Potential): matgl.apps.pes.Potential
            state_attr (tensor): State attribute
            compute_stress (bool): whether to calculate the stress
            stress_unit (str): stress unit either in "GPa" or "eV/A^3". Default: "GPa"
            stress_weight (float): conversion factor from GPa to eV/A^3, if it is set to 1.0, the unit is in GPa
            use_voigt (bool): whether the voigt notation is used for stress output
            **kwargs: Kwargs pass through to super().__init__().
        """
        super().__init__(**kwargs)
        self.potential = potential
        self.compute_stress = potential.calc_stresses
        self.compute_hessian = potential.calc_hessian
        self.compute_magmom = potential.calc_magmom

        # Handle stress unit conversion
        if stress_unit == "eV/A3":
            conversion_factor = units.GPa / (units.eV / units.Angstrom**3)  # Conversion factor from GPa to eV/A^3
        elif stress_unit == "GPa":
            conversion_factor = 1.0  # No conversion needed if stress is already in GPa
        else:
            raise ValueError(f"Unsupported stress_unit: {stress_unit}. Must be 'GPa' or 'eV/A3'.")

        self.conversion_factor = conversion_factor * stress_weight
        if self.conversion_factor == 1.0:
            logger.warning(
                "The stress unit is now in GPa. Please set stress_unit='eV/A3' "
                "if you want to use PESCalculator for other ASE applications."
            )
        elif math.isclose(
            self.conversion_factor, units.GPa / (units.eV / units.Angstrom**3), rel_tol=1e-4, abs_tol=1e-6
        ):
            logger.info("The stress unit is now in eV/A^3, which is the correct unit for an ASE Calculator.")
        else:
            raise ValueError(
                "Error: Invalid stress unit configuration: stress_weight corresponds to neither "
                "GPa nor eV/A^3. This is likely caused by setting both stress_unit and "
                "stress_weight. Please set stress_unit to 'GPa' or 'eV/A3' and "
                "stress_weight to 1.0."
            )
        self.state_attr = state_attr
        self.element_types: tuple[str, ...] = potential.model.element_types  # type: ignore[assignment,union-attr]
        self.cutoff: float = potential.model.cutoff  # type: ignore[assignment,union-attr]
        self.use_voigt = use_voigt
        self._atoms2graph = Atoms2Graph(self.element_types, self.cutoff)

    def calculate(  # type:ignore[override]
        self,
        atoms: Atoms,
        properties: list | None = None,
        system_changes: list | None = None,
    ):
        """
        Perform calculation for an input Atoms.

        Args:
            atoms (ase.Atoms): ase Atoms object
            properties (list): list of properties to calculate
            system_changes (list): monitor which properties of atoms were
                changed for new calculation. If not, the previous calculation
                results will be loaded.
        """
        properties = properties or ["energy"]
        system_changes = system_changes or all_changes
        super().calculate(atoms=atoms, properties=properties, system_changes=system_changes)
        graph, lattice, state_attr_default = self._atoms2graph.get_graph(atoms)
        if self.state_attr is not None:
            calc_result = self.potential(graph, lattice, self.state_attr)
        else:
            calc_result = self.potential(graph, lattice, state_attr_default)
        self.results.update(
            energy=calc_result[0].detach().cpu().numpy().item(),
            free_energy=calc_result[0].detach().cpu().numpy().item(),
            forces=calc_result[1].detach().cpu().numpy(),
        )
        if self.compute_stress:
            stresses_np = (
                full_3x3_to_voigt_6_stress(calc_result[2].detach().cpu().numpy())
                if self.use_voigt
                else calc_result[2].detach().cpu().numpy()
            )
            self.results.update(stress=stresses_np * self.conversion_factor)
        if self.compute_hessian:
            self.results.update(hessian=calc_result[3].detach().cpu().numpy())
        if self.compute_magmom:
            self.results.update(magmoms=calc_result[4].detach().cpu().numpy())


# for backward compatibility
class M3GNetCalculator(PESCalculator):
    """M3GNet potential Calculator for ASE."""

    def __init__(
        self,
        potential: Potential,
        state_attr: torch.Tensor | None = None,
        stress_weight: float = 1.0,
        **kwargs,
    ):
        """
        Init M3GNetCalculator with a M3GNet Potential.

        Args:
            potential (Potential): matgl.apps.pes.Potential
            state_attr (tensor): State attribute
            compute_stress (bool): whether to calculate the stress
            stress_weight (float): conversion factor from GPa to eV/A^3, if it is set to 1.0, the unit is in GPa
            **kwargs: Kwargs pass through to super().__init__().
        """
        super().__init__(potential=potential, state_attr=state_attr, stress_weight=stress_weight, **kwargs)


class Relaxer:
    """Relaxer is a class for structural relaxation."""

    def __init__(
        self,
        potential: Potential,
        state_attr: torch.Tensor | None = None,
        optimizer: Optimizer | str = "FIRE",
        relax_cell: bool = True,
        **kwargs,
    ):
        """
        Args:
            potential (Potential): a M3GNet potential, a str path to a saved model or a short name for saved model
            that comes with M3GNet distribution
            state_attr (torch.Tensor): State attr.
            optimizer (str or ase Optimizer): the optimization algorithm.
            Defaults to "FIRE"
            relax_cell (bool): whether to relax the lattice cell
            **kwargs: Kwargs pass through to super().__init__().
        """
        # Detect user-provided stress_weight
        if "stress_weight" in kwargs:
            warnings.warn(
                "Relaxer does not support user-defined stress_weight. "
                "The stress unit is fixed to 'eV/A3' for ASE compatibility, "
                "and the provided stress_weight will be ignored.",
                UserWarning,
                stacklevel=2,
            )
            kwargs.pop("stress_weight")
        self.optimizer: Optimizer = OPTIMIZERS[optimizer.lower()].value if isinstance(optimizer, str) else optimizer
        self.calculator = PESCalculator(
            potential=potential,
            state_attr=state_attr,
            stress_unit="eV/A3",
        )
        self.relax_cell = relax_cell
        self.ase_adaptor = AseAtomsAdaptor()

    def relax(
        self,
        atoms: Atoms | Structure | Molecule,
        fmax: float = 0.1,
        steps: int = 500,
        traj_file: str | None = None,
        interval: int = 1,
        verbose: bool = False,
        params_asecellfilter: dict | None = None,
        **kwargs,
    ):
        """
        Relax an input Atoms.

        Args:
            atoms (Atoms | Structure | Molecule): the atoms for relaxation
            fmax (float): total force tolerance for relaxation convergence.
            Here fmax is a sum of force and stress forces
            steps (int): max number of steps for relaxation
            traj_file (str): the trajectory file for saving
            interval (int): the step interval for saving the trajectories
            verbose (bool): Whether to have verbose output.
            params_asecellfilter (dict): Parameters to be passed to FrechetCellFilter. Allows
                setting of constant pressure or constant volume relaxations, for example. Refer to
                https://wiki.fysik.dtu.dk/ase/ase/filters.html#FrechetCellFilter for more information
            **kwargs: Kwargs pass-through to optimizer.
        """
        if isinstance(atoms, Structure | Molecule):
            atoms = self.ase_adaptor.get_atoms(atoms)
        atoms.set_calculator(self.calculator)
        stream = sys.stdout if verbose else io.StringIO()
        params_asecellfilter = params_asecellfilter or {}
        with contextlib.redirect_stdout(stream):
            obs = TrajectoryObserver(atoms)
            if self.relax_cell:
                atoms = FrechetCellFilter(atoms, **params_asecellfilter)  # type:ignore[assignment]

            optimizer = self.optimizer(atoms, **kwargs)  # type:ignore[operator]
            optimizer.attach(obs, interval=interval)
            optimizer.run(fmax=fmax, steps=steps)
            obs()
        if traj_file is not None:
            obs.save(traj_file)

        if isinstance(atoms, Filter):
            atoms = atoms.atoms

        final_structure: Structure | Molecule
        if isinstance(atoms, Atoms):
            if np.array(atoms.pbc).any():
                final_structure = self.ase_adaptor.get_structure(atoms)
            else:
                final_structure = self.ase_adaptor.get_molecule(atoms)  # type: ignore[assignment]
        elif isinstance(atoms, Structure | Molecule):
            final_structure = atoms
        else:
            raise TypeError(f"Unsupported atoms type: {type(atoms)}")

        return {
            "final_structure": final_structure,  # type:ignore[arg-type]
            "trajectory": obs,
        }


class TrajectoryObserver(collections.abc.Sequence):
    """Trajectory observer is a hook in the relaxation process that saves the
    intermediate structures.
    """

    def __init__(self, atoms: Atoms) -> None:
        """
        Init the Trajectory Observer from a Atoms.

        Args:
            atoms (Atoms): Structure to observe.
        """
        self.atoms = atoms
        self.energies: list[float] = []
        self.forces: list[np.ndarray] = []
        self.stresses: list[np.ndarray] = []
        self.atom_positions: list[np.ndarray] = []
        self.cells: list[np.ndarray] = []

    def __call__(self) -> None:
        """The logic for saving the properties of an Atoms during the relaxation."""
        self.energies.append(float(self.atoms.get_potential_energy()))
        self.forces.append(self.atoms.get_forces())
        if self.atoms.calc.compute_stress:
            self.stresses.append(self.atoms.get_stress())
        self.atom_positions.append(self.atoms.get_positions())
        if self.atoms.pbc.any():
            self.cells.append(self.atoms.get_cell()[:])

    def __getitem__(self, item):
        return self.energies[item], self.forces[item], self.stresses[item], self.cells[item], self.atom_positions[item]

    def __len__(self):
        return len(self.energies)

    def as_pandas(self) -> pd.DataFrame:
        """Returns: DataFrame of energies, forces, stresses, cells and atom_positions."""
        return pd.DataFrame(
            {
                "energies": self.energies,
                "forces": self.forces,
                "stresses": self.stresses,
                "cells": self.cells,
                "atom_positions": self.atom_positions,
            }
        )

    def save(self, filename: str) -> None:
        """Save the trajectory to file.

        Args:
            filename (str): filename to save the trajectory.
        """
        out = {
            "energy": self.energies,
            "forces": self.forces,
            "stresses": self.stresses,
            "atom_positions": self.atom_positions,
            "cell": self.cells,
            "atomic_number": self.atoms.get_atomic_numbers(),
        }
        with open(filename, "wb") as file:
            pickle.dump(out, file)


class MolecularDynamics:
    """Molecular dynamics class."""

    def __init__(
        self,
        atoms: Atoms,
        potential: Potential,
        state_attr: torch.Tensor | None = None,
        ensemble: Literal[
            "nve",
            "nvt",
            "nvt_langevin",
            "nvt_andersen",
            "nvt_bussi",
            "nvt_nose_hoover_chain",
            "npt",
            "npt_berendsen",
            "npt_nose_hoover",
            "npt_nose_hoover_chain",
        ] = "nvt",
        temperature: int = 300,
        timestep: float = 1.0,
        pressure: float = 1.01325 * units.bar,
        taut: float | None = None,
        taup: float | None = None,
        friction: float = 1.0e-2,
        andersen_prob: float = 1.0e-2,
        ttime: float = 25.0,
        pfactor: float = 75.0**2.0,
        external_stress: float | np.ndarray | None = None,
        compressibility_au: float | None = None,
        trajectory: Any = None,
        logfile: str | None = None,
        loginterval: int = 1,
        append_trajectory: bool = False,
        mask: tuple | np.ndarray | None = None,
        **kwargs,
    ):
        """
        Init the MD simulation.

        Args:
            atoms (Atoms): atoms to run the MD
            potential (Potential): potential for calculating the energy, force,
            stress of the atoms
            state_attr (torch.Tensor): State attr.
            ensemble (str): choose from "nve", "nvt", "nvt_langevin", "nvt_andersen", "nvt_bussi",
            "npt", "npt_berendsen", "npt_nose_hoover"
            temperature (float): temperature for MD simulation, in K
            timestep (float): time step in fs
            pressure (float): pressure in eV/A^3
            taut (float): time constant for temperature coupling
            taup (float): time constant for pressure coupling
            friction (float): friction coefficient for nvt_langevin, typically set to 1e-4 to 1e-2
            andersen_prob (float): random collision probability for nvt_andersen, typically set to 1e-4 to 1e-1
            ttime (float): Characteristic timescale of the thermostat, in ASE internal units
            pfactor (float): A constant in the barostat differential equation.
            external_stress (float): The external stress in eV/A^3.
                Either 3x3 tensor,6-vector or a scalar representing pressure
            compressibility_au (float): compressibility of the material in A^3/eV
            trajectory (str or Trajectory): Attach trajectory object
            logfile (str): open this file for recording MD outputs
            loginterval (int): write to log file every interval steps
            append_trajectory (bool): Whether to append to prev trajectory.
            mask (np.array): either a tuple of 3 numbers (0 or 1) or a symmetric 3x3 array indicating,
                which strain values may change for NPT simulations
            **kwargs: Kwargs pass-through to molecular dynamics.
        """
        # Detect user-provided stress_weight
        if "stress_weight" in kwargs:
            warnings.warn(
                "Relaxer does not support user-defined stress_weight. "
                "The stress unit is fixed to 'eV/A3' for ASE compatibility, "
                "and the provided stress_weight will be ignored.",
                UserWarning,
                stacklevel=2,
            )
            kwargs.pop("stress_weight")
        if isinstance(atoms, Structure | Molecule):
            atoms = AseAtomsAdaptor().get_atoms(atoms)
        self.atoms = atoms
        self.atoms.set_calculator(PESCalculator(potential=potential, state_attr=state_attr, stress_unit="eV/A3"))

        if taut is None:
            taut = 100 * timestep * units.fs
        if taup is None:
            taup = 1000 * timestep * units.fs

        if mask is None:
            mask = np.array([(1, 0, 0), (0, 1, 0), (0, 0, 1)])
        if external_stress is None:
            external_stress = 0.0

        if np.isclose(self.atoms.get_kinetic_energy(), 0.0, rtol=0, atol=1e-12):
            MaxwellBoltzmannDistribution(self.atoms, temperature_K=temperature)

        if ensemble.lower() == "nvt":
            self.dyn = NVTBerendsen(
                self.atoms,
                timestep * units.fs,
                temperature_K=temperature,
                taut=taut,
                trajectory=trajectory,
                logfile=logfile,
                loginterval=loginterval,
                append_trajectory=append_trajectory,
            )

        elif ensemble.lower() == "nve":
            self.dyn = VelocityVerlet(  # type:ignore[assignment]
                self.atoms,
                timestep * units.fs,
                trajectory=trajectory,
                logfile=logfile,
                loginterval=loginterval,
                append_trajectory=append_trajectory,
            )

        elif ensemble.lower() == "nvt_langevin":
            self.dyn = Langevin(  # type:ignore[assignment]
                self.atoms,
                timestep * units.fs,
                temperature_K=temperature,
                friction=friction / units.fs,
                trajectory=trajectory,
                logfile=logfile,
                loginterval=loginterval,
                append_trajectory=append_trajectory,
            )

        elif ensemble.lower() == "nvt_andersen":
            self.dyn = Andersen(  # type:ignore[assignment]
                self.atoms,
                timestep * units.fs,
                temperature_K=temperature,
                andersen_prob=andersen_prob,
                trajectory=trajectory,
                logfile=logfile,
                loginterval=loginterval,
                append_trajectory=append_trajectory,
            )

        elif ensemble.lower() == "nvt_bussi":
            self.dyn = Bussi(  # type:ignore[assignment]
                self.atoms,
                timestep * units.fs,
                temperature_K=temperature,
                taut=taut,
                trajectory=trajectory,
                logfile=logfile,
                loginterval=loginterval,
                append_trajectory=append_trajectory,
            )

        elif ensemble.lower() == "nvt_nose_hoover_chain":
            self.dyn = NoseHooverChainNVT(  # type:ignore[assignment]
                self.atoms,
                timestep * units.fs,
                temperature_K=temperature,
                tdamp=taut,
                trajectory=trajectory,
                logfile=logfile,
                loginterval=loginterval,
                append_trajectory=append_trajectory,
            )
        elif ensemble.lower() == "npt":
            """
            NPT ensemble default to Inhomogeneous_NPTBerendsen thermo/barostat
            This is a more flexible scheme that fixes three angles of the unit
            cell but allows three lattice parameter to change independently.
            """

            self.dyn = Inhomogeneous_NPTBerendsen(  # type:ignore[assignment]
                self.atoms,
                timestep * units.fs,
                temperature_K=temperature,
                pressure_au=pressure,
                taut=taut,
                taup=taup,
                compressibility_au=compressibility_au,
                trajectory=trajectory,
                logfile=logfile,
                loginterval=loginterval,
                append_trajectory=append_trajectory,
            )

        elif ensemble.lower() == "npt_berendsen":
            """

            This is a similar scheme to the Inhomogeneous_NPTBerendsen.
            This is a less flexible scheme that fixes the shape of the
            cell - three angles are fixed and the ratios between the three
            lattice constants.

            """

            self.dyn = NPTBerendsen(
                self.atoms,
                timestep * units.fs,
                temperature_K=temperature,
                pressure_au=pressure,
                taut=taut,
                taup=taup,
                compressibility_au=compressibility_au,
                trajectory=trajectory,
                logfile=logfile,
                loginterval=loginterval,
                append_trajectory=append_trajectory,
            )

        elif ensemble.lower() == "npt_nose_hoover":
            self.upper_triangular_cell()
            self.dyn = NPT(  # type:ignore[assignment]
                self.atoms,
                timestep * units.fs,
                temperature_K=temperature,
                externalstress=external_stress,  # type:ignore[arg-type]
                ttime=ttime * units.fs,
                pfactor=pfactor * units.fs,
                trajectory=trajectory,
                logfile=logfile,
                loginterval=loginterval,
                append_trajectory=append_trajectory,
                mask=mask,
            )
        elif ensemble.lower() == "npt_nose_hoover_chain":
            self.dyn = IsotropicMTKNPT(  # type:ignore[assignment]
                self.atoms,
                timestep * units.fs,
                temperature_K=temperature,
                tdamp=taut,
                pdamp=taup,
                pressure_au=pressure,
                trajectory=trajectory,
                logfile=logfile,
                loginterval=loginterval,
                append_trajectory=append_trajectory,
            )

        else:
            raise ValueError("Ensemble not supported")

        self.trajectory = trajectory
        self.logfile = logfile
        self.loginterval = loginterval
        self.timestep = timestep

    def run(self, steps: int):
        """Thin wrapper of ase MD run.

        Args:
            steps (int): number of MD steps
        """
        self.dyn.run(steps)

    def set_atoms(self, atoms: Atoms):
        """Set new atoms to run MD.

        Args:
            atoms (Atoms): new atoms for running MD.
        """
        if isinstance(atoms, Structure | Molecule):
            atoms = AseAtomsAdaptor().get_atoms(atoms)
        calculator = self.atoms.calc
        self.atoms = atoms
        self.dyn.atoms = atoms
        self.dyn.atoms.set_calculator(calculator)

    def upper_triangular_cell(self, verbose: bool | None = False) -> None:
        """Transform to upper-triangular cell.
        ASE Nose-Hoover implementation only supports upper-triangular cell
        while ASE's canonical description is lower-triangular cell.

        Args:
            verbose (bool): Whether to notify user about upper-triangular cell
                transformation. Default = False
        """
        if not NPT._isuppertriangular(self.atoms.get_cell()):
            a, b, c, alpha, beta, gamma = self.atoms.cell.cellpar()
            angles = np.radians((alpha, beta, gamma))
            sin_a, sin_b, _sin_g = np.sin(angles)
            cos_a, cos_b, cos_g = np.cos(angles)
            cos_p = (cos_g - cos_a * cos_b) / (sin_a * sin_b)
            cos_p = np.clip(cos_p, -1, 1)
            sin_p = (1 - cos_p**2) ** 0.5

            new_basis = [
                (a * sin_b * sin_p, a * sin_b * cos_p, a * cos_b),
                (0, b * sin_a, b * cos_a),
                (0, 0, c),
            ]

            self.atoms.set_cell(new_basis, scale_atoms=True)
            if verbose:
                print("Transformed to upper triangular unit cell.", flush=True)
