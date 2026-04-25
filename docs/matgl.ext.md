---
layout: default
title: matgl.ext.md
nav_exclude: true
---

# matgl.ext package

This package implements interfaces to external packages such as Pymatgen and the Atomic Simulation Environment.

## matgl.ext.ase module

Interfaces to the Atomic Simulation Environment package for dynamic simulations.

### *class* matgl.ext.ase.Atoms2Graph(element_types: tuple[str, …], cutoff: float = 5.0)

Bases: `GraphConverter`

Construct a DGL graph from ASE Atoms.

Init Atoms2Graph from element types and cutoff radius.

* **Parameters:**
  * **element_types** – List of elements present in dataset for graph conversion. This ensures all graphs are
    constructed with the same dimensionality of features.
  * **cutoff** – Cutoff radius for graph representation

#### get_graph(atoms: Atoms)

Get a DGL graph from an input Atoms.

* **Parameters:**
  **atoms** – Atoms object.
* **Returns:**
  DGL graph
  state_attr: state features
* **Return type:**
  g

### *class* matgl.ext.ase.M3GNetCalculator(potential: Potential, state_attr: torch.Tensor | None = None, stress_weight: float = 1.0, \*\*kwargs)

Bases: `Calculator`

M3GNet calculator for ASE.

Init M3GNetCalculator with a Potential.

* **Parameters:**
  * **potential** (*Potential*) – m3gnet.models.Potential
  * **state_attr** (*tensor*) – State attribute
  * **compute_stress** (*bool*) – whether to calculate the stress
  * **stress_weight** (*float*) – the stress weight.
  * **\*\*kwargs** – Kwargs pass through to super().      *\_init*\_().

#### calculate(atoms: Atoms | None = None, properties: list | None = None, system_changes: list | None = None)

Perform calculation for an input Atoms.

* **Parameters:**
  * **atoms** (*ase.Atoms*) – ase Atoms object
  * **properties** (*list*) – list of properties to calculate
  * **system_changes** (*list*) – monitor which properties of atoms were
    changed for new calculation. If not, the previous calculation
    results will be loaded.

#### implemented_properties\*: List[str]\*       *= (‘energy’, ‘free_energy’, ‘forces’, ‘stress’, ‘hessian’)*

Properties calculator can handle (energy, forces, …)

### *class* matgl.ext.ase.MolecularDynamics(atoms: Atoms, potential: Potential, state_attr: torch.Tensor | None = None, ensemble: Literal[‘nvt’, ‘npt’, ‘npt_berendsen’] = ‘nvt’, temperature: int = 300, timestep: float = 1.0, pressure: float = 6.324209121801212e-07, taut: float | None = None, taup: float | None = None, compressibility_au: float | None = None, trajectory: str | Trajectory | None = None, logfile: str | None = None, loginterval: int = 1, append_trajectory: bool = False)

Bases: `object`

Molecular dynamics class.

Init the MD simulation.

* **Parameters:**
  * **atoms** (*stress* *of* *the*) – atoms to run the MD
  * **potential** (*Potential*) – potential for calculating the energy, force,
  * **atoms** –
  * **state_attr** (*torch.Tensor*) – State attr.
  * **ensemble** (*str*) – choose from ‘nvt’ or ‘npt’. NPT is not tested,
  * **caution** (*use with extra*) –
  * **temperature** (*float*) – temperature for MD simulation, in K
  * **timestep** (*float*) – time step in fs
  * **pressure** (*float*) – pressure in eV/A^3
  * **taut** (*float*) – time constant for Berendsen temperature coupling
  * **taup** (*float*) – time constant for pressure coupling
  * **compressibility_au** (*float*) – compressibility of the material in A^3/eV
  * **trajectory** (*str* *or* *Trajectory*) – Attach trajectory object
  * **logfile** (*str*) – open this file for recording MD outputs
  * **loginterval** (*int*) – write to log file every interval steps
  * **append_trajectory** (*bool*) – Whether to append to prev trajectory.

#### run(steps: int)

Thin wrapper of ase MD run.

* **Parameters:**
  **steps** (*int*) – number of MD steps

#### set_atoms(atoms: Atoms)

Set new atoms to run MD.

* **Parameters:**
  **atoms** (*Atoms*) – new atoms for running MD.

### *class* matgl.ext.ase.OPTIMIZERS(value, names=None, \*, module=None, qualname=None, type=None, start=1, boundary=None)

Bases: `Enum`

An enumeration of optimizers for used in.

#### bfgs       *= <class ‘ase.optimize.bfgs.BFGS’>*

#### bfgslinesearch       *= <class ‘ase.optimize.bfgslinesearch.BFGSLineSearch’>*

#### fire       *= <class ‘ase.optimize.fire.FIRE’>*

#### lbfgs       *= <class ‘ase.optimize.lbfgs.LBFGS’>*

#### lbfgslinesearch       *= <class ‘ase.optimize.lbfgs.LBFGSLineSearch’>*

#### mdmin       *= <class ‘ase.optimize.mdmin.MDMin’>*

#### scipyfminbfgs       *= <class ‘ase.optimize.sciopt.SciPyFminBFGS’>*

#### scipyfmincg       *= <class ‘ase.optimize.sciopt.SciPyFminCG’>*

### *class* matgl.ext.ase.Relaxer(potential: Potential | None = None, state_attr: torch.Tensor | None = None, optimizer: Optimizer | str = ‘FIRE’, relax_cell: bool = True, stress_weight: float = 0.01)

Bases: `object`

Relaxer is a class for structural relaxation.

* **Parameters:**
  * **potential** (*Potential*) – a M3GNet potential, a str path to a saved model or a short name for saved model
  * **distribution** (*that comes with M3GNet*) –
  * **state_attr** (*torch.Tensor*) – State attr.
  * **optimizer** (*str* *or* *ase Optimizer*) – the optimization algorithm.
  * **“FIRE”** (*Defaults to*) –
  * **relax_cell** (*bool*) – whether to relax the lattice cell
  * **stress_weight** (*float*) – the stress weight for relaxation.

#### relax(atoms: Atoms, fmax: float = 0.1, steps: int = 500, traj_file: str | None = None, interval=1, verbose=False, \*\*kwargs)

Relax an input Atoms.

* **Parameters:**
  * **atoms** (*Atoms*) – the atoms for relaxation
  * **fmax** (*float*) – total force tolerance for relaxation convergence.
  * **forces** (*Here fmax is a sum* *of* *force and stress*) –
  * **steps** (*int*) – max number of steps for relaxation
  * **traj_file** (*str*) – the trajectory file for saving
  * **interval** (*int*) – the step interval for saving the trajectories
  * **verbose** (*bool*) – Whether to have verbose output.
  * **kwargs** – Kwargs pass-through to optimizer.

### *class* matgl.ext.ase.TrajectoryObserver(atoms: Atoms)

Bases: `Sequence`

Trajectory observer is a hook in the relaxation process that saves the
intermediate structures.

Init the Trajectory Observer from a Atoms.

* **Parameters:**
  **atoms** (*Atoms*) – Structure to observe.

#### as_pandas()

Returns: DataFrame of energies, forces, streeses, cells and atom_positions.

#### save(filename: str)

Save the trajectory to file.

* **Parameters:**
  **filename** (*str*) – filename to save the trajectory.

## matgl.ext.pymatgen module

Interface with pymatgen objects.

### *class* matgl.ext.pymatgen.Molecule2Graph(element_types: tuple[str, …], cutoff: float = 5.0)

Bases: `GraphConverter`

Construct a DGL graph from Pymatgen Molecules.

* **Parameters:**
  * **element_types** (*List* *of* *elements present in dataset for graph conversion. This ensures all graphs are*) – constructed with the same dimensionality of features.
  * **cutoff** (*Cutoff radius for graph representation*) –

#### get_graph(mol: Molecule)

Get a DGL graph from an input molecule.

* **Parameters:**
  **mol** – pymatgen molecule object
* **Returns:**
  (dgl graph, state features)

### *class* matgl.ext.pymatgen.Structure2Graph(element_types: tuple[str, …], cutoff: float = 5.0)

Bases: `GraphConverter`

Construct a DGL graph from Pymatgen Structure.

* **Parameters:**
  * **element_types** (*List* *of* *elements present in dataset for graph conversion. This ensures all graphs are*) – constructed with the same dimensionality of features.
  * **cutoff** (*Cutoff radius for graph representation*) –

#### get_graph(structure: Structure)

Get a DGL graph from an input Structure.

* **Parameters:**
  **structure** – pymatgen structure object
* **Returns:**
  g: DGL graph
  state_attr: state features

### matgl.ext.pymatgen.get_element_list(train_structures: list[pymatgen.core.structure.Structure | pymatgen.core.structure.Molecule])

Get the dictionary containing elements in the training set for atomic features.

* **Parameters:**
  **train_structures** – pymatgen Molecule/Structure object
* **Returns:**
  Tuple of elements covered in training set