# MatGL → LAMMPS pair_style

`pair_matgl` is a LAMMPS pair style that loads a TorchScript-compiled
**MatGL TensorNet or M3GNet** PES (PyG backend, extensive head) and uses
LibTorch to evaluate energies, forces, and the virial tensor on every
timestep.

This directory ships:

- `src/ML-MATGL/pair_matgl.{cpp,h}` — the CPU/serial pair style.
- `src/KOKKOS/pair_matgl_kokkos.{cpp,h}` — the Kokkos GPU/host variant
  (`pair_style matgl/kk`).
- `cmake/ML-MATGL.cmake` and `cmake/ML-MATGL-KOKKOS.cmake` — drop-in
  CMake snippets.
- `tests/in.matgl_si` — sample input deck for a single-point parity check.

The matgl CLI ships `mgl create-lammps-model`, which produces the `.pt`
artifact these pair styles consume, and `mgl lammps --patch <lammps-src-dir>`,
which drops these sources into a stock LAMMPS tree and wires up the build.

> **Status.** The CPU/serial pair style is exercised in CI
> (`.github/workflows/lammps-build.yml`). The Kokkos GPU variant builds
> but is not yet covered in CI, as GitHub-hosted runners have no GPU.

## Building

### 1. Export a LAMMPS-loadable model

```bash
# From your matgl checkout:
uv run mgl create-lammps-model \
    -m materialyze/TensorNet-MatPES-r2SCAN \
    -o tensornet_matpes_r2scan.pt \
    --dtype float32
```

The CLI prints `r_max`, `n_species`, the dtype, and the species list — all
of which you'll need for `pair_coeff`.

### 2. Build LAMMPS with the package

Drop the package into a stock LAMMPS source tree and configure:

```bash
# 1) Copy or symlink the source files.
ln -s /path/to/matgl/src/matgl/lammps/src/ML-MATGL <lammps>/src/ML-MATGL

# 2) Tell LAMMPS' CMake about the package.
echo 'include(/path/to/matgl/src/matgl/lammps/cmake/ML-MATGL.cmake)' \
    >> <lammps>/cmake/CMakeLists.txt

# 3) Configure + build. Match libtorch's CXX11 ABI to LAMMPS'.
cmake -B build -S <lammps>/cmake \
    -D PKG_ML-MATGL=ON \
    -D CMAKE_PREFIX_PATH=/path/to/libtorch \
    -D CMAKE_BUILD_TYPE=Release \
    -D BUILD_MPI=ON
cmake --build build -j 8
```

### 2b. Build the Kokkos GPU variant

The quickest way is `mgl lammps --patch`, which copies the CPU and Kokkos
pair-style sources into `<lammps>/src/` and `<lammps>/src/KOKKOS/` and wires
libtorch into the CMake build for you:

```bash
mgl lammps --patch <lammps>
```

The command is idempotent and prints the exact `cmake` invocation to run next.
It needs a full LAMMPS checkout (the in-tree `lib/kokkos` and `src/KOKKOS`)
and a matgl source checkout, since the `lammps/` tree ships only with the repo.

To do it by hand instead, enable Kokkos and append the matching snippet to
LAMMPS' CMake. CUDA example for an Ampere card (A100/A30):

```bash
echo 'include(/path/to/matgl/src/matgl/lammps/cmake/ML-MATGL-KOKKOS.cmake)' \
    >> <lammps>/cmake/CMakeLists.txt

cmake -B build -S <lammps>/cmake \
    -D PKG_ML-MATGL=ON \
    -D PKG_KOKKOS=ON \
    -D Kokkos_ENABLE_CUDA=ON \
    -D Kokkos_ARCH_AMPERE80=ON \
    -D CMAKE_PREFIX_PATH=/path/to/libtorch \
    -D CMAKE_CXX_COMPILER=<lammps>/lib/kokkos/bin/nvcc_wrapper \
    -D CMAKE_BUILD_TYPE=Release
cmake --build build -j 8
```

Run with:

```bash
mpirun -n 1 build/lmp -k on g 1 -sf kk -in in.matgl_si
```

`-sf kk` makes LAMMPS prefer Kokkos pair styles, so `pair_style matgl`
in your input deck dispatches to `matgl/kk` automatically. If you'd
rather force it explicitly, write `pair_style matgl/kk` instead.

**Single-GPU only.** Multi-rank Kokkos with libtorch is unreliable
(MACE issues #1294 and #322); the package emits a CMake message making
this explicit.

Tested with:

- LibTorch 2.2.x – 2.5.x (CXX11 ABI, CPU build).
- LAMMPS develop branch (Aug 2024 or newer for the `add_request` /
  `REQ_GHOST` neighbor-list API).
- C++17, MPI optional.

## LAMMPS input syntax

```lammps
units           metal
atom_style      atomic
atom_modify     map yes        # required: pair_matgl needs the atom map
newton          on             # required: ghost contributions

pair_style      matgl
pair_coeff      * * tensornet_matpes_r2scan.pt Si C O
```

`pair_coeff` arguments after the `.pt` path are **species symbols** in
LAMMPS atom-type order: type 1 = first symbol, type 2 = second, …

The cutoff (`r_max`) is read from the model — you don't pass it.

### Optional pair_style flags

```lammps
pair_style matgl no_domain_decomposition
```

Reserved for future single-rank optimisations (mirrors the MACE flag).
Currently a no-op.

## Limitations

- **No per-atom energies / virials.** `eflag_atom`, `vflag_atom`, and
  `compute … pe/atom` will error. The model returns a single
  `total_energy_local` scalar plus a 3×3 virial tensor; per-atom
  decompositions would require a different export.
- **`atom_style atomic` only** for now. Charged systems aren't supported
  (the model has no charge head).
- **TorchScript artifacts are dtype-specific.** Re-run
  `mgl create-lammps-model --dtype float64` to get a double-precision
  model; mixing dtypes between LAMMPS and the model will error at load
  time.
- **Multi-rank**: works for CPU MPI, but each rank loads the model
  independently (memory adds up). The `data_mean` buffer baked into the
  TorchScript is added once per rank — keep `data_mean = 0` (the default
  for trained MatGL PES models). Non-zero `data_mean` will over-count
  proportionally to the number of ranks.
- **No restart support.** The model lives on disk; `restart` files don't
  capture the path. Re-issue `pair_style` / `pair_coeff` after a restart.
- **TensorNet and M3GNet only.** The export path (`mgl
  create-lammps-model` / `LAMMPSMatGLModel`) supports the PyG TensorNet
  and M3GNet PES models with an extensive head (TensorNet must be
  no-Warp; both require `use_smooth=True`, and M3GNet requires
  `use_phi=False`). For M3GNet the three-body line graph is built inside
  the TorchScript module, so both the CPU and Kokkos pair styles run it
  with no extra handling. CHGNet, MEGNet, SO3Net, and QET are not yet
  wired into the LAMMPS export path.

## Continuous integration

`.github/workflows/lammps-build.yml` builds the **CPU** pair style on
every push that touches the `lammps/` tree, the Python wrapper, or the
workflow itself. The job runs inside the `lammps/lammps-build:ubuntu_latest`
public Docker image, downloads a CXX11-ABI libtorch, clones LAMMPS at a
pinned tag, builds with `PKG_ML-MATGL=ON`, exports a tiny in-tree model
through `LAMMPSMatGLModel`, runs the `in.matgl_si` deck, and diffs the
LAMMPS energy against the Python reference.

The Kokkos variant is **not** exercised in CI today — GitHub-hosted
runners have no GPU. Hardware-accelerated CI would need a self-hosted
CUDA runner.

## Verifying a build

```bash
cd src/matgl/lammps/tests
<lammps>/build/lmp -in in.matgl_si
```

The test deck prints energy, forces, and stress on a small Si supercell.
Compare against the Python reference:

```bash
uv run python tests/python_reference.py    # in this directory
```

Energies should match within `1e-5 eV`, forces within `1e-4 eV/Å`, and
stresses (when nonzero) within `1e-3 GPa`.

## Implementation notes

- The pair style requests a **full neighbor list with ghost atoms**
  (`REQ_FULL | REQ_GHOST`). The model expects edge indices that span both
  owned and ghost atoms.
- Every edge is folded back onto the *local* row of the atom it represents
  (via `atom->map(atom->tag[j])`) rather than pointing at the ghost row
  directly. TensorNet's message-passing layers need one consistent row
  per physical atom — a ghost row never propagates outgoing messages back
  to the atom it duplicates. The periodic image is recovered explicitly as
  an integer `unit_shifts` (the ghost/local position difference,
  transformed through the box's inverse deformation matrix and rounded to
  the nearest integer), rather than relying on LAMMPS' already-imaged
  ghost positions with `unit_shifts = 0`. Single-rank only: ghost atoms
  can be owned by a different MPI rank, so there is no local row to fold
  onto in a multi-rank run.
- Forces are accumulated for **all** atoms (owned + ghost). LAMMPS' usual
  `comm->reverse_comm` step then sums ghost contributions back to the
  rank that owns each atom. This requires `newton on`.
- Virials are written into the global `virial[6]` array directly as
  `virial -= va` (the model returns `virials = dE/dstrain = -W`, while
  LAMMPS' convention is `W = sum_i r_i ⊗ f_i`). We set
  `no_virial_fdotr_compute = 1` in the constructor so LAMMPS doesn't
  recompute the virial from forces.

## Reference

The Python wrapper that exports LAMMPS-loadable models is documented
inline at `src/matgl/ext/_lammps.py` in the matgl repo.
