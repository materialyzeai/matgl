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

The Python side (one repo up) ships `mgl create-lammps-model`, which
produces the `.pt` artifact these pair styles consume.

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

LAMMPS builds its style tables by scanning package directories and then
generating `style_pair.h`, and the generation happens roughly two thirds of
the way through `cmake/CMakeLists.txt` (`GenerateStyleHeaders(...)`, line 794
in `stable_22Jul2025_update5`). **Appending an `include()` to the END of that
file is therefore too late**: the sources compile and libtorch links, but the
style never reaches `style_pair.h` and LAMMPS rejects it at run time with

```
ERROR: Unrecognized pair style 'matgl' (src/force.cpp:275)
```

Register it the way LAMMPS registers its own packages instead:

```bash
# 1) Copy or symlink the source files into the LAMMPS src tree.
ln -s /path/to/matgl/lammps/src/ML-MATGL <lammps>/src/ML-MATGL

# 2) Add ML-MATGL to the package list, so LAMMPS' own per-package loop does
#    RegisterStyles + target_sources + include dir at the right point.
#    In <lammps>/cmake/CMakeLists.txt, inside set(STANDARD_PACKAGES ...):
#        ML-IAP
#      + ML-MATGL
#        ML-PACE
#    (`-D PKG_ML-MATGL=ON` then works like any other package flag.)

# 3) Link libtorch. This snippet only does find_package(Torch) and the link;
#    it must NOT also add the sources, or every file compiles twice under two
#    paths and the link fails on duplicate symbols.
echo 'include(/path/to/matgl/lammps/cmake/ML-MATGL.cmake)' \
    >> <lammps>/cmake/CMakeLists.txt

# 4) Configure + build. Match libtorch's CXX11 ABI to LAMMPS'.
cmake -B build -S <lammps>/cmake \
    -D PKG_ML-MATGL=ON \
    -D CMAKE_PREFIX_PATH=/path/to/libtorch \
    -D CMAKE_BUILD_TYPE=Release \
    -D BUILD_MPI=ON
cmake --build build -j 8
```

Check the registration before running anything:

```bash
grep matgl build/styles/style_pair.h     # expect pair_matgl.h (and _kokkos.h)
build/lmp -h | tr ' ' '\n' | grep '^matgl'
```

### 2b. Build the Kokkos GPU variant

To get the `matgl/kk` pair style, also enable Kokkos and append the
matching snippet to LAMMPS' CMake. CUDA example for an Ampere card
(A100/A30):

```bash
# Put the Kokkos sources where the KOKKOS package looks for them: its
# RegisterStylesExt(${KOKKOS_PKG_SOURCES_DIR} kokkos ...) scans
# <lammps>/src/KOKKOS for *_kokkos.h style headers and picks up matgl/kk
# automatically. A separate directory is not scanned.
cp /path/to/matgl/lammps/src/KOKKOS/pair_matgl_kokkos.* <lammps>/src/KOKKOS/

echo 'include(/path/to/matgl/lammps/cmake/ML-MATGL-KOKKOS.cmake)' \
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
mpirun -n 1 build/lmp -k on g 1 -sf kk -pk kokkos neigh half -in in.matgl_si
```

**`neigh half` is required, not optional.** `pair_matgl` needs `newton on`
(it folds periodic edges back onto local rows and needs ghost contributions),
and LAMMPS refuses `newton on` together with the Kokkos default `neigh full`:

```
ERROR: Must use 'newton off' with KOKKOS package option 'neigh full'
(src/KOKKOS/kokkos.cpp:693)
```

Equivalently, put `package kokkos neigh half` in the input deck before
`atom_style`.

`-sf kk` makes LAMMPS prefer Kokkos pair styles, so `pair_style matgl`
in your input deck dispatches to `matgl/kk` automatically. If you'd
rather force it explicitly, write `pair_style matgl/kk` instead.

**Single-GPU only.** Multi-rank Kokkos with libtorch is unreliable
(MACE issues #1294 and #322); the package emits a CMake message making
this explicit.

Tested with:

- LibTorch 2.2.x – 2.7.x (CXX11 ABI). **The Kokkos variant needs a CUDA
  build of libtorch**, not a CPU-only one: `pair_matgl_kokkos.cpp` selects
  `torch::Device(torch::kCUDA, gpu)` and wraps Kokkos device buffers as
  tensors without a copy, so a CPU-only libtorch silently runs the model on
  the host. The CPU build is what the CI job uses for the serial style.
- LAMMPS develop branch (Aug 2024 or newer for the `add_request` /
  `REQ_GHOST` neighbor-list API); verified on `stable_22Jul2025_update5`.
- C++17, MPI optional.

#### Troubleshooting: CUDA 12.9 and newer toolkits

libtorch's bundled Caffe2 CMake config predates two changes and each aborts
the generate step. Both are fixed by a small file injected before
`find_package(Torch)`, e.g. via
`-D CMAKE_PROJECT_lammps_INCLUDE=/path/to/fixups.cmake`:

- CUDA 12.9 removed the nvToolsExt shared library, so `FindCUDAToolkit` no
  longer defines `CUDA::nvToolsExt` while `Caffe2/public/cuda.cmake` still
  links it into `torch::nvtoolsext`. Declare it as a header-only interface
  target (the nvtx3 headers are still shipped):
  `add_library(CUDA::nvToolsExt INTERFACE IMPORTED GLOBAL)`.
- On a machine without MKL, Caffe2 leaves the literal
  `MKL_INCLUDE_DIR-NOTFOUND` inside torch's `INTERFACE_INCLUDE_DIRECTORIES`.
  Point `MKL_INCLUDE_DIR` at any existing directory.

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
cd lammps/tests
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
