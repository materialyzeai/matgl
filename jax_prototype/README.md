# JAX inference-path prototype for MatGL (TensorNet / QET, PyG)

A standalone prototype that reimplements the **inference path** (energy + forces
+ stress) of MatGL's PyG-backend **TensorNet** and **QET** models in **JAX**, and
benchmarks it against the eager-PyTorch baseline.

It answers one question: *does a portable, fusing-compiler backend (JAX/XLA) make
the MatGL MD / relaxation loop faster, on hardware where the NVIDIA-Warp kernels
are unavailable (CPU, Apple Silicon)?*

**Result: yes — 2.7–3.5× faster** for a full energy+forces+stress step on CPU,
with energies/forces/stresses matching PyTorch to float64 precision.

This directory is **not** part of the installed `matgl` package (it lives outside
`src/`, so it cannot affect `import matgl` or the default dependency closure). It
imports `matgl` only to load weights and to produce reference outputs.

## Status

| Stage | State |
|-------|-------|
| TensorNet forward (3 RBF types, O(3)/SO(3), intensive/extensive) | done, parity < 1e-6 (float64) |
| Potential — forces + stress via `jax.value_and_grad` | done, parity < 1e-6 |
| QET — charge-equilibration + electrostatics tail | done, parity < 1e-6 |
| Weight conversion (torch `state_dict` → JAX pytree) | done |
| `JAXPESCalculator` (ASE) + edge padding/bucketing | done |
| Benchmark harness (sizes × backends) | done |

21 parity tests pass (`tests/`). Not ported: the `use_warp=True` kernel path,
training, the per-atom Hessian loop (see *Limitations*).

## Install

JAX is an extra dependency. Install it into the existing matgl `.venv` (it is
independent of torch/PyG and does not conflict):

```bash
uv pip install --python .venv/bin/python "jax>=0.4.30"
```

For GPU/Apple acceleration install the platform wheel instead (`jax[cuda12]` on
NVIDIA, `jax-metal` on Apple Silicon) — the prototype code is unchanged.

## Layout

```
matgl_jax/
  _math.py        tensor algebra, cutoffs, scatter, layer/linear primitives
  _basis.py       radial bases (smooth + plain spherical Bessel, Gaussian)
  _tensornet.py   functional TensorNet forward_features + readout
  _qet.py         QET tail: chi/hardness/sigma, LinearQeq, electrostatics
  _potential.py   strain application + jitted (E, forces, stress) via value_and_grad
  _convert.py     torch state_dict -> JAX pytree
  _pad.py         edge padding / bucketing for shape-stable XLA compilation
  _calculator.py  JAXPESCalculator (ASE Calculator)
benchmarks/       bench.py + structures.py
tests/            test_roundtrip.py (TensorNet), test_qet.py (QET)
```

## Usage

```python
import jax
from matgl.apps.pes import Potential
from matgl.models import TensorNet
from matgl_jax import convert_potential, make_potential_fn

potential = Potential(model=TensorNet(...))          # any matgl TensorNet/QET Potential
params, cfg, extras = convert_potential(potential)   # torch -> JAX pytree
fn = make_potential_fn(params, cfg, extras)          # jitted (E, forces, stress)
e, forces, stress = fn(pos, strain, frac, lat3, pbc_offset, z, edge_index, batch, edge_mask)
```

As a drop-in ASE calculator (plugs into matgl's `MolecularDynamics` / `Relaxer`):

```python
from matgl_jax import JAXPESCalculator
atoms.calc = JAXPESCalculator(potential, stress_unit="eV/A3")
```

Run the tests / benchmark:

```bash
.venv/bin/python -m pytest jax_prototype/tests/
.venv/bin/python jax_prototype/benchmarks/bench.py            # TensorNet
.venv/bin/python jax_prototype/benchmarks/bench.py --model qet --torch-compile
```

## Results

### Parity (float64, random-weight models)

JAX vs PyTorch energy / forces / stress agree to `< 1e-6` across both RBF
families, O(3)/SO(3), intensive/extensive readouts, and every QET variant
(environment-dependent hardness, trainable sigma, magmom, single-atom systems).
Padded sentinel edges contribute exactly zero (no NaN leak).

### Benchmark (Apple Silicon, CPU, float32; one energy+forces+stress step)

TensorNet, `units=64`, `nblocks=2`, smooth spherical Bessel:

| system      | atoms | edges | eager ms | JAX ms | speedup |
|-------------|------:|------:|---------:|-------:|--------:|
| tiny-2      |     2 |   100 |     4.8  |   1.7  |  2.7×   |
| small-64    |    64 |  1792 |    22.3  |   6.4  |  3.5×   |
| medium-216  |   216 |  6048 |    48.8  |  17.9  |  2.7×   |
| large-512   |   512 | 14336 |    95.7  |  35.6  |  2.7×   |

QET is within noise of these numbers (it reuses `forward_features`). JAX also
beats `torch.compile` (Inductor) at every size. Graph build (pymatgen neighbour
list) is < 0.5 ms — negligible — so the speedup is essentially all model +
autograd. XLA compile is a one-time ~0.6–1.0 s, amortised over an MD trajectory.

The gain is kernel fusion + elimination of per-op Python dispatch: TensorNet runs
hundreds of small Cartesian-tensor ops per step, which `jax.jit` fuses into one
XLA program (forward + backward together).

## How it works

* **Whole-step fusion.** `make_potential_fn` wraps `jax.value_and_grad` of the
  energy in a single `jax.jit` — forward, force backward, and stress derivative
  compile to one XLA program.
* **Static shapes.** XLA needs fixed shapes; neighbour-list edge counts vary per
  MD step. `_pad.py` pads edges to a bucket capacity (sentinel self-loops with a
  non-zero PBC image — so `grad(||bond_vec||)` stays finite — masked to zero
  contribution). Atom count is constant within a trajectory, so one compilation
  is reused.
* **Weight conversion.** `_convert.py` maps the torch `state_dict` to a nested
  dict pytree; `nn.Linear` is stored transposed (`x @ W` instead of `x @ W.T`),
  LayerNorm / Embedding copied verbatim, Bessel-root constants rederived. A
  Warp-enabled TensorNet (the Warp embedding fuses `distance_proj1/2/3` into one
  `Linear`) converts to the same JAX pytree as its plain-PyG twin.
* **Faithful stress.** The strain leaf deforms both the PBC offshift *and* the
  atomic positions, reproducing `Potential.forward`'s two autograd leaves.

## Limitations

* CPU-measured; GPU/Apple-Metal numbers require the platform JAX wheel.
* The NVIDIA Warp kernels themselves are not ported — the JAX path *is* the
  portable alternative (on CUDA the comparison would be JAX/XLA vs Warp). A
  Warp-enabled TensorNet is still accepted: `convert_potential` reads only the
  weights (identical to the PyG variant bar the fused `distance_proj`), so the
  JAX result is the same whether or not the source model used Warp.
* Training and the per-atom Hessian loop are out of scope (the Hessian is a
  strong future JAX target via `jax.hessian` / forward-over-reverse).
* Non-smooth spherical Bessel is ported for `l ≤ 4` and only `max_l=1` (matgl's
  TensorNet itself is dimensionally inconsistent for non-smooth `max_l>1`).
* Single batch graph (`num_graphs=1`); batched inference is straightforward to
  add but unneeded for the MD/relax use case.

## Promotion path

If this graduates, the natural home is `src/matgl/ext/_jax_*` behind an optional
`jax` extra in `pyproject.toml`; the conversion + calculator API would not change.
