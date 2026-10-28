# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project layout

`src/matgl/` is a `src`-layout package (installed as `matgl`). The package targets Python 3.11+ and is built with setuptools; environment management is done with `uv` (`uv.lock` is checked in).

## Common commands

Environment setup (Python 3.11+ required):

```bash
uv venv && uv sync                 # install runtime deps + dev group (which includes the lint group: ruff, mypy)
```

Tests:

```bash
uv run pytest                                       # full suite
uv run pytest tests/models/test_tensornet.py        # one file
uv run pytest tests/models/test_tensornet.py::test_module  # one test
uv run pytest -k tensornet --cov=matgl              # keyword filter + coverage
```

`pyproject.toml` injects `--durations=30 --quiet -rXs --color=yes -p no:warnings` automatically (so don't add `-q`: the doubled quiet hides the pass/fail summary). `tests/conftest.py` calls `matgl.clear_cache(confirm=False)` at import time, so the user-level `~/.cache/matgl/` is wiped on every test run — be aware when running tests on a machine where you've manually cached models. An autouse fixture runs every test inside its own `tmp_path`, so anything written to relative paths (datasets, saved models, `lightning_logs`) never lands in the repo.

Lint/format/type-check (matches the `Lint` workflow):

```bash
uv run ruff check src
uv run ruff format src        # add --check for CI parity
uv run mypy -p matgl
```

Pre-commit (`uv run pre-commit run --all-files`) layers `ruff`, `mypy`, `codespell`, and `nbstripout`. `pre-commit-ci` skips `mypy`. Keep the `mirrors-mypy` hook `rev` in step with the mypy version in `uv.lock` so local hooks and the `Lint` workflow agree.

CLI entry point (installed by the package):

```bash
mgl relax   -i Li2O.cif -o Li2O_relax.cif
mgl predict -m M3GNet-MP-2018.6.1-Eform -i Li2O.cif
mgl md      -i Li2O.cif -e nvt -t 300 -n 1000
mgl clear -y
```

Docs/release helpers live in `tasks.py` (`invoke make-docs`, `invoke release <version>`). `changes.md` is the canonical changelog and is the source `release` reads from.

## Architecture

matgl uses PyTorch Geometric (PyG) exclusively as its graph backend. DGL is no longer
supported — all DGL code paths, backend-selection stubs, and compatibility shims have
been removed from the codebase. Do not reintroduce DGL imports, `MGL_BACKEND`-style
switches, or any abstraction that suggests multiple graph backends; PyG is the only
backend, and code/tests should assume PyG types (`torch_geometric.data.Data`,
`Batch`) directly.

### Public-vs-private convention (sklearn-style)

Modules prefixed with `_` are private. Public names are re-exported from each subpackage's `__init__.py` (e.g. `matgl.models.TensorNet`, `matgl.apps.pes.Potential`). When adding new code, import only from the exposed APIs and add new public names through the relevant `__init__.py`.

### Subpackage roles

- `matgl.models` — graph-network architectures (M3GNet, MEGNet, CHGNet, TensorNet, SO3Net, QET) plus `TransformedTargetModel` wrapper.
- `matgl.layers` — building blocks (embeddings, graph conv, readout, ZBL, atom-ref, basis functions, activations).
- `matgl.apps.pes.Potential` — wraps a graph model into an interatomic potential that returns energies/forces/stresses via autograd. Implementation in `apps/pes.py`.
- `matgl.graph` — graph construction and `MGLDataset` / data loaders.
- `matgl.ext` — external-library adaptors (`pymatgen` for `Structure`/`Molecule` → graph, `ase` for `Relaxer` / `MolecularDynamics` calculators, `alchmtk` for NVIDIA Alchemi toolkit ops, `lammps` for exporting TorchScript models to the `pair_matgl` LAMMPS plugin, `jax` for a JAX port of TensorNet/QET with an ASE calculator).
- Electrostatics (`LinearQeq` charge equilibration and the Gaussian-smeared Coulomb `ElectrostaticPotential` used by QET) live in `matgl.layers` (`layers/_fast_qeq.py`, `layers/_elec_pot.py`). `matgl.electrostatics` is only a deprecated re-export that warns on import; don't add code there.
- `matgl.data` — target transformers (`Normalizer`, `LogTransformer`; MSONable so `TransformedTargetModel` can be saved).
- `matgl.kernels` / `matgl.ops` — NVIDIA Warp GPU kernels and their pure-PyTorch reference ops; per-file ignores in `pyproject.toml` skip docstring/lint rules here.
- `matgl.utils` — `IOMixIn` (model save/load + HF Hub), training loops (`training.py`), spherical harmonics, math helpers, cutoff functions.

### Nested model pattern

End-user models often wrap a graph model: `Potential` wraps an `M3GNet`/`TensorNet` and exposes forces/stresses; `TransformedTargetModel` mirrors sklearn's `TransformedTargetRegressor` for fitting on transformed targets (e.g. log bulk modulus). Save/load goes through the outer wrapper — it dispatches to inner models automatically.

### Model serialization (IOMixIn)

Models subclass `torch.nn.Module` **and** `matgl.utils.io.IOMixIn`, and call `self.save_args(locals(), kwargs)` at the end of `__init__`. This is required so `model.save(path)` / `Model.load(path)` / `matgl.load_model(...)` can round-trip the constructor arguments via the standard triple `model.pt` + `state.pt` + `model.json`. Set a class-level `__version__: int` and bump it whenever architectural changes invalidate previously saved checkpoints.

`matgl.load_model(...)` resolves, in order: a local directory containing the three files, an `"owner/name"` Hugging Face Hub repo, then a bare name under the `materialyze` org. Hub downloads are cached under `~/.cache/matgl/`. Use `model.push_to_hub(...)` to publish.

### Pre-trained models

All pre-trained models live on Hugging Face Hub under the `materialyze` org, uploaded with
`model.push_to_hub(...)` and loaded with `matgl.load_model("materialyze/<model-name>")` or the bare-name form.
The former in-repo `pretrained_models/` directory has been removed; do not reintroduce it.

### Tests

`tests/` mirrors `src/matgl/`. `tests/conftest.py` provides session-scoped fixtures: pymatgen `Structure`/`Molecule` (`LiFePO4`, `BaNiO3`, `MoS`, `MoS2`, `Mo`, `NaCl`, `CH4`, `CO`, `Li3InCl6`, `AcAla3NHMe`) and pre-built graphs (`graph_<name>`). The `get_graph` helper in `conftest.py` builds a PyG graph for the given pymatgen structure.

## Conventions

- `from __future__ import annotations` is enforced by `ruff isort` (see `lint.isort.required-imports`) — don't remove it.
- Google-style docstrings; ruff `pydocstyle` is enabled with selected ignores. Tests, docs, examples, `kernels/`, and `ops/` have docstring requirements waived in `pyproject.toml`.
- Line length 120; ruff handles formatting (do not run black even though config exists).
- New features need tests in the matching `tests/<subpackage>/` directory.
