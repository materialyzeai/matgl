# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project layout

`src/matgl/` is a `src`-layout package (installed as `matgl`). The package targets Python 3.11+ and is built with setuptools; environment management is done with `uv` (`uv.lock` is checked in).

## Common commands

Environment setup (Python 3.11+ required):

```bash
uv venv && uv sync                 # install runtime deps + dev group
```

Tests:

```bash
uv run pytest                                       # full suite
uv run pytest tests/models/test_tensornet.py        # one file
uv run pytest tests/models/test_tensornet.py::test_module  # one test
uv run pytest -k tensornet --cov=matgl              # keyword filter + coverage
```

`pyproject.toml` injects `--durations=30 --quiet -rXs -p no:warnings` automatically. `tests/conftest.py` calls `matgl.clear_cache(confirm=False)` at import time, so the user-level `~/.cache/matgl/` is wiped on every test run — be aware when running tests on a machine where you've manually cached models.

Lint/format/type-check (matches the `Lint` workflow):

```bash
uv run ruff check src
uv run ruff format src        # add --check for CI parity
uv run mypy -p matgl
```

Pre-commit (`uv run pre-commit run --all-files`) layers `ruff`, `mypy`, `codespell`, and `nbstripout`. `pre-commit-ci` skips `mypy`.

CLI entry point (installed by the package):

```bash
mgl relax   -i Li2O.cif -o Li2O_relax.cif
mgl predict -m M3GNet-MP-2018.6.1-Eform -i Li2O.cif
mgl md      -i Li2O.cif -e nvt -t 300 -n 1000
mgl clear -y
```

Docs/release helpers live in `tasks.py` (`invoke make-docs`, `invoke release <version>`). `changes.md` is the canonical changelog and is the source `release` reads from.

## Architecture

matgl uses PyTorch Geometric (PyG) exclusively as its graph backend.

### Public-vs-private convention (sklearn-style)

Modules prefixed with `_` are private. Public names are re-exported from each subpackage's `__init__.py` (e.g. `matgl.models.TensorNet`, `matgl.apps.pes.Potential`). When adding new code, import only from the exposed APIs and add new public names through the relevant `__init__.py`.

### Subpackage roles

- `matgl.models` — graph-network architectures (M3GNet, MEGNet, CHGNet, TensorNet, SO3Net, QET, GRACE) plus `TransformedTargetModel` wrapper.
- `matgl.layers` — building blocks (embeddings, graph conv, readout, ZBL, atom-ref, basis functions, activations).
- `matgl.apps.pes.Potential` — wraps a graph model into an interatomic potential that returns energies/forces/stresses via autograd. Implementation in `_pes.py`.
- `matgl.graph` — graph construction and `MGLDataset` / data loaders.
- `matgl.ext` — external-library adaptors (`pymatgen` for `Structure`/`Molecule` → graph, `ase` for `Relaxer` / `MolecularDynamics` calculators, `alchmtk` for NVIDIA Alchemi toolkit ops).
- `matgl.electrostatics` — fast charge-equilibration and Gaussian-smeared Coulomb electrostatic potential utilities used by QET; split into `_fast_qeq.py` and `_elec_pot.py`.
- `matgl.kernels` / `matgl.ops` — NVIDIA Warp GPU kernels and their pure-PyTorch reference ops; per-file ignores in `pyproject.toml` skip docstring/lint rules here.
- `matgl.utils` — `IOMixIn` (model save/load + HF Hub), training loops (`training.py`), spherical harmonics, math helpers, cutoff functions.

### Nested model pattern

End-user models often wrap a graph model: `Potential` wraps an `M3GNet`/`TensorNet` and exposes forces/stresses; `TransformedTargetModel` mirrors sklearn's `TransformedTargetRegressor` for fitting on transformed targets (e.g. log bulk modulus). Save/load goes through the outer wrapper — it dispatches to inner models automatically.

### Model serialization (IOMixIn)

Models subclass `torch.nn.Module` **and** `matgl.utils.io.IOMixIn`, and call `self.save_args(locals(), kwargs)` at the end of `__init__`. This is required so `model.save(path)` / `Model.load(path)` / `matgl.load_model(...)` can round-trip the constructor arguments via the standard triple `model.pt` + `state.pt` + `model.json`. Set a class-level `__version__: int` and bump it whenever architectural changes invalidate previously saved checkpoints.

`matgl.load_model("owner/name")` loads from Hugging Face Hub; bare names resolve under the `materialyze` org or the locally cached `pretrained_models/` directory. Use `model.push_to_hub(...)` to publish.

### Pre-trained model layout

**The in-repo `pretrained_models/` directory is deprecated.** All new pre-trained models must be
uploaded to Hugging Face Hub (under the `materialyze` org) via `model.push_to_hub(...)`, and loaded
with `matgl.load_model("materialyze/<model-name>")` or the bare-name form. Do not add new model
directories to `pretrained_models/`.

The existing layout — `pretrained_models/<MODEL_NAME>/` holding `model.pt`, `state.pt`,
`model.json`, plus a `README.md` and a notebook documenting metrics — is retained only for the
already-shipped checkpoints that `mgl predict --choices` still discovers locally.

### Tests

`tests/` mirrors `src/matgl/`. `tests/conftest.py` provides session-scoped fixtures: pymatgen `Structure`/`Molecule` (`LiFePO4`, `BaNiO3`, `MoS2`, `Mo`, `CH4`, `CO`, `Li3InCl6`, `AcAla3NHMe`) and pre-built graphs (`graph_<name>`). The `get_graph` helper in `conftest.py` builds a PyG graph for the given pymatgen structure.

## Conventions

- `from __future__ import annotations` is enforced by `ruff isort` (see `lint.isort.required-imports`) — don't remove it.
- Google-style docstrings; ruff `pydocstyle` is enabled with selected ignores. Tests, docs, examples, `kernels/`, and `ops/` have docstring requirements waived in `pyproject.toml`.
- Line length 120; ruff handles formatting (do not run black even though config exists).
- New features need tests in the matching `tests/<subpackage>/` directory.
