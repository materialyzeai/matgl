---
layout: page
title: Change Log
nav_order: 3
---

# Change Log

## Unreleased
- **Fix: `SoftExponential` activation autograd correctness and NaN safety (#788).** `forward` now selects
  its `alpha < 0` / `alpha > 0` / `alpha ≈ 0` branches with `torch.where` instead of a Python `if` on the
  learnable `alpha` parameter. The old `if self.alpha < 0.0` forced a host-device sync and dropped the
  branch from the autograd graph (so `alpha` was effectively trapped in its initial sign region); the
  `alpha == 0.0` exact-float test was unreachable after the first optimizer step; and the `alpha < 0`
  formula produced NaN/Inf for sufficiently negative inputs. The log argument and denominators are now
  guarded so both the activation and `alpha.grad` stay finite. Values in the well-defined region are
  unchanged.
- **New: `matgl.utils.MCDropoutWrapper` for uncertainty-aware inference.** Enables Monte Carlo Dropout
  (Gal & Ghahramani, 2016) on any pretrained MatGL model (CHGNet, M3GNet, TensorNet, …) without
  retraining: the backbone stays in `eval()` while only the readout dropout is sampled, and
  `predict_uncertainty(structures, n_passes)` returns per-structure `(mean, std)` for acquisition
  functions such as UCB (`mean - lambda * std`). See issue #800.
- **Perf: backbone-once fast path for `predict_uncertainty` (`cache_backbone=True`, default).** Since
  MC Dropout only perturbs the readout, the deterministic backbone is evaluated once and only the
  cheap stochastic head is replayed `n_passes` times (vectorised), giving an ~`n_passes`× speed-up
  (~19× at `n_passes=20` on M3GNet, GPU). Numerically equivalent to the naive loop; engaged only when
  a probe proves the head is the model's terminal op, otherwise falls back automatically (e.g. CHGNet,
  which pools after dropout).

## 4.0.2
- **Bug fix: charges restored for charge-predicting potentials (QET) in the ASE calculators.**
  `PESCalculator.get_charges()` raised `PropertyNotImplementedError` because the PyG calculator never
  implemented the charge output of the removed DGL calculator, and `total_charge` was silently ignored:
  charged cells ran as neutral (QEq treats a missing total charge as 0) with no error. `PESCalculator`
  now exposes `"charges"` in results and accepts `total_charge` / `ext_pot` kwargs, falling back to
  `atoms.get_initial_charges()`. `Relaxer` and `MolecularDynamics` inherit this. `JAXPESCalculator`
  gains the same support; `make_potential_fn(with_charges=True)` (QET only) returns
  `(E, forces, stress, charges)` with `total_charge` as a traced argument, so varying it triggers no
  recompilation. **If you ran charged-cell MD or relaxations with QET on 4.0.0/4.0.1, those runs used
  a neutral cell.**
- **Bug fix: `state_attr` is cast to long before the `nn.Embedding` lookup in `EmbeddingBlock`**,
  fixing failures with float state attributes.
- Internal: the electrostatics modules (`LinearQeq`, electrostatic potential) now live in
  `matgl.layers`; no public API change. Backend-related redirects removed. Notebooks updated to
  remove DGL references.

## 4.0.1
- **Intermediate features now exposed via ``model.feature_dict``.** All ``MatGLModel`` subclasses
  (``TensorNet``, ``M3GNet``, ``CHGNet``, ``QET``, ``MEGNet``, ``SO3Net``, ``GRACE``, and the
  ``TransformedTargetModel`` wrapper) populate a ``feature_dict`` instance attribute on every
  ``forward`` call. The legacy ``return_all_layer_output`` (``forward``) and ``return_features``
  (in ``predict_structure`` for TensorNet / CHGNet / M3GNet, and in ``QET.__init__``) kwargs are
  **deprecated** and emit a ``DeprecationWarning`` when used. They will be **removed in matgl v5**.

## 4.0.0
- **DGL backend removed.** All DGL implementations have been deleted. matgl now targets PyTorch Geometric
  exclusively. The `MATGL_BACKEND` env var and the `ensure_backend()` helper have been removed.
  `matgl.set_backend()` is retained as a no-op stub for backwards compatibility and emits a
  `DeprecationWarning` when `"DGL"` is requested. Private `_*_pyg.py` modules have been renamed to drop
  the `_pyg` suffix. If you wish to use DGL, please install `matgl<=4`.
- **Optional JAX inference backend (`matgl.ext.jax`, experimental).** New subpackage that reimplements the inference path (energy + forces + stress) of the `TensorNet` and `QET` models in JAX. A pre-trained PyTorch potential is converted to a JAX parameter tree and JIT-compiled by XLA into one fused program, giving a portable (CPU / CUDA / Apple-Silicon) ~2-3.5x speedup over eager PyTorch for the MD / relaxation inner loop, with no NVIDIA-Warp dependency. `JAXPESCalculator` is a drop-in twin of `matgl.ext.ase.PESCalculator`. Outputs match the PyTorch reference to float64 precision. Requires the optional `jax` extra (`pip install matgl[jax]`); inference-only.
- **Bug fix: `QET` no longer crashes on single-atom structures.** Bare `torch.squeeze()` calls in `QET.forward` collapsed the size-1 node dimension of a one-atom input to a 0-d scalar, raising `IndexError` inside `ElectrostaticPotential`. These now use `.reshape(-1)`, which drops only trailing feature dimensions and never the node dimension.
- **Bug fix: `TensorNet` / `QET` no longer crash with the non-smooth `SphericalBessel` radial basis.** With `rbf_type="SphericalBessel"` and `use_smooth=False`, the basis emits `max_l * max_n` features, but the embedding / interaction input `Linear` layers were sized for only `max_n`, raising a shape-mismatch `RuntimeError` in the forward pass. The radial-basis width fed to those layers now correctly accounts for `max_l` in the non-smooth case (the smooth `SphericalBessel` and `Gaussian` bases are unaffected).

## 3.0.4
- **PyG `SO3Net`.** New `matgl.models._so3net_pyg.SO3Net` is the PyG counterpart of the existing DGL
  `SO3Net` and is now the implementation selected on the default PyG backend. The full public surface is
  preserved (`target_property` in `{atomwise, dipole_moment, polarizability, graph}`, `readout_type` in
  `{set2set, weighted_atom, reduce_atom}`, `correct_charges`, `predict_dipole_magnitude`,
  `use_vector_representation`, `return_vector_representation`). Forward now takes a PyG `Data`/`Batch` and
  aggregates per-graph via `scatter_add` / `bincount` instead of `dgl.readout_nodes` / `batch_num_nodes`.
- **DGL backend deprecated.** The DGL backend (`MATGL_BACKEND=DGL`) is deprecated and will be **removed in
  v4.0.0**. PyG is now the only supported backend for new work. `ensure_backend("DGL")` (called at import
  time when `MATGL_BACKEND=DGL`, and from `matgl.set_backend("DGL")`) now emits a `DeprecationWarning`.

## 3.0.3
- **GRACE (PyG) interatomic potential.** New `matgl.models.GRACE` (beta) joins TensorNet / M3GNet / MEGNet / QET / CHGNet / SO3Net on the PyG backend. (#779)
- **`matgl.utils.training.MGLPotentialTrainer` + `MGLDatasetLoader` (new, PyG-only).** First dataset-level Hugging Face integration in matgl: a configure-once / fit-when-asked trainer paired with a small dataset-factory class that hoists HF auth / cache config to one place. `__init__` stores hyperparameters; nothing heavy runs until `fit(dataset=...)`. (#782)
  - **`MGLDatasetLoader`** (defaults to HF `materialyze/matpes`): `loader = MGLDatasetLoader()` then `loader.matpes_dataset(version="r2SCAN-2025.2")` and `loader.matpes_element_refs(version="r2SCAN-2025.2", element_types=...)`. Override `repo_id` / `revision` / `token` / `cache_dir` in the constructor to point at a fork or a private mirror. Element references are reorderable to the caller's `element_types`. Stresses in the on-disk MatPES JSON (kbar, VASP compressive-positive) are converted to matgl's GPa compressive-negative convention automatically; pass `stress_unit="GPa"` to skip the conversion. For datasets (MatPES forks, custom DFT runs) already on disk, `loader.from_json("/path/to/file.json", ...)` skips the HF round trip entirely. The JSON file must use the same per-record schema as the MatPES dataset — `structure` (pymatgen-serialisable) + `energy` / `forces` / `stress` PES keys; any extra metadata fields are ignored. `stress_unit` defaults to `"kbar"` (MatPES on-disk convention) and is applied consistently with the HF path. The loader returns a raw `MGLDataset`; splitting + `MGLDataLoader` wrapping is the trainer's job (see `MGLPotentialTrainer`'s `frac_list` / `shuffle` / `random_state` `loader_kwargs`).
  - **`MGLPotentialTrainer`**: `MGLPotentialTrainer(model, accelerator="auto", max_epochs=100, ...)` accepts the full Lightning placement vocabulary (`"auto"` / `"cpu"` / `"gpu"` / `"cuda"` / `"mps"` / `"tpu"`). `trainer.fit(dataset, *, atomrefs=None, save_path=None)` is a small focused entry point:
    - `dataset` is a pre-built `MGLDataset` (random split inside `_build_dataloaders` via `frac_list` / `shuffle` / `random_state`) or a `{"train", "valid", "test"}` mapping of pre-built splits. Use `MGLDatasetLoader` above to build one.
    - `atomrefs` accepts `np.ndarray` / `AtomRef` instance / `None`. Use `MGLDatasetLoader().matpes_element_refs(...)` to download or `fit_element_refs(...)` to fit locally.
    - Loss-term toggling follows the constructor weights: set `stress_weight=0` for datasets without stress labels (cluster / dimer extxyz), and `magmom_weight` / `charge_weight` `> 0` only when the dataset carries those labels.
    - After fit, `trainer.potential` / `trainer.lit_module` / `trainer.trainer` / `trainer.loaders` / `trainer.dataset` / `trainer.atomrefs` are populated. Defaults: Huber loss with stress weight 0.1, batch size 32, lr 1e-3, 100 epochs, CosineAnnealingLR (`decay_steps=1000`, `decay_alpha=0.01`).
- **`MGLDataLoader` collate auto-detect (PyG).** When `collate_fn` is omitted, the loader now picks one from the training dataset's label keys (`collate_fn_graph` for property prediction, `collate_fn_pes` with stress / magmom / charge flags toggled to match labels), mirroring the DGL path. `Subset` (post-`split_dataset`) is peeled to reach the underlying `MGLDataset.labels`. Explicit `collate_fn=` always wins. (#782)
- **`fit_element_refs` training helper.** Convenience function that fits per-element energy offsets from pymatgen `Structure`s + energies via `np.linalg.lstsq`, returning an array that drops directly into `PotentialLightningModule(element_refs=...)` or `Potential(element_refs=...)`. (#780)
- **Performance speedups (no checkpoint or public-API changes).**
  - Cache spherical-Bessel basis constants (zeros, normalization factors) in `__init__` instead of recomputing them in every `forward`. (#787)
  - Lower-overhead `Potential` and `AtomRef` forward paths on PyG: hoist `.to(device)` / shape work out of the hot path, avoid redundant tensor allocations. (#783)
  - Port the same `Potential` / `AtomRef` speedups to DGL. (#786)
  - Opt-in `torch.compile` flag on `Potential` (PyG) for further inference speedups. (#784)
  - Low-risk speedups across `Structure2Graph` / `Molecule2Graph` converters, the training loop, and the ASE `PESCalculator`. (#781)
- **Bug fix (PyG): `Potential.forward` no longer mutates the input graph.** Previously the autograd `pos.requires_grad_(True)` / `cell.requires_grad_(True)` toggles were applied in place on the caller's `Data` object, which leaked grad-tracking state across reuses. The forward now operates on a shallow clone of the relevant tensors. (#785)
- Fleshed out module-level docstrings for `matgl.apps` and `matgl.layers`, and the `Potential` wrapper docstring (energy / force / stress / charge contract, stress unit, magmom / charge head gating).

## 3.0.2
- New `matgl.utils.callbacks.PredictionLogger` Lightning callback for capturing per-epoch energy and per-atom force   predictions, ground truth, and errors during `PotentialLightningModule` training. Pairs with   `add_sample_indices(dataset)` to keep `(n_epochs, n_samples)` log columns in a stable per-sample order across
  shuffled training epochs. The callback persists the cumulative log to disk every epoch end so it survives a
  walltime cut. `PredictionLogger` also logs stress and per-atom charge whenever the wrapped potential computes them
  (`model.calc_stresses` / `model.calc_charge`); pass `log_stress=False` or `log_charge=False` to opt out. New keys in the saved payload: `{train,val}_stress_{preds,labels,errors}` shape `(n_epochs, n_samples, 3, 3)` and
  `{train,val}_charge_{preds,labels,errors}` shape `(n_epochs, n_atoms)`. (#777)

## 3.0.1
- PyG charge-training parity for `QET`. `PotentialLightningModule` now accepts `charge_weight` and adds a per-atom
  charge loss term (`Charge_MAE` / `Charge_RMSE`) on top of energy/force/stress; `Potential.forward` (PyG) takes
  `total_charge` / `ext_pot` and returns equilibrated charges in the output tuple, mirroring the DGL pipeline.
- `MGLDataset` (PyG) gains `include_ref_charge` to attach per-atom `q_ref` onto each `Data` object (consumed by
  `LinearQeq`); `collate_fn_pes` (PyG) gains `include_charge` so per-atom charge labels propagate through batches.
- Renamed PyG layer classes for cross-backend consistency: `AtomRefPyG` -> `AtomRef`, `NuclearRepulsionPyG` ->
  `NuclearRepulsion` (matching the DGL counterparts).
- CI: bumped GitHub Actions to Node 24 versions (`actions/checkout@v5`, `actions/setup-python@v6`,
  `actions/upload-artifact@v5`, `actions/download-artifact@v5`, `astral-sh/setup-uv@v7`).

## 3.0.0
- **PyG `M3GNet` and `QET`.** New PyG implementations of `M3GNet` and `QET` join the existing PyG `TensorNet` and
  `MEGNet`, so all four core architectures now run on the default PyG backend without DGL.
- **Message-passing fix (TensorNet, M3GNet, QET).** Corrected the message-passing convention in the interaction
  and embedding blocks of `TensorNet`, `M3GNet`, and `QET` (both PyG and DGL): edge messages are now aggregated
  onto the source (center) node so each atom correctly collects information from its neighbors. Pre-trained
  weights generated under the old convention are no longer numerically valid. (#758, @kenko911)
- **New pre-trained weights on Hugging Face.** `TensorNet-PES-MatPES-PBE-2025.2`, the `M3GNet` and `QET` PyG
  potentials, and related models have been retrained against the corrected message-passing convention and
  re-released on the [`materialyze`](https://huggingface.co/materialyze) HF org, which is now the canonical
  source for all matgl pre-trained models.
- **Breaking — removed legacy GitHub `pretrained_models/` download fallback.** The `RemoteFile` class and the
  `PRETRAINED_MODELS_BASE_URL` config constant have been removed, and `get_available_pretrained_models` no longer
  accepts the `include_hf` / `include_github` arguments (it now always queries the `materialyze` HF org).
- Consolidated per-backend `tensornet` / `m3gnet` / `megnet` / `qet` test files; backend dispatch is via
  `matgl.config.BACKEND` with `pytest.skip` guarding backend-specific cases.

## 2.2.1
- Updated HuggingFace Repo Id to lowercase "materialyze".

## 2.2.0
- Fixed an incorrect message-passing convention in the PyG and DGL `TensorNet` interaction and embedding blocks.
  Edge messages are now aggregated onto the source (center) node so that each atom correctly collects information
  from its neighbors. Pre-trained PyG `TensorNet` weights have been re-released to match the corrected convention.
  (#758, @kenko911)
- Refreshed the PyG `TensorNet` README, added a missing `TrajectoryObserver`, improved `PESCalculator` stress-unit
  handling and logging, and tightened the related unit tests. (#758, @kenko911)

## 2.1.2
- Added Hugging Face Hub support for loading pre-trained models, with automatic fallback checking and respect for
  the `MATGL_CACHE` environment variable.
- Removed the deprecated `hubconf.py` (superseded by Hugging Face support).
- Added `TensorNetWrapper` integrating the NVIDIA `nvalchemi-toolkit` for fully GPU-resident MD/Relax workflows,
  including an example script for NVT MD. (#754)

## 2.1.1
- Merged `TensorNet` (PyG) and `TensorNetWarp` into a single `TensorNet` class with optional warp acceleration
  (`use_warp` parameter; auto-detected when `nvalchemi-toolkit-ops` is installed).
- Moved warp-accelerated `TensorEmbedding` and `TensorNetInteraction` layers to `matgl.layers._embedding_warp`
  and `matgl.layers._graph_convolution_warp`.
- Made `nvalchemiops` an optional dependency throughout: `_pymatgen_pyg`, `_ase_pyg`, and warp layer imports
  all fall back gracefully to pymatgen-based neighbor list construction when the package is absent.

## 2.1.0
- Bug fix for accidental change of default backend.
- Training module updated for QET support.

## 2.0.9
- Bug fix for missing Atoms2Graph export.

## 2.0.7
- Refactored PyG TensorNet embedding and interaction blocks to pure PyTorch for improved compatibility. (@kenko911)
- Improved handling of stress units in `PESCalculator`. (@kenko911)
- Enabled returning intermediate crystal features from CHGNet and TensorNet models. (@bowen-bd)
- Added GPU-accelerated neighbor list construction and improved CUDA neighbor list performance and retry logic.
  (@zubatyuk)
- Integrated NVIDIA TensorNet Warp CUDA kernels into the main branch. (@atulcthakur, @zubatyuk)
- Improved QET training support via updates to `Atoms2Graph`, `collate_fn_pes`, and `MGLDataset` (including
  `include_ref_charge`). (@kenko911)
- Documentation updates for QET, including references and DOI links. (@kenko911)

## 2.0.6
- Bug fix for CHGnet loading.

## 2.0.5
- Improved error messages for backend/model mismatch. Try to transparently handle simple situations.

## 2.0.4
- Bug fix for matgl.graph.data and matgl.graph.converter imports for different backends.

## 2.0.3
- Bug fix for matgl.ext.pymatgen import for different backends.

## 2.0.2
- QET (Charge-Equilibrated TensorNet) architecture and pre-trained weights are added!
- Begun a migration to Pytorch-Geometric over the now-deprecated DGL. So far, only vanilla TensorNet has been
  implemented in PYG). DGL models still work but require a manual setup (change of backend and installation of DGL).

## 1.2.7
- Use original custom RemoteFile rather than fsspec, which is very finicky with SSL connections.
- _create_directed_line_graph error handling (@bowen-bd)
- Update Import Alias for lightning (@jcwang587)
- Add nvt_nose_hoover to MD ensemble (@bowen-bd)
- Allow training of magmom when no line graph presents (@bowen-bd)
- Allow disable BondGraph in CHGNet (@bowen-bd)

## 1.2.6
- Fix missing torchdata dependency for Linux.

## 1.2.5
- Dependency pinning now is platform specific. Linux based systems can now work with latest DGL and torch.

## 1.2.3
- Fix dependency issues with DGL. Pinning to DGL<=2.1.0 for now, which have versions for all OSes.

## 1.2.1
- Bug fix for pbc dtype on Windows systems.

## 1.2.0
- Release of MatPES-based models.
- Pin DGL and PyTorch dependencies to 2.2.0 to ensure compatibility with Mac.

## 1.1.3
- Improve the memory efficiency and speed of three-body interactions. (@kenko911)
- FrechetCellFilter is added for variable cell relaxation in Relaxer class. (@kenko911)
- Smooth l1 loss function is added for training. (@kenko911)

## 1.1.2
- Move AtomRef Fitting to numpy to avoid bug (@BowenD-UCB)
- NVE ensemble added (@kenko911)
- Migrate from pytorch_lightning to lightning.

## 1.1.1
- Pin dependencies to support latest DGL 2.x. @kenko911

## 1.1.0
- Implementation of CHGnet + pre-trained models. (@BowenD-UCB)

## 1.0.0
- First 1.0.0 release to reflect the maturity of the matgl code! All changes below are the efforts of @kenko911.
- Equivariant TensorNet and SO3Net are now implemented in MatGL.
- Refactoring of M3GNetCalculator and M3GNetDataset into generic PESCalculator and MGLDataset for use with all models
  instead of just M3GNet.
- Training framework has been unified for all models.
- ZBL repulsive potentials has been implemented.


## 0.9.2
* Added Tensor Placement Calls For Ease of Training with PyTorch Lightning (@melo-gonzo).
* Allow extraction of intermediate outputs in "embedding", "gc_1", "gc_2", "gc_3", and "readout" layers for use as
  atom, bond, and structure features. (@JiQi535)

## 0.9.1
* Update Potential version numbers.

## 0.9.0

* set pbc_offsift and pos as float64 by @lbluque in https://github.com/materialsvirtuallab/matgl/pull/153
* Bump pytorch-lightning from 2.0.7 to 2.0.8 by @dependabot in https://github.com/materialsvirtuallab/matgl/pull/155
* add cpu() to avoid crash when using ase with GPU by @kenko911 in https://github.com/materialsvirtuallab/matgl/pull/156
* Added the united test for hessian in test_ase.py to improve the coverage score by @kenko911 in https://github.com/materialsvirtuallab/matgl/pull/157
* AtomRef Updates by @lbluque in https://github.com/materialsvirtuallab/matgl/pull/158
* Bump pymatgen from 2023.8.10 to 2023.9.2 by @dependabot in https://github.com/materialsvirtuallab/matgl/pull/160
* Remove torch.unique for finding the maximum three body index and little cleanup in united tests by @kenko911 in https://github.com/materialsvirtuallab/matgl/pull/161
* Bump pymatgen from 2023.9.2 to 2023.9.10 by @dependabot in https://github.com/materialsvirtuallab/matgl/pull/162
* Add united test for trainer.test and description in the example by @kenko911 in https://github.com/materialsvirtuallab/matgl/pull/165
* Bump pytorch-lightning from 2.0.8 to 2.0.9 by @dependabot in https://github.com/materialsvirtuallab/matgl/pull/167
* Sequence instead of list for inputs by @lbluque in https://github.com/materialsvirtuallab/matgl/pull/169
* Avoiding crashes for PES training without stresses and update pretrained models by @kenko911 in https://github.com/materialsvirtuallab/matgl/pull/168
* Bump pymatgen from 2023.9.10 to 2023.9.25 by @dependabot in https://github.com/materialsvirtuallab/matgl/pull/173
* Allow to choose distribution in xavier_init by @lbluque in https://github.com/materialsvirtuallab/matgl/pull/174
* An example for the simple training of M3GNet formation energy model is added by @kenko911 in https://github.com/materialsvirtuallab/matgl/pull/176
* Directed line graph by @lbluque in https://github.com/materialsvirtuallab/matgl/pull/178
* Bump pymatgen from 2023.9.25 to 2023.10.4 by @dependabot in https://github.com/materialsvirtuallab/matgl/pull/180
* Bump torch from 2.0.1 to 2.1.0 by @dependabot in https://github.com/materialsvirtuallab/matgl/pull/181
* Bump pymatgen from 2023.10.4 to 2023.10.11 by @dependabot in https://github.com/materialsvirtuallab/matgl/pull/183
* add testing to m3gnet potential training example by @lbluque in https://github.com/materialsvirtuallab/matgl/pull/179
* Update Training a MEGNet Formation Energy Model with PyTorch Lightning by @1152041831 in https://github.com/materialsvirtuallab/matgl/pull/185
* Bump pymatgen from 2023.10.11 to 2023.11.12 by @dependabot in https://github.com/materialsvirtuallab/matgl/pull/187
* dEdLat contribution for stress calculations is added and Universal Potentials are updated by @kenko911 in https://github.com/materialsvirtuallab/matgl/pull/189
* Bump torch from 2.1.0 to 2.1.1 by @dependabot in https://github.com/materialsvirtuallab/matgl/pull/190

## New Contributors

* @1152041831 made their first contribution in https://github.com/materialsvirtuallab/matgl/pull/185

**Full Changelog**: https://github.com/materialsvirtuallab/matgl/compare/v0.8.5...v0.8.6

## 0.8.3

* Extend the functionality of ASE-interface for molecular systems and include more different ensembles. (@kenko911)
* Improve the dgl graph construction and fix the if statements for stress and atomwise training. (@kenko911)
* Refactored MEGNetDataset and M3GNetDataset classes with optimizations.

## 0.8.5

* Bug fix for np.meshgrid. (@kenko911)

## 0.8.2

* Add site-wise predictions for Potential. (@lbluque)
* Enable CLI tool to be used for multi-fidelity models. (@kenko911)
* Minor fix for model version for DIRECT model.

## 0.8.1

* Fixed bug with loading of models trained with GPUs.
* Updated default model for relaxations to be the `M3GNet-MP-2021.2.8-DIRECT-PES model`.

## 0.8.0

* Fix a bug with use of set2set in M3Gnet implementation that affected intensive models such as the formation energy
  model. M3GNet model version is updated to 2 to invalidate previous models. Note that PES models are unaffected.
  (@kenko911)

## 0.7.1

* Minor optimizations for memory and isolated atom training (@kenko911)

## 0.7.0

* MatGL now supports structures with isolated atoms. (@JiQi535)
* Fourier expansion layer and generalize cutoff polynomial. (@lbluque)
* Radial bessel (zeroth order bessel). (@lbluque)

## 0.6.2

* Simple CLI tool `mgl` added.

## 0.6.1

* Bug fix for training loss_fn.

## 0.6.0

* Refactoring of training utilities. Added example for training an M3GNet potential.

## 0.5.6

* Minor internal refactoring of basis expansions into `_basis.py`. (@lbluque)

## 0.5.5

* Critical bug fix for code regression affecting pre-loaded models.

## 0.5.4

* M3GNet Formation energy model added, with example notebook.
* M3GNet.predict_structure method added.
* Massively improved documentation at http://matgl.ai.

## 0.5.3

* Minor doc and code usability improvements.

## 0.5.2

* Minor improvements to model versioning scheme.
* Added `matgl.get_available_pretrained_models()` to help with model discovery.
* Misc doc and error message improvements.

## 0.5.1

* Model versioning scheme implemented.
* Added convenience method to clear cache.

## 0.5.0

* Model serialization has been completely rewritten to make it easier to use models out of the box.
* Convenience method `matgl.load_model` is now the default way to load models.
* Added a TransformedTargetModel.
* Enable serialization of Potential.
* IMPORTANT: Pre-trained models have been reserialized. These models can only be used with v0.5.0+!

## 0.4.0

* Pre-trained M3GNet universal potential
* Pytorch lightning training utility.

## v0.3.0

* Major refactoring of MEGNet and M3GNet models and organization of internal implementations. Only key API are exposed
  via matgl.models or matgl.layers to hide internal implementations (which may change).
* Pre-trained models ported over to new implementation.
* Model download now implemented.

## v0.2.1

* Fixes for pre-trained model download.
* Speed up M3GNet 3-body computations.

## v0.2.0

* Pre-trained MEGNet models for formation energies and band gaps are now available.
* MEGNet model implemented with `predict_structure` convenience method.
* Example notebook demonstrating pre-trained model usage is available.

## v0.1.0

* Initial working version with m3gnet and megnet.
