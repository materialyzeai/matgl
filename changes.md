---
layout: page
title: Change Log
nav_order: 3
---

# Change Log

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
