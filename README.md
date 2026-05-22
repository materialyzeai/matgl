[![GitHub license](https://img.shields.io/github/license/materialyzeai/matgl)](https://github.com/materialyzeai/matgl/blob/main/LICENSE)
[![Lint](https://github.com/materialyzeai/matgl/workflows/Lint/badge.svg)](https://github.com/materialyzeai/matgl/workflows/Lint/badge.svg)
[![Test](https://github.com/materialyzeai/matgl/actions/workflows/test.yml/badge.svg)](https://github.com/materialyzeai/matgl/actions/workflows/test.yml)
[![Downloads](https://static.pepy.tech/badge/matgl)](https://pepy.tech/project/matgl)
[![codecov](https://codecov.io/gh/materialyzeai/matgl/branch/main/graph/badge.svg?token=3V3O79GODQ)](https://codecov.io/gh/materialyzeai/matgl)
[![PyPI](https://img.shields.io/pypi/v/matgl?logo=pypi&logoColor=white)](https://pypi.org/project/matgl?logo=pypi&logoColor=white)

# Materials Graph Library <img src="https://github.com/materialyzeai/matgl/blob/main/assets/MatGL.png?raw=true" alt="matgl" width="30%" style="float: right">

## Official Documentation

<https://matgl.ai>

## Introduction

MatGL (Materials Graph Library) is a graph deep learning library for materials science. Mathematical graphs are a
natural representation for a collection of atoms. Graph deep learning models have been shown to consistently deliver
exceptional performance as surrogate models for the prediction of materials properties. The goal is for MatGL to serve
as an extensible platform to develop and share materials graph deep learning models.

This first version of MatGL is a collaboration between the [Materialyze.AI][materialyze] and Intel Labs.

MatGL is part of the MatML ecosystem, which includes the [MatGL] (Materials Graph Library) and [maml] (MAterials
Machine Learning) packages, the [MatPES] (Materials Potential Energy Surface) dataset, and the [MatCalc] (Materials
Calculator).

## Status

Major milestones are summarized below. Please refer to the [changelog] for details.

- v3.0.0 (May 5 2026): PyG implementations of `M3GNet` and `QET`. Corrected message-passing convention in
  `TensorNet`, `M3GNet`, and `QET`. New pre-trained weights re-released on Hugging Face (`materialyze` org), which
  is now the canonical source for all matgl models.
- v2.0.0 (Nov 13 2025): [QET] architecture added. PYG backend is now the default.
- v1.3.0 (Aug 12 2025): Pretrained molecular potentials and PyG framework added.
- v1.1.0 (May 7 2024): Implementation of [CHGNet] + pre-trained models.
- v1.0.0 (Feb 14 2024): Implementation of [TensorNet] and [SO3Net].
- v0.5.1 (Jun 9 2023): Model versioning implemented.
- v0.5.0 (Jun 8 2023): Simplified saving and loading of models. Now models can be loaded with one line of code!
- v0.4.0 (Jun 7 2023): Near feature parity with original TF implementations. Re-trained M3Gnet universal potential now
  available.
- v0.1.0 (Feb 16 2023): Initial implementations of M3GNet and MEGNet architectures have been completed. Expect
  bugs!

## Major update: v3.0.0 (May 2026)

> **Deprecation notice.** The DGL backend is **deprecated** and will be **removed in v4.0.0**. New work should
> target the default PyG backend. Models that currently have only a DGL implementation will either be ported to
> PyG before v4.0.0 or dropped from matgl at that point — track [changes.md](changes.md) for status.

A bug in the message-passing convention of `TensorNet`, `M3GNet`, and `QET` (both PyG and DGL) has been corrected:
edge messages are now aggregated onto the source (center) node so each atom correctly collects information from
its neighbors. Pre-trained weights generated under the old convention are no longer numerically valid. New weights
— including `TensorNet-PES-MatPES-PBE-2025.2` and the PyG `M3GNet` / `QET` potentials — have been retrained
against the corrected convention and uploaded to the [`materialyze`](https://huggingface.co/materialyze) Hugging
Face org, which is now the canonical (and only) source for matgl pre-trained models. The legacy GitHub
`pretrained_models/` download fallback (`RemoteFile`, `PRETRAINED_MODELS_BASE_URL`) has been removed in this
release.

## Current Architectures

<div style="float: left; padding: 10px; width: 200px">
<img src="https://github.com/materialyzeai/matgl/blob/main/assets/MxGNet.png?raw=true" alt="m3gnet_schematic">
<p>Figure: Schematic of M3GNet/MEGNet</p>
</div>

Here, we summarize the currently implemented architectures in MatGL. It should be stressed that this is by no means
an exhaustive list, and we expect new architectures to be added by the core MatGL team as well as other contributors
in the future.

- [QET], pronounced as "ket", is a charge-equilibrated TensorNet architecture. It is an
  equivariant, charge-aware architecture that attains linear scaling with system size via an analytically solvable
  charge-equilibration scheme. A pre-trained QET-MatQ FP is available, which matches state-of-the-art FPs on standard
  materials property benchmarks but delivers qualitatively different predictions in systems dominated by charge
  transfer, e.g., NaCl–\ce{CaCl2} ionic liquid, reactive processes at the Li/\ce{Li6PS5Cl} solid-electrolyte interface,
  and supports simulations under applied electrochemical potentials.
- [TensorNet] is an O(3)-equivariant message-passing neural network architecture that leverages Cartesian tensor
  representations. It is a generalization of the [SO3Net] architecture, which is a minimalist SO(3)-equivariant neural
  network. In general, TensorNet has been shown to be much more data and parameter efficient than other equivariant
  architectures. It is currently the default architecture used in the [Materials Virtual Lab].
- [Crystal Hamiltonian Graph Network (CHGNet)][chgnet] is a graph neural network based MLIP. CHGNet involves atom
  graphs to capture atom bond relations and bond graph to capture angular information. It specializes in
  capturing the atomic charges through learning and predicting DFT atomic magnetic moments.
  See [original implementation][chgnetrepo]
- [Materials 3-body Graph Network (M3GNet)][m3gnet] is an invariant graph neural network architecture that
  incorporates 3-body interactions. An additional difference is the addition of the coordinates for atoms and
  the 3×3 lattice matrix in crystals, which are necessary for obtaining tensorial quantities such as forces and
  stresses via auto-differentiation. As a framework, M3GNet has diverse applications, including **Interatomic potential development.**
  With the same training data, M3GNet performs similarly to state-of-the-art
  machine learning interatomic potentials (MLIPs). However, a key feature of a graph representation is its
  flexibility to scale to diverse chemical spaces. One of the key accomplishments of M3GNet is the development of a
  [*foundation potential*][m3gnet] that can work across the entire periodic table of the elements by training on
  relaxations performed in the [Materials Project][mp]. Like the previous MEGNet architecture, M3GNet can be used to
  develop surrogate models for property predictions, achieving in many cases accuracies that are better or similar to
  other state-of-the-art ML models.
- [MatErials Graph Network (MEGNet)][megnet] is an implementation of DeepMind's [graph networks][graphnetwork] for
  machine learning in materials science. We have demonstrated its success in achieving low prediction errors in a broad
  array of properties in both [molecules and crystals][megnet]. New releases have included our recent work on
  [multi-fidelity materials property modeling][mfimegnet]. Figure 1 shows the sequential update steps of the graph
  network, whereby bonds, atoms, and global state attributes are updated using information from each other, generating
  an output graph.

For detailed performance benchmarks, please refer to the publications in the [References](#references) section.

## Installation

MatGL can be installed via pip:

```bash
pip install matgl
```

To enable the optional [JAX-accelerated inference backend](#jax-accelerated-inference-experimental), install the `jax`
extra:

```bash
pip install matgl[jax]
```

## Docker images

Docker images have now been built for matgl, together with LAMMPS support. They are available at the
[Materials Virtual Lab Docker Repository]. If you wish to use MatGL with LAMMPS, this is probably the easiest option.

## Usage

Pre-trained M3GNet universal potential and MEGNet models for the Materials Project formation energy and
multi-fidelity band gap are now available.

### Command line (from v0.6.2)

A CLI tool now provides the capability to perform quick relaxations or predictions using pre-trained models, as well
as other simple administrative tasks (e.g., clearing the cache). Some simple examples:

1. To perform a relaxation,

    ```bash
    mgl relax --infile Li2O.cif --outfile Li2O_relax.cif
    ```

2. To use one of the pre-trained property models,

    ```bash
    mgl predict --model M3GNet-Eform-MP-2018.6.1 --infile Li2O.cif
    ```

3. To clear the cache,

    ```bash
    mgl clear
    ```

For a full range of options, use `mgl -h`.

## Obtaining Models

Pre-trained MatGL models can be loaded from (and published to) the [Hugging Face Hub]. Any repo that contains the standard matgl
serialization artifacts (`model.pt`, `state.pt`, and `model.json`) can be loaded directly using `matgl.load_model` by
passing the repo id in `"owner/name"` form. Pre-trained models released by the [Materialyze] lab can be found under the
[Materialyze Hugging Face Organization](https://huggingface.co/materialyze).

```python
import matgl

# Load directly from a Hugging Face Hub repo id.
model = matgl.load_model("materialyze/TensorNet-PES-MatPES-2025.2")

# For materialyze org, you can also just use the bare model names directly.
model = matgl.load_model("TensorNet-PES-MatPES-2025.2")
```

To publish a trained model to the Hugging Face Hub, use `push_to_hub` (requires `huggingface-cli login` or a `token`):

```python
model.push_to_hub("your-username/your-matgl-model", private=False)
```

To list available matgl models, you can use the `HfApi`:

```python
from huggingface_hub import HfApi
hf = HfApi()
print(list(hf.list_models(filter="matgl")))
```

### Model Usage

The following is an example of a prediction of the formation energy for CsCl.

```python
from pymatgen.core import Lattice, Structure
import matgl

model = matgl.load_model("MEGNet-Eform-MP-2018.6.1")

# This is the structure obtained from the Materials Project.
struct = Structure.from_spacegroup("Pm-3m", Lattice.cubic(4.1437), ["Cs", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]])
eform = model.predict_structure(struct)
print(f"The predicted formation energy for CsCl is {float(eform.numpy()):.3f} eV/atom.")
```

### JAX-accelerated inference (experimental)

The optional `matgl.ext.jax` subpackage reimplements the **inference path** (energy + forces + stress) of the PyG-backend
**TensorNet** and **QET** models in [JAX](https://docs.jax.dev). A pre-trained PyTorch potential is converted to a JAX
parameter tree and JIT-compiled by XLA into a single fused program, giving a portable (CPU / CUDA / Apple-Silicon)
**~2-3.5x speedup** over eager PyTorch for the MD / relaxation inner loop — without the NVIDIA-Warp dependency. It
requires the `jax` extra (`pip install matgl[jax]`).

`JAXPESCalculator` is a drop-in twin of `matgl.ext.ase.PESCalculator` and plugs into the usual `MolecularDynamics` /
`Relaxer` workflows:

```python
import matgl
from matgl.ext.jax import JAXPESCalculator

potential = matgl.load_model("TensorNet-PES-MatPES-r2SCAN-2025.2")
atoms.calc = JAXPESCalculator(potential, stress_unit="eV/A3")  # any ASE Atoms
```

Energies, forces and stresses match the PyTorch reference to float64 precision. The backend is inference-only and
PyG-only; training still goes through the standard PyTorch path. See `dev/jax_benchmark.py` for the benchmark harness.

## Model Training

In the PES training, the unit of energies, forces and stresses (optional) in the training, validation and test sets is extremely important to be consistent with the unit used in MatGL.

- energies: a list of energies with unit eV.
- forces: a list of nx3 force matrix with unit eV/Å, where n is the number of atom in each structure. n does not need to be the same for all structures.
- stresses: a list of 3x3 stress matrices with unit GPa (optional)

Note: For stresses, we use the convention that compressive stress gives negative values. Stresses obtained from VASP calculations (default unit is kBar) should be multiplied by -0.1 to work directly with the model.

### `MGLPotentialTrainer`

`matgl.utils.training.MGLPotentialTrainer` is a high-level wrapper around `PotentialLightningModule` and `pl.Trainer` with sensible MatPES-tuned defaults (Huber loss, stress weight 0.1, Adam + CosineAnnealingLR). Dataset construction is delegated to a sibling `MGLDatasetLoader` factory; the trainer itself only consumes pre-built `MGLDataset`s.

#### Train a TensorNet on MatPES

```python
from matgl.models import TensorNet
from matgl import MGLDatasetLoader, MGLPotentialTrainer

# 1. Download the r2SCAN MatPES dataset + per-element isolated-atom offsets
#    from materialyze/matpes on Hugging Face. One loader holds the shared HF
#    auth / cache config; both calls go through it.
loader = MGLDatasetLoader()
ds = loader.matpes_dataset(version="R2SCAN-2025.2")
refs = loader.matpes_element_refs(version="R2SCAN-2025.2", element_types=ds.element_types)

# 2. Build the model on the same element_types as the dataset.
model = TensorNet(element_types=ds.element_types, is_intensive=False, cutoff=5.0)

# 3. Configure once, fit when asked.
trainer = MGLPotentialTrainer(
    model,
    energy_weight=1.0,
    force_weight=1.0,
    stress_weight=0.1,
    lr=1e-3,
    batch_size=32,
    max_epochs=200,
    accelerator="gpu",          # "auto" / "cpu" / "gpu" / "cuda" / "mps" / "tpu"
    devices=1,
)
potential = trainer.fit(dataset=ds, atomrefs=refs, save_path="./MatPES-TensorNet")
# trainer.potential / .lit_module / .trainer / .loaders / .dataset / .atomrefs
# are populated for inspection.
```

`MGLDatasetLoader()` defaults to the `materialyze/matpes` HF dataset repo; override `repo_id` / `revision` / `token` / `cache_dir` in the constructor to point at a fork or a private mirror.

#### Logging and callbacks

`MGLPotentialTrainer` instantiates `pl.Trainer` inside `fit()` and forwards everything in `trainer_kwargs` verbatim, so any Lightning `logger` / `callbacks` setup works unchanged. The `PotentialLightningModule` logs `Total_Loss`, `Energy_MAE`, `Force_MAE`, `Stress_MAE` (plus `Magmom_MAE` / `Charge_MAE` when those heads are active) and the matching `*_RMSE` keys, each prefixed with `train_` / `val_` / `test_` — those are the names a checkpoint / early-stopping / logger sees. For per-epoch dumps of every prediction / label / error in stable sample order, matgl ships `matgl.utils.callbacks.PredictionLogger`.

#### Fine-tune a pre-trained potential

`matgl.load_model(...)` returns a `Potential` whose inner graph model is `potential.model`. Pass that inner model into `MGLPotentialTrainer` to keep the pretrained weights as the initialisation; pair it with a low learning rate, fewer epochs, and (often) zero or reduced stress weight if the fine-tuning dataset doesn't carry stresses.

```python
from matgl import load_model, MGLDatasetLoader, MGLPotentialTrainer

# 1. Load the foundation potential and extract the inner graph model.
pretrained = load_model("TensorNet-PES-MatPES-r2SCAN-2025.2")
model = pretrained.model            # the bare TensorNet — pretrained weights intact

# 2. Build / load the fine-tuning dataset. Use MGLDatasetLoader for MatPES, or
#    construct an MGLDataset yourself from your own structures + labels.
loader = MGLDatasetLoader()
ds = loader.matpes_dataset(version="R2SCAN-2025.2", element_types=model.element_types)

# 3. Reuse the MatPES atomic references so the loss starts in the right energy
#    range. Reorder them to the model's element_types.
refs = loader.matpes_element_refs(version="R2SCAN-2025.2", element_types=model.element_types)

# 4. Fine-tune with a low LR and short schedule. inference_mode is set to False
#    automatically by MGLPotentialTrainer (autograd-driven force / stress).
trainer = MGLPotentialTrainer(
    model,
    energy_weight=1.0,
    force_weight=10.0,          # bump force weight; energies are already in scale
    stress_weight=0.0,          # set to 0 if your fine-tune set has no stress
    lr=1e-4,                    # one to two orders of magnitude lower than from-scratch
    decay_steps=200,
    max_epochs=50,
    accelerator="gpu",
)
finetuned = trainer.fit(dataset=ds, atomrefs=refs, save_path="./TensorNet-finetuned")
```

The same pattern works for any pretrained `Potential` from the [`materialyze`](https://huggingface.co/materialyze) HF organisation — extract `pretrained.model`, hand it to `MGLPotentialTrainer`, and `fit`. For datasets without stress labels (e.g. cluster / dimer extxyz), set `stress_weight=0` in the trainer constructor so the stress term is dropped from the loss.

## Tutorials

We wrote [tutorials] on how to use MatGL. These were generated from [Jupyter notebooks]
[jupyternb], which can be directly run on [Google Colab].

## Resources

- [API docs][apidocs] for all classes and methods.
- [Developer Guide](developer.md) outlines the key design elements of `matgl`, especially for developers wishing to
  train and contribute matgl models.
- AdvancedSoft has implemented a [LAMMPS interface](https://github.com/advancesoftcorp/lammps/tree/based-on-lammps_2Jun2022/src/ML-M3GNET)
  to both the TF and MatGL version of M3GNet.

## References

A manuscript for MatGL has been published in npj Computational Materials. please cite the following:
> **MatGL**
>
> Ko, T. W.; Deng, B.; Nassar, M.; Barroso-Luque, L.; Liu, R.; Qi, J.; Thakur, A. C.; Mishra, A. R.; Liu, E.; Ceder, G.; Miret, S.; Ong, S. P.
> *Materials Graph Library (MatGL), an Open-Source Graph Deep Learning Library for Materials Science and Chemistry.*
> npj Comput Mater 11, 253 (2025). DOI: [https://doi.org/10.1038/s41524-025-01742-y][matgl].

If you are using any of the pretrained models, please cite the relevant works below:

> **MEGNet**
>
> Chen, C.; Ye, W.; Zuo, Y.; Zheng, C.; Ong, S. P. *Graph Networks as a Universal Machine Learning Framework for
> Molecules and Crystals.* Chem. Mater. 2019, 31 (9), 3564–3572. DOI: [10.1021/acs.chemmater.9b01294][megnet].

> **Multi-fidelity MEGNet**
>
> Chen, C.; Zuo, Y.; Ye, W.; Li, X.; Ong, S. P. *Learning Properties of Ordered and Disordered Materials from
> Multi-Fidelity Data.* Nature Computational Science, 2021, 1, 46–53. DOI: [10.1038/s43588-020-00002-x][mfimegnet].

> **M3GNet**
>
> Chen, C., Ong, S.P. *A universal graph deep learning interatomic potential for the periodic table.* Nature
> Computational Science, 2023, 2, 718–728. DOI: [10.1038/s43588-022-00349-3][m3gnet].

>**CHGNet**
>
> Deng, B., Zhong, P., Jun, K. et al. *CHGNet: as a pretrained universal neural network potential for charge-informed atomistic modelling.*
> Nat Mach Intell 5, 1031–1041 (2023). DOI:[10.1038/s42256-023-00716-3][chgnet]

>**TensorNet**
>
> Simeon, G.  De Fabritiis, G. *Tensornet: Cartesian tensor representations for efficient learning of molecular potentials.*
> Adv. Neural Info. Process. Syst. 36, (2024). DOI: [10.48550/arXiv.2306.06482][tensornet]

>**SO3Net**
>
> Schütt, K. T., Hessmann, S. S. P., Gebauer, N. W. A., Lederer, J., Gastegger, M. *SchNetPack 2.0: A neural network toolbox for atomistic machine learning.*
> J. Chem. Phys. 158, 144801 (2023). DOI: [10.1063/5.0138367][so3net]

>**QET**
>
> Ko, T. W., Liu, R., Mishra, A. R., Yu, Z., Qi, J., Ong, S. P. *A Fast, Accurate, and Reactive Equivariant Foundation Potential.*
> arXiv preprint arXiv:2511.07249 (2025). DOI: [10.48550/arXiv.2511.07249][QET]

## FAQs

1. **The `M3GNet-MP-2021.2.8-PES` differs from the original TensorFlow (TF) implementation!**

   *Answer:* `M3GNet-MP-2021.2.8-PES` is a refitted model with some data improvements and minor architectural changes.
   Porting over the weights from the TF version to DGL/PyTorch is non-trivial. We have performed reasonable benchmarking
   to ensure that the new implementation reproduces the broad error characteristics of the original TF implementation
   (see [examples][jupyternb]). However, it is not expected to reproduce the TF version exactly. This refitted model
   serves as a baseline for future model improvements. We do not believe there is value in expending the resources
   to reproduce the TF version exactly.

2. **I am getting errors with `matgl.load_model()`!**

   *Answer:* The most likely reason is that you have a cached older version of the model. We often refactor models to
   ensure the best implementation. This can usually be solved by updating your `matgl` to the latest version
   and clearing your cache using the following command `mgl clear`. On the next run, the latest model will be
   downloaded. With effect from v0.5.2, we have implemented a model versioning scheme that will detect code vs model
   version conflicts and alert the user of such problems.

3. **What pre-trained models should I be using?**

   *Answer:* There is no one definitive answer. In general, the newer the architecture and dataset, the more likely
   the model performs better. However, it should also be noted that a model operating on a more diverse dataset may
   compromise on  performance on a specific system. The best way is to look at the READMEs included with each model
   and do some tests on the systems you are interested in.

4. **How do I contribute to matgl?**

   *Answer:* For code contributions, please fork and submit pull requests. You should read the
   [developer guide](developer.md) to understand the general design guidelines. We welcome pre-trained model
   contributions as well, which should also be submitted via PRs. Please follow the folder structure of the
   pretrained models. In particular, we expect all models to come with a `README.md` and notebook
   documenting its use and its key performance metrics. Also, we expect contributions to be on new properties
   or systems or to significantly outperform the existing models. We will develop an alternative means for model
   sharing in the future.

5. **None of your models do what I need. Where can I get help?**

   *Answer:* Please contact [Prof Ong][ongemail] with a brief description of your needs. For simple problems, we are
   glad to advise and point you in the right direction. For more complicated problems, we are always open to
   academic collaborations or projects. We also offer [consulting services][mqm] for companies with unique needs,
   including but not limited to custom data generation, model development and materials design.

## Acknowledgments

This work was primarily supported by the [Materials Project][mp], funded by the U.S. Department of Energy, Office of
Science, Office of Basic Energy Sciences, Materials Sciences and Engineering Division under contract no.
DE-AC02-05-CH11231: Materials Project program KC23MP. This work used the Expanse supercomputing cluster at the Extreme
Science and Engineering Discovery Environment (XSEDE), which is supported by National Science Foundation grant number
ACI-1548562.

We also acknowledge the NVIDIA Alchemi Team, specifically Roman Zubatyuk (@zubatyuk) and Alireza Moradzadeh (@moradza),
for their contributions to warp-acceleration for TensorNet, which yielded ~2-3x speed and memory usage improvements.

[m3gnetrepo]: https://github.com/materialyzeai/m3gnet "M3GNet repo"
[megnetrepo]: https://github.com/materialyzeai/megnet "MEGNet repo"
[materialyze]: http://materialyze.ai "Materialyze.AI website"
[changelog]: https://matgl.ai/changes "Changelog"
[graphnetwork]: https://arxiv.org/abs/1806.01261 "Deepmind's paper"
[megnet]: https://pubs.acs.org/doi/10.1021/acs.chemmater.9b01294 "MEGNet paper"
[mfimegnet]: https://nature.com/articles/s43588-020-00002-x "mfi MEGNet paper"
[m3gnet]: https://nature.com/articles/s43588-022-00349-3 "M3GNet paper"
[mp]: http://materialsproject.org "Materials Project"
[apidocs]: https://matgl.ai/matgl.html "MatGL API docs"
[doc]: https://matgl.ai "MatGL Documentation"
[google colab]: https://colab.research.google.com/ "Google Colab"
[jupyternb]: https://github.com/materialyzeai/matgl/tree/main/examples
[ongemail]: mailto:shyue@nus.edu.sg "Email"
[mqm]: https://materialsqm.com "MaterialsQM"
[tutorials]: https://matgl.ai/tutorials "Tutorials"
[matgl]: https://www.nature.com/articles/s41524-025-01742-y#citeas "MatGL"
[tensornet]: https://arxiv.org/abs/2306.06482 "TensorNet"
[qet]: https://arxiv.org/abs/2511.07249 "QET"
[so3net]: https://pubs.aip.org/aip/jcp/article-abstract/158/14/144801/2877924/SchNetPack-2-0-A-neural-network-toolbox-for "SO3Net"
[chgnet]: https://www.nature.com/articles/s42256-023-00716-3 "CHGNet"
[chgnetrepo]: https://github.com/CederGroupHub/chgnet "CHGNet repo"
[maml]: https://materialyzeai.github.io/maml/
[MatGL]: https://matgl.ai
[MatPES]: https://matpes.ai
[MatCalc]: https://matcalc.ai
[Materials Virtual Lab Docker Repository]: https://hub.docker.com/orgs/materialsvirtuallab/repositories
[Hugging Face Hub]: https://huggingface.co
