---
layout: default
title: matgl.utils.md
nav_exclude: true
---

# matgl.utils package

Implementation of various utility methods and classes.

## matgl.utils.cutoff module

Cutoff functions for constructing M3GNet potentials.

### matgl.utils.cutoff.cosine_cutoff(r: Tensor, cutoff: float)

Cosine cutoff function
:param r: radius distance tensor
:type r: torch.Tensor
:param cutoff: cutoff distance.
:type cutoff: float

Returns: cosine cutoff functions

### matgl.utils.cutoff.polynomial_cutoff(r: Tensor, cutoff: float, exponent: int = 3)

Envelope polynomial function that ensures a smooth cutoff.

Ensures first and second derivative vanish at cuttoff. As described in:
: [https://arxiv.org/abs/2003.03123](https://arxiv.org/abs/2003.03123)

* **Parameters:**
  * **r** (*torch.Tensor*) – radius distance tensor
  * **cutoff** (*float*) – cutoff distance.
  * **exponent** (*int*) – minimum exponent of the polynomial. Default is 3.
    The polynomial includes terms of order exponent, exponent + 1, exponent + 2.

Returns: polynomial cutoff function

## matgl.utils.io module

Provides utilities for managing models and data.

### *class* matgl.utils.io.IOMixIn

Bases: `object`

Mixin class for model saving and loading.

For proper usage, models should subclass nn.Module and IOMix and the save_args method should be called
immediately after the super().      *\_init*\_() call:

```default
super().__init__()
self.save_args(locals(), kwargs)
```

#### *classmethod* load(path: str | Path | dict, \*\*kwargs)

Load the model weights from a directory.

* **Parameters:**
  * **path** (*str*\*|**path**|\**dict*) –

Path to saved model or name of pre-trained model. If it is a dict, it is assumed to
be of the form:

```default
{
    "model.pt": path to model.pt file,
    "state.pt": path to state file,
    "model.json": path to model.json file
}
```

Otherwise, the search order is path, followed by download from PRETRAINED_MODELS_BASE_URL
(with caching).

```none
* **\*\*kwargs** – Additional kwargs passed to RemoteFile class. E.g., a useful one might be force_download if you
```

want to update the model.

Returns: model_object.

#### save(path: str | Path = ‘.’, metadata: dict | None = None, makedirs: bool = True)

Save model to a directory.

Three files will be saved.

* path/model.pt, which contains the torch serialized model args.
* path/state.pt, which contains the saved state_dict from the model.
* path/model.json, a txt version of model.pt that is purely meant for ease of reference.
* **Parameters:**
  * **path** – String or Path object to directory for model saving. Defaults to current working directory (“.”).
  * **metadata** – Any additional metadata to be saved into the model.json file. For example, a good use would be
    a description of model purpose, the training set used, etc.
  * **makedirs** – Whether to create the directory using os.makedirs(exist_ok=True). Note that if the directory
    already exists, makedirs will not do anything.

#### save_args(locals: dict, kwargs: dict | None = None)

Method to save args into a private \_init_args variable.

This should be called after super in the **init** method, e.g., self.save_args(locals(), kwargs).

* **Parameters:**
  * **locals** – The result of locals().
  * **kwargs** – kwargs passed to the class.

### *class* matgl.utils.io.RemoteFile(uri: str, cache_location: str | Path = PosixPath(‘/Users/shyue/.cache/matgl’), force_download: bool = False)

Bases: `object`

Handling of download of remote files to a local cache.

* **Parameters:**
  * **uri** – Uniform resource identifier.
  * **cache_location** – Directory to cache downloaded RemoteFile. By default, downloaded models are saved at
  * **$HOME/.matgl.** –
  * **force_download** – To speed up access, a model with the same name in the cache location will be used if
  * **re-download** (*present. If you want to force a*) –
  * **True.** (*set this to*) –

### matgl.utils.io.get_available_pretrained_models()

Checks Github for available pretrained_models for download. These can be used with load_model.

* **Returns:**
  List of available models.

### matgl.utils.io.load_model(path: Path, \*\*kwargs)

Convenience method to load a model from a directory or name.

* **Parameters:**
  * **path** (*str*\*|\**path*) – Path to saved model or name of pre-trained model. The search order is path, followed by
    download from PRETRAINED_MODELS_BASE_URL (with caching).
  * **\*\*kwargs** – Additional kwargs passed to RemoteFile class. E.g., a useful one might be force_download if you
    want to update the model.
* **Returns:**
  model_object if include_json is false. (model_object, dict) if include_json is True.
* **Return type:**
  Returns

## matgl.utils.maths module

Implementations of math functions.

### matgl.utils.maths.broadcast(input_tensor: Tensor, target_tensor: Tensor, dim: int)

Broadcast input tensor along a given dimension to match the shape of the target tensor.
Modified from torch_scatter library ([https://github.com/rusty1s/pytorch_scatter](https://github.com/rusty1s/pytorch_scatter)).

* **Parameters:**
  * **input_tensor** – The tensor to broadcast.
  * **target_tensor** – The tensor whose shape to match.
  * **dim** – The dimension along which to broadcast.
* **Returns:**
  resulting input tensor after broadcasting

### matgl.utils.maths.broadcast_states_to_atoms(g, state_feat)

Broadcast state attributes of shape [Ns, Nstate] to
bond attributes shape [Nb, Nstate].

* **Parameters:**
  * **g** – DGL graph
  * **state_feat** – state_feature

Returns: broadcasted state attributes

### matgl.utils.maths.broadcast_states_to_bonds(g, state_feat)

Broadcast state attributes of shape [Ns, Nstate] to
bond attributes shape [Nb, Nstate].

* **Parameters:**
  * **g** – DGL graph
  * **state_feat** – state_feature

Returns: broadcasted state attributes

### matgl.utils.maths.get_range_indices_from_n(ns)

Give ns = [2, 3], return [0, 1, 0, 1, 2].

* **Parameters:**
  **ns** – torch.Tensor, the number of atoms/bonds array

Returns: range indices

### matgl.utils.maths.get_segment_indices_from_n(ns)

Get segment indices from number array. For example if
ns = [2, 3], then the function will return [0, 0, 1, 1, 1].

* **Parameters:**
  **ns** – torch.Tensor, the number of atoms/bonds array
* **Return type:**
  object

Returns: segment indices tensor

### matgl.utils.maths.repeat_with_n(ns, n)

Repeat the first dimension according to n array.

* **Parameters:**
  * **ns** (*torch.tensor*) – tensor
  * **n** (*torch.tensor*) – a list of replications

Returns: repeated tensor

### matgl.utils.maths.scatter_sum(input_tensor: Tensor, segment_ids: Tensor, num_segments: int, dim: int)

Scatter sum operation along the specified dimension. Modified from the
torch_scatter library ([https://github.com/rusty1s/pytorch_scatter](https://github.com/rusty1s/pytorch_scatter)).

* **Parameters:**
  * **input_tensor** (*torch.Tensor*) – The input tensor to be scattered.
  * **segment_ids** (*torch.Tensor*) – Segment ID for each element in the input tensor.
  * **num_segments** (*int*) – The number of segments.
  * **dim** (*int*) – The dimension along which the scatter sum operation is performed (default: -1).
* **Returns:**
  resulting tensor

### matgl.utils.maths.spherical_bessel_roots(max_l: int, max_n: int)

Calculate the spherical Bessel roots. The n-th root of the l-th
spherical bessel function is the [l, n] entry of the return matrix.
The calculation is based on the fact that the n-root for l-th
spherical Bessel function j_l, i.e., z_{j, n} is in the range
[z_{j-1,n}, z_{j-1, n+1}]. On the other hand we know precisely the
roots for j0, i.e., sinc(x).

* **Parameters:**
  * **max_l** – max order of spherical bessel function
  * **max_n** – max number of roots

Returns: root matrix of size [max_l, max_n]

### matgl.utils.maths.unsorted_segment_fraction(data: Tensor, segment_ids: Tensor, num_segments: int)

Segment fraction
:param data: original data
:type data: torch.tensor
:param segment_ids: segment ids
:type segment_ids: torch.tensor
:param num_segments: number of segments
:type num_segments: int

* **Returns:**
  data after fraction.
* **Return type:**
  data (torch.tensor)

## matgl.utils.training module

Utils for training MatGL models.

### *class* matgl.utils.training.MatglLightningModuleMixin

Bases: `object`

Mix-in class implementing common functions for training.

#### configure_optimizers()

Configure optimizers.

#### on_test_model_eval(\*args, \*\*kwargs)

Executed on model testing.

* **Parameters:**
  * **\*args** – Pass-through
  * **\*\*kwargs** – Pass-through.

#### on_train_epoch_end()

Step scheduler every epoch.

#### predict_step(batch, batch_idx, dataloader_idx=0)

Prediction step.

* **Parameters:**
  * **batch** – Data batch.
  * **batch_idx** – Batch index.
  * **dataloader_idx** – Data loader index.
* **Returns:**
  Prediction

#### test_step(batch: tuple, batch_idx: int)

Test step.

* **Parameters:**
  * **batch** – Data batch.
  * **batch_idx** – Batch index.

#### training_step(batch: tuple, batch_idx: int)

Training step.

* **Parameters:**
  * **batch** – Data batch.
  * **batch_idx** – Batch index.
* **Returns:**
  Total loss.

#### validation_step(batch: tuple, batch_idx: int)

Validation step.

* **Parameters:**
  * **batch** – Data batch.
  * **batch_idx** – Batch index.

### *class* matgl.utils.training.ModelLightningModule(model, data_mean=None, data_std=None, loss: str = ‘mse_loss’, optimizer: Optimizer | None = None, scheduler=None, lr: float = 0.001, decay_steps: int = 1000, decay_alpha: float = 0.01, \*\*kwargs)

Bases: `MatglLightningModuleMixin`, `LightningModule`

A PyTorch.LightningModule for training MEGNet and M3GNet models.

Init ModelLightningModule with key parameters.

* **Parameters:**
  * **model** – Which type of the model for training
  * **data_mean** – average of training data
  * **data_std** – standard deviation of training data
  * **loss** – loss function used for training
  * **optimizer** – optimizer for training
  * **scheduler** – scheduler for training
  * **lr** – learning rate for training
  * **decay_steps** – number of steps for decaying learning rate
  * **decay_alpha** – parameter determines the minimum learning rate.
  * **\*\*kwargs** – Passthrough to parent init.

#### forward(g: dgl.DGLGraph, l_g: dgl.DGLGraph | None = None, state_attr: torch.Tensor | None = None)

* **Parameters:**
  * **g** – dgl Graph
  * **l_g** – Line graph
  * **state_attr** – State attribute.
* **Returns:**
  Model prediction.

#### loss_fn(loss: Module, labels: Tensor, preds: Tensor)

* **Parameters:**
  * **loss** – Loss function.
  * **labels** – Labels to compute the loss.
  * **preds** – Predictions.
* **Returns:**
  total_loss, “MAE”: mae, “RMSE”: rmse}
* **Return type:**
  {“Total_Loss”

#### step(batch: tuple)

* **Parameters:**
  **batch** – Batch of training data.
* **Returns:**
  results, batch_size

### *class* matgl.utils.training.PotentialLightningModule(model, element_refs: np.ndarray | None = None, energy_weight: float = 1.0, force_weight: float = 1.0, stress_weight: float | None = None, site_wise_weight: float | None = None, data_mean=None, data_std=None, calc_stress: bool = False, loss: str = ‘mse_loss’, optimizer: Optimizer | None = None, scheduler=None, lr: float = 0.001, decay_steps: int = 1000, decay_alpha: float = 0.01, \*\*kwargs)

Bases: `MatglLightningModuleMixin`, `LightningModule`

A PyTorch.LightningModule for training MatGL potentials.

This is slightly different from the ModelLightningModel due to the need to account for energy, forces and stress
losses.

Init PotentialLightningModule with key parameters.

* **Parameters:**
  * **model** – Which type of the model for training
  * **element_refs** – element offset for PES
  * **energy_weight** – relative importance of energy
  * **force_weight** – relative importance of force
  * **stress_weight** – relative importance of stress
  * **site_wise_weight** – relative importance of additional site-wise predictions.
  * **data_mean** – average of training data
  * **data_std** – standard deviation of training data
  * **calc_stress** – whether stress calculation is required
  * **loss** – loss function used for training
  * **optimizer** – optimizer for training
  * **scheduler** – scheduler for training
  * **lr** – learning rate for training
  * **decay_steps** – number of steps for decaying learning rate
  * **decay_alpha** – parameter determines the minimum learning rate.
  * **\*\*kwargs** – Passthrough to parent init.

#### forward(g: dgl.DGLGraph, l_g: dgl.DGLGraph | None = None, state_attr: torch.Tensor | None = None)

* **Parameters:**
  * **g** – dgl Graph
  * **l_g** – Line graph
  * **state_attr** – State attr.
* **Returns:**
  energy, force, stress, h

#### loss_fn(loss: Module, labels: tuple, preds: tuple, energy_weight: float | None = None, force_weight: float | None = None, stress_weight: float | None = None, site_wise_weight: float | None = None, num_atoms: int | None = None)

Compute losses for EFS.

* **Parameters:**
  * **loss** – Loss function.
  * **labels** – Labels.
  * **preds** – Predictions
  * **energy_weight** – Weight for energy loss.
  * **force_weight** – Weight for force loss.
  * **stress_weight** – Weight for stress loss.
  * **site_wise_weight** – Weight for site-wise loss.
  * **num_atoms** – Number of atoms.

Returns:

```default
{
    "Total_Loss": total_loss,
    "Energy_MAE": e_mae,
    "Force_MAE": f_mae,
    "Stress_MAE": s_mae,
    "Energy_RMSE": e_rmse,
    "Force_RMSE": f_rmse,
    "Stress_RMSE": s_rmse,
}
```

#### step(batch: tuple)

* **Parameters:**
  **batch** – Batch of training data.
* **Returns:**
  results, batch_size

### matgl.utils.training.xavier_init(model: Module)

Xavier initialization scheme for the model.

* **Parameters:**
  **model** (*nn.Module*) – The model to be Xavier-initialized.