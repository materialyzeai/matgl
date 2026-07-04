---
layout: default
title: Training a M3GNet Potential with PyTorch Lightning.md
nav_exclude: true
---

# Introduction

This notebook demonstrates how to fit a M3GNet potential using PyTorch Lightning with MatGL.


```python
from __future__ import annotations

import contextlib
import os
import shutil
import warnings
from functools import partial

import lightning as L
import numpy as np
from lightning.pytorch.loggers import CSVLogger
from pymatgen.ext.matproj import MPRester

import matgl
from matgl.config import DEFAULT_ELEMENTS
from matgl.ext.pymatgen import Structure2Graph
from matgl.graph.data import MGLDataLoader, MGLDataset, collate_fn_pes, split_dataset
from matgl.models import M3GNet
from matgl.utils.training import PotentialLightningModule

# To suppress warnings for clearer output
warnings.simplefilter("ignore")
```

For the purposes of demonstration, we will download all Si-O compounds in the Materials Project via the MPRester. The forces and stresses are set to zero, though in a real context, these would be non-zero and obtained from DFT calculations.


```python
# Obtain your API key here: https://next-gen.materialsproject.org/api
mpr = MPRester()
entries = mpr.get_entries_in_chemsys(["Si", "O"])
structures = [e.structure for e in entries]
energies = [e.energy for e in entries]
forces = [np.zeros((len(s), 3)).tolist() for s in structures]
stresses = [np.zeros((3, 3)).tolist() for s in structures]
labels = {
    "energies": energies,
    "forces": forces,
    "stresses": stresses,
}

print(f"{len(structures)} downloaded from MP.")
```

    407 downloaded from MP.


We will first setup the M3GNet model and the LightningModule.


```python
element_types = DEFAULT_ELEMENTS
converter = Structure2Graph(element_types=element_types, cutoff=5.0)
# M3GNet builds its three-body (line) graph on the fly inside forward() using the model's own
# threebody_cutoff, so the dataset does not pre-compute or pass a line graph.
dataset = MGLDataset(
    structures=structures,
    converter=converter,
    labels=labels,
)
train_data, val_data, test_data = split_dataset(
    dataset,
    frac_list=[0.8, 0.1, 0.1],
    shuffle=True,
    random_state=42,
)
# if you are not intended to use stress for training, switch include_stress=False!
my_collate_fn = partial(collate_fn_pes, include_stress=True)
train_loader, val_loader, test_loader = MGLDataLoader(
    train_data=train_data,
    val_data=val_data,
    test_data=test_data,
    collate_fn=my_collate_fn,
    batch_size=2,
    num_workers=0,
)
model = M3GNet(
    element_types=element_types,
    is_intensive=False,
)
# if you are not intended to use stress for training, set stress_weight=0.0!
lit_module = PotentialLightningModule(model=model, stress_weight=0.01)
```

Finally, we will initialize the Pytorch Lightning trainer and run the fitting. Here, the max_epochs is set to 2 just for demonstration purposes. In a real fitting, this would be a much larger number. Also, the `accelerator="cpu"` was set just to ensure compatibility with M1 Macs. In a real world use case, please remove the kwarg or set it to cuda for GPU based training.


```python
# If you wish to disable GPU or MPS (M1 mac) training, use the accelerator="cpu" kwarg.
logger = CSVLogger("logs", name="M3GNet_training")
# Inference mode = False is required for calculating forces, stress in test mode and prediction mode
trainer = L.Trainer(max_epochs=1, accelerator="cpu", logger=logger, inference_mode=False)
trainer.fit(model=lit_module, train_dataloaders=train_loader, val_dataloaders=val_loader)
```

    GPU available: True (mps), used: False
    TPU available: False, using: 0 TPU cores
    💡 Tip: For seamless cloud logging and experiment tracking, try installing [litlogger](https://pypi.org/project/litlogger/) to enable LitLogger, which logs metrics and artifacts automatically to the Lightning Experiments platform.
    💡 Tip: For seamless cloud uploads and versioning, try installing [litmodels](https://pypi.org/project/litmodels/) to enable LitModelCheckpoint, which syncs automatically with the Lightning model registry.



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">┏━━━┳━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━┳━━━━━━━┓
┃<span style="color: #800080; text-decoration-color: #800080; font-weight: bold">   </span>┃<span style="color: #800080; text-decoration-color: #800080; font-weight: bold"> Name  </span>┃<span style="color: #800080; text-decoration-color: #800080; font-weight: bold"> Type              </span>┃<span style="color: #800080; text-decoration-color: #800080; font-weight: bold"> Params </span>┃<span style="color: #800080; text-decoration-color: #800080; font-weight: bold"> Mode  </span>┃<span style="color: #800080; text-decoration-color: #800080; font-weight: bold"> FLOPs </span>┃
┡━━━╇━━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━╇━━━━━━━┩
│<span style="color: #7f7f7f; text-decoration-color: #7f7f7f"> 0 </span>│ mae   │ MeanAbsoluteError │      0 │ train │     0 │
│<span style="color: #7f7f7f; text-decoration-color: #7f7f7f"> 1 </span>│ rmse  │ MeanSquaredError  │      0 │ train │     0 │
│<span style="color: #7f7f7f; text-decoration-color: #7f7f7f"> 2 </span>│ model │ Potential         │  288 K │ train │     0 │
└───┴───────┴───────────────────┴────────┴───────┴───────┘
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold">Trainable params</span>: 288 K
<span style="font-weight: bold">Non-trainable params</span>: 0
<span style="font-weight: bold">Total params</span>: 288 K
<span style="font-weight: bold">Total estimated model params size (MB)</span>: 1
<span style="font-weight: bold">Modules in train mode</span>: 174
<span style="font-weight: bold">Modules in eval mode</span>: 0
<span style="font-weight: bold">Total FLOPs</span>: 0
</pre>




    Output()



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"></pre>




    ---------------------------------------------------------------------------

    IndexError                                Traceback (most recent call last)

    Cell In[8], line 5
          1 # If you wish to disable GPU or MPS (M1 mac) training, use the accelerator="cpu" kwarg.
          2 logger = CSVLogger("logs", name="M3GNet_training")
          3 # Inference mode = False is required for calculating forces, stress in test mode and prediction mode
          4 trainer = L.Trainer(max_epochs=1, accelerator="cpu", logger=logger, inference_mode=False)
    ----> 5 trainer.fit(model=lit_module, train_dataloaders=train_loader, val_dataloaders=val_loader)


    File ~/repos/matgl/.venv/lib/python3.12/site-packages/lightning/pytorch/trainer/trainer.py:584, in Trainer.fit(self, model, train_dataloaders, val_dataloaders, datamodule, ckpt_path, weights_only)
        582 self.training = True
        583 self.should_stop = False
    --> 584 call._call_and_handle_interrupt(
        585     self,
        586     self._fit_impl,
        587     model,
        588     train_dataloaders,
        589     val_dataloaders,
        590     datamodule,
        591     ckpt_path,
        592     weights_only,
        593 )


    File ~/repos/matgl/.venv/lib/python3.12/site-packages/lightning/pytorch/trainer/call.py:49, in _call_and_handle_interrupt(trainer, trainer_fn, *args, **kwargs)
         47     if trainer.strategy.launcher is not None:
         48         return trainer.strategy.launcher.launch(trainer_fn, *args, trainer=trainer, **kwargs)
    ---> 49     return trainer_fn(*args, **kwargs)
         51 except _TunerExitException:
         52     _call_teardown_hook(trainer)


    File ~/repos/matgl/.venv/lib/python3.12/site-packages/lightning/pytorch/trainer/trainer.py:630, in Trainer._fit_impl(self, model, train_dataloaders, val_dataloaders, datamodule, ckpt_path, weights_only)
        623     download_model_from_registry(ckpt_path, self)
        624 ckpt_path = self._checkpoint_connector._select_ckpt_path(
        625     self.state.fn,
        626     ckpt_path,
        627     model_provided=True,
        628     model_connected=self.lightning_module is not None,
        629 )
    --> 630 self._run(model, ckpt_path=ckpt_path, weights_only=weights_only)
        632 assert self.state.stopped
        633 self.training = False


    File ~/repos/matgl/.venv/lib/python3.12/site-packages/lightning/pytorch/trainer/trainer.py:1079, in Trainer._run(self, model, ckpt_path, weights_only)
       1074 self._signal_connector.register_signal_handlers()
       1076 # ----------------------------
       1077 # RUN THE TRAINER
       1078 # ----------------------------
    -> 1079 results = self._run_stage()
       1081 # ----------------------------
       1082 # POST-Training CLEAN UP
       1083 # ----------------------------
       1084 log.debug(f"{self.__class__.__name__}: trainer tearing down")


    File ~/repos/matgl/.venv/lib/python3.12/site-packages/lightning/pytorch/trainer/trainer.py:1121, in Trainer._run_stage(self)
       1119 if self.training:
       1120     with isolate_rng():
    -> 1121         self._run_sanity_check()
       1122     with torch.autograd.set_detect_anomaly(self._detect_anomaly):
       1123         self.fit_loop.run()


    File ~/repos/matgl/.venv/lib/python3.12/site-packages/lightning/pytorch/trainer/trainer.py:1150, in Trainer._run_sanity_check(self)
       1147 call._call_callback_hooks(self, "on_sanity_check_start")
       1149 # run eval step
    -> 1150 val_loop.run()
       1152 call._call_callback_hooks(self, "on_sanity_check_end")
       1154 # reset logger connector


    File ~/repos/matgl/.venv/lib/python3.12/site-packages/lightning/pytorch/loops/utilities.py:179, in _no_grad_context.<locals>._decorator(self, *args, **kwargs)
        177     context_manager = torch.no_grad
        178 with context_manager():
    --> 179     return loop_run(self, *args, **kwargs)


    File ~/repos/matgl/.venv/lib/python3.12/site-packages/lightning/pytorch/loops/evaluation_loop.py:146, in _EvaluationLoop.run(self)
        144     self.batch_progress.is_last_batch = data_fetcher.done
        145     # run step hooks
    --> 146     self._evaluation_step(batch, batch_idx, dataloader_idx, dataloader_iter)
        147 except StopIteration:
        148     # this needs to wrap the `*_step` call too (not just `next`) for `dataloader_iter` support
        149     break


    File ~/repos/matgl/.venv/lib/python3.12/site-packages/lightning/pytorch/loops/evaluation_loop.py:441, in _EvaluationLoop._evaluation_step(self, batch, batch_idx, dataloader_idx, dataloader_iter)
        435 hook_name = "test_step" if trainer.testing else "validation_step"
        436 step_args = (
        437     self._build_step_args_from_hook_kwargs(hook_kwargs, hook_name)
        438     if not using_dataloader_iter
        439     else (dataloader_iter,)
        440 )
    --> 441 output = call._call_strategy_hook(trainer, hook_name, *step_args)
        443 self.batch_progress.increment_processed()
        445 if using_dataloader_iter:
        446     # update the hook kwargs now that the step method might have consumed the iterator


    File ~/repos/matgl/.venv/lib/python3.12/site-packages/lightning/pytorch/trainer/call.py:329, in _call_strategy_hook(trainer, hook_name, *args, **kwargs)
        326     return None
        328 with trainer.profiler.profile(f"[Strategy]{trainer.strategy.__class__.__name__}.{hook_name}"):
    --> 329     output = fn(*args, **kwargs)
        331 # restore current_fx when nested context
        332 pl_module._current_fx_name = prev_fx_name


    File ~/repos/matgl/.venv/lib/python3.12/site-packages/lightning/pytorch/strategies/strategy.py:412, in Strategy.validation_step(self, *args, **kwargs)
        410 if self.model != self.lightning_module:
        411     return self._forward_redirection(self.model, self.lightning_module, "validation_step", *args, **kwargs)
    --> 412 return self.lightning_module.validation_step(*args, **kwargs)


    File ~/repos/matgl/src/matgl/utils/training.py:536, in PotentialLightningModule.validation_step(self, batch, batch_idx)
        524 def validation_step(self, batch: tuple, batch_idx: int) -> dict[str, Any]:
        525     """Validation step that exposes per-sample preds and labels for callbacks.
        526
        527     Args:
       (...)    534         stamped with :func:`matgl.utils.callbacks.add_sample_indices`).
        535     """
    --> 536     loss = super().validation_step(batch, batch_idx)
        537     return {
        538         "loss": loss,
        539         "preds": self._last_preds,
       (...)    542         "num_atoms": self._last_num_atoms,
        543     }


    File ~/repos/matgl/src/matgl/utils/training.py:84, in MatglLightningModuleMixin.validation_step(self, batch, batch_idx)
         77 def validation_step(self, batch: tuple, batch_idx: int) -> Any:
         78     """Validation step.
         79
         80     Args:
         81         batch: Data batch.
         82         batch_idx: Batch index.
         83     """
    ---> 84     results, batch_size = self.step(batch)  # type: ignore
         85     self.log_dict(  # type: ignore
         86         {f"val_{key}": val for key, val in results.items()},
         87         batch_size=batch_size,
       (...)     91         sync_dist=self.sync_dist,  # type: ignore
         92     )
         93     return results["Total_Loss"]


    File ~/repos/matgl/src/matgl/utils/training.py:475, in PotentialLightningModule.step(self, batch)
        473 if self.include_line_graph:
        474     g, lat, l_g, state_attr, *targets = batch
    --> 475     out = self(g=g, lat=lat, state_attr=state_attr, l_g=l_g)
        476 else:
        477     g, lat, state_attr, *targets = batch


    File ~/repos/matgl/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py:1778, in Module._wrapped_call_impl(self, *args, **kwargs)
       1776     return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
       1777 else:
    -> 1778     return self._call_impl(*args, **kwargs)


    File ~/repos/matgl/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py:1789, in Module._call_impl(self, *args, **kwargs)
       1784 # If we don't have any hooks, we want to skip the rest of the logic in
       1785 # this function, and just call forward.
       1786 if not (self._backward_hooks or self._backward_pre_hooks or self._forward_hooks or self._forward_pre_hooks
       1787         or _global_backward_pre_hooks or _global_backward_hooks
       1788         or _global_forward_hooks or _global_forward_pre_hooks):
    -> 1789     return forward_call(*args, **kwargs)
       1791 result = None
       1792 called_always_called_hooks = set()


    File ~/repos/matgl/src/matgl/utils/training.py:441, in PotentialLightningModule.forward(self, g, lat, l_g, state_attr)
        439         e, f, s, h, m = self.model(g=g, lat=lat, l_g=l_g, state_attr=state_attr)
        440         return e, f, s, h, m
    --> 441     e, f, s, h = self.model(g=g, lat=lat, l_g=l_g, state_attr=state_attr)
        442     return e, f, s, h
        443 if self.model.calc_charge:


    File ~/repos/matgl/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py:1778, in Module._wrapped_call_impl(self, *args, **kwargs)
       1776     return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
       1777 else:
    -> 1778     return self._call_impl(*args, **kwargs)


    File ~/repos/matgl/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py:1789, in Module._call_impl(self, *args, **kwargs)
       1784 # If we don't have any hooks, we want to skip the rest of the logic in
       1785 # this function, and just call forward.
       1786 if not (self._backward_hooks or self._backward_pre_hooks or self._forward_hooks or self._forward_pre_hooks
       1787         or _global_backward_pre_hooks or _global_backward_hooks
       1788         or _global_forward_hooks or _global_forward_pre_hooks):
    -> 1789     return forward_call(*args, **kwargs)
       1791 result = None
       1792 called_always_called_hooks = set()


    File ~/repos/matgl/src/matgl/apps/pes.py:299, in Potential.forward(self, g, lat, state_attr, l_g, total_charge, ext_pot)
        288 autograd_ctx = nullcontext() if needs_autograd else torch.no_grad()
        289 with autograd_ctx:
        290     total_energies = (
        291         self.model(
        292             g=g,
        293             state_attr=state_attr,
        294             l_g=l_g,
        295             total_charge=total_charge,
        296             ext_pot=ext_pot,
        297         )
        298         if self.calc_charge
    --> 299         else self.model(g=g, state_attr=state_attr, l_g=l_g)
        300     )
        301     total_energies = self.data_std * total_energies + self.data_mean
        303     if self.calc_repuls:


    File ~/repos/matgl/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py:1778, in Module._wrapped_call_impl(self, *args, **kwargs)
       1776     return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
       1777 else:
    -> 1778     return self._call_impl(*args, **kwargs)


    File ~/repos/matgl/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py:1789, in Module._call_impl(self, *args, **kwargs)
       1784 # If we don't have any hooks, we want to skip the rest of the logic in
       1785 # this function, and just call forward.
       1786 if not (self._backward_hooks or self._backward_pre_hooks or self._forward_hooks or self._forward_pre_hooks
       1787         or _global_backward_pre_hooks or _global_backward_hooks
       1788         or _global_forward_hooks or _global_forward_pre_hooks):
    -> 1789     return forward_call(*args, **kwargs)
       1791 result = None
       1792 called_always_called_hooks = set()


    File ~/repos/matgl/src/matgl/models/_m3gnet.py:242, in M3GNet.forward(self, g, state_attr, l_g, return_all_layer_output)
        240     l_g = create_line_graph(edge_index, bond_dist, bond_vec, pbc_offshift, num_nodes, self.threebody_cutoff)
        241 else:
    --> 242     l_g = ensure_line_graph_compatibility(l_g, bond_dist, bond_vec, pbc_offshift, self.threebody_cutoff)
        244 angles = compute_theta_and_phi(l_g["bond_vec"], l_g["bond_dist"], l_g["line_edge_index"])
        245 three_body_basis = self.basis_expansion(angles["triple_bond_lengths"], angles["cos_theta"], angles["phi"])


    File ~/repos/matgl/src/matgl/graph/_compute.py:249, in ensure_line_graph_compatibility(line_graph, bond_dist, bond_vec, pbc_offset, threebody_cutoff)
        246 valid_dist = bond_dist[valid]
        247 valid_vec = bond_vec[valid]
    --> 249 n_lg_nodes = line_graph["bond_dist"].size(0)
        250 if n_lg_nodes == valid_dist.size(0):
        251     new_bond_dist = valid_dist


    IndexError: too many indices for tensor of dimension 2



```python
# test the model, remember to set inference_mode=False in trainer (see above)
trainer.test(dataloaders=test_loader)
```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    Cell In[9], line 2
          1 # test the model, remember to set inference_mode=False in trainer (see above)
    ----> 2 trainer.test(dataloaders=test_loader)


    File ~/repos/matgl/.venv/lib/python3.12/site-packages/lightning/pytorch/trainer/trainer.py:821, in Trainer.test(self, model, dataloaders, ckpt_path, verbose, datamodule, weights_only)
        819 self.state.status = TrainerStatus.RUNNING
        820 self.testing = True
    --> 821 return call._call_and_handle_interrupt(
        822     self, self._test_impl, model, dataloaders, ckpt_path, verbose, datamodule, weights_only
        823 )


    File ~/repos/matgl/.venv/lib/python3.12/site-packages/lightning/pytorch/trainer/call.py:49, in _call_and_handle_interrupt(trainer, trainer_fn, *args, **kwargs)
         47     if trainer.strategy.launcher is not None:
         48         return trainer.strategy.launcher.launch(trainer_fn, *args, trainer=trainer, **kwargs)
    ---> 49     return trainer_fn(*args, **kwargs)
         51 except _TunerExitException:
         52     _call_teardown_hook(trainer)


    File ~/repos/matgl/.venv/lib/python3.12/site-packages/lightning/pytorch/trainer/trainer.py:861, in Trainer._test_impl(self, model, dataloaders, ckpt_path, verbose, datamodule, weights_only)
        859 if _is_registry(ckpt_path) and module_available("litmodels"):
        860     download_model_from_registry(ckpt_path, self)
    --> 861 ckpt_path = self._checkpoint_connector._select_ckpt_path(
        862     self.state.fn, ckpt_path, model_provided=model_provided, model_connected=self.lightning_module is not None
        863 )
        864 results = self._run(model, ckpt_path=ckpt_path, weights_only=weights_only)
        865 # remove the tensors from the test results


    File ~/repos/matgl/.venv/lib/python3.12/site-packages/lightning/pytorch/trainer/connectors/checkpoint_connector.py:108, in _CheckpointConnector._select_ckpt_path(self, state_fn, ckpt_path, model_provided, model_connected)
        106         ckpt_path = self._ckpt_path
        107 else:
    --> 108     ckpt_path = self._parse_ckpt_path(
        109         state_fn,
        110         ckpt_path,
        111         model_provided=model_provided,
        112         model_connected=model_connected,
        113     )
        114 return ckpt_path


    File ~/repos/matgl/.venv/lib/python3.12/site-packages/lightning/pytorch/trainer/connectors/checkpoint_connector.py:175, in _CheckpointConnector._parse_ckpt_path(self, state_fn, ckpt_path, model_provided, model_connected)
        170     if self.trainer.fast_dev_run:
        171         raise ValueError(
        172             f'You cannot execute `.{fn}(ckpt_path="best")` with `fast_dev_run=True`.'
        173             f" Please pass an exact checkpoint path to `.{fn}(ckpt_path=...)`"
        174         )
    --> 175     raise ValueError(
        176         f'`.{fn}(ckpt_path="best")` is set but `ModelCheckpoint` is not configured to save the best model.'
        177     )
        178 # load best weights
        179 ckpt_path = getattr(self.trainer.checkpoint_callback, "best_model_path", None)


    ValueError: `.test(ckpt_path="best")` is set but `ModelCheckpoint` is not configured to save the best model.



```python
# save trained model
model_export_path = "./trained_model/"
lit_module.model.save(model_export_path)

# load trained model
model = matgl.load_model(path=model_export_path)
```

## Finetuning a pre-trained M3GNet
In the previous cells, we demonstrated the process of training an M3GNet from scratch. Next, let's see how to perform additional training on an M3GNet that has already been trained using Materials Project data.


```python
# download a pre-trained M3GNet foundation potential from Hugging Face
m3gnet_nnp = matgl.load_model("M3GNet-PES-MatPES-PBE-2025.2")
model_pretrained = m3gnet_nnp.model
# IMPORTANT: graph node indices are positions into ``element_types``, not atomic numbers, so the
# ordering used to build the dataset MUST match the pre-trained model's ``element_types`` exactly.
# The dataset above was built with DEFAULT_ELEMENTS; confirm the pre-trained model agrees. When
# fine-tuning any foundation model, always build your Structure2Graph with
# element_types=model_pretrained.element_types (do NOT use get_element_list(structures), which only
# covers the elements in your data and yields a different index ordering).
assert tuple(model_pretrained.element_types) == tuple(element_types), (
    "element_types mismatch: rebuild the dataset with element_types=model_pretrained.element_types"
)
# obtain element energy offset (a per-element vector in model_pretrained.element_types order)
property_offset = m3gnet_nnp.element_refs.property_offset
# you should test whether including the original property_offset helps improve training and validation accuracy
lit_module_finetune = PotentialLightningModule(model=model_pretrained, element_refs=property_offset, lr=1e-4)
```


```python
# If you wish to disable GPU or MPS (M1 mac) training, use the accelerator="cpu" kwarg.
logger = CSVLogger("logs", name="M3GNet_finetuning")
trainer = L.Trainer(max_epochs=1, accelerator="cpu", logger=logger, inference_mode=False)
trainer.fit(model=lit_module_finetune, train_dataloaders=train_loader, val_dataloaders=val_loader)
```


```python
# save trained model
model_save_path = "./finetuned_model/"
lit_module_finetune.model.save(model_save_path)
# load trained model
trained_model = matgl.load_model(path=model_save_path)
```


```python
# This code just performs cleanup for this notebook.

for fn in ("pyg_graph.pt", "lattice.pt", "pyg_line_graph.pt", "state_attr.pt", "labels.json"):
    with contextlib.suppress(FileNotFoundError):
        os.remove(fn)

shutil.rmtree("logs")
shutil.rmtree("trained_model")
shutil.rmtree("finetuned_model")
```