---
layout: default
title: API Documentation
nav_order: 5
---

# matgl package

MatGL (Materials Graph Library) is a graph deep learning library for materials science.

## Subpackages

* [matgl.apps package](matgl.apps.md)
  * matgl.apps.pes module
    * `Potential`
      * `Potential.forward()`
* [matgl.data package](matgl.data.md)
  * matgl.data.transformer module
    * `LogTransformer`
      * `LogTransformer.inverse_transform()`
      * `LogTransformer.transform()`
    * `Normalizer`
      * `Normalizer.from_data()`
      * `Normalizer.inverse_transform()`
      * `Normalizer.transform()`
    * `Transformer`
      * `Transformer.inverse_transform()`
      * `Transformer.transform()`
* [matgl.ext package](matgl.ext.md)
  * matgl.ext.ase module
    * `Atoms2Graph`
      * `Atoms2Graph.get_graph()`
    * `M3GNetCalculator`
      * `M3GNetCalculator.calculate()`
      * `M3GNetCalculator.implemented_properties`
    * `MolecularDynamics`
      * `MolecularDynamics.run()`
      * `MolecularDynamics.set_atoms()`
    * `OPTIMIZERS`
      * `OPTIMIZERS.bfgs`
      * `OPTIMIZERS.bfgslinesearch`
      * `OPTIMIZERS.fire`
      * `OPTIMIZERS.lbfgs`
      * `OPTIMIZERS.lbfgslinesearch`
      * `OPTIMIZERS.mdmin`
      * `OPTIMIZERS.scipyfminbfgs`
      * `OPTIMIZERS.scipyfmincg`
    * `Relaxer`
      * `Relaxer.relax()`
    * `TrajectoryObserver`
      * `TrajectoryObserver.as_pandas()`
      * `TrajectoryObserver.save()`
  * [matgl.ext.torch_sim module](matgl.ext.torch_sim.md)
    * `MatGLModel`
  * matgl.ext.pymatgen module
    * `Molecule2Graph`
      * `Molecule2Graph.get_graph()`
    * `Structure2Graph`
      * `Structure2Graph.get_graph()`
    * `get_element_list()`
* [matgl.graph package](matgl.graph.md)
  * matgl.graph.compute module
    * `compute_3body()`
    * `compute_pair_vector_and_distance()`
    * `compute_theta()`
    * `compute_theta_and_phi()`
    * `create_line_graph()`
  * matgl.graph.converters module
    * `GraphConverter`
      * `GraphConverter.get_graph()`
      * `GraphConverter.get_graph_from_processed_structure()`
  * matgl.graph.data module
    * `MGLDataset`
      * `MGLDataset.has_cache()`
      * `MGLDataset.load()`
      * `MGLDataset.process()`
      * `MGLDataset.save()`
    * `MEGNetDataset`
      * `MEGNetDataset.has_cache()`
      * `MEGNetDataset.load()`
      * `MEGNetDataset.process()`
      * `MEGNetDataset.save()`
    * `MGLDataLoader()`
    * `collate_fn()`
    * `collate_fn_efs()`
* [matgl.layers package](matgl.layers.md)
  * matgl.layers._activations module
    * `ActivationFunction`
      * `ActivationFunction.sigmoid`
      * `ActivationFunction.softexp`
      * `ActivationFunction.softplus`
      * `ActivationFunction.softplus2`
      * `ActivationFunction.swish`
      * `ActivationFunction.tanh`
    * `SoftExponential`
      * `SoftExponential.forward()`
    * `SoftPlus2`
      * `SoftPlus2.forward()`
  * matgl.layers._atom_ref module
    * `AtomRef`
      * `AtomRef.fit()`
      * `AtomRef.forward()`
      * `AtomRef.get_feature_matrix()`
  * matgl.layers._basis module
    * `FourierExpansion`
      * `FourierExpansion.forward()`
    * `GaussianExpansion`
      * `GaussianExpansion.forward()`
      * `GaussianExpansion.reset_parameters()`
    * `RadialBesselFunction`
      * `RadialBesselFunction.forward()`
    * `SphericalBesselFunction`
      * `SphericalBesselFunction.rbf_j0()`
    * `SphericalBesselWithHarmonics`
      * `SphericalBesselWithHarmonics.forward()`
    * `SphericalHarmonicsFunction`
    * `spherical_bessel_smooth()`
  * matgl.layers._bond module
    * `BondExpansion`
      * `BondExpansion.forward()`
  * matgl.layers._core module
    * `EdgeSet2Set`
      * `EdgeSet2Set.forward()`
      * `EdgeSet2Set.reset_parameters()`
    * `GatedMLP`
      * `GatedMLP.forward()`
    * `MLP`
      * `MLP.depth`
      * `MLP.forward()`
      * `MLP.in_features`
      * `MLP.last_linear`
      * `MLP.out_features`
  * matgl.layers._embedding module
    * `EmbeddingBlock`
      * `EmbeddingBlock.forward()`
  * matgl.layers._graph_convolution module
    * `M3GNetBlock`
      * `M3GNetBlock.forward()`
    * `M3GNetGraphConv`
      * `M3GNetGraphConv.edge_update_()`
      * `M3GNetGraphConv.forward()`
      * `M3GNetGraphConv.from_dims()`
      * `M3GNetGraphConv.node_update_()`
      * `M3GNetGraphConv.state_update_()`
    * `MEGNetBlock`
      * `MEGNetBlock.forward()`
    * `MEGNetGraphConv`
      * `MEGNetGraphConv.edge_update_()`
      * `MEGNetGraphConv.forward()`
      * `MEGNetGraphConv.from_dims()`
      * `MEGNetGraphConv.node_update_()`
      * `MEGNetGraphConv.state_update_()`
  * matgl.layers._readout module
    * `ReduceReadOut`
      * `ReduceReadOut.forward()`
    * `Set2SetReadOut`
      * `Set2SetReadOut.forward()`
    * `WeightedReadOut`
      * `WeightedReadOut.forward()`
    * `WeightedReadOutPair`
      * `WeightedReadOutPair.forward()`
  * matgl.layers._three_body module
    * `ThreeBodyInteractions`
      * `ThreeBodyInteractions.forward()`
    * `combine_sbf_shf()`
* [matgl.models package](matgl.models.md)
  * matgl.models._m3gnet module
    * `M3GNet`
      * `M3GNet.forward()`
      * `M3GNet.predict_structure()`
  * matgl.models._megnet module
    * `MEGNet`
      * `MEGNet.forward()`
      * `MEGNet.predict_structure()`
  * matgl.models._wrappers module
    * `TransformedTargetModel`
      * `TransformedTargetModel.forward()`
      * `TransformedTargetModel.predict_structure()`
* [matgl.utils package](matgl.utils.md)
  * matgl.utils.cutoff module
    * `cosine_cutoff()`
    * `polynomial_cutoff()`
  * matgl.utils.io module
    * `IOMixIn`
      * `IOMixIn.load()`
      * `IOMixIn.save()`
      * `IOMixIn.save_args()`
    * `RemoteFile`
    * `get_available_pretrained_models()`
    * `load_model()`
  * matgl.utils.maths module
    * `broadcast()`
    * `broadcast_states_to_atoms()`
    * `broadcast_states_to_bonds()`
    * `get_range_indices_from_n()`
    * `get_segment_indices_from_n()`
    * `repeat_with_n()`
    * `scatter_sum()`
    * `spherical_bessel_roots()`
    * `unsorted_segment_fraction()`
  * matgl.utils.training module
    * `MatglLightningModuleMixin`
      * `MatglLightningModuleMixin.configure_optimizers()`
      * `MatglLightningModuleMixin.on_test_model_eval()`
      * `MatglLightningModuleMixin.on_train_epoch_end()`
      * `MatglLightningModuleMixin.predict_step()`
      * `MatglLightningModuleMixin.test_step()`
      * `MatglLightningModuleMixin.training_step()`
      * `MatglLightningModuleMixin.validation_step()`
    * `ModelLightningModule`
      * `ModelLightningModule.forward()`
      * `ModelLightningModule.loss_fn()`
      * `ModelLightningModule.step()`
    * `PotentialLightningModule`
      * `PotentialLightningModule.forward()`
      * `PotentialLightningModule.loss_fn()`
      * `PotentialLightningModule.step()`
    * `xavier_init()`

## matgl.cli module

Command line interface for matgl.

### matgl.cli.clear_cache(args)

Clear cache command.

* **Parameters:**
  **args** – Args from CLI.

### matgl.cli.main()

Handle main.

### matgl.cli.predict_structure(args)

Use MatGL models to perform predictions on structures.

* **Parameters:**
  **args** – Args from CLI.

### matgl.cli.relax_structure(args)

Relax crystals.

* **Parameters:**
  **args** – Args from CLI.

## matgl.config module

Global configuration variables for matgl.

### matgl.config.clear_cache(confirm: bool = True)

Deletes all files in the matgl.cache. This is used to clean out downloaded models.

* **Parameters:**
  **confirm** – Whether to ask for confirmation. Default is True.
