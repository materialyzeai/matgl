---
layout: default
title: matgl.graph.md
nav_exclude: true
---

# matgl.graph package

Package for creating and manipulating graphs.

## matgl.graph.compute module

Computing various graph based operations.

### matgl.graph.compute.compute_3body(g: DGLGraph)

Calculate the three body indices from pair atom indices.

* **Parameters:**
  **g** – DGL graph
* **Returns:**
  DGL graph containing three body information from graph
  triple_bond_indices (np.ndarray): bond indices that form three-body
  n_triple_ij (np.ndarray): number of three-body angles for each bond
  n_triple_i (np.ndarray): number of three-body angles each atom
  n_triple_s (np.ndarray): number of three-body angles for each structure
* **Return type:**
  l_g

### matgl.graph.compute.compute_pair_vector_and_distance(g: DGLGraph)

Calculate bond vectors and distances using dgl graphs.

Args:
g: DGL graph

Returns:
bond_vec (torch.tensor): bond distance between two atoms
bond_dist (torch.tensor): vector from src node to dst node

### matgl.graph.compute.compute_theta(edges: EdgeBatch, cosine: bool = False)

User defined dgl function to calculate bond angles from edges in a graph.

* **Parameters:**
  * **edges** – DGL graph edges
  * **cosine** – Whether to return the cosine of the angle or the angle itself
* **Returns:**
  Dictionary containing bond angles and distances
* **Return type:**
  dict[str, torch.Tensor]

### matgl.graph.compute.compute_theta_and_phi(edges: EdgeBatch)

Calculate bond angle Theta and Phi using dgl graphs.

Args:
edges: DGL graph edges

Returns:
cos_theta: torch.Tensor
phi: torch.Tensor
triple_bond_lengths (torch.tensor):

### matgl.graph.compute.create_line_graph(g: DGLGraph, threebody_cutoff: float)

Calculate the three body indices from pair atom indices.

* **Parameters:**
  * **g** – DGL graph
  * **threebody_cutoff** (*float*) – cutoff for three-body interactions
* **Returns:**
  DGL graph containing three body information from graph
* **Return type:**
  l_g

## matgl.graph.converters module

Tools to convert materials representations from Pymatgen and other codes to DGLGraphs.

### *class* matgl.graph.converters.GraphConverter

Bases: `object`

Abstract base class for converters from input crystals/molecules to graphs.

#### *abstract* get_graph(structure)

Args:
structure: Input crystals or molecule.

Returns:
DGLGraph object, state_attr

#### get_graph_from_processed_structure(structure, src_id, dst_id, images, lattice_matrix, element_types, cart_coords)

Construct a dgl graph from processed structure and bond information.

* **Parameters:**
  * **structure** – Input crystals or molecule of pymatgen structure or molecule types.
  * **src_id** – site indices for starting point of bonds.
  * **dst_id** – site indices for destination point of bonds.
  * **images** – the periodic image offsets for the bonds.
  * **lattice_matrix** – lattice information of the structure.
  * **element_types** – Element symbols of all atoms in the structure.
  * **cart_coords** – Cartisian coordinates of all atoms in the structure.
* **Returns:**
  DGLGraph object, state_attr

## matgl.graph.data module

Tools to construct a dataset of DGL graphs.

### *class* matgl.graph.data.MGLDataset(converter: GraphConverter, threebody_cutoff: float, structures: list, energies: list | None = None, forces: list | None = None, stresses: list | None = None, labels: list | None = None, name=’M3GNETDataset’, label_name: str | None = None, graph_labels: list | None = None)

Bases: `DGLDataset`

Create a dataset including dgl graphs.

* **Parameters:**
  * **converter** – dgl graph converter
  * **threebody_cutoff** – cutoff for three body
  * **structures** – Pymatgen structure
  * **energies** – Target energies
  * **forces** – Target forces
  * **stresses** – Target stresses
  * **labels** – target properties
  * **name** – name of dataset
  * **label_name** – name of target properties
  * **graph_labels** – state attributes.

#### has_cache(filename: str = ‘dgl_graph.bin’)

Check if the dgl_graph.bin exists or not
:param : filename: Name of file storing dgl graphs

Returns: True if file exists.

#### load(filename: str = ‘dgl_graph.bin’, filename_line_graph: str = ‘dgl_line_graph.bin’, filename_state_attr: str = ‘state_attr.pt’)

Load dgl graphs from files.

* **Parameters:**
  * **filename** – Name of file storing dgl graphs
  * **filename_line_graph** – Name of file storing dgl line graphs
  * **filename_state_attr** – Name of file storing state attrs.

#### process()

Convert Pymatgen structure into dgl graphs.

#### save(filename: str = ‘dgl_graph.bin’, filename_line_graph: str = ‘dgl_line_graph.bin’, filename_state_attr: str = ‘state_attr.pt’)

Save dgl graphs
Args:
:filename: Name of file storing dgl graphs
:filename_state_attr: Name of file storing graph attrs.

### *class* matgl.graph.data.MEGNetDataset(structures: list, labels: list, label_name: str, converter: GraphConverter, initial: float = 0.0, final: float = 5.0, num_centers: int = 100, width: float = 0.5, name: str = ‘MEGNETDataset’, graph_labels: list | None = None)

Bases: `DGLDataset`

Create a dataset including dgl graphs.

* **Parameters:**
  * **structures** – Pymatgen structure
  * **labels** – property values
  * **label_name** – label name
  * **converter** – Transformer for converting structures to DGL graphs, e.g., Pmg2Graph.
  * **initial** – initial distance for Gaussian expansions
  * **final** – final distance for Gaussian expansions
  * **num_centers** – number of Gaussian functions
  * **width** – width of Gaussian functions
  * **name** – Name of dataset
  * **graph_labels** – graph attributes either integers and floating point numbers.

#### has_cache(filename: str = ‘dgl_graph.bin’)

Check if the dgl_graph.bin exists or not
:param : filename: Name of file storing dgl graphs

Returns: True if file exists.

#### load(filename: str = ‘dgl_graph.bin’, filename_state_attr: str = ‘state_attr.pt’)

Load dgl graphs
Args:
:filename: Name of file storing dgl graphs
:filename: Name of file storing state attrs.

#### process()

Convert Pymatgen structure into dgl graphs.

#### save(filename: str = ‘dgl_graph.bin’, filename_state_attr: str = ‘state_attr.pt’)

Save dgl graphs
Args:
:filename: Name of file storing dgl graphs
:filename_state_attr: Name of file storing graph attrs.

### matgl.graph.data.MGLDataLoader(train_data: Subset, val_data: Subset, collate_fn: Callable, batch_size: int, num_workers: int, use_ddp: bool = False, pin_memory: bool = False, test_data: Subset | None = None, generator: Generator | None = None)

Dataloader for MEGNet training.

* **Parameters:**
  * **train_data** (*dgl.data.utils.Subset*) – Training dataset.
  * **val_data** (*dgl.data.utils.Subset*) – Validation dataset.
  * **collate_fn** (*Callable*) – Collate function.
  * **batch_size** (*int*) – Batch size.
  * **num_workers** (*int*) – Number of workers.
  * **use_ddp** (*bool*\*,\* *optional*) – Whether to use DDP. Defaults to False.
  * **pin_memory** (*bool*\*,\* *optional*) – Whether to pin memory. Defaults to False.
  * **test_data** (*dgl.data.utils.Subset*       *|* *None*\*,\* *optional*) – Test dataset. Defaults to None.
  * **generator** (*torch.Generator*       *|* *None*\*,\* *optional*) – Random number generator. Defaults to None.
* **Returns:**
  Train, validation and test data loaders. Test data
  : loader is None if test_data is None.
* **Return type:**
  tuple[GraphDataLoader, …]

### matgl.graph.data.collate_fn(batch, include_line_graph: bool = False)

Merge a list of dgl graphs to form a batch.

### matgl.graph.data.collate_fn_efs(batch)

Merge a list of dgl graphs to form a batch.