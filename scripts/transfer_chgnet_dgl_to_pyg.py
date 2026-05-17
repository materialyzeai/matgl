#!/usr/bin/env python
"""Transfer weights from CHGNet-DGL to CHGNet-PyG and validate on 100 MatPES structures.

Usage:
    python transfer_chgnet_dgl_to_pyg.py [DGL_MODEL_NAME]

    DGL_MODEL_NAME defaults to CHGNet-PES-MatPES-r2SCAN-2025.2.10.
    The PyG model is saved to ~/CHGNet-PyG-<suffix>/ where <suffix> is
    the part after "CHGNet-PES-" in the DGL model name.

Outputs:
    - ~/CHGNet-PyG-<suffix>/  (saved PyG model)
    - Prints per-structure comparison of E/F/S/M between DGL and PyG
"""
from __future__ import annotations

import os
import sys

os.environ["MATGL_BACKEND"] = "DGL"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import json
import numpy as np
import torch
from pathlib import Path

dgl_model_name = sys.argv[1] if len(sys.argv) > 1 else "CHGNet-PES-MatPES-r2SCAN-2025.2.10"
# Derive PyG save name: replace "CHGNet-PES-" prefix with "CHGNet-PyG-"
suffix = dgl_model_name.removeprefix("CHGNet-PES-")
pyg_model_name = f"CHGNet-PyG-{suffix}"

print(f"DGL model : {dgl_model_name}")
print(f"PyG target: {pyg_model_name}")

# ── 1. Load pretrained DGL model ──────────────────────────────────────────────
import matgl
print(f"\nLoading {dgl_model_name} (DGL) ...")
dgl_potential = matgl.load_model(dgl_model_name)
dgl_model = dgl_potential.model
dgl_sd = dgl_model.state_dict()

# Detect capabilities from the DGL potential
calc_magmom = getattr(dgl_potential, "calc_magmom", False)
calc_stresses = getattr(dgl_potential, "calc_stresses", True)
print(f"  calc_magmom={calc_magmom}, calc_stresses={calc_stresses}")

# Read init args from cached model.json
cache_root = Path.home() / ".cache/matgl"
# Match the cached directory by a fragment of the model name
fragment = suffix.replace("MatPES-", "").replace(".", r"*")  # e.g. "r2SCAN*"
candidates = list(cache_root.glob(f"**/{dgl_model_name}*/**/model.json"))
if not candidates:
    # fallback: glob by fragment
    candidates = sorted(cache_root.glob("**/model.json"))
    candidates = [p for p in candidates if dgl_model_name.split("-")[3] in str(p)]
if not candidates:
    raise FileNotFoundError(f"Cannot find model.json for {dgl_model_name} under {cache_root}")
model_json_path = candidates[0]
print(f"  model.json: {model_json_path}")
with open(model_json_path) as f:
    cfg = json.load(f)
init_args = cfg["kwargs"]["model"]["init_args"]
print(f"  DGL params: {sum(p.numel() for p in dgl_model.parameters()):,}")
print(f"  Element types: {len(init_args['element_types'])} elements")

# ── 2. Build PyG model with same init_args ─────────────────────────────────────
os.environ["MATGL_BACKEND"] = "PYG"
from matgl.models._chgnet_pyg import CHGNet as CHGNetPyG

pyg_init = {k: v for k, v in init_args.items() if k != "error_handling"}
pyg_init["element_types"] = tuple(pyg_init["element_types"])
pyg_init["atom_conv_hidden_dims"] = tuple(pyg_init["atom_conv_hidden_dims"])
pyg_init["bond_update_hidden_dims"] = tuple(pyg_init["bond_update_hidden_dims"])
pyg_init["bond_conv_hidden_dims"] = tuple(pyg_init["bond_conv_hidden_dims"])
pyg_init["angle_update_hidden_dims"] = tuple(pyg_init["angle_update_hidden_dims"])
pyg_init["final_hidden_dims"] = tuple(pyg_init["final_hidden_dims"])

pyg_model = CHGNetPyG(**pyg_init)
pyg_total = sum(p.numel() for p in pyg_model.parameters())
dgl_total = sum(p.numel() for p in dgl_model.parameters())
print(f"  PyG params: {pyg_total:,}")
if pyg_total != dgl_total:
    print(f"  INFO: DGL has {dgl_total - pyg_total:,} extra params (e.g. graph norms not in PyG) — OK")
print("  ✓ Architecture initialised")

# ── 3. Transfer weights: remap DGL key structure to PyG key structure ──────────
def remap_key(k: str) -> str:
    # conv_layer → conv  (block attribute rename)
    k = k.replace(".conv_layer.", ".conv.")
    # GatedMLPNorm: DGL .layers (inner MLPNorm) → PyG .value (inner _MLPNorm)
    #               DGL .gates  (inner MLPNorm) → PyG .gate  (inner _MLPNorm)
    # Do norm_layers → norms before layers → value to avoid double substitution.
    k = k.replace(".layers.norm_layers.", ".value.norms.")
    k = k.replace(".gates.norm_layers.", ".gate.norms.")
    k = k.replace(".layers.layers.", ".value.layers.")
    k = k.replace(".gates.layers.", ".gate.layers.")
    return k

remapped_sd = {remap_key(k): v for k, v in dgl_sd.items()}

# Verify key coverage
pyg_sd = pyg_model.state_dict()
dgl_only = set(remapped_sd.keys()) - set(pyg_sd.keys())   # DGL-only (e.g. atom_norm, bond_norm)
pyg_only = set(pyg_sd.keys()) - set(remapped_sd.keys())   # PyG-only (missing from DGL)
if dgl_only:
    print(f"  INFO: {len(dgl_only)} DGL-only keys skipped (e.g. graph norms not in PyG)")
if pyg_only:
    print(f"  ERROR: PyG keys with no DGL source: {pyg_only}")
    raise RuntimeError("PyG has parameter keys not covered by DGL — check remap_key()")

# Shape check for matched keys
shape_mismatches = []
for k in remapped_sd:
    if k in pyg_sd and remapped_sd[k].shape != pyg_sd[k].shape:
        shape_mismatches.append(f"  {k}: DGL {list(remapped_sd[k].shape)} vs PyG {list(pyg_sd[k].shape)}")
if shape_mismatches:
    print("  SHAPE MISMATCHES:")
    for s in shape_mismatches:
        print(s)
    raise RuntimeError("Shape mismatch prevents weight transfer")

# Only load keys PyG expects (DGL extras like atom_norm are silently dropped)
filtered_sd = {k: v for k, v in remapped_sd.items() if k in pyg_sd}
pyg_model.load_state_dict(filtered_sd, strict=True)
print(f"  ✓ Weights transferred successfully ({len(filtered_sd)} keys)")

# ── 4. Save PyG model ──────────────────────────────────────────────────────────
from matgl.apps._pes_pyg import Potential as PyGPotential

data_mean = dgl_potential.data_mean
data_std = dgl_potential.data_std
element_refs_tensor = dgl_potential.element_refs.property_offset if dgl_potential.element_refs is not None else None

pyg_potential = PyGPotential(
    model=pyg_model,
    data_mean=data_mean,
    data_std=data_std,
    element_refs=element_refs_tensor,
    calc_forces=True,
    calc_stresses=calc_stresses,
    calc_hessian=False,
    calc_magmom=calc_magmom,
)

save_path = Path.home() / pyg_model_name
save_path.mkdir(exist_ok=True)
pyg_potential.save(str(save_path))
print(f"  ✓ PyG Potential saved to {save_path}")
print(f"  Load later: matgl.load_model('{save_path}')")
print(f"  Push to Hub: pot.push_to_hub('materialyze/{pyg_model_name}')")

# ── 5. Reload via matgl.load_model (the intended public API) ─────────────────
pyg_pot_loaded = matgl.load_model(str(save_path))
pyg_pot_loaded.eval()
print("  ✓ PyG Potential reloaded via matgl.load_model successfully")

# ── 6. Ionic relaxation sanity check ─────────────────────────────────────────
from pymatgen.core import Structure, Lattice
from matgl.ext._ase_pyg import Relaxer

print("\nRunning ionic relaxation on Li2O ...")
li2o = Structure(
    Lattice.cubic(4.6),
    ["Li", "Li", "O"],
    [[0.25, 0.25, 0.25], [0.75, 0.75, 0.75], [0.0, 0.0, 0.0]],
)
relaxer = Relaxer(potential=pyg_pot_loaded, relax_cell=False)
result = relaxer.relax(li2o, fmax=0.1, steps=50)
print(f"  ✓ Relaxation done. Final energy: {result['trajectory'].energies[-1]:.4f} eV")

# ── 7. Compare predictions with DGL on 100 MatPES structures ─────────────────
print("\n" + "="*60)
print(f"Comparing DGL vs PyG predictions on 100 structures")
print("="*60)

from matgl.ext._pymatgen_dgl import Structure2Graph as DGLStructure2Graph
from matgl.ext._pymatgen_pyg import Structure2Graph as PyGStructure2Graph

elem_types = init_args["element_types"]
dgl_conv = DGLStructure2Graph(element_types=elem_types, cutoff=init_args["cutoff"])
pyg_conv = PyGStructure2Graph(element_types=elem_types, cutoff=init_args["cutoff"])

dgl_potential.eval()
pyg_pot_loaded.eval()

# Try loading from MatPES dataset or use pymatgen test structures
try:
    from monty.serialization import loadfn

    dataset_candidates = [
        "/data/bdeng/project/MatPES/data/2024.11.18/2024_11_18_MatPES-20240214-r2SCAN-training-data.json.gz",
        "/data/bdeng/project/MatPES/data/matpes_r2scan_test.json",
        "/data/bdeng/project/MatPES/data/2024.11.18/2024_11_18_MatPES-20240214-PBE-training-data.json.gz",
        "/data/bdeng/project/MatPES/data/matpes_pbe_test.json",
    ]
    structures_data = None
    for ds_path in dataset_candidates:
        if os.path.exists(ds_path):
            print(f"  Loading dataset from {ds_path} ...")
            entries = loadfn(ds_path)
            if isinstance(entries, dict):
                entry_list = list(entries.values())
            else:
                entry_list = list(entries)
            n_total = len(entry_list)
            test_entries = entry_list[int(n_total * 0.95): int(n_total * 0.95) + 100]
            structures_data = [(e["structure"], e.get("energy"), e.get("forces"), e.get("stress")) for e in test_entries]
            print(f"  Got {len(structures_data)} test structures")
            break
    if structures_data is None:
        raise FileNotFoundError("No MatPES dataset found")

except (ImportError, FileNotFoundError, KeyError) as exc:
    print(f"  MatPES dataset not available ({exc}), using pymatgen built-in structures")
    test_structures = [
        Structure(Lattice.cubic(4.0), ["Li", "Li", "O"], [[0.25,0.25,0.25],[0.75,0.75,0.75],[0,0,0]]),
        Structure(Lattice.cubic(3.9), ["Fe", "O"], [[0,0,0],[0.5,0.5,0.5]]),
        Structure(Lattice.cubic(4.2), ["Ni", "O"], [[0,0,0],[0.5,0.5,0.5]]),
        Structure(Lattice.cubic(5.0), ["Cu", "Cu"], [[0,0,0],[0.5,0.5,0.5]]),
        Structure(Lattice.cubic(3.5), ["Fe"], [[0,0,0]]),
    ]
    structures_data = [(s, None, None, None) for s in test_structures]

# Run comparison
e_diffs, f_diffs, s_diffs, m_diffs = [], [], [], []
n_success = 0

with torch.no_grad():
    for i, (structure, true_e, true_f, true_s) in enumerate(structures_data):
        try:
            # DGL forward
            g_dgl, lat_dgl, _ = dgl_conv.get_graph(structure)
            g_dgl.edata["pbc_offshift"] = torch.matmul(g_dgl.edata["pbc_offset"], lat_dgl[0])
            g_dgl.ndata["pos"] = g_dgl.ndata["frac_coords"] @ lat_dgl[0]
            with torch.enable_grad():
                dgl_out = dgl_potential(g_dgl, lat_dgl)
            e_dgl = dgl_out[0].detach().item()
            f_dgl = dgl_out[1].detach().numpy()
            s_dgl = dgl_out[2].detach().numpy() if dgl_out[2] is not None else None
            m_dgl = dgl_out[-1].detach().numpy() if len(dgl_out) >= 5 and dgl_out[-1] is not None else None

            # PyG forward
            g_pyg, lat_pyg, _ = pyg_conv.get_graph(structure)
            g_pyg.pbc_offshift = torch.matmul(g_pyg.pbc_offset, lat_pyg[0])
            g_pyg.pos = g_pyg.frac_coords @ lat_pyg[0]
            with torch.enable_grad():
                pyg_out = pyg_pot_loaded(g_pyg, lat_pyg)
            e_pyg = pyg_out[0].detach().item()
            f_pyg = pyg_out[1].detach().numpy()
            s_pyg = pyg_out[2].detach().numpy() if pyg_out[2] is not None else None
            m_pyg = pyg_out[-1].detach().numpy() if len(pyg_out) >= 5 and pyg_out[-1] is not None else None

            e_diffs.append(abs(e_dgl - e_pyg))
            f_diffs.append(np.mean(np.abs(f_dgl - f_pyg)))
            if s_dgl is not None and s_pyg is not None:
                s_diffs.append(np.mean(np.abs(s_dgl - s_pyg)))
            if m_dgl is not None and m_pyg is not None:
                m_diffs.append(np.mean(np.abs(m_dgl - m_pyg)))

            n_success += 1
            if (i + 1) % 20 == 0:
                print(f"  [{i+1:3d}] E_diff={e_diffs[-1]:.2e} eV, F_diff={f_diffs[-1]:.2e} eV/Å")

        except Exception as exc:
            import traceback
            print(f"  [{i:3d}] SKIP: {exc}")
            if i == 0:
                traceback.print_exc()

print(f"\n{'='*60}")
print(f"Structures compared: {n_success}")
if e_diffs:
    print(f"Energy MAE (DGL vs PyG): {np.mean(e_diffs)*1000:.4f} meV")
    print(f"Energy max (DGL vs PyG): {np.max(e_diffs)*1000:.4f} meV")
if f_diffs:
    print(f"Force  MAE (DGL vs PyG): {np.mean(f_diffs)*1000:.4f} meV/Å")
if s_diffs:
    print(f"Stress MAE (DGL vs PyG): {np.mean(s_diffs)*1000:.4f} meV/Å³")
if m_diffs:
    print(f"Magmom MAE (DGL vs PyG): {np.mean(m_diffs):.6f} μB")
print(f"{'='*60}")
