#!/usr/bin/env python
"""Relax 10 perturbed MatPES structures with DGL and PyG CHGNet, compare results.

Usage:
    python compare_relaxation_dgl_pyg.py

For each of 10 randomly chosen test structures, applies a 0.1 Å random perturbation,
then relaxes with all four models: DGL-r2SCAN, PyG-r2SCAN, DGL-PBE, PyG-PBE.
Reports final energies and structure RMSD between DGL and PyG for each functional.
"""
from __future__ import annotations

import os
import sys
import random

os.environ["MATGL_BACKEND"] = "DGL"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import torch
from pathlib import Path

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import matgl
from matgl.ext._ase_dgl import Relaxer as DGLRelaxer

# ── 1. Load all four models and move to GPU ───────────────────────────────────
print(f"Device: {DEVICE}")
print("Loading DGL models ...")
dgl_r2scan = matgl.load_model("CHGNet-PES-MatPES-r2SCAN-2025.2.10").to(DEVICE)
dgl_pbe    = matgl.load_model("CHGNet-PES-MatPES-PBE-2025.2.10").to(DEVICE)
dgl_r2scan.eval(); dgl_pbe.eval()
print(f"  DGL r2SCAN: {sum(p.numel() for p in dgl_r2scan.model.parameters()):,} params  device={next(dgl_r2scan.parameters()).device}")
print(f"  DGL PBE   : {sum(p.numel() for p in dgl_pbe.model.parameters()):,} params  device={next(dgl_pbe.parameters()).device}")

# Switch env before importing PyG-specific modules
os.environ["MATGL_BACKEND"] = "PYG"
from matgl.ext._ase_pyg import Relaxer as PyGRelaxer

print("Loading PyG models ...")
pyg_r2scan = matgl.load_model(str(Path.home() / "CHGNet-PyG-MatPES-r2SCAN-2025.2.10")).to(DEVICE)
pyg_pbe    = matgl.load_model(str(Path.home() / "CHGNet-PyG-MatPES-PBE-2025.2.10")).to(DEVICE)
pyg_r2scan.eval(); pyg_pbe.eval()
print(f"  PyG r2SCAN: {sum(p.numel() for p in pyg_r2scan.model.parameters()):,} params  device={next(pyg_r2scan.parameters()).device}")
print(f"  PyG PBE   : {sum(p.numel() for p in pyg_pbe.model.parameters()):,} params  device={next(pyg_pbe.parameters()).device}")

# ── 2. Build Relaxers ─────────────────────────────────────────────────────────
FMAX  = 0.01
STEPS = 1000
relaxers = {
    "DGL-r2SCAN": DGLRelaxer(potential=dgl_r2scan, optimizer="FIRE", relax_cell=True),
    "DGL-PBE"   : DGLRelaxer(potential=dgl_pbe,    optimizer="FIRE", relax_cell=True),
    "PyG-r2SCAN": PyGRelaxer(potential=pyg_r2scan,  optimizer="FIRE", relax_cell=True),
    "PyG-PBE"   : PyGRelaxer(potential=pyg_pbe,     optimizer="FIRE", relax_cell=True),
}

# ── 3. Load 10 random MatPES test structures ──────────────────────────────────
from pymatgen.core import Structure, Lattice

try:
    from monty.serialization import loadfn
    candidates = [
        "/data/bdeng/project/MatPES/data/2024.11.18/2024_11_18_MatPES-20240214-r2SCAN-training-data.json.gz",
        "/data/bdeng/project/MatPES/data/matpes_r2scan_test.json",
    ]
    entries = None
    for path in candidates:
        if os.path.exists(path):
            print(f"\nLoading MatPES dataset from {path} ...")
            raw = loadfn(path)
            entries = list(raw.values()) if isinstance(raw, dict) else list(raw)
            break
    if entries is None:
        raise FileNotFoundError("No MatPES dataset found")
    n = len(entries)
    # pick 10 random structures from the last 5% (test split)
    rng = random.Random(42)
    test_pool = entries[int(n * 0.95):]
    chosen = rng.sample(test_pool, 10)
    structures = [e["structure"] for e in chosen]
    print(f"  Selected 10 structures from {len(test_pool)}-entry test split")
except (ImportError, FileNotFoundError) as exc:
    print(f"  MatPES not available ({exc}), using 10 synthetic structures")
    structures = [
        Structure(Lattice.cubic(4.0), ["Li", "Li", "O"], [[0.25,0.25,0.25],[0.75,0.75,0.75],[0,0,0]]),
        Structure(Lattice.cubic(3.9), ["Fe", "O"], [[0,0,0],[0.5,0.5,0.5]]),
        Structure(Lattice.cubic(4.2), ["Ni", "O"], [[0,0,0],[0.5,0.5,0.5]]),
        Structure(Lattice.cubic(5.0), ["Cu", "Cu"], [[0,0,0],[0.5,0.5,0.5]]),
        Structure(Lattice.cubic(3.5), ["Fe"], [[0,0,0]]),
        Structure(Lattice.cubic(4.0), ["Mn", "O"], [[0,0,0],[0.5,0.5,0.5]]),
        Structure(Lattice.cubic(4.5), ["Co", "O"], [[0,0,0],[0.5,0.5,0.5]]),
        Structure(Lattice.cubic(4.0), ["Mg", "O"], [[0,0,0],[0.5,0.5,0.5]]),
        Structure(Lattice.cubic(3.8), ["Ti", "O"], [[0,0,0],[0.5,0.5,0.5]]),
        Structure(Lattice.cubic(4.1), ["Zn", "O"], [[0,0,0],[0.5,0.5,0.5]]),
    ]

# ── 4. Perturb structures ─────────────────────────────────────────────────────
PERTURB_DIST = 0.1  # Å
rng_np = np.random.default_rng(42)

perturbed = []
for s in structures:
    p = s.copy()
    p.perturb(PERTURB_DIST)
    perturbed.append(p)
print(f"\nPerturbed {len(perturbed)} structures by up to {PERTURB_DIST} Å")

# ── 5. Relax each structure with all four models ───────────────────────────────

def struct_rmsd(s1: Structure, s2: Structure) -> float:
    """Max Cartesian displacement between matching sites (Å)."""
    c1 = np.array([s1.lattice.get_cartesian_coords(s.frac_coords) for s in s1])
    c2 = np.array([s2.lattice.get_cartesian_coords(s.frac_coords) for s in s2])
    return float(np.sqrt(np.mean(np.sum((c1 - c2)**2, axis=1))))


results = {name: [] for name in relaxers}

print(f"\nRelaxing with fmax={FMAX}, steps={STEPS} ...")
print(f"{'Struct':>8}  {'Model':>12}  {'E/atom(eV)':>12}  {'Steps':>6}  {'Converged':>10}")
print("-" * 60)

for i, struct in enumerate(perturbed):
    natoms = len(struct)
    for name, relaxer in relaxers.items():
        try:
            result = relaxer.relax(struct, fmax=FMAX, steps=STEPS)
            traj = result["trajectory"]
            final_e = traj.energies[-1] / natoms
            n_steps = len(traj.energies) - 1
            converged = n_steps < STEPS
            final_struct = result["final_structure"]
            results[name].append({
                "energy": final_e,
                "steps": n_steps,
                "converged": converged,
                "structure": final_struct,
            })
            print(f"  [{i:2d}]  {name:>12}  {final_e:12.6f}  {n_steps:6d}  {'YES' if converged else 'NO':>10}")
        except Exception as exc:
            print(f"  [{i:2d}]  {name:>12}  FAILED: {exc}")
            results[name].append(None)

# ── 6. Compare DGL vs PyG per functional ──────────────────────────────────────
print("\n" + "="*70)
print("DGL vs PyG comparison")
print("="*70)

for functional in ("r2SCAN", "PBE"):
    dgl_key = f"DGL-{functional}"
    pyg_key = f"PyG-{functional}"
    dgl_res = results[dgl_key]
    pyg_res = results[pyg_key]

    e_diffs, rmsds, step_diffs = [], [], []
    for i, (d, p) in enumerate(zip(dgl_res, pyg_res)):
        if d is None or p is None:
            continue
        e_diff = abs(d["energy"] - p["energy"]) * 1000  # meV/atom
        rmsd = struct_rmsd(d["structure"], p["structure"]) * 1000  # mÅ
        e_diffs.append(e_diff)
        rmsds.append(rmsd)
        step_diffs.append(abs(d["steps"] - p["steps"]))

    print(f"\n  {functional}: {len(e_diffs)}/10 structures compared")
    if e_diffs:
        print(f"  Energy MAE   (DGL vs PyG): {np.mean(e_diffs):.4f} meV/atom  (max {np.max(e_diffs):.4f})")
        print(f"  Struct RMSD  (DGL vs PyG): {np.mean(rmsds):.2f} mÅ          (max {np.max(rmsds):.2f})")
        print(f"  Steps diff   (DGL vs PyG): {np.mean(step_diffs):.1f} steps avg")
        if np.max(e_diffs) < 1.0:
            print(f"  ✓ PASS — energy difference < 1 meV/atom")
        else:
            print(f"  ✗ FAIL — energy difference >= 1 meV/atom")

print("\n" + "="*70)
