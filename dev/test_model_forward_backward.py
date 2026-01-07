# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
"""Compare forward/backward/double-backward between matgl-main and current TensorNet."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import torch
from pymatgen.core import Structure


# =============================================================================
# Configuration
# =============================================================================

DEFAULT_MATGL_MAIN_PATH = str(Path(__file__).parent.parent / "matgl-main" / "src")

MODEL_CONFIG = {
    "units": 64,
    "nblocks": 2,
    "num_rbf": 32,
    "cutoff": 5.0,
    "rbf_type": "Gaussian",
    "activation_type": "swish",
    "equivariance_invariance_group": "O(3)",
    "is_intensive": False,
    "ntargets": 1,
}


# =============================================================================
# Utilities
# =============================================================================

def clear_matgl_modules() -> None:
    """Remove all matgl modules from sys.modules."""
    for mod in [k for k in sys.modules if k.startswith("matgl")]:
        del sys.modules[mod]


def print_section(title: str) -> None:
    """Print a section header."""
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def load_structure(path: str) -> Structure:
    """Load structure from file using pymatgen."""
    return Structure.from_file(path)


def get_element_types(structure: Structure) -> tuple[str, ...]:
    """Extract sorted unique element symbols from structure."""
    return tuple(sorted({site.specie.symbol for site in structure}))


def build_graph(
    converter: Any,
    structure: Structure,
    device: torch.device,
    compute_bond: Any = None,
    requires_grad: bool = False,
) -> Any:
    """Build graph from structure."""
    graph, lat, _ = converter.get_graph(structure)
    pos = graph.frac_coords @ lat[0]
    graph.pos = pos.clone().detach().requires_grad_(requires_grad) if requires_grad else pos
    graph.pbc_offshift = graph.pbc_offset @ lat[0]

    if compute_bond is not None:
        bond_vec, bond_dist = compute_bond(graph)
        graph.bond_vec = bond_vec
        graph.bond_dist = bond_dist

    return graph.to(device)


# =============================================================================
# Comparison Functions
# =============================================================================

def compare_tensors(name: str, t1: torch.Tensor, t2: torch.Tensor, atol: float = 1e-6) -> bool:
    """Compare two tensors, return True if matching."""
    if t1.shape != t2.shape:
        print(f"  {name}: SHAPE MISMATCH {t1.shape} vs {t2.shape}")
        return False

    if torch.allclose(t1, t2, atol=atol):
        print(f"  {name}: MATCH")
        return True

    diff = (t1 - t2).abs()
    print(f"  {name}: DIFF (max={diff.max():.2e}, mean={diff.mean():.2e})")
    return False


def compare_weights(ref_model: Any, cur_model: Any) -> bool:
    """Compare model weights, handling distance_proj1/2/3 -> distance_proj mapping."""
    print_section("Weight Comparison")

    ref_sd, cur_sd = ref_model.state_dict(), cur_model.state_dict()
    all_match = True

    # Handle merged distance_proj layers
    dp_keys = [f"tensor_embedding.distance_proj{i}" for i in range(1, 4)]
    if f"{dp_keys[0]}.weight" in ref_sd:
        ref_w = torch.cat([ref_sd[f"{k}.weight"] for k in dp_keys], dim=0)
        ref_b = torch.cat([ref_sd[f"{k}.bias"] for k in dp_keys], dim=0)

        print("\n--- distance_proj (merged) ---")
        all_match &= compare_tensors("weight", ref_w, cur_sd["tensor_embedding.distance_proj.weight"])
        all_match &= compare_tensors("bias", ref_b, cur_sd["tensor_embedding.distance_proj.bias"])

    # Compare remaining parameters
    skip = {f"{k}.{p}" for k in dp_keys for p in ("weight", "bias")}
    print("\n--- Other Parameters ---")

    for key in sorted(cur_sd):
        if "distance_proj" in key:
            continue
        if key in ref_sd:
            all_match &= compare_tensors(key, ref_sd[key], cur_sd[key])
        else:
            print(f"  {key}: NOT IN REFERENCE")

    for key in sorted(ref_sd):
        if key not in skip and key not in cur_sd:
            print(f"  {key}: IN REFERENCE ONLY")
            all_match = False

    print(f"\n{'=' * 70}\nResult: {'ALL MATCH' if all_match else 'MISMATCH'}")
    return all_match


def compare_forward(
    ref_model: Any, cur_model: Any, ref_graph: Any, cur_graph: Any, device: torch.device
) -> bool:
    """Compare forward pass outputs."""
    print_section("Forward Pass")

    ref_model.eval()
    cur_model.eval()
    state_attr = torch.tensor([0.0, 0.0], device=device)

    ref_e = ref_model(g=ref_graph, state_attr=state_attr)
    cur_e = cur_model(g=cur_graph, state_attr=state_attr)
    diff = abs(float(ref_e) - float(cur_e))

    print(f"Reference: {float(ref_e):.10f}")
    print(f"Current:   {float(cur_e):.10f}")
    print(f"Diff:      {diff:.2e}")

    match = diff < 1e-5
    print(f"Result:    {'PASS' if match else 'FAIL'}")
    return match


def compare_backward(
    ref_model: Any, cur_model: Any, ref_graph: Any, cur_graph: Any, device: torch.device
) -> tuple[bool, torch.Tensor, torch.Tensor, Any, Any]:
    """Compare backward pass (forces = -dE/dpos)."""
    print_section("Backward Pass (Forces)")

    ref_model.train()
    cur_model.train()
    state_attr = torch.tensor([0.0, 0.0], device=device)

    def get_forces(model, graph):
        energy = model(g=graph, state_attr=state_attr)
        return -torch.autograd.grad(energy, graph.pos, create_graph=True, retain_graph=True)[0]

    ref_f = get_forces(ref_model, ref_graph)
    cur_f = get_forces(cur_model, cur_graph)

    print(f"Reference: mean={ref_f.mean():.6f}, std={ref_f.std():.6f}")
    print(f"Current:   mean={cur_f.mean():.6f}, std={cur_f.std():.6f}")

    diff = (ref_f - cur_f).abs()
    print(f"Diff:      max={diff.max():.2e}, mean={diff.mean():.2e}")

    match = diff.max().item() < 1e-5
    print(f"Result:    {'PASS' if match else 'FAIL'}")
    return match, ref_f, cur_f, ref_graph, cur_graph


def compare_double_backward(
    ref_forces: torch.Tensor, cur_forces: torch.Tensor, ref_graph: Any, cur_graph: Any
) -> bool:
    """Compare Hessian-vector product: d(F·v)/dpos."""
    print_section("Double Backward (Hessian-Vector Product)")

    torch.manual_seed(123)
    v = torch.randn_like(ref_forces)

    ref_Hv = torch.autograd.grad((ref_forces * v).sum(), ref_graph.pos, retain_graph=True)[0]
    cur_Hv = torch.autograd.grad((cur_forces * v).sum(), cur_graph.pos, retain_graph=True)[0]

    print(f"Reference: mean={ref_Hv.mean():.6f}, std={ref_Hv.std():.6f}")
    print(f"Current:   mean={cur_Hv.mean():.6f}, std={cur_Hv.std():.6f}")

    if ref_Hv.abs().max() < 1e-10 or cur_Hv.abs().max() < 1e-10:
        print("WARNING: Hessian-vector product is nearly zero")

    diff = (ref_Hv - cur_Hv).abs()
    print(f"Diff:      max={diff.max():.2e}, mean={diff.mean():.2e}")

    match = diff.max().item() < 1e-4
    print(f"Result:    {'PASS' if match else 'FAIL'}")
    return match


# =============================================================================
# Main
# =============================================================================

def main(structure_path: str, matgl_main_path: str, seed: int = 42) -> bool:
    """Run all comparison tests."""
    print_section("TensorNet Comparison: matgl-main vs Current")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Seed: {seed}, Device: {device}")
    print(f"matgl-main path: {matgl_main_path}")

    structure = load_structure(structure_path)
    element_types = get_element_types(structure)
    print(f"Structure: {structure_path} ({len(structure)} atoms, elements: {element_types})")

    model_config = {**MODEL_CONFIG, "element_types": element_types}

    # Load reference model (matgl-main)
    clear_matgl_modules()
    sys.path.insert(0, matgl_main_path)

    from matgl.models._tensornet_pyg import TensorNet as RefTensorNet
    from matgl.ext._pymatgen_pyg import Structure2Graph as RefConverter
    from matgl.graph._compute_pyg import compute_pair_vector_and_distance as ref_compute_bond

    torch.manual_seed(seed)
    ref_model = RefTensorNet(**model_config).to(device)
    ref_converter = RefConverter(element_types=element_types, cutoff=MODEL_CONFIG["cutoff"])

    ref_graph = build_graph(ref_converter, structure, device, ref_compute_bond)
    ref_graph_grad = build_graph(ref_converter, structure, device, ref_compute_bond, requires_grad=True)

    sys.path.pop(0)

    # Load current model (src)
    clear_matgl_modules()

    from matgl.models._tensornet_pyg import TensorNet as CurTensorNet
    from matgl.ext._pymatgen_pyg import Structure2Graph as CurConverter

    torch.manual_seed(seed)
    cur_model = CurTensorNet(**model_config).to(device)
    cur_converter = CurConverter(element_types=element_types, cutoff=MODEL_CONFIG["cutoff"])

    cur_graph = build_graph(cur_converter, structure, device)
    cur_graph_grad = build_graph(cur_converter, structure, device, requires_grad=True)

    print(f"Models: {sum(p.numel() for p in ref_model.parameters())} params each")

    # Run comparisons
    results = {
        "Weights": compare_weights(ref_model, cur_model),
        "Forward": compare_forward(ref_model, cur_model, ref_graph, cur_graph, device),
    }

    back_ok, ref_f, cur_f, ref_g, cur_g = compare_backward(
        ref_model, cur_model, ref_graph_grad, cur_graph_grad, device
    )
    results["Backward"] = back_ok
    results["Double Backward"] = compare_double_backward(ref_f, cur_f, ref_g, cur_g)

    # Summary
    print_section("SUMMARY")
    all_pass = all(results.values())
    for name, passed in results.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")

    print(f"\n{'=' * 70}")
    print("ALL TESTS PASSED" if all_pass else "SOME TESTS FAILED")
    print("=" * 70)

    assert all_pass, "Model comparison tests failed"
    return all_pass


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compare TensorNet implementations")
    parser.add_argument(
        "--structure", "-s",
        required=True,
        help="Path to structure file (any format supported by pymatgen)",
    )
    parser.add_argument(
        "--matgl-main-path",
        default=os.environ.get("MATGL_MAIN_PATH", DEFAULT_MATGL_MAIN_PATH),
        help="Path to matgl-main/src (default: $MATGL_MAIN_PATH or ../matgl-main/src)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()
    main(structure_path=args.structure, matgl_main_path=args.matgl_main_path, seed=args.seed)
