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
"""Compare forward/backward/double-backward between matgl-main and current TensorNet."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import torch
from pymatgen.core import Structure
from torch_geometric.data import Batch

DEFAULT_MATGL_MAIN_PATH = str(Path(__file__).parent.parent / "matgl-main" / "src")

BATCH_SIZE = 13

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


def clear_matgl_modules() -> None:
    """Remove all matgl modules from sys.modules."""
    for mod in [k for k in sys.modules if k.startswith("matgl")]:
        del sys.modules[mod]


def print_section(title: str) -> None:
    """Print a formatted section header."""
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def load_structure(path: str) -> Structure:
    """Load structure from file."""
    return Structure.from_file(path)


def get_element_types(structure: Structure) -> tuple[str, ...]:
    """Extract sorted unique element symbols."""
    return tuple(sorted({site.species_string for site in structure}))


def build_graph(
    converter: Any,
    structure: Structure,
    device: torch.device,
    compute_bond: Any = None,
    requires_grad: bool = False,
) -> Any:
    """Build graph from structure with optional gradient tracking."""
    graph, lat, _ = converter.get_graph(structure)
    pos = graph.frac_coords @ lat[0]
    graph.pos = pos.clone().detach().requires_grad_(requires_grad) if requires_grad else pos
    graph.pbc_offshift = graph.pbc_offset @ lat[0]

    if compute_bond is not None:
        bond_vec, bond_dist = compute_bond(graph)
        graph.bond_vec = bond_vec
        graph.bond_dist = bond_dist

    return graph.to(device)


def build_batched_graph(
    converter: Any,
    structure: Structure,
    device: torch.device,
    compute_bond: Any = None,
    requires_grad: bool = False,
    batch_size: int = BATCH_SIZE,
) -> Any:
    """Build batched graph by repeating the same structure multiple times."""
    graphs = []
    for _ in range(batch_size):
        graph, lat, _ = converter.get_graph(structure)
        pos = graph.frac_coords @ lat[0]
        graph.pos = pos.clone().detach().requires_grad_(requires_grad) if requires_grad else pos.clone()
        graph.pbc_offshift = (graph.pbc_offset @ lat[0]).clone()

        if compute_bond is not None:
            bond_vec, bond_dist = compute_bond(graph)
            graph.bond_vec = bond_vec.clone()
            graph.bond_dist = bond_dist.clone()

        # Clone all tensor attributes to ensure independence
        for key in list(graph.keys()):
            val = graph[key]
            if isinstance(val, torch.Tensor):
                graph[key] = val.clone()

        graphs.append(graph)

    batched = Batch.from_data_list(graphs)
    return batched.to(device)


def compare_tensors(name: str, t1: torch.Tensor, t2: torch.Tensor, atol: float = 1e-6) -> bool:
    """Compare two tensors element-wise."""
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
    """Compare model weights with distance_proj layer remapping."""
    print_section("Weight Comparison")

    ref_sd, cur_sd = ref_model.state_dict(), cur_model.state_dict()
    all_match = True

    # Handle distance_proj layers
    print("--- distance_proj ---")
    dp_keys = [f"tensor_embedding.distance_proj{i}" for i in range(1, 4)]
    skip = set()

    if f"{dp_keys[0]}.weight" in ref_sd:
        # Reference has separate distance_proj1/2/3 -> merge and compare
        ref_w = torch.cat([ref_sd[f"{k}.weight"] for k in dp_keys], dim=0)
        ref_b = torch.cat([ref_sd[f"{k}.bias"] for k in dp_keys], dim=0)
        skip = {f"{k}.{p}" for k in dp_keys for p in ("weight", "bias")}

        all_match &= compare_tensors("weight", ref_w, cur_sd["tensor_embedding.distance_proj.weight"])
        all_match &= compare_tensors("bias", ref_b, cur_sd["tensor_embedding.distance_proj.bias"])
    elif "tensor_embedding.distance_proj.weight" in ref_sd:
        # Reference has merged distance_proj -> compare directly
        skip = {"tensor_embedding.distance_proj.weight", "tensor_embedding.distance_proj.bias"}

        all_match &= compare_tensors(
            "weight",
            ref_sd["tensor_embedding.distance_proj.weight"],
            cur_sd["tensor_embedding.distance_proj.weight"],
        )
        all_match &= compare_tensors(
            "bias",
            ref_sd["tensor_embedding.distance_proj.bias"],
            cur_sd["tensor_embedding.distance_proj.bias"],
        )
    else:
        print("  WARNING: distance_proj not found in reference model")

    print("\n--- Other Parameters ---")

    for key in sorted(cur_sd):
        if "distance_proj" in key:
            continue
        if key in ref_sd:
            all_match &= compare_tensors(key, ref_sd[key], cur_sd[key])
        else:
            print(f"  {key}: NOT IN REFERENCE")

    for key in sorted(ref_sd):
        if key not in skip and key not in cur_sd and "distance_proj" not in key:
            print(f"  {key}: IN REFERENCE ONLY")
            all_match = False

    print(f"\n{'=' * 70}\nResult: {'ALL MATCH' if all_match else 'MISMATCH'}")
    return all_match


def compare_forward(
    ref_model: Any, cur_model: Any, ref_graph: Any, cur_graph: Any, device: torch.device, batch_size: int = BATCH_SIZE
) -> bool:
    """Compare forward pass energy predictions for batched graphs."""
    print_section("Forward Pass (Batched)")

    ref_model.eval()
    cur_model.eval()
    state_attr = torch.tensor([[0.0, 0.0]] * batch_size, device=device)

    ref_e = ref_model(g=ref_graph, state_attr=state_attr)
    cur_e = cur_model(g=cur_graph, state_attr=state_attr)

    print(f"Reference energies: {ref_e.detach().cpu().numpy()}")
    print(f"Current energies:   {cur_e.detach().cpu().numpy()}")

    diff = (ref_e - cur_e).abs()
    print(f"Diff:      max={diff.max():.2e}, mean={diff.mean():.2e}")

    match = diff.max().item() < 1e-5
    print(f"Result:    {'PASS' if match else 'FAIL'}")
    return match


def compare_backward(
    ref_model: Any, cur_model: Any, ref_graph: Any, cur_graph: Any, device: torch.device, batch_size: int = BATCH_SIZE
) -> bool:
    """Compare forces (F = -dE/dpos) for batched graphs."""
    print_section("Backward Pass (Forces, Batched)")

    ref_model.train()
    cur_model.train()
    state_attr = torch.tensor([[0.0, 0.0]] * batch_size, device=device)

    def get_forces(model, graph):
        energy = model(g=graph, state_attr=state_attr)
        # Sum energies to get scalar for gradient
        total_energy = energy.sum()
        return -torch.autograd.grad(total_energy, graph.pos, create_graph=True)[0]

    ref_f = get_forces(ref_model, ref_graph)
    cur_f = get_forces(cur_model, cur_graph)

    print(f"Reference: mean={ref_f.mean():.6f}, std={ref_f.std():.6f}")
    print(f"Current:   mean={cur_f.mean():.6f}, std={cur_f.std():.6f}")

    diff = (ref_f - cur_f).abs()
    print(f"Diff:      max={diff.max():.2e}, mean={diff.mean():.2e}")

    match = diff.max().item() < 1e-5
    print(f"Result:    {'PASS' if match else 'FAIL'}")
    return match


def compare_double_backward(
    ref_model: Any, cur_model: Any, ref_graph: Any, cur_graph: Any, device: torch.device, batch_size: int = BATCH_SIZE
) -> bool:
    """Compare position gradients via loss = sum(forces^2) for batched graphs."""
    print_section("Double Backward (Position Gradients, Batched)")

    ref_model.train()
    cur_model.train()
    state_attr = torch.tensor([[0.0, 0.0]] * batch_size, device=device)

    ref_graph.pos.retain_grad()
    cur_graph.pos.retain_grad()

    # Reference
    ref_energy = ref_model(g=ref_graph, state_attr=state_attr)
    ref_total_energy = ref_energy.sum()
    ref_forces = torch.autograd.grad(ref_total_energy, ref_graph.pos, create_graph=True)[0]
    ref_loss = (ref_forces * ref_forces).sum()
    ref_loss.backward()
    ref_pos_grad = ref_graph.pos.grad.clone()

    # Current
    cur_energy = cur_model(g=cur_graph, state_attr=state_attr)
    cur_total_energy = cur_energy.sum()
    cur_forces = torch.autograd.grad(cur_total_energy, cur_graph.pos, create_graph=True)[0]
    cur_loss = (cur_forces * cur_forces).sum()
    cur_loss.backward()
    cur_pos_grad = cur_graph.pos.grad.clone()

    forces_diff = (ref_forces - cur_forces).abs()
    print(f"Forces:    max_diff={forces_diff.max():.2e}, mean_diff={forces_diff.mean():.2e}")

    print(f"Reference pos.grad: mean={ref_pos_grad.mean():.6f}, std={ref_pos_grad.std():.6f}")
    print(f"Current pos.grad:   mean={cur_pos_grad.mean():.6f}, std={cur_pos_grad.std():.6f}")

    if ref_pos_grad.abs().max() < 1e-10 or cur_pos_grad.abs().max() < 1e-10:
        print("WARNING: Position gradient is nearly zero")

    diff = (ref_pos_grad - cur_pos_grad).abs()
    print(f"Diff:      max={diff.max():.2e}, mean={diff.mean():.2e}")

    match = diff.max().item() < 1e-4
    print(f"Result:    {'PASS' if match else 'FAIL'}")
    return match


def compare_param_gradients(
    ref_model: Any, cur_model: Any, ref_graph: Any, cur_graph: Any, device: torch.device, batch_size: int = BATCH_SIZE
) -> bool:
    """Compare gradients on all model parameters after double backward (forces loss)."""
    print_section("Parameter Gradients (Double Backward, Batched)")

    ref_model.train()
    cur_model.train()
    state_attr = torch.tensor([[0.0, 0.0]] * batch_size, device=device)

    # Zero gradients
    ref_model.zero_grad()
    cur_model.zero_grad()

    # Double backward: compute forces, then loss = sum(forces^2)
    # Reference
    ref_energy = ref_model(g=ref_graph, state_attr=state_attr)
    ref_total_energy = ref_energy.sum()
    ref_forces = torch.autograd.grad(ref_total_energy, ref_graph.pos, create_graph=True)[0]
    ref_loss = (ref_forces * ref_forces).sum()
    ref_loss.backward()

    # Current
    cur_energy = cur_model(g=cur_graph, state_attr=state_attr)
    cur_total_energy = cur_energy.sum()
    cur_forces = torch.autograd.grad(cur_total_energy, cur_graph.pos, create_graph=True)[0]
    cur_loss = (cur_forces * cur_forces).sum()
    cur_loss.backward()

    print(f"Reference loss: {ref_loss.item():.6f}")
    print(f"Current loss:   {cur_loss.item():.6f}")

    # Build mapping for distance_proj layers (merged in current, separate in reference)
    ref_sd = dict(ref_model.named_parameters())
    cur_sd = dict(cur_model.named_parameters())

    all_match = True
    max_diff_overall = 0.0
    mismatched_params = []

    # Handle merged distance_proj layers
    print("--- distance_proj (merged) ---")
    dp_keys = [f"tensor_embedding.distance_proj{i}" for i in range(1, 4)]
    skip_ref_keys = set()
    skip_cur_keys = set()
    for suffix in [".weight", ".bias"]:
        ref_grads = []
        for dp_key in dp_keys:
            key = dp_key + suffix
            if key in ref_sd:
                skip_ref_keys.add(key)
                if ref_sd[key].grad is not None:
                    ref_grads.append(ref_sd[key].grad)

        cur_key = "tensor_embedding.distance_proj" + suffix
        skip_cur_keys.add(cur_key)

        if not ref_grads:
            # Reference doesn't have separate distance_proj layers, compare directly
            ref_key = cur_key
            if ref_key in ref_sd:
                ref_param = ref_sd[ref_key]
                cur_param = cur_sd.get(cur_key)
                if ref_param.grad is None and (cur_param is None or cur_param.grad is None):
                    print(f"  distance_proj{suffix}: NO GRAD (both)")
                elif ref_param.grad is None:
                    print(f"  distance_proj{suffix}: NO GRAD (reference)")
                    all_match = False
                elif cur_param is None or cur_param.grad is None:
                    print(f"  distance_proj{suffix}: NO GRAD (current)")
                    all_match = False
                else:
                    diff = (ref_param.grad - cur_param.grad).abs()
                    max_diff = diff.max().item()
                    max_diff_overall = max(max_diff_overall, max_diff)
                    if max_diff > 5e-5:
                        mismatched_params.append(f"distance_proj{suffix}")
                        all_match = False
                        print(f"  distance_proj{suffix}: DIFF (max={max_diff:.2e})")
                    else:
                        print(f"  distance_proj{suffix}: MATCH (max={max_diff:.2e})")
            else:
                print(f"  distance_proj{suffix}: NOT FOUND IN REFERENCE")
        else:
            # Reference has separate layers, concatenate and compare
            ref_grad = torch.cat(ref_grads, dim=0)
            if cur_key in cur_sd and cur_sd[cur_key].grad is not None:
                cur_grad = cur_sd[cur_key].grad
                if ref_grad.shape == cur_grad.shape:
                    diff = (ref_grad - cur_grad).abs()
                    max_diff = diff.max().item()
                    max_diff_overall = max(max_diff_overall, max_diff)
                    if max_diff > 5e-5:
                        mismatched_params.append(f"distance_proj{suffix}")
                        all_match = False
                        print(f"  distance_proj{suffix}: DIFF (max={max_diff:.2e})")
                    else:
                        print(f"  distance_proj{suffix}: MATCH (max={max_diff:.2e})")
                else:
                    print(f"  distance_proj{suffix}: SHAPE MISMATCH {ref_grad.shape} vs {cur_grad.shape}")
                    all_match = False
            else:
                print(f"  distance_proj{suffix}: NO GRAD (current)")
                all_match = False

    # Compare other parameters
    print("\n--- Other Parameters ---")
    for cur_key, cur_param in cur_sd.items():
        if "distance_proj" in cur_key:
            continue

        if cur_key in ref_sd:
            ref_param = ref_sd[cur_key]
            if ref_param.grad is None and cur_param.grad is None:
                print(f"  {cur_key}: NO GRAD (both)")
                continue
            if ref_param.grad is None:
                print(f"  {cur_key}: NO GRAD (reference)")
                all_match = False
                continue
            if cur_param.grad is None:
                print(f"  {cur_key}: NO GRAD (current)")
                all_match = False
                continue

            if ref_param.grad.shape != cur_param.grad.shape:
                print(f"  {cur_key}: SHAPE MISMATCH {ref_param.grad.shape} vs {cur_param.grad.shape}")
                all_match = False
                continue

            diff = (ref_param.grad - cur_param.grad).abs()
            max_diff = diff.max().item()
            max_diff_overall = max(max_diff_overall, max_diff)

            if max_diff > 5e-5:
                mismatched_params.append(cur_key)
                all_match = False
                print(f"  {cur_key}: DIFF (max={max_diff:.2e}, mean={diff.mean():.2e})")
            else:
                print(f"  {cur_key}: MATCH (max={max_diff:.2e})")
        else:
            print(f"  {cur_key}: NOT IN REFERENCE")

    # Check for params in reference only
    for ref_key in ref_sd:
        if ref_key not in skip_ref_keys and ref_key not in cur_sd:
            print(f"  {ref_key}: IN REFERENCE ONLY")

    print(f"\nMax diff overall: {max_diff_overall:.2e}")
    if mismatched_params:
        print(f"Mismatched params: {mismatched_params}")

    print(f"Result:    {'PASS' if all_match else 'FAIL'}")
    return all_match


def main(structure_path: str, matgl_main_path: str, seed: int = 42, pretrained_path: str | None = None) -> bool:
    """Run all comparison tests between reference and current implementations."""
    print_section("TensorNet Comparison: matgl-main vs Current")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Seed: {seed}, Device: {device}, Batch size: {BATCH_SIZE}")
    print(f"matgl-main path: {matgl_main_path}")
    if pretrained_path:
        print(f"Pretrained model: {pretrained_path}")

    structure = load_structure(structure_path)
    element_types = get_element_types(structure)
    print(f"Structure: {structure_path} ({len(structure)} atoms, elements: {element_types})")

    # Reference model (matgl-main)
    clear_matgl_modules()
    sys.path.insert(0, matgl_main_path)

    from matgl.ext._pymatgen_pyg import Structure2Graph as RefConverter
    from matgl.graph._compute_pyg import compute_pair_vector_and_distance as ref_compute_bond
    from matgl.models._tensornet_pyg import TensorNet as RefTensorNet
    from matgl.utils.io import load_model as ref_load_model

    if pretrained_path:
        # Load pre-trained model (Potential wrapper contains TensorNet)
        ref_potential = ref_load_model(pretrained_path)
        ref_model = ref_potential.model.to(device)
        ref_cutoff = ref_model.cutoff
        ref_element_types = ref_model.element_types
    else:
        model_config = {**MODEL_CONFIG, "element_types": element_types}
        torch.manual_seed(seed)
        ref_model = RefTensorNet(**model_config).to(device)
        ref_cutoff = MODEL_CONFIG["cutoff"]
        ref_element_types = element_types

    ref_converter = RefConverter(element_types=ref_element_types, cutoff=ref_cutoff)

    # Build batched graphs for reference model
    ref_graph = build_batched_graph(ref_converter, structure, device, ref_compute_bond)
    ref_graph_grad = build_batched_graph(ref_converter, structure, device, ref_compute_bond, requires_grad=True)
    ref_graph_grad2 = build_batched_graph(ref_converter, structure, device, ref_compute_bond, requires_grad=True)
    ref_graph_param = build_batched_graph(ref_converter, structure, device, ref_compute_bond, requires_grad=True)

    sys.path.pop(0)

    # Current model (src)
    clear_matgl_modules()

    from matgl.ext._pymatgen_pyg import Structure2Graph as CurConverter
    from matgl.models._tensornet_pyg import TensorNet as CurTensorNet
    from matgl.utils.io import load_model as cur_load_model

    if pretrained_path:
        # Load pre-trained model (Potential wrapper contains TensorNet)
        cur_potential = cur_load_model(pretrained_path)
        cur_model = cur_potential.model.to(device)
        cur_cutoff = cur_model.cutoff
        cur_element_types = cur_model.element_types
    else:
        model_config = {**MODEL_CONFIG, "element_types": element_types}
        torch.manual_seed(seed)
        cur_model = CurTensorNet(**model_config).to(device)
        cur_cutoff = MODEL_CONFIG["cutoff"]
        cur_element_types = element_types

    cur_converter = CurConverter(element_types=cur_element_types, cutoff=cur_cutoff)

    # Build batched graphs for current model
    cur_graph = build_batched_graph(cur_converter, structure, device)
    cur_graph_grad = build_batched_graph(cur_converter, structure, device, requires_grad=True)
    cur_graph_grad2 = build_batched_graph(cur_converter, structure, device, requires_grad=True)
    cur_graph_param = build_batched_graph(cur_converter, structure, device, requires_grad=True)

    print(f"Models: {sum(p.numel() for p in ref_model.parameters())} params each")
    print(f"Batched graph: {ref_graph.num_nodes} nodes, {ref_graph.num_edges} edges")

    # Run comparisons
    results = {
        "Weights": compare_weights(ref_model, cur_model),
        "Forward": compare_forward(ref_model, cur_model, ref_graph, cur_graph, device),
        "Backward": compare_backward(ref_model, cur_model, ref_graph_grad, cur_graph_grad, device),
        "Double Backward": compare_double_backward(ref_model, cur_model, ref_graph_grad2, cur_graph_grad2, device),
        "Param Gradients": compare_param_gradients(ref_model, cur_model, ref_graph_param, cur_graph_param, device),
    }

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
        "--structure",
        "-s",
        required=True,
        help="Path to structure file (any format supported by pymatgen)",
    )
    parser.add_argument(
        "--matgl-main-path",
        default=os.environ.get("MATGL_MAIN_PATH", DEFAULT_MATGL_MAIN_PATH),
        help="Path to matgl-main/src (default: $MATGL_MAIN_PATH or ../matgl-main/src)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--pretrained",
        "-p",
        default=None,
        help="Path to pretrained model directory (e.g., pretrained_models/TensorNet-MatPES-PBE-v2025.1-PES)",
    )

    args = parser.parse_args()
    main(
        structure_path=args.structure,
        matgl_main_path=args.matgl_main_path,
        seed=args.seed,
        pretrained_path=args.pretrained,
    )
