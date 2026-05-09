"""Reference timing for upstream tensorpotential GRACE.

Mirrors ``bench_train_infer.py``'s GRACE workload (LiFePO4 2x2x2 supercell,
batch=4 -> ~224 atoms/batch, 5 A cutoff) but runs the *upstream* gracemaker
(``tensorpotential``, TF-Keras, JIT-compiled) instead of matgl's PyG GRACE.

Two presets are timed:

  * ``LINEAR`` configured to ~213K params (matches matgl's GRACE param
    count) so the per-step compute is roughly comparable.
  * ``FS-small`` -- the canonical user-facing GRACE preset; reported as
    well so the comparison reflects what gracemaker users actually run.

Usage::

    .venv/bin/python scripts/bench/bench_train_infer_tensorpotential.py
"""

from __future__ import annotations

import argparse
import os
import time
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

# Quiet TensorFlow as much as possible before importing.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
# tensorpotential 0.5.9 sets TF_USE_LEGACY_KERAS=1, but tf_keras isn't installed
# in this env. Force standalone keras_3 (default) so tf.keras.optimizers works.
os.environ["TF_USE_LEGACY_KERAS"] = "0"
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # force CPU for parity
warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402  -- env vars must be set before tf import
import tensorflow as tf  # noqa: E402
from pymatgen.core import Lattice, Structure  # noqa: E402
from pymatgen.io.ase import AseAtomsAdaptor  # noqa: E402

tf.get_logger().setLevel("ERROR")
tf.config.experimental.enable_tensor_float_32_execution(False)

from tensorpotential import constants  # noqa: E402
from tensorpotential.data.databuilder import (  # noqa: E402
    GeometricalDataBuilder,
    construct_batches,
)
from tensorpotential.potentials import get_preset, get_preset_settings  # noqa: E402
from tensorpotential.tpmodel import (  # noqa: E402
    ComputeBatchEnergyForcesVirials,
    ComputeStructureEnergyAndForcesAndVirial,
    TPModel,
)

if TYPE_CHECKING:
    from ase import Atoms

# ---------------------------------------------------------------------------
# Same LiFePO4 supercell builder as bench_train_infer.py.
# ---------------------------------------------------------------------------


def build_lifepo4_atoms(supercell: int) -> Atoms:
    lat = Lattice.from_parameters(a=10.32, b=6.0, c=4.69, alpha=90.0, beta=90.0, gamma=90.0)
    species = ["Li", "Fe", "P", "O", "O", "O", "O"]
    coords = [
        [0.0, 0.0, 0.0],
        [0.28, 0.25, 0.97],
        [0.09, 0.25, 0.42],
        [0.10, 0.25, 0.74],
        [0.46, 0.25, 0.21],
        [0.16, 0.05, 0.28],
        [0.16, 0.45, 0.28],
    ]
    s = Structure(lat, species, coords)
    s.make_supercell([supercell, supercell, supercell])
    atoms = AseAtomsAdaptor.get_atoms(s)
    # Synthetic E / F / S so the Reference data builder doesn't choke.
    atoms.info["energy"] = 0.0
    forces = np.zeros((len(atoms), 3))
    stress = np.zeros(6)
    # Attach a SinglePointCalculator-equivalent: ASE will look at .calc.
    from ase.calculators.singlepoint import SinglePointCalculator

    atoms.calc = SinglePointCalculator(atoms, energy=0.0, forces=forces, stress=stress)
    return atoms


# ---------------------------------------------------------------------------
# Build a TPModel from a preset.
# ---------------------------------------------------------------------------


def build_model(preset_name: str, element_map: dict[str, int], cutoff: float, **preset_kwargs: Any) -> TPModel:
    build_fn = get_preset(preset_name)
    # Let preset-supplied rcut win (FS small/medium hard-code rcut=7).
    rcut = preset_kwargs.pop("rcut", cutoff)
    instructor = build_fn(
        element_map=element_map,
        rcut=rcut,
        avg_n_neigh=1.0,
        constant_out_shift=0.0,
        constant_out_scale=1.0,
        atomic_shift_map=None,
        **preset_kwargs,
    )
    instructions = instructor.instruction_dict  # dict[str, TPInstruction]
    model = TPModel(
        instructions=instructions,
        compute_function=ComputeStructureEnergyAndForcesAndVirial(),
        train_function=ComputeBatchEnergyForcesVirials(),
    )
    model.build(float_dtype=tf.float32, jit_compile=True, input_dtype=tf.float32)
    return model


def count_params(model: TPModel) -> int:
    return int(sum(int(np.prod(v.shape)) for v in model.trainable_variables))


# ---------------------------------------------------------------------------
# Build a single batch and convert to TF tensors.
# ---------------------------------------------------------------------------


def build_batch(
    atoms: Atoms,
    element_map: dict[str, int],
    cutoff: float,
    batch_size: int,
) -> dict[str, tf.Tensor]:
    # Only the geometric data builder — labels are injected by the train step.
    # (ReferenceEnergyForcesStressesDataBuilder hits a numpy>=2 bug in
    # tensorpotential 0.5.9's join_to_batch path; we don't need it.)
    builders = [
        GeometricalDataBuilder(elements_map=element_map, cutoff=cutoff, is_fit_stress=True, float_dtype="float32"),
    ]
    atom_list = [atoms.copy() for _ in range(batch_size)]

    batches, _ = construct_batches(
        df_or_ase_atoms_list=atom_list,
        data_builders=builders,
        batch_size=batch_size,
        max_n_buckets=None,
        return_padding_stats=True,
        verbose=False,
    )
    batches = list(batches)
    assert len(batches) == 1, f"expected 1 batch, got {len(batches)}"
    raw = batches[0]
    # Cast to TF tensors with the right dtypes.
    out: dict[str, tf.Tensor] = {}
    for k, v in raw.items():
        arr = np.asarray(v)
        if np.issubdtype(arr.dtype, np.integer):
            out[k] = tf.constant(arr, dtype=tf.int32)
        else:
            out[k] = tf.constant(arr, dtype=tf.float32)
    return out


# ---------------------------------------------------------------------------
# Train / infer step closures.
# ---------------------------------------------------------------------------


def make_train_step(model: TPModel, batch: dict[str, tf.Tensor]) -> Any:
    # Synthetic targets (we're timing, not fitting). Shapes follow what the
    # train_function returns: E -> (n_struct, 1), F -> (n_atoms, 3), V -> (n_struct, 6).
    n_atoms = int(batch[constants.ATOMIC_MU_I].shape[0])
    n_struct = int(batch[constants.N_STRUCTURES_BATCH_TOTAL].numpy())
    e_lab = tf.zeros((n_struct, 1), dtype=tf.float32)
    f_lab = tf.zeros((n_atoms, 3), dtype=tf.float32)
    v_lab = tf.zeros((n_struct, 6), dtype=tf.float32)

    train_vars = list(model.variables_to_train)
    # Hand-rolled Adam — keras 3's Adam doesn't accept raw tf.Variable from
    # tf.Module without wrapping. We just need timing-equivalent state.
    lr = tf.constant(1e-3, dtype=tf.float32)
    beta1 = tf.constant(0.9, dtype=tf.float32)
    beta2 = tf.constant(0.999, dtype=tf.float32)
    eps = tf.constant(1e-8, dtype=tf.float32)
    m_state = [tf.Variable(tf.zeros_like(v), trainable=False) for v in train_vars]
    v_state = [tf.Variable(tf.zeros_like(v), trainable=False) for v in train_vars]
    t_state = tf.Variable(0.0, dtype=tf.float32, trainable=False)

    @tf.function(jit_compile=True)
    def step(input_data: dict[str, tf.Tensor]) -> tf.Tensor:
        with tf.GradientTape() as tape:
            preds = model(input_data, training=True)
            loss = (
                tf.reduce_mean(tf.square(preds[constants.PREDICT_TOTAL_ENERGY] - e_lab))
                + tf.reduce_mean(tf.square(preds[constants.PREDICT_FORCES] - f_lab))
                + tf.reduce_mean(tf.square(preds[constants.PREDICT_VIRIAL] - v_lab))
            )
        grads = tape.gradient(loss, train_vars)
        t_state.assign_add(1.0)
        bias_corr1 = 1.0 - tf.pow(beta1, t_state)
        bias_corr2 = 1.0 - tf.pow(beta2, t_state)
        for var, g, m, v in zip(train_vars, grads, m_state, v_state, strict=True):
            if g is None:
                continue
            m.assign(beta1 * m + (1.0 - beta1) * g)
            v.assign(beta2 * v + (1.0 - beta2) * tf.square(g))
            m_hat = m / bias_corr1
            v_hat = v / bias_corr2
            var.assign_sub(lr * m_hat / (tf.sqrt(v_hat) + eps))
        return loss

    def run_step() -> None:
        _ = step(batch).numpy()

    return run_step


def make_infer_step(model: TPModel, batch: dict[str, tf.Tensor]) -> Any:
    @tf.function(jit_compile=True)
    def step(input_data: dict[str, tf.Tensor]) -> dict[str, tf.Tensor]:
        return model(input_data, training=False)

    def run_step() -> None:
        out = step(batch)
        # Force materialization.
        _ = (
            float(tf.reduce_sum(out[constants.PREDICT_TOTAL_ENERGY])),
            float(tf.reduce_mean(tf.abs(out[constants.PREDICT_FORCES]))),
            float(tf.reduce_mean(tf.abs(out[constants.PREDICT_VIRIAL]))),
        )

    return run_step


# ---------------------------------------------------------------------------
# Timing helpers (same logic as bench_train_infer.py).
# ---------------------------------------------------------------------------


@dataclass
class TimingResult:
    median_ms: float
    p10_ms: float
    p90_ms: float
    n: int


def _percentile(times: list[float], q: float) -> float:
    if not times:
        return float("nan")
    times = sorted(times)
    idx = max(0, min(len(times) - 1, int(q * (len(times) - 1))))
    return times[idx]


def time_loop(fn: Any, *, warmup: int, steps: int) -> TimingResult:
    for _ in range(warmup):
        fn()
    times: list[float] = []
    for _ in range(steps):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1e3)
    return TimingResult(
        median_ms=_percentile(times, 0.5),
        p10_ms=_percentile(times, 0.1),
        p90_ms=_percentile(times, 0.9),
        n=steps,
    )


# ---------------------------------------------------------------------------
# Driver.
# ---------------------------------------------------------------------------


# LINEAR with hyperparameters mirroring matgl's GRACE defaults
# (n_rad_base=8, n_rad_max=144, lmax=3, max_order=3, embedding_size=128).
# Single-block so the upstream params come out smaller — included as the
# closest single-block comparison.
LINEAR_DEFAULT = {
    "n_rad_base": 8,
    "n_rad_max": 144,
    "embedding_size": 128,
    "lmax": 3,
    "max_order": 3,
}


# Canonical user-facing FS presets — what gracemaker users actually run.
# We materialise the size-keyed settings dict from get_preset_settings.
def _fs_kwargs(size: str) -> dict:
    settings = get_preset_settings("FS")
    if settings is None or size not in settings:
        return {}
    return dict(settings[size])


FS_SMALL = _fs_kwargs("small")
FS_MEDIUM = _fs_kwargs("medium")
FS_LARGE = _fs_kwargs("large")


def run(args: argparse.Namespace) -> None:
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    atoms = build_lifepo4_atoms(args.supercell)
    element_symbols = sorted(set(atoms.get_chemical_symbols()))
    element_map = {s: i for i, s in enumerate(element_symbols)}
    cutoff = args.cutoff

    n_atoms = len(atoms) * args.batch_size
    print("=" * 78)
    print("tensorpotential GRACE reference benchmark")
    print("  device       : CPU (TF, jit_compile=True, float32)")
    print(f"  structure    : LiFePO4 {args.supercell}x{args.supercell}x{args.supercell} ({len(atoms)} atoms / cell)")
    print(f"  batch_size   : {args.batch_size}  ->  {n_atoms} atoms/batch")
    print(f"  cutoff       : {cutoff} A")
    print(f"  warmup/steps : {args.warmup}/{args.steps}")
    print(f"  element_map  : {element_map}")
    print("=" * 78)

    print()
    header = f"  {'preset':<22s} {'#params':>10s} {'train ms (med)':>16s} "
    header += f"{'p10':>8s} {'p90':>8s} {'infer ms (med)':>17s} {'p10':>8s} {'p90':>8s}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    presets: list[tuple[str, str, dict]] = [
        ("LINEAR (matgl-like)", "LINEAR", LINEAR_DEFAULT),
        ("FS small", "FS", FS_SMALL),
        ("FS medium", "FS", FS_MEDIUM),
        ("FS large", "FS", FS_LARGE),
    ]

    for label, preset_name, preset_kwargs in presets:
        # FS small/medium set their own rcut; build the matching neighbor list.
        eff_cutoff = float(preset_kwargs.get("rcut", cutoff))
        batch = build_batch(atoms, element_map, eff_cutoff, args.batch_size)
        try:
            model = build_model(preset_name, element_map, eff_cutoff, **preset_kwargs)
        except Exception as exc:
            print(f"  {label:<22s}    FAILED to build: {type(exc).__name__}: {exc}")
            continue

        n_params = count_params(model)

        train_step = make_train_step(model, batch)
        train_t = time_loop(train_step, warmup=args.warmup, steps=args.steps)

        infer_step = make_infer_step(model, batch)
        infer_t = time_loop(infer_step, warmup=args.warmup, steps=args.steps)

        print(
            f"  {label:<22s} {n_params:>10,d}"
            f"   {train_t.median_ms:>13.2f}  {train_t.p10_ms:>7.2f} {train_t.p90_ms:>7.2f}"
            f"   {infer_t.median_ms:>14.2f}  {infer_t.p10_ms:>7.2f} {infer_t.p90_ms:>7.2f}"
        )

        del model, train_step, infer_step, batch

    print()
    print("Notes:")
    print("  - tensorpotential uses tf.function(jit_compile=True) for both train & infer.")
    print("  - 'train ms' = forward + MSE(E,F,V) + backward + Adam.apply_gradients.")
    print("  - 'infer ms' = forward via train_function path (E,F,V via tape over bond vectors).")
    print("  - First call is excluded by the warmup loop, so XLA compile time is amortized.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--supercell", type=int, default=2)
    p.add_argument("--cutoff", type=float, default=5.0)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--steps", type=int, default=15)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
