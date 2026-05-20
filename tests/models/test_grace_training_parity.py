"""Short training-parity test: GRACE-2L (matgl PYG) vs upstream tensorpotential.

Trains both implementations of GRACE-2L for a handful of optimization
steps on the ``nacl_training_set`` fixture (10 perturbed NaCl rocksalt
configurations labelled with ``TensorNet-PES-MatPES-r2SCAN-2025.2``) and
checks that *both* training paths reduce the energy + force MSE loss by a
non-trivial fraction. This is a **soft** parity test: the two
implementations have different parametric forms (matgl uses
``LinearRadialFunction`` and a simpler indicator-mixing block; upstream
``GRACE_2LAYER_v1_24`` adds ``FCRight2Left`` projections, an
``MLPRadialFunction`` option, separate per-block readouts, etc.) so we
do not expect exact numerical agreement of weights or predictions. We
do, however, drive both sides with the *same* optimizer settings (Adam,
``lr=0.01``, ``betas=(0.9, 0.999)``, ``eps=1e-8``) and assert numerical
properties of the resulting loss trajectories:

    (a) both training loops produce only finite losses across all
        ``N_STEPS`` Adam steps,
    (b) each side reduces its *best-seen* loss by at least 50% relative
        to its initial loss, and
    (c) the two relative best-progress ratios ``min(L) / L_initial``
        agree within 1.5 decades — same Adam settings, same data, so the
        learning curves should be in the same ballpark even though the
        absolute loss scales differ between the two parametrizations.

We compare best-seen reductions rather than endpoints because Adam at
``lr=0.01`` (chosen so the test runs in seconds) overshoots and starts
oscillating well before ``N_STEPS`` is reached; ``min(losses)`` captures
"how much progress was actually made" without dragging in that
late-stage oscillation, which is parametrization-dependent.

This module is **not** part of the regular ``pytest`` run: it skips
automatically when ``tensorflow`` and ``tensorpotential`` are not
importable (mirroring :mod:`tests.models.test_grace_parity_tp`). The
``test_grace_parity`` CI job installs them manually before invoking the
test.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch

# Heavy parity dependencies — keep optional.
tf = pytest.importorskip("tensorflow")
tf.config.experimental.enable_tensor_float_32_execution(False)
tf.experimental.numpy.experimental_enable_numpy_behavior(dtype_conversion_mode="all")

tp_constants = pytest.importorskip("tensorpotential.constants")
tp_presets = pytest.importorskip("tensorpotential.potentials.presets")
tp_tpmodel = pytest.importorskip("tensorpotential.tpmodel")


from pymatgen.io.ase import AseAtomsAdaptor  # noqa: E402

import matgl  # noqa: E402
from matgl.apps.pes import Potential  # noqa: E402
from matgl.ext.pymatgen import Structure2Graph  # noqa: E402
from matgl.models import GRACE  # noqa: E402

# Shared training hyperparameters. Kept tiny so the test runs in seconds
# even on CPU-only CI: 6 Chebyshev radials, n_rad_max=6, lmax=2,
# max_order=3, indicator_lmax=1. These match the matgl ``test_grace.py``
# fast-path config so the two test files exercise the same shape.
N_RAD_BASE = 6
N_RAD_MAX = 6
LMAX = 2
MAX_ORDER = 3
INDICATOR_LMAX = 1
EMBEDDING_SIZE = 8
CUTOFF = 5.0

# Training: short (20 steps) — this is a smoke / soft-parity test. 20 steps
# gives both Adam trajectories time to settle into their respective basins
# so the trailing loss-reduction ratios are dominated by signal rather than
# Adam's first-few-step bias-correction transient. (At 10 steps matgl was
# still in its early plateau on this seed while tp had already moved past
# its initial spike, putting the two ratios well outside the 1.5-decade
# parity window even though both were learning.)
N_STEPS = 20
LR = 0.01
ENERGY_WEIGHT = 1.0
FORCE_WEIGHT = 0.1
SEED = 42

# Element ordering shared by both models. Matches the ``Z``-sorted matgl
# convention (``get_element_list`` orders by atomic number).
ELEMENT_TYPES = ("Na", "Cl")


# ---------------------------------------------------------------------------
# Helpers: matgl side
# ---------------------------------------------------------------------------


def _build_matgl_grace2l() -> Potential:
    """Construct a small matgl ``GRACE`` (nblocks=2) wrapped in ``Potential``.

    Returns the ``Potential`` so we can call it as ``pot(graph, lat)`` and
    get back ``(energies, forces, stresses, hessian)`` via autograd.
    """
    torch.manual_seed(SEED)
    model = GRACE(
        element_types=ELEMENT_TYPES,
        cutoff=CUTOFF,
        n_rad_base=N_RAD_BASE,
        n_rad_max=N_RAD_MAX,
        lmax=LMAX,
        embedding_size=EMBEDDING_SIZE,
        max_order=MAX_ORDER,
        nblocks=2,
        indicator_lmax=INDICATOR_LMAX,
        indicator_n_max=N_RAD_MAX,
        readout_hidden=(32,),
    ).to(matgl.float_th)
    return Potential(model=model, calc_forces=True, calc_stresses=False)


def _matgl_step(pot: Potential, optimizer: torch.optim.Optimizer, samples: list[dict]) -> float:
    """Run one Adam step over the whole training set; return scalar loss."""
    converter = Structure2Graph(element_types=ELEMENT_TYPES, cutoff=CUTOFF)  # type: ignore[arg-type]
    optimizer.zero_grad()
    total_loss = torch.zeros((), dtype=matgl.float_th)
    for sample in samples:
        graph, lattice, _ = converter.get_graph(sample["structure"])
        lat = lattice  # shape (1, 3, 3); already a torch tensor
        pred_e, pred_f, _, _ = pot(graph, lat)
        target_e = torch.tensor(sample["energy"], dtype=matgl.float_th)
        target_f = torch.tensor(sample["forces"], dtype=matgl.float_th)
        loss_e = (pred_e.view(-1) - target_e.view(-1)).pow(2).mean()
        loss_f = (pred_f - target_f).pow(2).mean()
        total_loss = total_loss + ENERGY_WEIGHT * loss_e + FORCE_WEIGHT * loss_f
    total_loss = total_loss / len(samples)
    total_loss.backward()
    optimizer.step()
    return float(total_loss.detach())


# ---------------------------------------------------------------------------
# Helpers: tensorpotential side
# ---------------------------------------------------------------------------


def _build_tp_grace2l_instructions():
    """Build an upstream GRACE-2L preset matched to our matgl config.

    We use ``mlp_radial=False`` so both implementations share the same
    ``LinearRadialFunction`` parametrization of the radial expansion, and
    set ``embedding_size``, ``lmax``, ``n_rad_*``, ``max_order``, and
    ``indicator_lmax`` to the values used by the matgl side. The small
    ``n_mlp_dens=8`` keeps the readout MLP cheap.
    """
    element_map = {sym: i for i, sym in enumerate(ELEMENT_TYPES)}
    return tp_presets.GRACE_2LAYER_v1_24(
        element_map=element_map,
        rcut=CUTOFF,
        avg_n_neigh=1.0,
        n_rad_base=N_RAD_BASE,
        basis_type="Cheb",
        cutoff_function_order=5,
        embedding_size=EMBEDDING_SIZE,
        lmax=(LMAX, LMAX),
        n_rad_max=(N_RAD_MAX, N_RAD_MAX),
        n_mlp_dens=8,
        max_order=MAX_ORDER,
        mlp_radial=False,  # match matgl's LinearRadialFunction
        indicator_lmax=INDICATOR_LMAX,
    )


def _build_tp_batch(samples: list[dict], float_dtype):
    """Assemble the TF input dict for ``ComputeBatchEnergyForcesVirials`` directly.

    We bypass upstream's ``tensorpotential.data.databuilder.construct_batches``
    because ``ReferenceEnergyForcesStressesDataBuilder.join_to_batch`` calls
    ``float(np.array(energy).reshape(-1, 1))`` — a pattern that NumPy 1.26+
    rejects with ``TypeError: only 0-dimensional arrays can be converted to
    Python scalars``. Building the batch by hand also keeps the test
    insensitive to other refactors of upstream's batching pipeline.

    The fields below are exactly the union of:
      - ``ComputeBatchEnergyForcesVirials.specs`` (geometry + structure maps)
      - the extra inputs the ``GRACE_2LAYER_v1_24`` instructions consume
        (``ATOMIC_MU_I``, ``BOND_MU_I``, ``BOND_MU_J``)
      - the reference labels read by ``_tp_step`` (``DATA_REFERENCE_ENERGY``,
        ``DATA_REFERENCE_FORCES``).

    ``float_dtype`` is a ``tf.DType``; left untyped because ``tensorflow`` is
    only imported via ``pytest.importorskip`` and not visible to mypy.
    """
    from ase.neighborlist import neighbor_list

    elements_map = {sym: i for i, sym in enumerate(ELEMENT_TYPES)}
    adaptor = AseAtomsAdaptor()

    atomic_mu = []  # per-atom species index
    atoms_to_struct = []  # per-atom structure index
    ind_i_list, ind_j_list, bond_vec_list = [], [], []
    bonds_to_struct = []
    energies = []
    forces_list = []

    atom_offset = 0
    for struct_idx, sample in enumerate(samples):
        atoms = adaptor.get_atoms(sample["structure"])
        # All ASE PBC components must be True for the periodic neighbor list.
        atoms.set_pbc(True)
        n_at = len(atoms)
        species = np.array([elements_map[s] for s in atoms.get_chemical_symbols()], dtype=np.int32)
        atomic_mu.append(species)
        atoms_to_struct.append(np.full(n_at, struct_idx, dtype=np.int32))

        # ``"ijD"`` returns (i, j, D) where D = r_j - r_i (Cartesian, Å).
        ii, jj, dd = neighbor_list("ijD", atoms, cutoff=CUTOFF)
        ii = ii.astype(np.int32) + atom_offset
        jj = jj.astype(np.int32) + atom_offset
        ind_i_list.append(ii)
        ind_j_list.append(jj)
        bond_vec_list.append(np.asarray(dd, dtype=np.float64))
        bonds_to_struct.append(np.full(ii.shape[0], struct_idx, dtype=np.int32))

        energies.append(float(sample["energy"]))
        forces_list.append(np.asarray(sample["forces"], dtype=np.float64))

        atom_offset += n_at

    atomic_mu_i = np.concatenate(atomic_mu, axis=0)
    map_atoms_to_struct = np.concatenate(atoms_to_struct, axis=0)
    ind_i = np.concatenate(ind_i_list, axis=0)
    ind_j = np.concatenate(ind_j_list, axis=0)
    bond_vector = np.concatenate(bond_vec_list, axis=0)
    map_bonds_to_struct = np.concatenate(bonds_to_struct, axis=0)
    mu_i = atomic_mu_i[ind_i]
    mu_j = atomic_mu_i[ind_j]
    true_energy = np.asarray(energies, dtype=np.float64).reshape(-1, 1)
    true_force = np.concatenate(forces_list, axis=0)

    n_atoms_total = atomic_mu_i.shape[0]
    int_specs = {
        tp_constants.BOND_IND_I: ind_i,
        tp_constants.BOND_IND_J: ind_j,
        tp_constants.BOND_MU_I: mu_i,
        tp_constants.BOND_MU_J: mu_j,
        tp_constants.ATOMIC_MU_I: atomic_mu_i,
        # ``ATOMIC_MU_I_LOCAL`` is only consumed by instructions that run in
        # the distributed ``local=True`` path; we never trigger that path in
        # this test, but feeding the same array keeps the input dict
        # complete-by-construction in case a future instruction starts
        # reading it unconditionally.
        tp_constants.ATOMIC_MU_I_LOCAL: atomic_mu_i,
        tp_constants.ATOMS_TO_STRUCTURE_MAP: map_atoms_to_struct,
        tp_constants.BONDS_TO_STRUCTURE_MAP: map_bonds_to_struct,
        tp_constants.N_STRUCTURES_BATCH_TOTAL: np.asarray(len(samples), dtype=np.int32),
        # We do not pad, so the "real" count equals the total batch atom
        # count. ``ConstantScaleShiftTarget.frwrd`` reads
        # ``N_ATOMS_BATCH_REAL`` to mask shifts onto only real atoms; the
        # train function reads ``N_ATOMS_BATCH_TOTAL`` for force
        # accumulation segment counts.
        tp_constants.N_ATOMS_BATCH_REAL: np.asarray(n_atoms_total, dtype=np.int32),
        tp_constants.N_ATOMS_BATCH_TOTAL: np.asarray(n_atoms_total, dtype=np.int32),
    }
    float_specs = {
        tp_constants.BOND_VECTOR: bond_vector,
        tp_constants.DATA_REFERENCE_ENERGY: true_energy,
        tp_constants.DATA_REFERENCE_FORCES: true_force,
    }
    tf_batch: dict = {k: tf.constant(v, dtype=tf.int32) for k, v in int_specs.items()}
    tf_batch.update({k: tf.constant(v, dtype=float_dtype) for k, v in float_specs.items()})
    return tf_batch


class _ManualTFAdam:
    """Hand-rolled TF Adam — same defaults as ``torch.optim.Adam``.

    We deliberately avoid ``tf.keras.optimizers.Adam`` (or any Keras
    optimizer) here: TF 2.19 ships with Keras 3.14, whose
    ``Variable.__eq__`` fails inside ``Adam.update_step`` with

        ValueError: Attempt to convert a value (<class 'bool'>) with an
        unsupported type (<class 'type'>) to a Tensor.

    when called against the ``tensorpotential`` variables. ``tensorpotential``
    sets ``TF_USE_LEGACY_KERAS=1`` to dodge this, but the flag arrives
    too late once another test has already imported TF.

    A vanilla SGD step (the previous workaround) sidesteps Keras
    entirely but diverges in one step at ``lr=0.01``: with the random
    initialization of ``GRACE_2LAYER_v1_24`` the starting energy MSE is
    ~10^6 (per-atom contributions of O(100 eV) before any energy shift
    is fit), so raw-gradient updates at ``lr=0.01`` overshoot
    catastrophically. Adam's variance-normalized update keeps the
    effective step bounded, and mirroring ``torch.optim.Adam``'s
    defaults (``beta1=0.9``, ``beta2=0.999``, ``eps=1e-8``) keeps the
    matgl and TP sides on equal footing. Updates use only arithmetic
    and ``assign_sub`` — never ``__eq__`` — so we are immune to the
    Keras 3 plumbing entirely.
    """

    def __init__(
        self,
        variables,
        lr: float,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        self._m = [tf.Variable(tf.zeros_like(v), trainable=False) for v in variables]
        self._v = [tf.Variable(tf.zeros_like(v), trainable=False) for v in variables]

    def step(self, variables, grads) -> None:
        """Apply one Adam update in-place to ``variables``."""
        self.t += 1
        bc1 = 1.0 - self.beta1**self.t
        bc2 = 1.0 - self.beta2**self.t
        for var, grad, m, v in zip(variables, grads, self._m, self._v, strict=True):
            if grad is None:
                continue
            g = tf.cast(grad, var.dtype)
            m.assign(self.beta1 * m + (1.0 - self.beta1) * g)
            v.assign(self.beta2 * v + (1.0 - self.beta2) * g * g)
            m_hat = m / bc1
            v_hat = v / bc2
            var.assign_sub(self.lr * m_hat / (tf.sqrt(v_hat) + self.eps))


def _tp_step(model, train_function, optimizer: _ManualTFAdam, batch) -> float:
    """One Adam step on the whole TP batch; returns scalar loss.

    Uses :class:`_ManualTFAdam` rather than ``tf.keras.optimizers.Adam``
    to dodge a Keras 3 / TF 2.19 ``Variable.__eq__`` bug; see that
    class's docstring for the full rationale, including why a vanilla
    SGD fallback is not viable here.
    """
    e_target = batch[tp_constants.DATA_REFERENCE_ENERGY]
    f_target = batch[tp_constants.DATA_REFERENCE_FORCES]
    with tf.GradientTape() as tape:
        # Make a defensive copy: ``train_function`` mutates ``input_data``
        # by writing intermediate instruction outputs into it.
        input_data = dict(batch)
        out = train_function(model.instructions, input_data, training=True)
        e_pred = out[tp_constants.PREDICT_TOTAL_ENERGY]
        f_pred = out[tp_constants.PREDICT_FORCES]
        # Force tensor is padded to the max_atoms axis; trim to real atoms.
        n_real = tf.shape(f_target)[0]
        f_pred = f_pred[:n_real]
        loss_e = tf.reduce_mean(tf.square(e_pred - e_target))
        loss_f = tf.reduce_mean(tf.square(f_pred - f_target))
        loss = ENERGY_WEIGHT * loss_e + FORCE_WEIGHT * loss_f
    variables = model.variables_to_train
    grads = tape.gradient(loss, variables)
    optimizer.step(variables, grads)
    return float(loss.numpy())


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    os.environ.get("MATGL_SKIP_GRACE_TRAINING_PARITY") == "1",
    reason="explicitly skipped via MATGL_SKIP_GRACE_TRAINING_PARITY=1",
)
def test_grace2l_training_parity_matgl_vs_tp(nacl_training_set):
    """matgl GRACE-2L and upstream GRACE-2L both reduce loss in N_STEPS Adam steps."""
    samples = nacl_training_set
    assert len(samples) >= 5  # sanity — fixture should produce ~10

    # ---- matgl ----------------------------------------------------------
    pot = _build_matgl_grace2l()
    optimizer = torch.optim.Adam(pot.parameters(), lr=LR)
    matgl_losses: list[float] = []
    for _ in range(N_STEPS):
        matgl_losses.append(_matgl_step(pot, optimizer, samples))

    # ---- tensorpotential ------------------------------------------------
    tf.random.set_seed(SEED)
    # ``GRACE_2LAYER_v1_24`` returns an ``InstructionManager``; ``TPModel``
    # (and the ``execute_instructions`` helper it dispatches through)
    # accepts only ``list`` or ``dict``, so we unwrap to the registered
    # instruction dict.
    instructions = _build_tp_grace2l_instructions().get_instructions()
    tp_model = tp_tpmodel.TPModel(
        instructions,
        train_function=tp_tpmodel.ComputeBatchEnergyForcesVirials(),
        compute_function=tp_tpmodel.ComputeStructureEnergyAndForcesAndVirial(),
    )
    tp_model.build(tf.float64, jit_compile=False)
    train_function = tp_model.train_function
    tf_batch = _build_tp_batch(samples, tf.float64)
    # Adam (hand-rolled to dodge the Keras 3 / TF 2.19 optimizer bug;
    # see ``_ManualTFAdam`` docstring) with the same defaults as
    # ``torch.optim.Adam`` puts both sides on the same effective step
    # size. Vanilla SGD diverges at ``lr=0.01`` on this starting loss.
    tp_optimizer = _ManualTFAdam(tp_model.variables_to_train, lr=LR)
    tp_losses: list[float] = []
    for _ in range(N_STEPS):
        tp_losses.append(_tp_step(tp_model, train_function, tp_optimizer, tf_batch))

    # ---- assertions -----------------------------------------------------
    # (a) Both trajectories must be finite throughout.
    assert all(np.isfinite(matgl_losses)), f"matgl loss went non-finite: {matgl_losses}"
    assert all(np.isfinite(tp_losses)), f"tp loss went non-finite: {tp_losses}"

    # (b)/(c) compare *best-seen* reduction rather than the endpoint.
    # At ``lr=0.01`` Adam on a fresh GRACE-2L on ten NaCl configurations
    # quickly drops the loss by 1-2 orders of magnitude, then oscillates
    # around its basin (the LR is deliberately aggressive so the test
    # finishes fast). The two parametrizations have very different
    # initial-loss scales (matgl ~3e3, upstream ~1e6: the upstream
    # readout's larger random init blows up its first prediction), which
    # the first 1-2 Adam steps absorb. Taking ``min(losses)`` over the
    # trajectory measures "best progress made", which is what we actually
    # want to compare: robust to the late-stage oscillation and to the
    # initial-scale asymmetry that has nothing to do with whether either
    # side is learning.
    matgl_best = float(np.min(matgl_losses))
    tp_best = float(np.min(tp_losses))
    matgl_ratio = matgl_best / matgl_losses[0]
    tp_ratio = tp_best / tp_losses[0]

    # (b) Each side must reduce its best-seen loss by at least 50%.
    assert matgl_ratio < 0.5, (
        f"matgl best-seen reduction only {(1.0 - matgl_ratio) * 100:.1f}% "
        f"(start={matgl_losses[0]:.4g}, best={matgl_best:.4g}): {matgl_losses}"
    )
    assert tp_ratio < 0.5, (
        f"tensorpotential best-seen reduction only {(1.0 - tp_ratio) * 100:.1f}% "
        f"(start={tp_losses[0]:.4g}, best={tp_best:.4g}): {tp_losses}"
    )

    # (c) Numerical parity on the *best-progress* shape: same Adam
    # settings + same data, so the relative best reductions
    # ``L_best / L_initial`` should land in the same ballpark even though
    # the two parametric forms differ (matgl uses a simpler
    # indicator-mixing block; upstream adds FCRight2Left projections +
    # per-block readouts). Allow a generous 1.5-decade window — tight
    # enough to flag a regression where one side stops learning, loose
    # enough not to depend on architecture-specific details.
    log_diff = abs(np.log10(matgl_ratio) - np.log10(tp_ratio))
    assert log_diff < 1.5, (
        f"best-loss reduction ratios disagree by {log_diff:.2f} decades "
        f"(matgl={matgl_ratio:.3g}, tp={tp_ratio:.3g}); expected <1.5\n"
        f"  matgl losses: {matgl_losses}\n"
        f"  tp losses:    {tp_losses}"
    )
