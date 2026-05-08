"""Smoke test for the ``nacl_training_set`` session fixture.

Verifies the artifact loads, has the documented shape, and is a
self-consistent dataset (matching atom counts, finite labels). Skips when
the artifact file is absent — the actual skip happens inside the fixture
itself.
"""

from __future__ import annotations

import numpy as np
from pymatgen.core import Structure


def test_nacl_training_set_shape(nacl_training_set):
    samples = nacl_training_set
    assert len(samples) == 10
    for sample in samples:
        s = sample["structure"]
        assert isinstance(s, Structure)
        n = len(s)
        assert sample["forces"].shape == (n, 3)
        assert sample["stress"].shape == (3, 3)
        assert np.isfinite(sample["energy"])
        assert np.all(np.isfinite(sample["forces"]))
        assert np.all(np.isfinite(sample["stress"]))


def test_nacl_training_set_first_is_ground_state(nacl_training_set):
    """First entry is the unperturbed rocksalt cell — forces should be ~0."""
    s = nacl_training_set[0]
    assert np.max(np.abs(s["forces"])) < 1e-4
