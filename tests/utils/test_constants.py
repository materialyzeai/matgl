from __future__ import annotations

import numpy as np
import pytest
import scipy.constants as const

from matgl.utils.constants import COULOMB_CONSTANT, EV_PER_ANG3_TO_GPA


def test_coulomb_constant():
    # Electrostatic energy (eV) between two unit charges (e) 1 Ang apart:
    # e / (4 pi eps0) converted from J.m/e^2 to eV.Ang/e^2 (/e for J->eV, *1e10 for m->Ang).
    expected = const.elementary_charge / (4 * np.pi * const.epsilon_0) * 1e10
    assert pytest.approx(expected, rel=1e-6) == COULOMB_CONSTANT


def test_ev_per_ang3_to_gpa():
    # 1 eV/Ang^3 = e[J] / 1e-30 m^3 Pa = e * 1e30 Pa = e * 1e21 GPa.
    expected = const.elementary_charge * 1e21
    assert pytest.approx(expected, rel=1e-6) == EV_PER_ANG3_TO_GPA
