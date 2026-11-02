import numpy as np
import pytest

from thermokit import equivalent_radius, spreading_resistance


def test_half_space_limit():
    # Isoflux disk on a half-space, mean temperature: R = 8 / (3 π² k a)
    k, a = 1.0, 1.0
    eps = 0.01
    b = a / eps
    r = spreading_resistance(np.pi * a**2, np.pi * b**2, 10 * b, k, np.inf)
    exact = 8 / (3 * np.pi**2 * k * a)
    # flux tube correction is roughly (1 - 1.41 ε)
    assert r == pytest.approx(exact * (1 - 1.41 * eps), rel=0.005)


def test_no_spreading_when_source_covers_plate():
    assert spreading_resistance(1e-4, 1e-4, 2e-3, 390, 1e3) == 0.0


def test_source_larger_than_plate_rejected():
    with pytest.raises(ValueError):
        spreading_resistance(2e-4, 1e-4, 2e-3, 390, 1e3)


def test_spreading_falls_as_biot_rises():
    # φ goes from coth(λτ) at Bi → 0 to tanh(λτ) at Bi → ∞
    args = (25e-6, 1600e-6, 3e-3, 390)
    rs = [spreading_resistance(*args, h) for h in (1e2, 1e3, 1e4, 1e5)]
    assert np.all(np.diff(rs) < 0)


def test_thicker_plate_spreads_better_at_low_biot():
    rs = [spreading_resistance(25e-6, 1600e-6, t, 390, 200.0) for t in (1e-3, 3e-3, 6e-3)]
    assert np.all(np.diff(rs) < 0)


def test_higher_conductivity_spreads_better():
    al = spreading_resistance(1e-4, 1e-3, 3e-3, 200, 2e3)
    cu = spreading_resistance(1e-4, 1e-3, 3e-3, 390, 2e3)
    assert cu < al


def test_equivalent_radius():
    assert equivalent_radius(np.pi * 4.0) == pytest.approx(2.0)
