"""Spreading resistance of a small heat source on a larger plate."""

import numpy as np
from scipy.special import j0, j1, jn_zeros

_N_TERMS = 2000
_ZEROS = jn_zeros(1, _N_TERMS)


def equivalent_radius(area):
    """Radius of the circle with the same area (m)."""
    return np.sqrt(area / np.pi)


def spreading_resistance(source_area, plate_area, thickness, k, h):
    """Spreading resistance based on the mean source temperature, K/W.

    Isoflux circular source of radius a centred on a circular plate of
    radius b. Top face outside the source and the rim are adiabatic, the
    bottom face sees a uniform film coefficient h (pass np.inf for an
    isothermal bottom). Exact Fourier-Bessel series solution, see
    Yovanovich, Culham & Teertstra, IEEE Trans. CPMT-A 21(1), 1998.

    Square or rectangular footprints are mapped to circles of equal area,
    which is accurate to a few percent for aspect ratios up to about 2.

    The result excludes the 1D terms t/(k A_p) and 1/(h A_p).
    """
    a = equivalent_radius(source_area)
    b = equivalent_radius(plate_area)
    if a > b:
        raise ValueError("source area is larger than plate area")
    if np.isclose(a, b):
        return 0.0

    lam = _ZEROS
    m = lam / b
    x = 0.0 if np.isinf(h) else k * m / h
    th = np.tanh(m * thickness)
    phi = (th + x) / (1.0 + x * th)

    terms = j1(lam * a / b) ** 2 * phi / (lam**3 * j0(lam) ** 2)
    return 4.0 * b / (np.pi * a**2 * k) * terms.sum()
