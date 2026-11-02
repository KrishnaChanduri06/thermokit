import ht
import numpy as np
import pytest

from thermokit.duct import f_re_shah_london, rectangular_duct


@pytest.mark.parametrize("alpha", [1.0, 0.5, 0.25, 0.1])
def test_friction_matches_shah_london(alpha):
    assert rectangular_duct(alpha, 1.0).f_re == pytest.approx(f_re_shah_london(alpha), rel=0.005)


@pytest.mark.parametrize("alpha", [1.0, 0.5, 0.25, 0.1])
def test_four_wall_nusselt_matches_shah_london(alpha):
    nu = rectangular_duct(alpha, 1.0).nu_fd
    assert nu == pytest.approx(ht.Nu_laminar_rectangular_Shan_London(alpha), rel=0.005)


def test_orientation_does_not_matter_with_all_walls_heated():
    assert rectangular_duct(0.4, 1.0).nu_fd == pytest.approx(rectangular_duct(1.0, 0.4).nu_fd, rel=1e-6)


def test_three_heated_walls_below_four_for_tall_channel():
    four = rectangular_duct(231, 713).nu_fd
    three = rectangular_duct(231, 713, heated=("left", "right", "bottom")).nu_fd
    # unheated short wall raises Nu for a tall narrow channel (5.29 vs 4.84)
    assert three == pytest.approx(5.29, abs=0.03)
    assert three > four


def test_entry_decays_to_fully_developed():
    d = rectangular_duct(0.5, 1.0)
    assert np.all(np.diff(d.nu_x) < 1e-9)
    assert d.nu_x[-1] == pytest.approx(d.nu_fd, rel=1e-3)
    assert d.nu_local(10.0) == d.nu_fd


@pytest.mark.parametrize("x_pp, expected", [(9.5e-5, 32.63), (1.9e-3, 12.82), (9.5e-3, 8.82)])
def test_parallel_plate_thermal_entry(x_pp, expected):
    # Shah & London (1978), parallel plates, both walls at uniform flux
    d = rectangular_duct(40, 1, heated=("bottom", "top"))
    dh = 2 * 40 / 41
    nu = d.nu_local(x_pp / (dh / 2) ** 2) * 2 / dh
    assert nu == pytest.approx(expected, rel=0.03)


def test_bad_walls():
    with pytest.raises(ValueError):
        rectangular_duct(1, 1, heated=())
    with pytest.raises(ValueError):
        rectangular_duct(1, 1, heated=("front",))
