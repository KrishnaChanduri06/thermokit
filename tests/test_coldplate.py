import numpy as np
import pytest

from thermokit import ColdPlate, fluid

UM = 1e-6


@pytest.fixture
def plate():
    return ColdPlate(width=10e-3, length=40e-3, channel_width=200 * UM, channel_height=700 * UM,
                     fin_width=200 * UM, base_thickness=1e-3, k=390)


def test_channel_count(plate):
    assert plate.n_channels == 25


def test_energy_balance(plate):
    r = plate.solve(0.5, 300.0, t_in=25.0)
    f = fluid("water", 25.0)
    assert r.t_out == pytest.approx(25 + 300 / (f.rho * 0.5 / 60000 * f.cp))
    assert r.t_bulk[0] == 25.0 and r.t_bulk[-1] == pytest.approx(r.t_out)


def test_temperatures_ordered(plate):
    r = plate.solve(0.5, 300.0)
    assert np.all(r.t_base > r.t_wall)
    assert np.all(r.t_wall > r.t_bulk)
    assert r.r_th > r.r_th_mean


def test_developing_flow_beats_fully_developed(plate):
    dev = plate.solve(0.5, 300.0, wall_viscosity=False)
    fd = plate.solve(0.5, 300.0, developing=False, wall_viscosity=False)
    assert dev.r_th < fd.r_th
    assert dev.dp == pytest.approx(fd.dp)


def test_heated_liquid_lowers_pressure_drop(plate):
    hot = plate.solve(0.5, 300.0)
    iso = plate.solve(0.5, 300.0, wall_viscosity=False)
    assert hot.viscosity_ratio < 1
    assert hot.dp < iso.dp
    assert hot.r_th < iso.r_th


def test_more_flow_lowers_resistance_raises_dp(plate):
    a, b = plate.solve(0.3, 300.0), plate.solve(1.0, 300.0)
    assert b.r_th < a.r_th
    assert b.dp > a.dp


def test_flow_for_re_round_trip(plate):
    flow = plate.flow_for_re(800, 25.0)
    r = plate.solve(flow, 1e-3, t_in=25.0)  # negligible heating, mean T = inlet T
    assert r.re == pytest.approx(800, rel=1e-3)


def test_regimes():
    mini = ColdPlate(width=50e-3, length=80e-3, channel_width=2e-3, channel_height=4e-3,
                     fin_width=1.5e-3, base_thickness=3e-3, k=390)
    assert mini.solve(30.0, 1000.0).regime == "turbulent"
    assert mini.solve(8.0, 1000.0).regime == "transitional"
    assert mini.solve(8.0, 1000.0, regime="laminar").regime == "laminar"
    with pytest.raises(ValueError):
        mini.solve(8.0, 1000.0, regime="slug")


def test_transition_is_continuous():
    mini = ColdPlate(width=50e-3, length=80e-3, channel_width=2e-3, channel_height=4e-3,
                     fin_width=1.5e-3, base_thickness=3e-3, k=390)
    flow = mini.flow_for_re(2300, 25.0)
    below = mini.solve(flow * 0.995, 1e-3, wall_viscosity=False)
    above = mini.solve(flow * 1.005, 1e-3, wall_viscosity=False)
    assert below.regime == "laminar" and above.regime == "transitional"
    assert above.r_th == pytest.approx(below.r_th, rel=0.03)
    assert above.dp == pytest.approx(below.dp, rel=0.03)


def test_glycol_needs_fraction(plate):
    with pytest.raises(ValueError):
        plate.solve(0.5, 100.0, coolant="ethylene glycol")
    r = plate.solve(0.5, 100.0, coolant="ethylene glycol", fraction=0.3)
    w = plate.solve(0.5, 100.0)
    assert r.r_th > w.r_th


def test_bonded_lid_helps(plate):
    bonded = ColdPlate(**{**plate.__dict__, "lid": "bonded", "n_channels": None})
    assert bonded.solve(0.5, 300.0).r_th < plate.solve(0.5, 300.0).r_th
