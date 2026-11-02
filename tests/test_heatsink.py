import numpy as np
import pytest

from thermokit import PlateFinHeatSink, fin_count_sweep, fluid, solve, teertstra_nu

mm = 1e-3


@pytest.fixture
def sink():
    return PlateFinHeatSink(50 * mm, 50 * mm, 25 * mm, 1 * mm, 12, 4 * mm, 200)


def test_teertstra_limits():
    pr = 0.71
    lo = 1e-3
    assert teertstra_nu(lo, pr) == pytest.approx(lo * pr / 2, rel=1e-3)
    hi = 1e5
    dev = 0.664 * np.sqrt(hi) * pr ** (1 / 3) * np.sqrt(1 + 3.65 / np.sqrt(hi))
    assert teertstra_nu(hi, pr) == pytest.approx(dev, rel=1e-3)


def test_cannot_beat_the_air_heat_capacity(sink):
    # at very low flow the air leaves at base temperature, R -> 1/(m_dot cp)
    flow = 1e-6
    air = fluid("air", 25.0)
    r = sink.forced(flow)
    assert r.r_th >= 1 / (air.rho * flow * air.cp) * 0.999


def test_more_flow_lower_resistance(sink):
    a, b = sink.forced(2e-3), sink.forced(6e-3)
    assert b.r_th < a.r_th
    assert b.dp > a.dp


def test_fan_operating_point_balances_pressure(sink):
    fq = np.array([0, 2, 4, 6, 8]) * 1e-3
    fp = np.array([60, 48, 33, 16, 0.0])
    r = sink.operating_point(fq, fp)
    assert r.dp == pytest.approx(np.interp(r.flow, fq, fp), rel=1e-4)


def test_weak_fan_gives_a_trickle(sink):
    r = sink.operating_point([0, 1e-4], [0.01, 0.0])
    assert r.flow < 1e-4
    assert r.r_th > sink.forced(4e-3).r_th * 5


def test_fan_curve_must_reach_free_delivery(sink):
    with pytest.raises(ValueError):
        sink.operating_point([0, 1e-3], [200.0, 150.0])


def test_natural_convection_converges(sink):
    r = sink.natural(power=10.0)
    assert r.base_rise == pytest.approx(10.0 * r.r_th, rel=1e-3)
    assert 1.0 < r.r_th < 10.0


def test_sweep_skips_impossible_counts(sink):
    results = fin_count_sweep(sink, [4, 8, 12, 60], mode="forced", flow=3e-3)
    assert [r.sink.n_fins for r in results] == [4, 8, 12]


def test_spacing_and_geometry_checks():
    with pytest.raises(ValueError):
        PlateFinHeatSink(10 * mm, 50 * mm, 20 * mm, 1 * mm, 11, 3 * mm)


def test_layers_feed_the_stack(sink):
    r = sink.forced(4e-3)
    res = solve(sink.layers(15 * mm * 15 * mm, r), power=20.0, ambient=30.0)
    assert res.total > r.r_th
    assert res.junction == pytest.approx(30 + 20 * res.total)
