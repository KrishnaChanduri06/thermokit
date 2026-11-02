import pytest

from thermokit import Contact, Convection, Fixed, Slab, Spreader, solve


def test_slab_and_film_by_hand():
    area = 0.02 * 0.02
    layers = [
        Slab("plate", 2e-3, 200.0, area),
        Convection("air", 50.0, area),
    ]
    res = solve(layers, power=10.0, ambient=25.0)
    r_hand = 2e-3 / (200 * area) + 1 / (50 * area)
    assert res.total == pytest.approx(r_hand)
    assert res.junction == pytest.approx(25 + 10 * r_hand)


def test_node_temperatures():
    layers = [Fixed("jc", 0.2), Contact("tim", 1e-5, 1e-4), Fixed("sink", 0.5)]
    res = solve(layers, power=20.0, ambient=30.0)
    t = res.temperatures
    assert t[-1] == 30.0
    assert t[0] == pytest.approx(30 + 20 * (0.2 + 0.1 + 0.5))
    assert t[1] == pytest.approx(30 + 20 * (0.1 + 0.5))
    assert res.shares.sum() == pytest.approx(1.0)


def test_spreader_with_full_coverage_is_a_slab():
    area = 1e-3
    spreader = solve([Spreader("base", area, area, 4e-3, 390), Fixed("fins", 0.3)], 1.0)
    slab = solve([Slab("base", 4e-3, 390, area), Fixed("fins", 0.3)], 1.0)
    assert spreader.total == pytest.approx(slab.total)


def test_spreader_sees_downstream_cooling():
    base = Spreader("base", 1e-4, 2.5e-3, 3e-3, 390)
    weak = solve([base, Fixed("sink", 2.0)], 1.0).resistances[0]
    strong = solve([base, Fixed("sink", 0.05)], 1.0).resistances[0]
    assert strong < weak
