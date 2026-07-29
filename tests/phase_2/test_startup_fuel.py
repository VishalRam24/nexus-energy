"""Phase 2.x — ``Generator.startup_fuel_cost`` accounting."""

from __future__ import annotations

import numpy as np

import nexus_energy as ne


def _cycling_system(startup_cost=0.0, startup_fuel_cost=0.0):
    T = 4
    demand = np.array([100.0, 0.0, 100.0, 0.0])
    sys = ne.EnergySystem("startup_fuel")
    sys.set_timesteps(T)
    b = sys.add_bus("e")
    sys.add_load("ld", bus=b, amount=demand)
    sys.add_generator(
        "coal", bus=b, capacity=100.0, marginal_cost=30.0, p_min=40.0,
        committable=True,
        startup_cost=startup_cost,
        startup_fuel_cost=startup_fuel_cost,
    )
    sys.add_generator("peaker", bus=b, capacity=100.0, marginal_cost=200.0)
    return sys


def test_startup_fuel_cost_additive_to_startup_cost():
    """A $10 startup_cost + $5 startup_fuel_cost should price each
    start at $15 — same objective as $15 startup_cost alone."""
    sys_a = _cycling_system(startup_cost=15.0, startup_fuel_cost=0.0)
    sys_b = _cycling_system(startup_cost=10.0, startup_fuel_cost=5.0)
    res_a = sys_a.optimise()
    res_b = sys_b.optimise()
    assert res_a.status == "optimal"
    assert res_b.status == "optimal"
    assert abs(res_a.total_cost - res_b.total_cost) < 1e-6


def test_startup_fuel_cost_raises_total_cost():
    """Raising ``startup_fuel_cost`` from 0 to a large value makes cycling
    the committable unit strictly more expensive; the LP either pays more
    or switches to the peaker. Cost must go up either way."""
    base = _cycling_system(startup_cost=10.0, startup_fuel_cost=0.0)
    with_fuel = _cycling_system(startup_cost=10.0, startup_fuel_cost=1000.0)
    c_base = base.optimise().total_cost
    c_fuel = with_fuel.optimise().total_cost
    assert c_fuel > c_base + 1e-6


def test_startup_fuel_cost_zero_is_no_regression():
    """Default 0.0 must match the pre-Phase-2.x behaviour exactly."""
    sys = _cycling_system(startup_cost=10.0, startup_fuel_cost=0.0)
    res = sys.optimise()
    assert res.status == "optimal"
    gen = sys._generators[0]
    starts = sum(res._raw.value(v) for v in gen._startup_vars[1:])
    assert starts >= 1.0 - 1e-6
