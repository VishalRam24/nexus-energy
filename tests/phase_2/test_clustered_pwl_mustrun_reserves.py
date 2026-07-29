"""Clustered UC, PWL heat rate, must-run, regulation reserve tests."""

from __future__ import annotations

import numpy as np

import nexus_energy as ne


def test_clustered_uc_scales_like_n_units():
    """3 identical 100 MW units lumped as clustered: u ∈ [0, 3]. Demand 250
    over 3 hours → expect u = 2.5 (two units fully + half of third) since
    p_min = 0 makes partial cluster dispatch legal."""
    demand = np.array([250, 250, 250])
    sys = ne.EnergySystem("cl")
    sys.set_timesteps(3)
    b = sys.add_bus("e")
    sys.add_load("ld", bus=b, amount=demand)
    sys.add_generator("fleet", bus=b, capacity=100, marginal_cost=30,
                      committable=True, clustered=True, n_units=3,
                      startup_cost=0.0)
    result = sys.optimise()
    assert result.status == "optimal"
    gen = sys._generators[0]
    u = np.array([result._raw.value(v) for v in gen._status_vars])
    # u ≤ 3 and u ≥ ceil(demand/cap) only when p_min=0 — upper-bound check:
    assert u.max() <= 3.0 + 1e-6
    # With p_min=0, the LP can set u to exactly demand/cap = 2.5 (fractional).
    assert all(u > 2.5 - 1e-6)


def test_pwl_heat_rate_selects_cheap_segment_first():
    """Two-segment PWL cost: 0–50 MW @ $10/MWh, 50–100 MW @ $40/MWh.
    At demand=30 LP should fill segment 0 only → total fuel = 300."""
    sys = ne.EnergySystem("pwl")
    sys.set_timesteps(1)
    b = sys.add_bus("e")
    sys.add_load("ld", bus=b, amount=30.0)
    sys.add_generator(
        "g", bus=b, capacity=100, marginal_cost=0.0,
        heat_rate_segments=[(0.0, 10.0), (50.0, 10.0), (100.0, 40.0)],
    )
    result = sys.optimise()
    assert result.status == "optimal"
    # Cost = 30 MW * 10 $/MWh * 1 h = 300.
    assert abs(result.total_cost - 300.0) < 1e-6


def test_pwl_uses_expensive_segment_only_when_needed():
    """Demand 80 MW → fill 50 at $10 then 30 at $40 = 500 + 1200 = 1700."""
    sys = ne.EnergySystem("pwl2")
    sys.set_timesteps(1)
    b = sys.add_bus("e")
    sys.add_load("ld", bus=b, amount=80.0)
    sys.add_generator(
        "g", bus=b, capacity=100, marginal_cost=0.0,
        heat_rate_segments=[(0.0, 10.0), (50.0, 10.0), (100.0, 40.0)],
    )
    result = sys.optimise()
    assert result.status == "optimal"
    # 50*10 + 30*40 = 500 + 1200 = 1700.
    assert abs(result.total_cost - 1700.0) < 1e-6


def test_must_run_forces_on():
    """Must-run committable at p_min=50 forces dispatch even with zero
    demand (slack sink absorbs). Check u=1 at every t."""
    demand = np.array([0, 0, 0])
    sys = ne.EnergySystem("mr")
    sys.set_timesteps(3)
    b = sys.add_bus("e")
    sys.add_load("ld", bus=b, amount=demand)
    sys.add_generator("baseload", bus=b, capacity=100, marginal_cost=5,
                      p_min=50, committable=True, must_run=True)
    # Negative-cost sink via a cheap "generator" with negative output is not
    # available; model must_run along with a storage that soaks up energy.
    sys.add_storage("sink", bus=b, power_capacity=100, energy_capacity=1000,
                    efficiency_charge=1.0, efficiency_discharge=1.0,
                    cyclic=False)
    result = sys.optimise()
    assert result.status == "optimal"
    gen = sys._generators[0]
    u = np.array([result._raw.value(v) for v in gen._status_vars])
    assert (u >= 1.0 - 1e-6).all()


def test_regulation_up_reserve_requirement_met():
    """System requires reg-up = 10% of load. Two gens: gen_a with reg_up_max=0.2,
    gen_b with reg_up_max=0.0. Expect reg_a[t] >= 10 per unit load."""
    demand = np.array([100, 100, 100])
    sys = ne.EnergySystem("reg")
    sys.set_timesteps(3)
    b = sys.add_bus("e")
    sys.add_load("ld", bus=b, amount=demand)
    sys.add_generator("a", bus=b, capacity=100, marginal_cost=20, reg_up_max=0.2)
    sys.add_generator("b", bus=b, capacity=100, marginal_cost=10)
    sys.set_regulation_reserve(up_fraction=0.1)
    result = sys.optimise()
    assert result.status == "optimal"
    a = sys._generators[0]
    reg_a = np.array([result._raw.value(v) for v in a._reg_up_vars])
    assert (reg_a + 1e-6 >= 10.0).all()
    # reg_up_max caps at 20% × 100 = 20.
    assert (reg_a <= 20.0 + 1e-6).all()
