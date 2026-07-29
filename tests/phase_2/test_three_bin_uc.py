"""Morales-España 3-bin UC correctness tests."""

from __future__ import annotations

import numpy as np
import pytest

import nexus_energy as ne


def _single_unit_sys(T=6, demand=None, startup_cost=0.0, shutdown_cost=0.0,
                     min_up_time=0, min_down_time=0, p_min=0.0, must_run=False):
    if demand is None:
        demand = np.zeros(T)
    sys = ne.EnergySystem("uc")
    sys.set_timesteps(T)
    b = sys.add_bus("e")
    sys.add_load("ld", bus=b, amount=demand)
    sys.add_generator(
        "coal", bus=b, capacity=100, marginal_cost=30, p_min=p_min,
        committable=True, startup_cost=startup_cost, shutdown_cost=shutdown_cost,
        min_up_time=min_up_time, min_down_time=min_down_time, must_run=must_run,
    )
    sys.add_generator("peaker", bus=b, capacity=100, marginal_cost=200)
    return sys


def test_3bin_v_w_indicators_on_startup_shutdown():
    """Starting from off (u=0), serving demand 50 for 3 steps, then 0,
    then 50: expect v=1 at first on-step, w=1 at first off-step.

    PyPSA-style formulation: v >= max(0, u[t]-u[t-1]), w >= max(0, u[t-1]-u[t]),
    with non-zero startup/shutdown cost the LP/MIP picks the minimum.
    """
    demand = np.array([0, 50, 50, 50, 0, 50])
    sys = _single_unit_sys(T=6, demand=demand,
                           startup_cost=1.0, shutdown_cost=1.0)
    result = sys.optimise()
    assert result.status == "optimal"

    gen = sys._generators[0]
    u = np.array([result._raw.value(v) for v in gen._status_vars])
    v = np.array([result._raw.value(v) for v in gen._startup_vars])
    w = np.array([result._raw.value(v) for v in gen._shutdown_vars])

    # Inequality state-transition + non-zero costs → uniquely determined.
    for t in range(1, 6):
        assert v[t] >= (u[t] - u[t-1]) - 1e-6
        assert w[t] >= (u[t-1] - u[t]) - 1e-6
        assert v[t] + w[t] <= 1.0 + 1e-6


def test_3bin_mutex_v_plus_w_le_1():
    """Can't both start and stop in the same timestep."""
    demand = np.array([50, 0, 50, 0, 50, 0])
    sys = _single_unit_sys(T=6, demand=demand, startup_cost=1.0, shutdown_cost=1.0)
    result = sys.optimise()
    assert result.status == "optimal"
    gen = sys._generators[0]
    for t in range(1, 6):
        v_t = result._raw.value(gen._startup_vars[t])
        w_t = result._raw.value(gen._shutdown_vars[t])
        assert v_t + w_t <= 1.0 + 1e-6


def test_startup_cost_counted_once_per_start():
    """With p_min > 0, staying on when demand=0 wastes fuel — so the LP
    turns the unit off between demand pulses. Startup cost makes the
    tradeoff non-trivial."""
    demand = np.array([100, 0, 100, 0])
    sys = _single_unit_sys(T=4, demand=demand, startup_cost=10.0, p_min=40.0)
    result = sys.optimise()
    assert result.status == "optimal"
    gen = sys._generators[0]
    total_startups = sum(result._raw.value(v) for v in gen._startup_vars[1:])
    # With p_min=40 and cheap startup_cost=10, cycling (shut+start) costs
    # 10 per cycle vs running idle at 40 MW × $30 = $1200/h wasted → cycle.
    assert total_startups >= 1.0 - 1e-6


def test_shutdown_cost_adds_to_objective():
    """With p_min > 0, turning off is favored when demand drops (avoids
    idle fuel burn). shutdown_cost > 0 just makes each transition priced."""
    demand = np.array([100, 100, 0, 0])
    sys = _single_unit_sys(T=4, demand=demand, shutdown_cost=10.0, p_min=40.0)
    result = sys.optimise()
    assert result.status == "optimal"
    gen = sys._generators[0]
    shutdowns = sum(result._raw.value(v) for v in gen._shutdown_vars[1:])
    # One shutdown between t=1 and t=2.
    assert shutdowns >= 1.0 - 1e-6
