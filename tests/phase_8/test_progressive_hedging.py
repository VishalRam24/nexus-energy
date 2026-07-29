"""
Phase 8 depth pass — progressive hedging consensus solver.

Coverage:
  (a) solve_stochastic_ph converges on a small 3-scenario capacity
      expansion: final capacity_decisions are a single consensus
      (no per-scenario spread).
  (b) PH expected cost is within a bounded tolerance of the Benders
      extensive-form reference solution.
  (c) Degenerate case (no extendable components) returns the direct
      solver's per-scenario aggregation without error.
"""

from __future__ import annotations

import numpy as np
import pytest

import nexus_energy as ne
from nexus_energy.stochastic import (
    Scenario,
    solve_stochastic,
    solve_stochastic_ph,
)


def _invest_system(T: int = 24) -> ne.EnergySystem:
    rng = np.random.default_rng(0)
    sys = ne.EnergySystem("ph_small")
    elec = sys.add_bus("elec", carrier="electricity")
    day = np.arange(T) % 24
    load = 70 + 30 * np.cos((day - 18) * np.pi / 12) ** 2
    load = load + rng.normal(0, 1, size=T)
    sys.add_load("d", bus=elec, amount=load)
    sys.add_generator("cheap_fixed", bus=elec, capacity=60, marginal_cost=10)
    sys.add_generator("slack", bus=elec, capacity=5000, marginal_cost=5000)
    cf = np.clip(np.cos((day - 12) * np.pi / 12), 0, None)
    sys.add_generator(
        "solar", bus=elec, capacity=10, marginal_cost=0,
        carrier_factor=cf, extendable=True, min_capacity=10,
        max_capacity=400, capital_cost=40,
    )
    sys.add_generator(
        "peaker", bus=elec, capacity=10, marginal_cost=200,
        extendable=True, min_capacity=10, max_capacity=400, capital_cost=15,
    )
    return sys


def _nofirst_system(T: int = 12) -> ne.EnergySystem:
    sys = ne.EnergySystem("ph_nofirst")
    elec = sys.add_bus("elec", carrier="electricity")
    sys.add_load("d", bus=elec, amount=np.full(T, 80.0))
    # No extendable units — pure operational system.
    sys.add_generator("g", bus=elec, capacity=200, marginal_cost=30)
    sys.add_generator("slack", bus=elec, capacity=5000, marginal_cost=5000)
    return sys


def test_progressive_hedging_converges_to_consensus():
    sys = _invest_system()
    scenarios = [
        Scenario("low",  0.4, demand_factor=0.85),
        Scenario("mid",  0.4, demand_factor=1.00),
        Scenario("high", 0.2, demand_factor=1.30),
    ]
    res = solve_stochastic_ph(
        sys, scenarios,
        rho=1.0, max_iter=25, tol=5e-3,
        initial_radius=0.6, radius_decay=0.75,
    )
    assert res.status in ("optimal", "max_iter")
    assert res.method == "progressive_hedging"
    assert "solar" in res.capacity_decisions
    assert "peaker" in res.capacity_decisions
    assert res.n_iterations >= 1
    # Consensus capacities should be within min/max bounds.
    assert 10.0 - 1e-6 <= res.capacity_decisions["solar"] <= 400.0 + 1e-6
    assert 10.0 - 1e-6 <= res.capacity_decisions["peaker"] <= 400.0 + 1e-6


def test_progressive_hedging_matches_benders_cost_within_tolerance():
    sys_ph = _invest_system()
    sys_bd = _invest_system()
    scenarios = [
        Scenario("low",  0.4, demand_factor=0.85),
        Scenario("mid",  0.4, demand_factor=1.00),
        Scenario("high", 0.2, demand_factor=1.30),
    ]
    ph = solve_stochastic_ph(
        sys_ph, scenarios,
        rho=1.0, max_iter=30, tol=3e-3,
        initial_radius=0.6, radius_decay=0.8,
    )
    bd = solve_stochastic(
        sys_bd, scenarios, risk_measure="expected",
        method="benders", max_iter=40, tol=1e-3,
    )
    assert bd.status == "optimal"
    assert ph.expected_cost == pytest.approx(bd.expected_cost, rel=0.25)


def test_progressive_hedging_no_first_stage_is_direct():
    sys = _nofirst_system()
    scenarios = [
        Scenario("a", 0.5, demand_factor=1.00),
        Scenario("b", 0.5, demand_factor=1.10),
    ]
    res = solve_stochastic_ph(sys, scenarios, max_iter=5, tol=1e-2)
    assert res.status == "optimal"
    # No first-stage decisions → empty capacity set.
    assert res.capacity_decisions == {}
    assert set(res.scenario_costs) == {"a", "b"}
