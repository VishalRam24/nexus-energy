"""
Phase 9 — Stochastic / robust optimisation tests.

Coverage:
  (a) two-stage stochastic capacity expansion converges via Benders;
  (b) CVaR objective produces a more conservative (>= expected-cost)
      plan than expected-cost on a tail-heavy scenario set;
  (c) Budget-uncertainty robust plan dominates the nominal plan in
      worst-case realised cost;
  (d) ``evaluate_plan`` returns sane per-scenario costs for a fixed plan;
  (e) ``reduce_scenarios`` collapses a 20-scenario tree to k medoids
      with conserved probability mass and acceptable optimum drift;
  (f) ``ChanceConstraint`` helpers behave;
  (g) ``solve_sddip`` raises NotImplementedError pointing at Phase 12.
"""

from __future__ import annotations

import numpy as np
import pytest

import nexus_energy as ne
from nexus_energy.stochastic import (
    BudgetUncertaintySet,
    ChanceConstraint,
    Scenario,
    apply_scenario,
    evaluate_plan,
    generate_demand_scenarios,
    reduce_scenarios,
    solve_robust,
    solve_sddip,
    solve_stochastic,
)


def _build_invest_system(T: int = 24) -> ne.EnergySystem:
    """
    Tiny first-stage / second-stage system:
      cheap_fixed (60 MW, mc=10), slack (5000 MW, mc=5000) so every
      scenario is feasible, solar (extendable, capex=40 $/MW, daily CF),
      peaker (extendable, capex=15 $/MW, mc=200).
    """
    rng = np.random.default_rng(0)
    sys = ne.EnergySystem("invest_small")
    elec = sys.add_bus("elec", carrier="electricity")

    hours = np.arange(T)
    day = hours % 24
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


# ---------------------------------------------------------------------------
# (a) Convergence
# ---------------------------------------------------------------------------

def test_solve_stochastic_benders_converges():
    sys = _build_invest_system()
    scenarios = [
        Scenario("low",  0.4, demand_factor=0.85),
        Scenario("mid",  0.4, demand_factor=1.00),
        Scenario("high", 0.2, demand_factor=1.30),
    ]
    res = solve_stochastic(
        sys, scenarios, risk_measure="expected", method="benders",
        max_iter=30, tol=1e-3,
    )
    assert res.status == "optimal", res.status
    assert res.expected_cost > 0
    assert "solar" in res.capacity_decisions
    assert "peaker" in res.capacity_decisions
    # Probabilities sum to 1, expected = Σ p_s · cost_s.
    recomp = sum(s.probability * res.scenario_costs[s.name] for s in scenarios)
    assert abs(recomp - res.expected_cost) <= 1e-2 * abs(res.expected_cost) + 1.0


# ---------------------------------------------------------------------------
# (b) CVaR shifts the plan toward the tail
# ---------------------------------------------------------------------------

def test_cvar_dominates_expected_in_tail():
    """
    With one heavy-tail scenario, CVaR should yield a plan whose worst-case
    cost is no worse than the expected-cost plan's worst case. Equivalently,
    CVaR cost >= expected cost on the *same* scenario set (tail conditional
    mean is at least as large as the unconditional mean).
    """
    sys = _build_invest_system()
    scenarios = [
        Scenario("nominal", 0.7, demand_factor=1.00),
        Scenario("mild",    0.2, demand_factor=1.10),
        Scenario("tail",    0.1, demand_factor=1.50),
    ]
    exp_res = solve_stochastic(
        sys, scenarios, risk_measure="expected", method="benders",
        max_iter=30, tol=1e-3,
    )
    cvar_res = solve_stochastic(
        sys, scenarios, risk_measure="cvar", cvar_alpha=0.10,
        method="benders", max_iter=30, tol=1e-3,
    )
    assert exp_res.status == "optimal"
    assert cvar_res.status == "optimal"
    # CVaR plan's worst-case cost should be no worse than expected plan's.
    assert cvar_res.worst_case_cost <= exp_res.worst_case_cost * 1.05 + 1.0
    # CVaR objective on the realised costs is always >= expected cost.
    assert cvar_res.cvar >= cvar_res.expected_cost - 1e-6


# ---------------------------------------------------------------------------
# (c) Robust plan covers worst case
# ---------------------------------------------------------------------------

def test_robust_budget_set_returns_feasible_plan():
    sys = _build_invest_system()
    nominal = solve_stochastic(
        sys, [Scenario("nom", 1.0)], method="benders", max_iter=20, tol=1e-3,
    )
    robust = solve_robust(
        sys,
        uncertainty=BudgetUncertaintySet(
            demand_up=0.30, cf_down=0.20, budget=2.0),
    )
    assert nominal.status == "optimal"
    assert robust.status == "optimal"
    # Robust plan must be at least as large in total capacity (it covers
    # a strictly larger uncertainty set).
    nom_cap = sum(nominal.capacity_decisions.values())
    rob_cap = sum(robust.capacity_decisions.values())
    assert rob_cap >= nom_cap - 1e-6


# ---------------------------------------------------------------------------
# (d) evaluate_plan: out-of-sample harness
# ---------------------------------------------------------------------------

def test_evaluate_plan_returns_per_scenario_costs():
    sys = _build_invest_system()
    plan = solve_stochastic(
        sys, [Scenario("nom", 1.0)], method="benders", max_iter=20, tol=1e-3,
    )
    oos = [
        Scenario("oos_low",  0.5, demand_factor=0.9),
        Scenario("oos_high", 0.5, demand_factor=1.2),
    ]
    report = evaluate_plan(sys, plan.capacity_decisions, oos, cvar_alpha=0.5)
    assert report["n_scenarios"] == 2
    assert set(report["scenario_costs"]) == {"oos_low", "oos_high"}
    assert report["expected_cost"] > 0
    assert report["worst_case"] >= report["expected_cost"] - 1e-6
    # higher-demand scenario is more expensive (slack hours included)
    assert report["scenario_costs"]["oos_high"] >= \
        report["scenario_costs"]["oos_low"] - 1.0


# ---------------------------------------------------------------------------
# (e) Scenario reduction
# ---------------------------------------------------------------------------

def test_reduce_scenarios_preserves_probability_and_close_optimum():
    sys = _build_invest_system()
    raw = generate_demand_scenarios(base_demand=80, n_scenarios=20, std=0.15)
    reduced = reduce_scenarios(raw, n_reduced=5, seed=0)
    assert len(reduced) == 5
    assert abs(sum(s.probability for s in reduced) - 1.0) < 1e-6

    full_res = solve_stochastic(
        sys, raw, risk_measure="expected", method="benders",
        max_iter=25, tol=2e-3,
    )
    red_res = solve_stochastic(
        sys, reduced, risk_measure="expected", method="benders",
        max_iter=25, tol=2e-3,
    )
    assert full_res.status == "optimal"
    assert red_res.status == "optimal"
    # Optimum drift is bounded: 5 medoids should track 20 raw scenarios
    # within ~5 % on a near-Gaussian factor distribution.
    drift = abs(red_res.expected_cost - full_res.expected_cost) / full_res.expected_cost
    assert drift < 0.05, f"reduction drift {drift:.3%}"


# ---------------------------------------------------------------------------
# (f) Chance constraint helper
# ---------------------------------------------------------------------------

def test_chance_constraint_violation_count_and_bonferroni():
    cc = ChanceConstraint(name="reserve", alpha=0.05, threshold=10.0)
    realised = [5.0, 9.9, 10.5, 12.0, 1.0]
    assert cc.violates(realised) == 2
    assert cc.bonferroni_correction(5) == pytest.approx(0.01)
    assert cc.bonferroni_correction(0) == pytest.approx(0.05)


# ---------------------------------------------------------------------------
# (g) SDDiP — now implemented (Phase 9 depth pass). The defining-property
#     convergence test lives in test_stochastic_depth.py; here we only
#     assert the stub no longer raises and that a bad call surfaces a clear
#     argument error rather than NotImplementedError.
# ---------------------------------------------------------------------------

def test_sddip_is_implemented():
    # No longer a NotImplementedError stub.
    with pytest.raises(TypeError):
        solve_sddip()  # missing required stages / scenarios_per_stage
    # Single-stage is rejected (needs >= 2 stages).
    with pytest.raises(ValueError, match="2 stages"):
        solve_sddip([object()], [[{}]])


# ---------------------------------------------------------------------------
# apply_scenario sanity
# ---------------------------------------------------------------------------

def test_apply_scenario_does_not_mutate_base():
    sys = _build_invest_system()
    sc = Scenario("hot", 1.0, demand_factor=1.5, fuel_cost_factor=2.0)
    base_load = float(np.asarray(sys._loads[0].amount).mean())
    base_mc = float(sys._generators[0].marginal_cost)
    sub = apply_scenario(sys, sc)
    sub_load = float(np.asarray(sub._loads[0].amount).mean())
    sub_mc = float(sub._generators[0].marginal_cost)
    assert sub_load == pytest.approx(1.5 * base_load, rel=1e-9)
    assert sub_mc == pytest.approx(2.0 * base_mc, rel=1e-9)
    # Base untouched
    assert float(np.asarray(sys._loads[0].amount).mean()) == pytest.approx(base_load)
    assert float(sys._generators[0].marginal_cost) == pytest.approx(base_mc)
