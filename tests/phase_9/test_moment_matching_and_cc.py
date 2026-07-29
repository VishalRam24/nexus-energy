"""
Phase 9 depth pass — moment-matching scenarios + native SAA chance
constraints.

Coverage:
  (a) generate_moment_matching_scenarios reproduces target mean/cov on
      the realised sample batch.
  (b) generate_moment_matching_scenarios respects custom field_names
      and probability vectors.
  (c) ChanceConstraint.saa_quantile_threshold returns the empirical
      (1-α)-quantile and is monotone in α.
  (d) ChanceConstraint.violation_probability respects non-uniform
      scenario weights.
  (e) solve_saa_chance_constrained with tight α yields a firmer plan
      than loose α (more firm capacity in worst-tail coverage).
"""

from __future__ import annotations

import numpy as np
import pytest

import nexus_energy as ne
from nexus_energy.stochastic import (
    ChanceConstraint,
    Scenario,
    generate_moment_matching_scenarios,
    solve_saa_chance_constrained,
)


# ---------------------------------------------------------------------------
# (a) Moment matching reproduces target mean/cov
# ---------------------------------------------------------------------------

def test_moment_matching_reproduces_target_moments():
    target_mean = np.array([1.0, 1.05, 0.95])
    # Positive-definite target cov with non-trivial off-diagonals.
    target_cov = np.array([
        [0.04, 0.01, -0.005],
        [0.01, 0.03, 0.002],
        [-0.005, 0.002, 0.02],
    ])
    sc = generate_moment_matching_scenarios(
        target_mean=target_mean,
        target_cov=target_cov,
        n_scenarios=40,
        seed=7,
    )
    X = np.array([
        [s.demand_factor, s.carrier_factor_scale, s.fuel_cost_factor]
        for s in sc
    ], dtype=float)
    probs = np.array([s.probability for s in sc])
    assert abs(probs.sum() - 1.0) < 1e-9

    emp_mean = X.mean(axis=0)
    emp_cov = np.cov(X, rowvar=False, bias=False)

    np.testing.assert_allclose(emp_mean, target_mean, atol=1e-6)
    np.testing.assert_allclose(emp_cov, target_cov, atol=1e-6)


# ---------------------------------------------------------------------------
# (b) Custom field mapping
# ---------------------------------------------------------------------------

def test_moment_matching_custom_field_names_and_probs():
    target_mean = np.array([1.0, 0.9])
    target_cov = np.array([[0.02, 0.0], [0.0, 0.01]])
    probs = np.ones(8) / 8.0
    sc = generate_moment_matching_scenarios(
        target_mean=target_mean,
        target_cov=target_cov,
        n_scenarios=8,
        seed=11,
        field_names=("demand_factor", "fuel_cost_factor"),
        probability=probs,
    )
    # carrier_factor_scale should stay at its default (1.0) since we did
    # not request it be driven by the joint distribution.
    for s in sc:
        assert s.carrier_factor_scale == pytest.approx(1.0)
    assert sum(s.probability for s in sc) == pytest.approx(1.0)

    # Dimension-mismatch guard.
    with pytest.raises(ValueError):
        generate_moment_matching_scenarios(
            target_mean=np.array([1.0, 1.0]),
            target_cov=np.array([[0.01]]),   # 1x1 vs 2-d mean
            n_scenarios=5,
        )


# ---------------------------------------------------------------------------
# (c) Chance constraint quantile helper
# ---------------------------------------------------------------------------

def test_chance_constraint_saa_quantile_threshold():
    samples = [10.0, 12.0, 11.0, 13.0, 14.0, 15.0, 20.0, 25.0, 18.0, 17.0]
    cc_loose = ChanceConstraint(name="cc", alpha=0.20, threshold=0.0)
    cc_tight = ChanceConstraint(name="cc", alpha=0.05, threshold=0.0)
    q_loose = cc_loose.saa_quantile_threshold(samples)
    q_tight = cc_tight.saa_quantile_threshold(samples)
    # Tighter α must cover a higher quantile.
    assert q_tight >= q_loose
    # Sorted: [10, 11, 12, 13, 14, 15, 17, 18, 20, 25]. α=0.20 → cum ≥ 0.80
    # first at element 8 (value 18). α=0.05 → cum ≥ 0.95 first at 10th (25).
    assert q_loose == 18.0
    assert q_tight == 25.0


def test_chance_constraint_violation_probability_weighted():
    cc = ChanceConstraint(name="cc", alpha=0.10, threshold=100.0)
    samples = [50.0, 110.0, 105.0, 80.0]
    probs = [0.4, 0.3, 0.2, 0.1]
    # Violators are indices 1 and 2 (110, 105). Combined weight 0.3 + 0.2.
    assert cc.violation_probability(samples, probs) == pytest.approx(0.5)
    # Uniform default weights: 2 / 4 = 0.5.
    assert cc.violation_probability(samples) == pytest.approx(0.5)
    # bonferroni correction divides alpha.
    assert cc.bonferroni_correction(4) == pytest.approx(0.025)


# ---------------------------------------------------------------------------
# (e) Native SAA CC produces a firmer plan with tighter α
# ---------------------------------------------------------------------------

def _build_cc_test_system(T: int = 24) -> ne.EnergySystem:
    """
    Tiny planning system with one dispatchable + one VRE candidate + a
    peaker for firming. The peaker is cheap to install but expensive to
    dispatch — CC should push it up as α tightens to cover rare peaks.
    """
    rng = np.random.default_rng(0)
    sys = ne.EnergySystem("cc_small")
    elec = sys.add_bus("elec", carrier="electricity")

    hours = np.arange(T)
    day = hours % 24
    load = 70 + 30 * np.cos((day - 18) * np.pi / 12) ** 2
    load += rng.normal(0, 1, size=T)
    sys.add_load("d", bus=elec, amount=load)

    sys.add_generator("cheap_fixed", bus=elec, capacity=60,
                      marginal_cost=10, tech="gas")
    sys.add_generator("slack", bus=elec, capacity=5000, marginal_cost=5000)

    cf = np.clip(np.cos((day - 12) * np.pi / 12), 0, None)
    sys.add_generator(
        "solar", bus=elec, capacity=10, marginal_cost=0,
        carrier_factor=cf, extendable=True, min_capacity=10,
        max_capacity=400, capital_cost=40, tech="solar",
    )
    sys.add_generator(
        "peaker", bus=elec, capacity=10, marginal_cost=200,
        extendable=True, min_capacity=10, max_capacity=400,
        capital_cost=15, tech="peaker",
    )
    return sys


def test_saa_cc_tight_alpha_forces_firmer_plan():
    sys_loose = _build_cc_test_system()
    sys_tight = _build_cc_test_system()
    scenarios = [
        Scenario("low",   0.3, demand_factor=0.85),
        Scenario("mid",   0.4, demand_factor=1.00),
        Scenario("high",  0.2, demand_factor=1.25),
        Scenario("spike", 0.1, demand_factor=1.60),
    ]
    firm_credit = {"peaker": 1.0, "gas": 1.0}

    res_loose = solve_saa_chance_constrained(
        sys_loose, scenarios,
        reserve_margin=0.10,
        firm_credit=firm_credit,
        alpha=0.20,            # allow violations up to 20%
        method="benders", max_iter=25,
    )
    res_tight = solve_saa_chance_constrained(
        sys_tight, scenarios,
        reserve_margin=0.10,
        firm_credit=firm_credit,
        alpha=0.05,            # must cover 95% of peak-load scenarios
        method="benders", max_iter=25,
    )
    assert res_loose.status == "optimal"
    assert res_tight.status == "optimal"
    # Tighter α → firmer plan → peaker build ≥ loose-α peaker build.
    # Totals firm credit = sum over credited techs; compare peaker here.
    loose_peaker = res_loose.capacity_decisions.get("peaker", 0.0)
    tight_peaker = res_tight.capacity_decisions.get("peaker", 0.0)
    assert tight_peaker >= loose_peaker - 1e-3
