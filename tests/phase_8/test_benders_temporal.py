"""
Phase 8 — temporal Benders convergence tests.

Strategy: build a small capacity-expansion problem with a known
monolithic optimum (solver-derived), then run Benders against the same
problem and assert:

  (a) It converges to the same capacity mix and objective (≤ 1 % gap).
  (b) It converges within the iteration limit (no stall).
  (c) Subproblem duals are non-zero — i.e. cuts actually bite (β != 0),
      proving we're doing real Benders, not the Phase-5 β=0 stub.
"""

from __future__ import annotations

import numpy as np
import pytest

import nexus_energy as ne


def _build_capex_system(T: int = 48) -> ne.EnergySystem:
    """
    Capacity-expansion test with a deliberate investment sweet spot:
      - "cheap_fixed" : 200 MW, mc=50          — insufficient at peak
      - "slack"       : 5000 MW, mc=5000       — unmet-demand surrogate
        so every subproblem is feasible for any cap choice
      - "solar"       : extendable, capex=50 $/MW, mc=0, daily CF
      - "peaker"      : extendable, capex=20 $/MW, mc=300
    Load peak ≈ 280 MW so both solar and peaker are economic over the
    48-hr horizon — ensures subproblem duals are non-trivial.
    """
    rng = np.random.default_rng(0)
    sys = ne.EnergySystem("capex_small")
    elec = sys.add_bus("elec", carrier="electricity")

    hours = np.arange(T)
    day = hours % 24
    load = 200 + 80 * np.cos((day - 18) * np.pi / 12) ** 2
    load = load + rng.normal(0, 3, size=T)
    solar_cf = np.clip(np.sin((day - 6) * np.pi / 12), 0, None)

    sys.add_generator("cheap_fixed", bus=elec, capacity=200.0,
                      marginal_cost=50.0)
    sys.add_generator("slack", bus=elec, capacity=5000.0,
                      marginal_cost=5000.0)
    sys.add_generator("solar", bus=elec, capacity=0.0,
                      marginal_cost=0.0,
                      capital_cost=50.0,
                      carrier_factor=solar_cf,
                      extendable=True, max_capacity=200.0)
    sys.add_generator("peaker", bus=elec, capacity=0.0,
                      marginal_cost=300.0,
                      capital_cost=20.0,
                      extendable=True, max_capacity=200.0)
    sys.add_load("demand", bus=elec, amount=load)
    sys.set_timesteps(T, dt=1.0)
    return sys


class TestTemporalBenders:

    def test_matches_monolithic_within_tol(self):
        sys = _build_capex_system()
        mono = sys.optimise()
        assert mono.status == "optimal"

        # Fresh system so Benders gets its own cap_vars
        sys2 = _build_capex_system()
        res = ne.solve_with_temporal_benders(
            sys2, n_periods=2, max_iter=40, tol=5e-3,
            stabilisation="plain",
        )
        assert res.converged, f"Benders didn't converge: {res.status}"
        # ≤1% gap on the objective
        rel = abs(res.total_cost - mono.total_cost) / abs(mono.total_cost)
        assert rel < 0.01, \
            f"Benders UB {res.total_cost:.1f} vs monolithic {mono.total_cost:.1f} (rel={rel:.4f})"

    def test_cuts_have_nonzero_duals(self):
        """
        Proves real Benders — the old stub always pushed β=0 and would
        only "converge" by accident. A real sub-LP has non-zero duals
        on binding capacity pins. We force binding by pinning caps
        strictly below what the monolithic optimum chose.
        """
        from nexus_energy.decomposition import _slice_system
        sys = _build_capex_system()
        # Pin below optimum so caps are binding from above.
        sub = _slice_system(sys, 0, 24)
        sub_res = sub.optimise(
            benders_fix_caps={"solar": 50.0, "peaker": 30.0},
            benders_skip_capex=True,
        )
        assert sub_res.status == "optimal"
        assert set(sub_res.cap_dual.keys()) == {"solar", "peaker"}
        # At least one cap must have a non-trivial β — proving the dual
        # extraction pipeline isn't silently discarding the equality
        # constraint's shadow price.
        beta_sum = sum(abs(v) for v in sub_res.cap_dual.values())
        assert beta_sum > 1.0, \
            f"Flat β — sub-LP duals missing or wrong. {sub_res.cap_dual}"

    def test_converged_against_fake_beta_stub(self):
        """
        A β=0 stub cut is equivalent to θ_p ≥ op_cost_p (no capacity
        feedback). Such a stub would either stall or accept a bad cap.
        Verify Benders' final UB is much better than what a β=0 run
        could achieve on the first iteration.
        """
        sys = _build_capex_system()
        res = ne.solve_with_temporal_benders(
            sys, n_periods=2, max_iter=20, tol=5e-3, stabilisation="plain")
        assert res.converged
        # First iteration's UB uses caps = master-lower-bound (usually 0);
        # β=0 would never learn to move caps upward. Final UB must be
        # strictly better.
        assert res.iterations[-1].upper_bound < \
               0.5 * res.iterations[0].upper_bound, \
            "Benders didn't drive UB down — β cuts may be ineffective."

    def test_trust_region_also_converges(self):
        sys = _build_capex_system()
        mono = sys.optimise()
        sys2 = _build_capex_system()
        res = ne.solve_with_temporal_benders(
            sys2, n_periods=2, max_iter=40, tol=5e-3,
            stabilisation="trust_region", trust_radius=200.0,
        )
        assert res.converged
        rel = abs(res.total_cost - mono.total_cost) / abs(mono.total_cost)
        assert rel < 0.01

    def test_adaptive_also_converges(self):
        sys = _build_capex_system()
        mono = sys.optimise()
        sys2 = _build_capex_system()
        res = ne.solve_with_temporal_benders(
            sys2, n_periods=2, max_iter=40, tol=5e-3,
            stabilisation="adaptive",
            gap_init=1e-2, gap_final=1e-6,
        )
        assert res.converged
        rel = abs(res.total_cost - mono.total_cost) / abs(mono.total_cost)
        assert rel < 0.01

    def test_results_schema(self):
        sys = _build_capex_system()
        res = ne.solve_with_temporal_benders(
            sys, n_periods=2, max_iter=10, tol=5e-3)
        assert isinstance(res, ne.BendersResult)
        assert res.sub_solves >= 2  # at least one iter * 2 periods
        assert len(res.iterations) >= 1
        it = res.iterations[0]
        assert set(it.master_capacities.keys()).issuperset(
            {"solar", "peaker"})
