"""
Phase 9 depth pass — defining-property tests for the stochastic/robust
algorithms added in stochastic.py:

  9.1 SDDiP            — LB monotone ↑ AND converges to the deterministic-
                         equivalent MILP optimum on a tiny binary instance.
  9.2 General-form CC  — Big-M indicator MILP covers ≥ (1-α) probability
                         mass and tightens (more capacity) as α shrinks.
  9.3 Wasserstein DRO  — reduces to SAA as ε→0; worst-case cost monotone ↑ in ε.
  9.4 Risk-averse cuts — CVaR change-of-measure plan has lower worst-case
                         cost (higher worst-case coverage) than risk-neutral.
  2.5 Forced outage    — generated scenarios have empirical outage frequency
                         ≈ the forced-outage rate.

All instances are TINY (≤ 3 stages, ≤ 4 scenarios, ≤ 12 timesteps) per the
24GB-RAM constraint.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

import nexus_opt as nx

from nexus_energy.stochastic import (
    StageProblem,
    cvar_change_of_measure,
    generate_forced_outage_scenarios,
    solve_general_chance_constrained,
    solve_risk_averse_benders,
    solve_sddip,
    solve_wasserstein_dro,
)


# ===========================================================================
# 9.1 SDDiP
# ===========================================================================
#
# Two-stage binary capacity-expansion toy:
#   Stage 0: build x ∈ {0,1} units of a generator at capex C0 each, cap 2.
#            State threaded forward = installed units 'b'.
#   Stage 1: demand d_s must be served. Available capacity = 5·b (each unit
#            = 5 MW). Served by installed gen at mc=1, or by expensive
#            slack at mc=20. Recourse cost = 1·min(d,5b) + 20·max(0,d-5b).
#
# DE optimum we can enumerate over b ∈ {0,1,2}.

UNIT_MW = 5.0
C0 = 8.0          # capex per unit (stage-0 immediate)
MC_GEN = 1.0
MC_SLACK = 20.0
DEMANDS = [4.0, 9.0]          # two stage-1 realisations
DEM_PROB = [0.5, 0.5]


def _make_sddip_stages():
    """Build the 2-stage StageProblem list for the toy above."""
    stage1_scen = [{"d": d} for d in DEMANDS]

    def build_stage0(model, z_prev, stage_idx):
        # Decide installed units b ∈ {0,1,2} via two binaries (b = b0 + b1).
        b0 = model.binary("b0")
        b1 = model.binary("b1")
        b = b0 + b1
        # Outgoing state 'b' must live in a continuous-but-integer-valued
        # var so the copy/relaxation arithmetic is clean.
        bvar = model.variable("b_state", lower=0.0, upper=2.0)
        model.add(bvar - b == 0.0, name="b_def")
        stage_cost = C0 * b0 + C0 * b1
        return stage_cost, {"b": bvar}

    def build_stage1(model, z_prev, stage_idx):
        b = z_prev["b"]                       # incoming installed units
        k = build_stage1._active[0]
        d = stage1_scen[k]["d"]
        gen = model.variable("gen", lower=0.0, upper=1e6)
        slack = model.variable("slack", lower=0.0, upper=1e6)
        # gen ≤ UNIT_MW · b
        model.add(UNIT_MW * b - gen >= 0.0, name="cap")
        # gen + slack ≥ d  (serve demand)
        model.add(gen + slack >= d, name="serve")
        stage_cost = MC_GEN * gen + MC_SLACK * slack
        # No outgoing state at the leaf.
        return stage_cost, {}

    s0 = StageProblem(build=build_stage0, state_names=("b",),
                      state_bounds={"b": (0.0, 2.0)})
    s1 = StageProblem(build=build_stage1, state_names=("b",),
                      state_bounds={"b": (0.0, 2.0)})
    # _active set by the solver; default 0.
    s1.build._active = [0]
    return [s0, s1], [[{}], stage1_scen]


def _deterministic_equivalent_sddip():
    """Enumerate b ∈ {0,1,2}; pick min of capex + E[recourse]."""
    best = float("inf")
    for b in (0, 1, 2):
        capex = C0 * b
        exp_rec = 0.0
        for d, p in zip(DEMANDS, DEM_PROB):
            served = min(d, UNIT_MW * b)
            slack = max(0.0, d - UNIT_MW * b)
            exp_rec += p * (MC_GEN * served + MC_SLACK * slack)
        best = min(best, capex + exp_rec)
    return best


def test_sddip_lower_bound_monotone_and_converges():
    stages, scen = _make_sddip_stages()
    de = _deterministic_equivalent_sddip()
    res = solve_sddip(
        stages, scen,
        stage_probabilities=[[1.0], DEM_PROB],
        max_iter=30, n_forward=2, lagrangian_iters=80,
        lagrangian_step=4.0, seed=1,
        deterministic_equivalent=de, tol=1e-4, verbose=False,
    )
    assert res.status == "optimal", res.status
    lbs = res.lower_bounds
    assert len(lbs) >= 1
    # (i) Monotone non-decreasing lower bounds.
    for a, b in zip(lbs, lbs[1:]):
        assert b >= a - 1e-7, f"LB decreased: {a} -> {b}"
    # (ii) Converges to the deterministic-equivalent MILP optimum.
    print(f"SDDiP: DE={de:.4f} finalLB={lbs[-1]:.4f} UB={res.upper_bound:.4f} "
          f"gap={res.gap}")
    assert lbs[-1] <= de + 1e-4
    assert abs(lbs[-1] - de) <= 1e-2 * (1.0 + abs(de)), (
        f"LB {lbs[-1]} did not reach DE {de}")


# ===========================================================================
# 9.2 General-form chance constraints (Big-M binaries)
# ===========================================================================

def test_general_cc_coverage_and_alpha_monotone():
    # Random peak loads; one rare spike.
    peaks = [80.0, 90.0, 100.0, 160.0]
    probs = [0.4, 0.3, 0.2, 0.1]
    credits = {"firm": 1.0}
    capex = {"firm": 1.0}
    bounds = {"firm": (0.0, 500.0)}

    # alpha = 0.1: may drop ≤ 10% mass → must cover the 0.9 mass below the
    # spike, i.e. firm ≥ 100 (covers peaks 80/90/100, drops the 0.1 spike).
    r_loose = solve_general_chance_constrained(
        peak_loads=peaks, probabilities=probs, credits=credits,
        capex=capex, cap_bounds=bounds, alpha=0.10,
    )
    # alpha = 0.0: must cover everything → firm ≥ 160.
    r_tight = solve_general_chance_constrained(
        peak_loads=peaks, probabilities=probs, credits=credits,
        capex=capex, cap_bounds=bounds, alpha=0.0,
    )
    assert r_loose["status"] == "optimal"
    assert r_tight["status"] == "optimal"
    print(f"CC loose: firm={r_loose['firm_capacity']:.1f} "
          f"cover={r_loose['coverage']:.3f} z={r_loose['indicators']}")
    print(f"CC tight: firm={r_tight['firm_capacity']:.1f} "
          f"cover={r_tight['coverage']:.3f} z={r_tight['indicators']}")
    # Coverage ≥ 1-α for both.
    assert r_loose["coverage"] >= 1.0 - 0.10 - 1e-9
    assert r_tight["coverage"] >= 1.0 - 1e-9
    # Tighter α (cover the spike) needs more firm capacity.
    assert r_tight["firm_capacity"] >= r_loose["firm_capacity"] - 1e-6
    # Loose plan covers the 0.9-mass requirement: firm ≥ 100.
    assert r_loose["firm_capacity"] >= 100.0 - 1e-6
    assert r_tight["firm_capacity"] >= 160.0 - 1e-6


# ===========================================================================
# 9.3 Wasserstein DRO
# ===========================================================================
#
# Loss(cap, ξ) = newsvendor-style piecewise-linear convex in ξ:
#   piece A (under-build):  c_o·(ξ - cap)   ... slope_xi = +c_o
#   piece B (over-build):   c_u·(cap - ξ)   ... slope_xi = -c_u
# plus build cost c_b·cap on both pieces.
# Decision cap ∈ [0, 20]; samples of demand ξ.

def _dro_pieces():
    c_b, c_o, c_u = 1.0, 5.0, 2.0
    # piece A: c_b·cap + c_o·(ξ - cap) = (c_b - c_o)·cap + c_o·ξ
    # piece B: c_b·cap + c_u·(cap - ξ) = (c_b + c_u)·cap - c_u·ξ
    loss_slopes = [
        [c_b - c_o, c_o],   # [slope_cap, slope_xi]
        [c_b + c_u, -c_u],
    ]
    loss_intercepts = [[0.0], [0.0]]
    return loss_slopes, loss_intercepts


def test_wasserstein_dro_saa_limit_and_eps_monotone():
    slopes, intercepts = _dro_pieces()
    samples = [6.0, 8.0, 10.0, 12.0, 14.0]

    # SAA reference: ε = 0 should equal min_cap (1/N) Σ_i ℓ(cap, ξ_i).
    r0 = solve_wasserstein_dro(
        loss_slopes=slopes, loss_intercepts=intercepts, samples=samples,
        cap_bounds=(0.0, 20.0), epsilon=0.0,
    )
    assert r0["status"] == "optimal"

    # Brute-force SAA optimum over a fine cap grid for cross-check.
    def saa_cost(cap):
        tot = 0.0
        for xi in samples:
            pieces = [intercepts[k][0] + slopes[k][0] * cap + slopes[k][1] * xi
                      for k in range(len(slopes))]
            tot += max(pieces)
        return tot / len(samples)

    grid = np.linspace(0, 20, 2001)
    saa_opt = min(saa_cost(c) for c in grid)
    print(f"DRO ε=0: obj={r0['worst_case_cost']:.5f} SAA_grid={saa_opt:.5f} "
          f"cap={r0['cap']:.3f}")
    assert abs(r0["worst_case_cost"] - saa_opt) <= 1e-2

    # Monotonicity: worst-case cost non-decreasing in ε.
    eps_list = [0.0, 0.5, 1.0, 2.0, 4.0]
    wc = []
    for eps in eps_list:
        r = solve_wasserstein_dro(
            loss_slopes=slopes, loss_intercepts=intercepts, samples=samples,
            cap_bounds=(0.0, 20.0), epsilon=eps,
        )
        assert r["status"] == "optimal"
        wc.append(r["worst_case_cost"])
    print(f"DRO worst-case vs ε {eps_list}: "
          f"{[round(w,4) for w in wc]}")
    for a, b in zip(wc, wc[1:]):
        assert b >= a - 1e-7, f"DRO worst-case decreased with ε: {a}->{b}"
    # Strictly more conservative at the largest ε (loss is ξ-sensitive).
    assert wc[-1] > wc[0] + 1e-3


# ===========================================================================
# 9.4 Risk-averse Benders cuts (nested CVaR change-of-measure)
# ===========================================================================

def test_cvar_change_of_measure_reweights_tail():
    costs = [10.0, 20.0, 100.0]
    probs = [1 / 3, 1 / 3, 1 / 3]
    q = cvar_change_of_measure(costs, probs, alpha=1 / 3)
    # α = 1/3 → CVaR is the single worst scenario → all mass on cost=100.
    assert q[2] == pytest.approx(1.0, abs=1e-9)
    assert q[0] == pytest.approx(0.0, abs=1e-9)
    assert sum(q) == pytest.approx(1.0)


def test_risk_averse_plan_lower_worst_case():
    # Single capacity 'cap' ∈ [0, 20], capex small. Per-scenario recourse:
    #   serve demand d_s; gen up to cap at mc 1, slack at mc 20.
    #   cost_s(cap) = 1·min(d_s,cap) + 20·max(0, d_s-cap)
    #   dual wrt cap (subgradient): -19 if cap < d_s else -1  (cost falls
    #   by 19 per extra unit while slack is active).
    demands = [5.0, 7.0, 9.0, 18.0]   # last is a heavy tail
    probs = [0.4, 0.3, 0.2, 0.1]
    capex = 2.0

    def scen_costs(cap):
        out = []
        for d in demands:
            served = min(d, cap)
            slack = max(0.0, d - cap)
            out.append(MC_GEN * served + MC_SLACK * slack)
        return out

    def cap_dual(cap):
        # d(cost_s)/d(cap): if slack active (cap < d): 1 - 20 = -19, else
        # still serving from gen (cap ≥ d): increasing cap doesn't change
        # served (= d) so derivative 0; but to give Benders a valid
        # subgradient at the kink use -19 when cap<d, 0 otherwise.
        return [(-19.0 if cap < d - 1e-9 else 0.0) for d in demands]

    rn = solve_risk_averse_benders(
        scenario_costs_fn=scen_costs, capex=capex, cap_bounds=(0.0, 20.0),
        probabilities=probs, cap_dual_fn=cap_dual,
        alpha=0.2, risk_lambda=0.0, max_iter=40,   # risk-neutral
    )
    ra = solve_risk_averse_benders(
        scenario_costs_fn=scen_costs, capex=capex, cap_bounds=(0.0, 20.0),
        probabilities=probs, cap_dual_fn=cap_dual,
        alpha=0.2, risk_lambda=1.0, max_iter=40,   # pure CVaR_0.2
    )
    assert rn["status"] == "optimal"
    assert ra["status"] == "optimal"
    print(f"risk-neutral: cap={rn['cap']:.3f} wc={rn['worst_case_cost']:.3f}")
    print(f"risk-averse : cap={ra['cap']:.3f} wc={ra['worst_case_cost']:.3f}")
    # Risk-averse over-weights the tail → builds at least as much capacity.
    assert ra["cap"] >= rn["cap"] - 1e-6
    # And achieves a worst-case cost no worse (higher coverage of the tail).
    assert ra["worst_case_cost"] <= rn["worst_case_cost"] + 1e-6
    # On this tail-heavy instance the risk-averse plan is strictly firmer.
    assert ra["cap"] > rn["cap"] + 1e-3


# ===========================================================================
# 2.5 Forced-outage scenario generation
# ===========================================================================

def test_forced_outage_bernoulli_frequency():
    gens = {"g1": 0.10, "g2": 0.30}
    N = 4000
    scen = generate_forced_outage_scenarios(gens, n_scenarios=N, seed=7)
    assert len(scen) == N
    assert abs(sum(s.probability for s in scen) - 1.0) < 1e-9
    # Empirical outage frequency per generator (override capacity == 0).
    freq = {g: 0 for g in gens}
    for s in scen:
        for (kind, name, field_), val in s.overrides.items():
            if kind == "gen" and field_ == "capacity" and val == 0.0:
                freq[name] += 1
    for g, q in gens.items():
        emp = freq[g] / N
        print(f"FOR {g}: target={q:.3f} empirical={emp:.3f}")
        assert abs(emp - q) < 0.03, f"{g}: {emp} vs {q}"


def test_forced_outage_markov_downtime_fraction():
    gens = {"g1": 0.20}
    scen = generate_forced_outage_scenarios(
        gens, n_scenarios=200, n_timesteps=12, mttr=3.0, seed=3)
    # Average down-fraction across scenarios ≈ FOR.
    downs = []
    for s in scen:
        avail = s.overrides[("gen", "g1", "availability")]
        downs.append(1.0 - float(np.mean(avail)))
    emp = float(np.mean(downs))
    print(f"Markov FOR g1: target=0.20 empirical_downfrac={emp:.3f}")
    assert abs(emp - 0.20) < 0.06
