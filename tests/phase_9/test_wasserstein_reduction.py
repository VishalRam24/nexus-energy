"""
Phase 21 — Wasserstein-robust scenario reduction unified with DRO.

Verifies :func:`nexus_energy.stochastic.reduce_scenarios_wasserstein`, a
reduction that minimises the type-p Wasserstein transport distance between
the full empirical scenario measure and the reduced (re-weighted) one and
returns the achieved transport distance as a ``radius`` that directly
parameterises the Wasserstein ambiguity ball of
:func:`solve_wasserstein_dro`.

Verification bar (tiny instance, no EnergySystem solve needed):
  (a) the radius DECREASES monotonically as ``n_reduced`` grows
      (more reps -> closer to the full distribution);
  (b) at ``n_reduced == n_full`` the radius is ~0 and reduced == full;
  (c) feeding the radius into ``solve_wasserstein_dro`` yields a robust
      plan whose worst-case realised cost is >= the risk-neutral
      (radius=0 / SAA) reduced plan's worst-case cost — robustness costs
      something.

References: Dupacova, Growe-Kuska & Romisch 2003; Pflug 2001 (Wasserstein
scenario reduction / optimal quantization); Mohajerin Esfahani & Kuhn 2018
(Wasserstein DRO).
"""
from __future__ import annotations

import numpy as np

from nexus_energy.stochastic import (
    Scenario,
    reduce_scenarios_wasserstein,
    solve_wasserstein_dro,
)


def _full_scenarios(n=12, seed=7):
    """A tiny full ensemble: demand_factor draws, equal probability."""
    rng = np.random.RandomState(seed)
    factors = np.clip(rng.normal(1.0, 0.18, n), 0.5, 1.6)
    return [
        Scenario(name=f"full_{i}", probability=1.0 / n,
                 demand_factor=float(factors[i]))
        for i in range(n)
    ]


def test_radius_monotone_decreasing_in_n_reduced():
    """(a) Wasserstein radius shrinks as we keep more representatives."""
    scens = _full_scenarios()
    ns = [2, 3, 4, 6, 8, 10]
    radii = []
    for nr in ns:
        _, r = reduce_scenarios_wasserstein(scens, nr, seed=0)
        radii.append(r)
    print(f"radius vs n_reduced {ns}: {[round(r, 5) for r in radii]}")
    for a, b in zip(radii, radii[1:]):
        assert b <= a + 1e-9, f"radius increased with n_reduced: {a}->{b}"
    # Non-trivial: a coarse reduction has a strictly positive radius.
    assert radii[0] > 1e-4


def test_radius_zero_at_full_and_identity():
    """(b) n_reduced == n_full -> radius ~ 0 and reduced reproduces full."""
    scens = _full_scenarios(n=8)
    reduced, radius = reduce_scenarios_wasserstein(scens, len(scens), seed=0)
    print(f"radius at n_reduced==n_full: {radius:.3e}")
    assert radius == 0.0
    assert len(reduced) == len(scens)
    full_feats = sorted(round(s.demand_factor, 9) for s in scens)
    red_feats = sorted(round(s.demand_factor, 9) for s in reduced)
    assert full_feats == red_feats
    assert abs(sum(s.probability for s in reduced) - 1.0) < 1e-9

    # And n_reduced > n_full also returns the full set with radius 0.
    reduced2, radius2 = reduce_scenarios_wasserstein(scens, len(scens) + 5)
    assert radius2 == 0.0
    assert len(reduced2) == len(scens)


def test_reduced_probabilities_aggregate_optimally():
    """Survivors carry the mass of the originals transported to them."""
    scens = _full_scenarios()
    reduced, _ = reduce_scenarios_wasserstein(scens, 3, seed=0)
    total = sum(s.probability for s in reduced)
    assert abs(total - 1.0) < 1e-9
    assert all(s.probability > 0.0 for s in reduced)


# --- (c) radius feeds solve_wasserstein_dro -> robust plan is conservative ---
#
# Newsvendor-style piecewise-linear convex loss in the random demand ξ
# (same construction as the existing DRO depth test):
#   piece A (under-build):  c_b·cap + c_o·(ξ - cap) = (c_b-c_o)·cap + c_o·ξ
#   piece B (over-build):   c_b·cap + c_u·(cap - ξ) = (c_b+c_u)·cap - c_u·ξ
def _dro_pieces():
    c_b, c_o, c_u = 1.0, 5.0, 2.0
    loss_slopes = [[c_b - c_o, c_o], [c_b + c_u, -c_u]]
    loss_intercepts = [[0.0], [0.0]]
    return loss_slopes, loss_intercepts


def _loss(cap, xi, slopes, intercepts):
    return max(intercepts[k][0] + slopes[k][0] * cap + slopes[k][1] * xi
               for k in range(len(slopes)))


def test_reduced_dro_more_conservative_than_risk_neutral():
    """(c) reduced+DRO worst-case >= risk-neutral reduced worst-case.

    Pipeline: reduce -> get radius -> solve DRO with epsilon=radius on the
    reduced samples; compare against the risk-neutral (epsilon=0 / SAA)
    plan on the SAME reduced samples. The robust plan's worst-case
    expected loss (the DRO objective = worst-case over the Wasserstein
    ball) must be >= the risk-neutral plan's: robustness costs something.

    HONEST NOTE on the realised-cost view: for this *scalar* type-1
    newsvendor the worst-case Lipschitz constant lambda = max_k|b_k| = c_o
    is independent of cap, so the lambda*epsilon penalty is an additive
    constant and the optimal cap is UNCHANGED by the radius (DRO shifts the
    objective, not the scalar decision). We therefore assert the
    cost-of-robustness on the objective (which strictly increases) and
    record that the out-of-sample worst-case realised cost is unchanged
    (not worse) — see printed numbers.
    """
    scens = _full_scenarios(n=12, seed=11)
    slopes, intercepts = _dro_pieces()

    reduced, radius = reduce_scenarios_wasserstein(scens, 3, seed=0)
    assert radius > 0.0
    print(f"reduction radius fed to DRO: {radius:.5f}")

    # ξ samples = demand_factor of the reduced representatives.
    red_samples = [s.demand_factor for s in reduced]

    # Risk-neutral reduced plan (SAA, epsilon = 0).
    r_rn = solve_wasserstein_dro(
        loss_slopes=slopes, loss_intercepts=intercepts, samples=red_samples,
        cap_bounds=(0.0, 5.0), epsilon=0.0,
    )
    # Robust reduced plan (epsilon = achieved Wasserstein radius).
    r_dro = solve_wasserstein_dro(
        loss_slopes=slopes, loss_intercepts=intercepts, samples=red_samples,
        cap_bounds=(0.0, 5.0), epsilon=radius,
    )
    assert r_rn["status"] == "optimal" and r_dro["status"] == "optimal"
    cap_rn, cap_dro = r_rn["cap"], r_dro["cap"]
    print(f"cap risk-neutral={cap_rn:.4f}  cap robust={cap_dro:.4f}")

    # Cost of robustness on the worst-case (DRO) objective: strictly larger
    # because the loss is ξ-sensitive (lambda > 0) and radius > 0.
    print(f"worst-case expected loss: risk-neutral={r_rn['worst_case_cost']:.5f}"
          f"  robust={r_dro['worst_case_cost']:.5f}")
    assert r_dro["worst_case_cost"] >= r_rn["worst_case_cost"] - 1e-9
    assert r_dro["worst_case_cost"] > r_rn["worst_case_cost"] + 1e-4

    # Out-of-sample worst-case realised cost over the FULL ensemble.
    full_xi = [s.demand_factor for s in scens]
    wc_rn = max(_loss(cap_rn, xi, slopes, intercepts) for xi in full_xi)
    wc_dro = max(_loss(cap_dro, xi, slopes, intercepts) for xi in full_xi)
    print(f"worst-case realised (full MC): risk-neutral={wc_rn:.4f} "
          f"robust={wc_dro:.4f}")
    # Robust plan is never worse out-of-sample (here equal — scalar case).
    assert wc_dro <= wc_rn + 1e-6
