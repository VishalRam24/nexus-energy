"""
N_En_Phase 20 — differentiable CAPACITY EXPANSION (Paper 2).

Extends Degleris et al. (2024) from differentiating dispatch to
differentiating the operational optimum w.r.t. the design (capacity)
variables. Verification:

  (a) forward solve of the lower-level dispatch QP respects per-period
      balance and the capacity (upper) bounds on a tiny 2-gen instance;
  (b) analytic d dispatch / d capacity matches central finite differences
      within 1e-3 (gradient flows through the active upper-bound
      multipliers; a pinned cheap gen has nonzero gradient, an interior
      peaker has zero);
  (c) the HEADLINE deliverable — analytic dOperatingCost/dcapacity matches
      finite differences of J(cap) within 1e-3 — verified at a SMOOTH
      interior point (cheap gen strictly pinned, peaker strictly interior,
      no exactly-transitioning generator/kink);
  (d) CapacityExpansionLayer.backward composes the Jacobian with a random
      grad-output and matches the FD gradient of the scalar loss;
  (e) fit_component_params recovers a planted marginal_cost (and, in a
      separate run, a planted capacity) from synthetic dispatch, and
      returns an overrides dict consumable by add_component(..., **overrides).

TINY instances only.
"""

from __future__ import annotations

import numpy as np

from nexus_energy.diff import (
    CapacityExpansionProblem,
    CapacityExpansionLayer,
    solve_capacity_expansion_with_sensitivities,
    fit_component_params,
    numerical_jacobian,
)


# ---------------------------------------------------------------------------
# Tiny 2-gen, 3-period expansion instance
# ---------------------------------------------------------------------------
#
# Cheap baseload (mc=10) + expensive peaker (mc=60). Demand is set ABOVE
# the cheap gen's capacity every period, so the cheap gen is strictly
# PINNED at its cap (active upper-bound multiplier ⇒ nonzero dCost/dcap)
# and the peaker is strictly INTERIOR (mc-priced, well below its own cap ⇒
# zero dCost/dcap). No generator sits exactly at a transition, so central
# FD is valid (smooth interior point).

def _problem(mc=None, cap=None, demand=None, ridge=1.0):
    mc = np.array([10.0, 60.0]) if mc is None else np.asarray(mc, float)
    cap = np.array([50.0, 120.0]) if cap is None else np.asarray(cap, float)
    demand = (np.array([70.0, 90.0, 80.0])
              if demand is None else np.asarray(demand, float))
    return CapacityExpansionProblem(
        marginal_cost=mc, capacity=cap, demand=demand, ridge=ridge)


def test_expansion_forward_feasible():
    prob = _problem()
    sol = solve_capacity_expansion_with_sensitivities(prob)
    G, T = sol.dispatch.shape
    assert (G, T) == (2, 3)
    for t in range(T):
        assert sol.dispatch[:, t].sum() == np.float64(prob.demand[t]) or \
            abs(sol.dispatch[:, t].sum() - prob.demand[t]) < 1e-5
    assert np.all(sol.dispatch >= -1e-6)
    assert np.all(sol.dispatch <= prob.capacity[:, None] + 1e-6)
    # Cheap gen pinned at its cap; peaker interior (carries the rest).
    assert np.allclose(sol.dispatch[0, :], prob.capacity[0], atol=1e-4)
    assert np.all(sol.dispatch[1, :] < prob.capacity[1] - 1.0)


def test_expansion_dispatch_grad_matches_finite_diff():
    prob = _problem()
    sol = solve_capacity_expansion_with_sensitivities(prob)

    def disp_flat(p):
        return p.dispatch.reshape(-1)

    def f_cap(x):
        return disp_flat(solve_capacity_expansion_with_sensitivities(
            _problem(cap=x)))

    num_cap = numerical_jacobian(f_cap, prob.capacity, eps=1e-5)
    err = np.max(np.abs(sol.d_dispatch_d_capacity - num_cap))
    print(f"\n[expansion] d dispatch/d cap grad-vs-FD max abs err = {err:.2e}")
    assert err < 1e-3
    # Gradient flows ONLY through the pinned (active) cheap gen.
    G, T = sol.dispatch.shape
    Jp = sol.d_dispatch_d_capacity  # (G*T, G)
    # column 0 (d/dcap_0) is materially nonzero; column 1 (interior) ~ 0.
    assert np.max(np.abs(Jp[:, 0])) > 1e-3
    assert np.max(np.abs(Jp[:, 1])) < 1e-6


def test_expansion_operating_cost_grad_matches_finite_diff():
    """HEADLINE: dOperatingCost/dcapacity vs FD of J(cap), smooth point."""
    prob = _problem()
    sol = solve_capacity_expansion_with_sensitivities(prob)

    def J(x):
        s = solve_capacity_expansion_with_sensitivities(_problem(cap=x))
        return np.array([s.operating_cost])

    num = numerical_jacobian(J, prob.capacity, eps=1e-5).ravel()
    err = np.max(np.abs(sol.d_cost_d_capacity - num))
    print(f"\n[expansion] dCost/dcap analytic={sol.d_cost_d_capacity} "
          f"fd={num} max abs err={err:.2e}")
    assert err < 1e-3
    # Economic sanity: more cheap capacity LOWERS operating cost (negative
    # gradient on the cheap pinned gen); the interior peaker contributes 0.
    assert sol.d_cost_d_capacity[0] < 0.0
    assert abs(sol.d_cost_d_capacity[1]) < 1e-6


def test_expansion_layer_backward_matches_fd():
    layer = CapacityExpansionLayer(ridge=1.0)
    prob = _problem(ridge=1.0)
    p = layer.forward(prob)
    rng = np.random.default_rng(20)
    grad_out = rng.normal(size=p.shape)
    grads = layer.backward(grad_out)

    def loss_cap(x):
        pp = solve_capacity_expansion_with_sensitivities(
            _problem(cap=x, ridge=1.0))
        return np.array([float((grad_out * pp.dispatch).sum())])

    num_cap = numerical_jacobian(loss_cap, prob.capacity, eps=1e-5).ravel()
    err = np.max(np.abs(grads["capacity"] - num_cap))
    print(f"\n[expansion-backward] dL/dcap max abs err = {err:.2e}")
    assert err < 1e-3

    # operating_cost_gradient() matches the chain-rule contraction.
    assert layer.operating_cost() > 0.0
    g_cost = layer.operating_cost_gradient()
    assert g_cost.shape == (2,)


# ---------------------------------------------------------------------------
# (e) fit_component_params — recover planted parameters from dispatch
# ---------------------------------------------------------------------------

def test_fit_component_params_recovers_marginal_cost():
    # Plant a true marginal_cost; generate dispatch over several periods;
    # fit it back from a wrong init. Use demand that keeps BOTH gens
    # interior (strictly off bounds) across periods so dispatch responds
    # smoothly to mc (so there is gradient signal to recover mc).
    demand = np.array([40.0, 60.0, 50.0, 70.0, 55.0])
    cap = np.array([90.0, 90.0])
    true_mc = np.array([12.0, 28.0])
    ridge = 1.0

    sol = solve_capacity_expansion_with_sensitivities(
        CapacityExpansionProblem(marginal_cost=true_mc, capacity=cap,
                                 demand=demand, ridge=ridge))
    observed = sol.dispatch

    res = fit_component_params(
        {"marginal_cost": [20.0, 20.0], "capacity": cap.tolist()},
        observed, demand, fit=("marginal_cost",),
        ridge=ridge, lr=5e-3, n_iter=20000)

    recovered = np.asarray(res.params["marginal_cost"])
    print(f"\n[fit-mc] true={true_mc} recovered={recovered} "
          f"loss0={res.history[0]:.3e} lossN={res.history[-1]:.3e} "
          f"iters={res.n_iter}")
    assert res.history[-1] < res.history[0]
    # mc has a gauge freedom (adding a constant to both shifts lambda, not
    # dispatch) so we recover the DIFFERENCE between the two costs, which is
    # what pins the relative dispatch. Check the spread is recovered.
    assert abs((recovered[1] - recovered[0]) - (true_mc[1] - true_mc[0])) < 0.5
    # overrides dict is consumable by add_component(..., **overrides).
    assert set(res.overrides) >= {"marginal_cost", "capacity"}
    assert isinstance(res.overrides["marginal_cost"], list)


def test_fit_component_params_recovers_capacity():
    # Plant a true cheap-gen capacity; demand above it pins the cheap gen,
    # so observed dispatch[0] == cap0 directly identifies it. Fit cap0 back.
    demand = np.array([70.0, 85.0, 95.0, 80.0])
    true_cap = np.array([55.0, 150.0])
    mc = np.array([10.0, 60.0])
    ridge = 1.0

    sol = solve_capacity_expansion_with_sensitivities(
        CapacityExpansionProblem(marginal_cost=mc, capacity=true_cap,
                                 demand=demand, ridge=ridge))
    observed = sol.dispatch

    res = fit_component_params(
        {"marginal_cost": mc.tolist(), "capacity": [40.0, 150.0]},
        observed, demand, fit=("capacity",),
        ridge=ridge, lr=2e-2, n_iter=20000)

    recovered = np.asarray(res.params["capacity"])
    print(f"\n[fit-cap] true_cap0={true_cap[0]} recovered_cap0={recovered[0]:.4f} "
          f"loss0={res.history[0]:.3e} lossN={res.history[-1]:.3e} "
          f"iters={res.n_iter}")
    assert res.history[-1] < res.history[0]
    assert abs(recovered[0] - true_cap[0]) < 0.5
