"""
Phase 12.1 / 12.3 — multi-bus multi-period differentiable dispatch and
the demand-elasticity recovery example.

Coverage:
  (a) multi-bus, multi-period forward solve respects per-bus balance,
      gen bounds and line limits;
  (b) analytic d dispatch / d {mc, capacity, demand, line_limit} match
      central finite differences within 1e-3 on a tiny 2-bus, 3-period
      instance (interior + bound-active);
  (c) MultiBusDispatchLayer.backward composes the Jacobian with a random
      grad-output and matches the finite-difference gradient of the
      scalar loss;
  (d) fit_demand_elasticity recovers a planted linear elasticity from
      synthetic (price, dispatch) pairs.
"""

from __future__ import annotations

import numpy as np
import pytest

from nexus_energy.diff import (
    MultiBusDispatchProblem,
    MultiBusDispatchLayer,
    solve_multibus_dispatch_with_sensitivities,
    solve_dispatch_with_sensitivities,
    fit_demand_elasticity,
    numerical_jacobian,
)


# ---------------------------------------------------------------------------
# Tiny 2-bus, 3-period instance
# ---------------------------------------------------------------------------

def _problem(mc=None, cap=None, demand=None, flim=None, ridge=1.0):
    # Bus 0 has a cheap+mid gen, bus 1 has an expensive gen and most load,
    # connected by one line. Forces flow across the line.
    gen_bus = np.array([0, 0, 1])
    mc = np.array([10.0, 30.0, 80.0]) if mc is None else np.asarray(mc, float)
    cap = np.array([60.0, 60.0, 60.0]) if cap is None else np.asarray(cap, float)
    line_from = np.array([0])
    line_to = np.array([1])
    flim = np.array([40.0]) if flim is None else np.asarray(flim, float)
    # (B=2, T=3). Demands chosen so no generator/line sits exactly at a
    # bound kink: t0 interior, t1 line-saturated with gen2 strictly on,
    # t2 interior. Central FD is only valid away from kinks.
    demand = (np.array([[20.0, 25.0, 30.0],
                        [50.0, 70.0, 45.0]])
              if demand is None else np.asarray(demand, float))
    return MultiBusDispatchProblem(
        gen_bus=gen_bus, marginal_cost=mc, capacity=cap,
        line_from=line_from, line_to=line_to, line_limit=flim,
        demand=demand, n_buses=2, ridge=ridge,
    )


def test_multibus_forward_feasible():
    prob = _problem()
    sol = solve_multibus_dispatch_with_sensitivities(prob)
    G, T = sol.dispatch.shape
    assert (G, T) == (3, 3)
    # Per-bus balance: gen + incoming flow = demand.
    A = np.array([[-1.0], [1.0]])  # line 0: from bus0 (-1), to bus1 (+1)
    GB = np.array([[1.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    for t in range(T):
        bal = GB @ sol.dispatch[:, t] + A @ sol.flows[:, t]
        assert np.allclose(bal, prob.demand[:, t], atol=1e-6)
    # Bounds respected.
    assert np.all(sol.dispatch >= -1e-6)
    assert np.all(sol.dispatch <= prob.capacity[:, None] + 1e-6)
    assert np.all(np.abs(sol.flows) <= prob.line_limit[:, None] + 1e-6)


def test_multibus_gradients_match_finite_diff():
    prob = _problem()
    sol = solve_multibus_dispatch_with_sensitivities(prob)
    G, T = sol.dispatch.shape

    def disp_flat(p):
        return p.dispatch.reshape(-1)

    # d/d mc
    def f_mc(x):
        return disp_flat(solve_multibus_dispatch_with_sensitivities(
            _problem(mc=x)))
    num_mc = numerical_jacobian(f_mc, prob.marginal_cost, eps=1e-5)
    err_mc = np.max(np.abs(sol.d_dispatch_d_mc - num_mc))

    # d/d capacity
    def f_cap(x):
        return disp_flat(solve_multibus_dispatch_with_sensitivities(
            _problem(cap=x)))
    num_cap = numerical_jacobian(f_cap, prob.capacity, eps=1e-5)
    err_cap = np.max(np.abs(sol.d_dispatch_d_capacity - num_cap))

    # d/d demand (flatten (b, t))
    def f_dem(x):
        d = x.reshape(2, 3)
        return disp_flat(solve_multibus_dispatch_with_sensitivities(
            _problem(demand=d)))
    num_dem = numerical_jacobian(f_dem, prob.demand.reshape(-1), eps=1e-5)
    err_dem = np.max(np.abs(sol.d_dispatch_d_demand - num_dem))

    # d/d line_limit
    def f_flim(x):
        return disp_flat(solve_multibus_dispatch_with_sensitivities(
            _problem(flim=x)))
    num_flim = numerical_jacobian(f_flim, prob.line_limit, eps=1e-5)
    err_flim = np.max(np.abs(sol.d_dispatch_d_linelimit - num_flim))

    print(f"\n[multibus] grad-vs-FD max abs err: mc={err_mc:.2e} "
          f"cap={err_cap:.2e} demand={err_dem:.2e} flim={err_flim:.2e}")
    assert err_mc < 1e-3
    assert err_cap < 1e-3
    assert err_dem < 1e-3
    assert err_flim < 1e-3


def test_multibus_layer_backward_matches_fd():
    layer = MultiBusDispatchLayer(ridge=1.0)
    prob = _problem(ridge=1.0)
    p = layer.forward(prob)
    rng = np.random.default_rng(1)
    grad_out = rng.normal(size=p.shape)
    grads = layer.backward(grad_out)

    def loss_mc(x):
        pp = solve_multibus_dispatch_with_sensitivities(_problem(mc=x, ridge=1.0))
        return np.array([float((grad_out * pp.dispatch).sum())])
    num_mc = numerical_jacobian(loss_mc, prob.marginal_cost, eps=1e-5).ravel()
    err = np.max(np.abs(grads["marginal_cost"] - num_mc))
    print(f"\n[multibus-backward] dL/dmc max abs err = {err:.2e}")
    assert err < 1e-3


# ---------------------------------------------------------------------------
# 12.3 — demand-elasticity recovery
# ---------------------------------------------------------------------------

def test_recover_demand_elasticity():
    mc = np.array([10.0, 30.0, 80.0])
    cap = np.array([100.0, 100.0, 100.0])
    base_demand = 150.0
    ref_price = 30.0
    true_elast = 1.5
    ridge = 1.0

    rng = np.random.default_rng(0)
    prices = np.linspace(15.0, 70.0, 8)
    observed = []
    for pr in prices:
        d = base_demand - true_elast * (pr - ref_price)
        p, _ = solve_dispatch_with_sensitivities(mc, cap, max(d, 0.0), ridge=ridge)
        observed.append(p)
    observed = np.array(observed)

    res = fit_demand_elasticity(
        prices, observed, mc, cap, base_demand, ref_price,
        ridge=ridge, lr=2e-4, n_iter=2000, elasticity_init=0.0,
    )
    print(f"\n[elasticity] true={true_elast} recovered={res.elasticity:.4f} "
          f"final_loss={res.history[-1]:.3e} iters={res.n_iter}")
    assert abs(res.elasticity - true_elast) < 0.05
    # Loss decreased monotonically-ish to near zero.
    assert res.history[-1] < 1e-2
