"""
Phase 12.2 — differentiable multi-period storage dispatch + smoothed
commitment surrogate smoke tests.

Coverage:
  (a) storage forward solve respects per-period balance, SOC continuity,
      and all box bounds on a tiny 2-gen, 4-period instance;
  (b) analytic d dispatch / d {mc, capacity, demand, soc_init} match
      central finite differences within 1e-3 — INCLUDING the off-diagonal
      d dispatch_t / d demand_{t'} that exists ONLY because the SOC
      couples periods (verified non-zero), and d/d soc_init;
  (c) StorageDispatchLayer.backward composes the Jacobian with a random
      grad-output and matches the finite-difference scalar-loss gradient;
  (d) smooth_commitment / SmoothCommitmentLayer gradients are finite and
      match finite differences; fit_commitment_threshold decreases its
      fitting loss and recovers a planted threshold.

TINY instances only.
"""

from __future__ import annotations

import numpy as np
import pytest

from nexus_energy.diff import (
    StorageDispatchProblem,
    StorageDispatchLayer,
    solve_storage_dispatch_with_sensitivities,
    SmoothCommitmentLayer,
    smooth_commitment,
    fit_commitment_threshold,
    numerical_jacobian,
)


# ---------------------------------------------------------------------------
# Tiny 2-gen, 4-period storage instance
# ---------------------------------------------------------------------------

def _problem(mc=None, cap=None, demand=None, soc_init=None, ridge=1.0,
             eta_c=0.95, eta_d=0.95, plim=40.0):
    # Cheap baseload + expensive peaker; a peaky demand so the storage
    # discharges steadily over the horizon — keeping charge/discharge
    # strictly off their bounds (so central FD is valid, no kinks). A
    # generous soc_init + soc_max keeps SOC interior too.
    mc = np.array([10.0, 60.0]) if mc is None else np.asarray(mc, float)
    cap = np.array([80.0, 80.0]) if cap is None else np.asarray(cap, float)
    demand = (np.array([60.0, 75.0, 95.0, 70.0])
              if demand is None else np.asarray(demand, float))
    soc_init = 60.0 if soc_init is None else float(soc_init)
    return StorageDispatchProblem(
        marginal_cost=mc, capacity=cap, demand=demand,
        charge_eff=eta_c, discharge_eff=eta_d,
        power_limit=plim, soc_max=200.0, soc_init=soc_init,
        ridge=ridge,
    )


def test_storage_forward_feasible():
    prob = _problem()
    sol = solve_storage_dispatch_with_sensitivities(prob)
    G, T = sol.dispatch.shape
    assert (G, T) == (2, 4)
    # Per-period balance: gen + discharge − charge = demand.
    for t in range(T):
        bal = sol.dispatch[:, t].sum() + sol.discharge[t] - sol.charge[t]
        assert bal == pytest.approx(prob.demand[t], abs=1e-5)
    # SOC continuity.
    prev = prob.soc_init
    for t in range(T):
        expected = prev + prob.charge_eff * sol.charge[t] \
            - sol.discharge[t] / prob.discharge_eff
        assert sol.soc[t] == pytest.approx(expected, abs=1e-5)
        prev = sol.soc[t]
    # Bounds.
    assert np.all(sol.dispatch >= -1e-6)
    assert np.all(sol.dispatch <= prob.capacity[:, None] + 1e-6)
    assert np.all(sol.charge >= -1e-6) and np.all(sol.charge <= prob.power_limit + 1e-6)
    assert np.all(sol.discharge >= -1e-6) and np.all(sol.discharge <= prob.power_limit + 1e-6)
    assert np.all(sol.soc >= -1e-6) and np.all(sol.soc <= prob.soc_max + 1e-6)


def test_storage_gradients_match_finite_diff():
    prob = _problem()
    sol = solve_storage_dispatch_with_sensitivities(prob)

    def disp_flat(p):
        return p.dispatch.reshape(-1)

    def f_mc(x):
        return disp_flat(solve_storage_dispatch_with_sensitivities(_problem(mc=x)))
    num_mc = numerical_jacobian(f_mc, prob.marginal_cost, eps=1e-5)
    err_mc = np.max(np.abs(sol.d_dispatch_d_mc - num_mc))

    def f_cap(x):
        return disp_flat(solve_storage_dispatch_with_sensitivities(_problem(cap=x)))
    num_cap = numerical_jacobian(f_cap, prob.capacity, eps=1e-5)
    err_cap = np.max(np.abs(sol.d_dispatch_d_capacity - num_cap))

    def f_dem(x):
        return disp_flat(solve_storage_dispatch_with_sensitivities(_problem(demand=x)))
    num_dem = numerical_jacobian(f_dem, prob.demand, eps=1e-5)
    err_dem = np.max(np.abs(sol.d_dispatch_d_demand - num_dem))

    def f_soc(x):
        return disp_flat(solve_storage_dispatch_with_sensitivities(
            _problem(soc_init=float(x[0]))))
    num_soc = numerical_jacobian(f_soc, np.array([prob.soc_init]), eps=1e-5).ravel()
    err_soc = np.max(np.abs(sol.d_dispatch_d_soc_init - num_soc))

    # The defining inter-temporal property: dispatch in period t responds
    # to demand in OTHER periods t' (off-diagonal of d/d demand) BECAUSE
    # the SOC couples them. Confirm at least one off-diagonal block is
    # materially non-zero.
    G, T = sol.dispatch.shape
    Jd = sol.d_dispatch_d_demand  # (G*T, T)
    offdiag_max = 0.0
    for g in range(G):
        for t in range(T):
            for tp in range(T):
                if tp != t:
                    offdiag_max = max(offdiag_max, abs(Jd[g * T + t, tp]))

    print(f"\n[storage] grad-vs-FD max abs err: mc={err_mc:.2e} cap={err_cap:.2e} "
          f"demand={err_dem:.2e} soc_init={err_soc:.2e} | "
          f"intertemporal off-diag max |d disp/d dem_t'|={offdiag_max:.3f}")
    assert err_mc < 1e-3
    assert err_cap < 1e-3
    assert err_dem < 1e-3
    assert err_soc < 1e-3
    assert offdiag_max > 1e-3  # SOC coupling is real, not zero


def test_storage_layer_backward_matches_fd():
    layer = StorageDispatchLayer(ridge=1.0)
    prob = _problem(ridge=1.0)
    p = layer.forward(prob)
    rng = np.random.default_rng(2)
    grad_out = rng.normal(size=p.shape)
    grads = layer.backward(grad_out)

    def loss_dem(x):
        pp = solve_storage_dispatch_with_sensitivities(_problem(demand=x, ridge=1.0))
        return np.array([float((grad_out * pp.dispatch).sum())])
    num_dem = numerical_jacobian(loss_dem, prob.demand, eps=1e-5).ravel()
    err = np.max(np.abs(grads["demand"] - num_dem))

    def loss_soc(x):
        pp = solve_storage_dispatch_with_sensitivities(
            _problem(soc_init=float(x[0]), ridge=1.0))
        return np.array([float((grad_out * pp.dispatch).sum())])
    num_soc = numerical_jacobian(loss_soc, np.array([prob.soc_init]), eps=1e-5).ravel()
    err_soc = abs(grads["soc_init"] - float(num_soc[0]))

    print(f"\n[storage-backward] dL/ddemand max abs err = {err:.2e} "
          f"dL/dsoc_init err = {err_soc:.2e}")
    assert err < 1e-3
    assert err_soc < 1e-3


# ---------------------------------------------------------------------------
# N_En_Phase 20 — efficiency (constraint-matrix) Jacobians
# ---------------------------------------------------------------------------

def test_storage_eta_gradients_match_finite_diff():
    prob = _problem()
    sol = solve_storage_dispatch_with_sensitivities(prob)
    # Storage must actually be active for η to matter in this fixture.
    assert sol.discharge.max() > 1.0

    def f_eta_c(x):
        return solve_storage_dispatch_with_sensitivities(
            _problem(eta_c=float(x[0]))).dispatch.reshape(-1)

    def f_eta_d(x):
        return solve_storage_dispatch_with_sensitivities(
            _problem(eta_d=float(x[0]))).dispatch.reshape(-1)

    num_c = numerical_jacobian(f_eta_c, np.array([0.95]), eps=1e-6).ravel()
    num_d = numerical_jacobian(f_eta_d, np.array([0.95]), eps=1e-6).ravel()
    err_c = np.max(np.abs(sol.d_dispatch_d_charge_eff - num_c))
    err_d = np.max(np.abs(sol.d_dispatch_d_discharge_eff - num_d))
    print(f"\n[storage-eta] grad-vs-FD max abs err: eta_c={err_c:.2e} "
          f"eta_d={err_d:.2e} |J_eta_d|max={np.abs(num_d).max():.3f}")
    assert err_c < 1e-3
    assert err_d < 1e-3
    # η_d must matter (storage discharges in this fixture).
    assert np.abs(sol.d_dispatch_d_discharge_eff).max() > 1e-3


def test_storage_eta_gradient_pinned_charge():
    # Force charging pinned at the power limit while DISCHARGE stays
    # interior (battery drained over two expensive periods) — exercises
    # the rhs path where pinned charge contributes to the equality term
    # but not the reduced system, with a non-zero gradient (more η_c →
    # more stored energy → less peaker).
    kw = dict(mc=np.array([10.0, 200.0]), cap=np.array([60.0, 100.0]),
              demand=np.array([30.0, 30.0, 80.0, 80.0]),
              soc_init=0.0, plim=5.0)
    prob = _problem(**kw)
    sol = solve_storage_dispatch_with_sensitivities(prob)
    assert sol.charge.max() > 5.0 - 1e-6  # pinned at plim somewhere
    # At least one discharge period strictly interior (drained battery).
    assert np.any((sol.discharge > 1e-3) & (sol.discharge < 5.0 - 1e-3))

    def f_eta_c(x):
        return solve_storage_dispatch_with_sensitivities(
            _problem(eta_c=float(x[0]), **kw)).dispatch.reshape(-1)

    num_c = numerical_jacobian(f_eta_c, np.array([0.95]), eps=1e-6).ravel()
    err_c = np.max(np.abs(sol.d_dispatch_d_charge_eff - num_c))
    print(f"\n[storage-eta-pinned] err_eta_c={err_c:.2e} "
          f"|J|max={np.abs(num_c).max():.3f}")
    assert err_c < 1e-3
    assert np.abs(sol.d_dispatch_d_charge_eff).max() > 1e-4


def test_storage_eta_gradient_idle_storage():
    # No arbitrage value (flat demand, constant costs) → battery idle →
    # efficiencies have exactly zero influence. The honest-zero case the
    # auto-calibration gate relies on ("data silent ⇒ don't update").
    prob = _problem(demand=np.array([50.0, 50.0, 50.0, 50.0]), soc_init=0.0)
    sol = solve_storage_dispatch_with_sensitivities(prob)
    assert np.abs(sol.charge).max() < 1e-6
    assert np.abs(sol.discharge).max() < 1e-6
    assert np.abs(sol.d_dispatch_d_charge_eff).max() < 1e-8
    assert np.abs(sol.d_dispatch_d_discharge_eff).max() < 1e-8


def test_storage_layer_backward_eta():
    layer = StorageDispatchLayer(ridge=1.0)
    prob = _problem(ridge=1.0)
    p = layer.forward(prob)
    rng = np.random.default_rng(7)
    grad_out = rng.normal(size=p.shape)
    grads = layer.backward(grad_out)

    def loss_eta(x):
        pp = solve_storage_dispatch_with_sensitivities(
            _problem(eta_c=float(x[0]), ridge=1.0))
        return np.array([float((grad_out * pp.dispatch).sum())])
    num = numerical_jacobian(loss_eta, np.array([0.95]), eps=1e-6).ravel()[0]
    err = abs(grads["charge_eff"] - num)
    print(f"\n[storage-backward-eta] dL/deta_c analytic="
          f"{grads['charge_eff']:.5f} fd={num:.5f} err={err:.2e}")
    assert err < 1e-3


# ---------------------------------------------------------------------------
# (d) smoothed commitment surrogate
# ---------------------------------------------------------------------------

def test_smooth_commitment_gradients_finite_and_match_fd():
    rng = np.random.default_rng(3)
    signal = rng.uniform(0.0, 1.0, size=6)
    cap = np.full(6, 50.0)
    tau = 0.5
    layer = SmoothCommitmentLayer(sharpness=8.0)
    p = layer.forward(signal, tau, cap)
    assert np.all(np.isfinite(p))
    grad_out = rng.normal(size=p.shape)
    grads = layer.backward(grad_out)
    # All gradients finite.
    assert np.isfinite(grads["threshold"])
    assert np.isfinite(grads["sharpness"])
    assert np.all(np.isfinite(grads["signal"]))

    # FD check on dL/dtau.
    def loss_tau(x):
        u = smooth_commitment(signal, float(x[0]), 8.0)
        pp = u * cap
        return np.array([float((grad_out * pp).sum())])
    num = numerical_jacobian(loss_tau, np.array([tau]), eps=1e-6).ravel()[0]
    err = abs(grads["threshold"] - num)
    print(f"\n[commit] dL/dtau analytic={grads['threshold']:.4f} "
          f"fd={num:.4f} err={err:.2e}")
    assert err < 1e-3


def test_fit_commitment_threshold_decreases_loss():
    # Plant a true threshold; generate committed power under a sharp
    # surrogate; fit it back from a wrong init.
    rng = np.random.default_rng(4)
    signal = np.linspace(0.0, 1.0, 30)
    cap = 1.0  # commitment fraction scale
    true_tau = 0.6
    observed = cap * smooth_commitment(signal, true_tau, sharpness=8.0)

    res = fit_commitment_threshold(
        signal, observed, cap, sharpness=8.0,
        lr=2e-3, n_iter=4000, threshold_init=0.2)
    print(f"\n[commit-fit] true_tau={true_tau} recovered={res.threshold:.4f} "
          f"loss0={res.history[0]:.3e} lossN={res.history[-1]:.3e} "
          f"iters={res.n_iter}")
    # Loss decreased and threshold recovered.
    assert res.history[-1] < res.history[0]
    assert res.history[-1] < 1e-1 * res.history[0]
    assert abs(res.threshold - true_tau) < 0.05
