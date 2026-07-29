"""N_En_Phase 19.x.2 — storage in the MULTIBUS diff layer.

Stacked multibus+storage QP with IFT sensitivities, windowed
(representative-day) scope. FD-exactness on a 2-bus, 3-gen, 1-line,
1-storage, 6-period instance; consistency with the single-bus storage
solver when the network is trivial.
"""

from __future__ import annotations

import numpy as np
import pytest

from nexus_energy.diff import (
    MultiBusStorageProblem,
    StorageDispatchProblem,
    numerical_jacobian,
    solve_multibus_storage_dispatch_with_sensitivities,
    solve_storage_dispatch_with_sensitivities,
)

T = 6
RIDGE = 1.0


def _problem(mc=None, eta_c=0.95, eta_d=0.90, soc_init=0.0):
    # Bus 0: cheap baseload; bus 1: peaker + most load + the BATTERY
    # (behind the congested line, so charging off-peak imports cheap power
    # and discharging at the peak displaces the 90-€ peaker).
    mc = np.array([10.0, 35.0, 90.0]) if mc is None else np.asarray(mc, float)
    return MultiBusStorageProblem(
        gen_bus=np.array([0, 0, 1]),
        marginal_cost=mc,
        capacity=np.array([60.0, 40.0, 70.0]),
        line_from=np.array([0]), line_to=np.array([1]),
        line_limit=np.array([35.0]),
        demand=np.array([[20.0, 24.0, 28.0, 22.0, 26.0, 21.0],
                         [40.0, 47.0, 88.0, 84.0, 49.0, 43.0]]),
        n_buses=2,
        sto_bus=np.array([1]),
        charge_eff=np.array([eta_c]),
        discharge_eff=np.array([eta_d]),
        power_limit=np.array([18.0]),
        soc_max=np.array([60.0]),
        soc_init=np.array([soc_init]),
        ridge=RIDGE,
    )


def test_forward_feasible_and_storage_active():
    p = _problem()
    sol = solve_multibus_storage_dispatch_with_sensitivities(p)
    # Balance per bus per period (incl. storage and line terms).
    for t in range(T):
        b0 = (sol.dispatch[0, t] + sol.dispatch[1, t]
              - sol.flows[0, t])
        b1 = (sol.dispatch[2, t] + sol.flows[0, t]
              + sol.discharge[0, t] - sol.charge[0, t])
        assert b0 == pytest.approx(p.demand[0, t], abs=1e-5)
        assert b1 == pytest.approx(p.demand[1, t], abs=1e-5)
    # SOC continuity.
    prev = 0.0
    for t in range(T):
        expected = prev + 0.95 * sol.charge[0, t] - sol.discharge[0, t] / 0.90
        assert sol.soc[0, t] == pytest.approx(expected, abs=1e-5)
        prev = sol.soc[0, t]
    # The peak periods make storage worthwhile (charges early via cheap
    # bus-0 capacity, discharges into the line-limited peak).
    assert sol.charge.max() > 0.5
    assert sol.discharge.max() > 0.5


def test_gradients_fd_exact():
    p = _problem()
    sol = solve_multibus_storage_dispatch_with_sensitivities(p)
    eps = 1e-6

    def disp(prob):
        return solve_multibus_storage_dispatch_with_sensitivities(
            prob, jacobians=()).dispatch.reshape(-1)

    # d/d mc_1 (mid-cost gen)
    def f_mc(x):
        return disp(_problem(mc=np.array([10.0, float(x[0]), 90.0])))
    num = numerical_jacobian(f_mc, np.array([35.0]), eps=1e-5).ravel()
    err_mc = np.max(np.abs(sol.d_dispatch_d_mc[:, 1] - num))

    # d/d eta_c
    def f_ec(x):
        return disp(_problem(eta_c=float(x[0])))
    num = numerical_jacobian(f_ec, np.array([0.95]), eps=eps).ravel()
    err_ec = np.max(np.abs(sol.d_dispatch_d_charge_eff[:, 0] - num))

    # d/d eta_d
    def f_ed(x):
        return disp(_problem(eta_d=float(x[0])))
    num = numerical_jacobian(f_ed, np.array([0.90]), eps=eps).ravel()
    err_ed = np.max(np.abs(sol.d_dispatch_d_discharge_eff[:, 0] - num))

    # d/d soc_init
    def f_s0(x):
        return disp(_problem(soc_init=float(x[0])))
    num = numerical_jacobian(f_s0, np.array([0.0 + 5.0]), eps=1e-5).ravel()
    sol5 = solve_multibus_storage_dispatch_with_sensitivities(
        _problem(soc_init=5.0))
    err_s0 = np.max(np.abs(sol5.d_dispatch_d_soc_init[:, 0] - num))

    print(f"\n[mb-storage] FD err: mc={err_mc:.2e} eta_c={err_ec:.2e} "
          f"eta_d={err_ed:.2e} soc0={err_s0:.2e}")
    assert err_mc < 1e-3
    assert err_ec < 1e-3
    assert err_ed < 1e-3
    assert err_s0 < 1e-3
    # η actually matters here.
    assert np.abs(sol.d_dispatch_d_discharge_eff).max() > 1e-3


def test_soc_trace_blocks_fd_exact():
    p = _problem()
    sol = solve_multibus_storage_dispatch_with_sensitivities(p)
    eps = 1e-6

    def soc_flat(prob):
        return solve_multibus_storage_dispatch_with_sensitivities(
            prob, jacobians=()).soc.reshape(-1)

    def f_ec(x):
        return soc_flat(_problem(eta_c=float(x[0])))
    num = numerical_jacobian(f_ec, np.array([0.95]), eps=eps).ravel()
    err = np.max(np.abs(sol.d_soc_d_charge_eff[:, 0] - num))
    print(f"\n[mb-storage-soc] FD err eta_c={err:.2e}")
    assert err < 1e-3


def test_matches_single_bus_solver_on_trivial_network():
    # B=1, L=0 → must agree with the dedicated single-bus storage solver.
    mb = MultiBusStorageProblem(
        gen_bus=np.array([0, 0]),
        marginal_cost=np.array([10.0, 60.0]),
        capacity=np.array([80.0, 80.0]),
        line_from=np.array([], dtype=int), line_to=np.array([], dtype=int),
        line_limit=np.array([]),
        demand=np.array([[60.0, 75.0, 95.0, 70.0]]),
        n_buses=1,
        sto_bus=np.array([0]),
        charge_eff=np.array([0.95]), discharge_eff=np.array([0.95]),
        power_limit=np.array([40.0]), soc_max=np.array([200.0]),
        soc_init=np.array([60.0]), ridge=RIDGE)
    sol_mb = solve_multibus_storage_dispatch_with_sensitivities(
        mb, jacobians=())
    sb = StorageDispatchProblem(
        marginal_cost=np.array([10.0, 60.0]), capacity=np.array([80.0, 80.0]),
        demand=np.array([60.0, 75.0, 95.0, 70.0]),
        charge_eff=0.95, discharge_eff=0.95,
        power_limit=40.0, soc_max=200.0, soc_init=60.0, ridge=RIDGE)
    sol_sb = solve_storage_dispatch_with_sensitivities(sb)
    assert np.max(np.abs(sol_mb.dispatch - sol_sb.dispatch)) < 1e-5
    assert np.max(np.abs(sol_mb.soc[0] - sol_sb.soc)) < 1e-5
