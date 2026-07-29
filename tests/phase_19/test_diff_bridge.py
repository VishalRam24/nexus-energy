"""N_En_Phase 19.A — from_pypsa → multibus diff layer composition.

Acceptance (INVERSE_CALIBRATION_FLAGSHIP.md, Build plan Phase A):
wire ``from_pypsa`` output into ``solve_multibus_dispatch_with_sensitivities``
on a 3-bus slice, transport mode, and confirm the gradients are FD-exact
END-TO-END THROUGH THE BRIDGE (perturb the PyPSA network / the CO₂ price,
re-import, re-solve → central differences match the analytic Jacobians).

Also covers the two diff-layer extensions Phase A required:
  * time-varying availability (VRE p_max_pu, incl. a zero-availability hour);
  * asymmetric line bounds (unidirectional DC link → line_min = 0).
Plus the 3-bus plant-and-recover: a hidden CO₂ price is recovered by
Gauss-Newton through the bridge.
"""

from __future__ import annotations

import numpy as np
import pytest

pypsa = pytest.importorskip("pypsa")
import pandas as pd  # noqa: E402

from nexus_energy.diff import solve_multibus_dispatch_with_sensitivities  # noqa: E402
from nexus_energy.diff_bridge import (  # noqa: E402
    d_dispatch_d_co2_price,
    fit_co2_price,
    multibus_problem_from_system,
)
from nexus_energy.pypsa_compat import from_pypsa  # noqa: E402

T = 6
RIDGE = 1.0


def make_network(mc_coal=18.0, d0_scale=1.0):
    n = pypsa.Network()
    n.set_snapshots(pd.date_range("2025-01-01", periods=T, freq="h"))
    n.add("Carrier", "coal", co2_emissions=0.34)
    n.add("Carrier", "gas", co2_emissions=0.20)
    n.add("Carrier", "wind", co2_emissions=0.0)
    for b in ("b0", "b1", "b2"):
        n.add("Bus", b, carrier="AC")
    n.add("Generator", "coal0", bus="b0", p_nom=80.0, carrier="coal",
          marginal_cost=mc_coal, efficiency=0.38)
    n.add("Generator", "gas1", bus="b1", p_nom=60.0, carrier="gas",
          marginal_cost=45.0, efficiency=0.50)
    # Wind has one zero-availability hour → exercises the frozen lb==ub path.
    wind_cf = np.array([0.62, 0.31, 0.0, 0.47, 0.83, 0.55])
    n.add("Generator", "wind2", bus="b2", p_nom=70.0, carrier="wind",
          marginal_cost=1.3, p_max_pu=wind_cf)
    n.add("Load", "d0", bus="b0",
          p_set=d0_scale * np.array([33.0, 41.0, 37.0, 29.0, 44.0, 38.0]))
    n.add("Load", "d1", bus="b1",
          p_set=np.array([27.0, 22.0, 31.0, 35.0, 24.0, 28.0]))
    n.add("Load", "d2", bus="b2",
          p_set=np.array([18.0, 23.0, 26.0, 17.0, 21.0, 19.0]))
    # AC lines (x > 0 on purpose: transport mode must override DC-OPF routing)
    n.add("Line", "L01", bus0="b0", bus1="b1", x=0.1, r=0.01, s_nom=35.0)
    n.add("Line", "L12", bus0="b1", bus1="b2", x=0.1, r=0.01, s_nom=25.0)
    # Unidirectional DC link b0 → b2
    n.add("Link", "dc02", bus0="b0", bus1="b2", p_nom=15.0)
    return n


def bridge_for(co2_price=0.0, **net_kw):
    sys = from_pypsa(make_network(**net_kw), line_model="transport")
    return multibus_problem_from_system(sys, co2_price=co2_price, ridge=RIDGE)


def test_bridge_structure():
    br = bridge_for()
    p = br.problem
    assert p.n_buses == 3
    assert p.demand.shape == (3, T)
    assert br.gen_names == ["coal0", "gas1", "wind2"]
    # Emission factors are tCO₂/MWh_el = carrier co2 / efficiency.
    np.testing.assert_allclose(br.emission, [0.34 / 0.38, 0.20 / 0.50, 0.0])
    # AC lines symmetric, DC link unidirectional.
    lm = dict(zip(br.line_names, p.line_min))
    ll = dict(zip(br.line_names, p.line_limit))
    assert lm["line_L01"] == -35.0 and ll["line_L01"] == 35.0
    assert lm["line_L12"] == -25.0 and ll["line_L12"] == 25.0
    assert lm["dc02"] == 0.0 and ll["dc02"] == 15.0
    # Wind availability row carried through (incl. the zero hour).
    np.testing.assert_allclose(p.availability[2],
                               [0.62, 0.31, 0.0, 0.47, 0.83, 0.55])


def test_forward_feasible_and_wind_zero_hour():
    br = bridge_for()
    sol = solve_multibus_dispatch_with_sensitivities(br.problem)
    p = br.problem
    # Balance per bus per period.
    B, L, G = 3, 3, 3
    A = np.zeros((B, L))
    for l in range(L):
        A[p.line_to[l], l] += 1.0
        A[p.line_from[l], l] -= 1.0
    GB = np.zeros((B, G))
    for g in range(G):
        GB[p.gen_bus[g], g] = 1.0
    for t in range(T):
        np.testing.assert_allclose(
            GB @ sol.dispatch[:, t] + A @ sol.flows[:, t],
            p.demand[:, t], atol=1e-6)
    # Availability respected; zero-cf hour → exactly zero wind.
    assert np.all(sol.dispatch[2] <= 70.0 * p.availability[2] + 1e-6)
    assert abs(sol.dispatch[2, 2]) < 1e-9
    # Unidirectional link never flows backwards.
    dc = br.line_names.index("dc02")
    assert np.all(sol.flows[dc] >= -1e-9)


def test_gradients_fd_exact_end_to_end():
    """Central FD through (rebuild pypsa net → from_pypsa → bridge → solve)."""
    br = bridge_for()
    sol = solve_multibus_dispatch_with_sensitivities(br.problem)
    eps = 1e-5

    def dispatch_for(**kw):
        b = bridge_for(**kw)
        return solve_multibus_dispatch_with_sensitivities(b.problem).dispatch

    # d/d marginal_cost of coal0 — perturb the PYPSA attribute itself.
    fd = (dispatch_for(mc_coal=18.0 + eps) -
          dispatch_for(mc_coal=18.0 - eps)).reshape(-1) / (2 * eps)
    err_mc = np.max(np.abs(sol.d_dispatch_d_mc[:, 0] - fd))

    # d/d CO₂ price at price=0 — the calibration parameter.
    fd_p = (dispatch_for(co2_price=eps) -
            dispatch_for(co2_price=-eps)).reshape(-1) / (2 * eps)
    J_price = d_dispatch_d_co2_price(sol, br.emission)
    err_price = np.max(np.abs(J_price - fd_p))

    # Same check at a strictly positive price (different active sets).
    br25 = bridge_for(co2_price=25.0)
    sol25 = solve_multibus_dispatch_with_sensitivities(br25.problem)
    fd_p25 = (dispatch_for(co2_price=25.0 + eps) -
              dispatch_for(co2_price=25.0 - eps)).reshape(-1) / (2 * eps)
    err_price25 = np.max(np.abs(
        d_dispatch_d_co2_price(sol25, br25.emission) - fd_p25))

    print(f"\n[bridge-FD] mc={err_mc:.2e} price@0={err_price:.2e} "
          f"price@25={err_price25:.2e}")
    assert err_mc < 1e-4
    assert err_price < 1e-4
    assert err_price25 < 1e-4


def test_gradients_fd_exact_problem_level():
    """FD on the bridged problem's own fields (capacity, demand, line limit)."""
    br = bridge_for()
    p = br.problem
    sol = solve_multibus_dispatch_with_sensitivities(p)
    eps = 1e-5
    from dataclasses import replace

    def disp(prob):
        return solve_multibus_dispatch_with_sensitivities(prob).dispatch

    # capacity of gas1 (index 1)
    cap_hi = p.capacity.copy(); cap_hi[1] += eps
    cap_lo = p.capacity.copy(); cap_lo[1] -= eps
    fd = (disp(replace(p, capacity=cap_hi)) -
          disp(replace(p, capacity=cap_lo))).reshape(-1) / (2 * eps)
    assert np.max(np.abs(sol.d_dispatch_d_capacity[:, 1] - fd)) < 1e-4

    # nameplate capacity of wind2 (index 2) — chains through availability
    cap_hi = p.capacity.copy(); cap_hi[2] += eps
    cap_lo = p.capacity.copy(); cap_lo[2] -= eps
    fd = (disp(replace(p, capacity=cap_hi)) -
          disp(replace(p, capacity=cap_lo))).reshape(-1) / (2 * eps)
    assert np.max(np.abs(sol.d_dispatch_d_capacity[:, 2] - fd)) < 1e-4

    # demand at bus 1, period 3
    d_hi = p.demand.copy(); d_hi[1, 3] += eps
    d_lo = p.demand.copy(); d_lo[1, 3] -= eps
    fd = (disp(replace(p, demand=d_hi)) -
          disp(replace(p, demand=d_lo))).reshape(-1) / (2 * eps)
    assert np.max(np.abs(sol.d_dispatch_d_demand[:, 1 * T + 3] - fd)) < 1e-4

    # line limit of the symmetric AC line L12
    j = br.line_names.index("line_L12")
    f_hi = p.line_limit.copy(); f_hi[j] += eps
    f_lo = p.line_limit.copy(); f_lo[j] -= eps
    fd = (disp(replace(p, line_limit=f_hi)) -
          disp(replace(p, line_limit=f_lo))).reshape(-1) / (2 * eps)
    assert np.max(np.abs(sol.d_dispatch_d_linelimit[:, j] - fd)) < 1e-4

    # limit of the unidirectional dc02 (line_min stays 0)
    j = br.line_names.index("dc02")
    f_hi = p.line_limit.copy(); f_hi[j] += eps
    f_lo = p.line_limit.copy(); f_lo[j] -= eps
    fd = (disp(replace(p, line_limit=f_hi)) -
          disp(replace(p, line_limit=f_lo))).reshape(-1) / (2 * eps)
    assert np.max(np.abs(sol.d_dispatch_d_linelimit[:, j] - fd)) < 1e-4


def test_plant_and_recover_co2_price_3bus():
    """Hide a CO₂ price, observe dispatch, recover it through the bridge."""
    true_price = 25.0
    sys = from_pypsa(make_network(), line_model="transport")
    br = multibus_problem_from_system(sys, co2_price=true_price, ridge=RIDGE)
    observed = solve_multibus_dispatch_with_sensitivities(br.problem).dispatch

    res = fit_co2_price(sys, observed, ridge=RIDGE, price_bounds=(0.0, 200.0))
    print(f"\n[recover] true={true_price} recovered={res.price:.6f} "
          f"solves={res.n_solves} loss={res.history[-1][1]:.3e}")
    assert res.converged
    assert abs(res.price - true_price) < 1e-3
    assert res.n_solves <= 25


def test_out_of_scope_rejections():
    n = make_network()
    sys_dc = from_pypsa(n, line_model="auto")  # x>0 → dc_opf links
    with pytest.raises(ValueError, match="transport"):
        multibus_problem_from_system(sys_dc)

    n2 = make_network()
    n2.generators.loc["coal0", "p_nom_extendable"] = True
    sys_ext = from_pypsa(n2, line_model="transport")
    with pytest.raises(ValueError, match="extendable"):
        multibus_problem_from_system(sys_ext)

    n3 = make_network()
    n3.add("StorageUnit", "batt", bus="b1", p_nom=10.0, max_hours=4.0)
    sys_sto = from_pypsa(n3, line_model="transport")
    with pytest.raises(ValueError, match="storage"):
        multibus_problem_from_system(sys_sto)
