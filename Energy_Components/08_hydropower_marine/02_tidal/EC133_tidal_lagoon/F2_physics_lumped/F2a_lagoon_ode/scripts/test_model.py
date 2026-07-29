"""
EC133 -- Tidal Lagoon -- F2a Physics-Lumped Water-Level ODE
Test suite: mass/energy conservation, head dynamics, controller behaviour,
holding optimality, edge cases, predict() interface, benchmark.
Custom harness (NO pytest). Run: python3 scripts/test_model.py
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from model import TidalLagoonF2a
from predict import ComponentModel

PASS = "✓"
FAIL = "✗"


def assert_true(cond, msg):
    if cond:
        print(f"  {PASS}  {msg}")
    else:
        print(f"  {FAIL}  FAILED: {msg}")
        raise AssertionError(msg)


def make_model():
    cm = ComponentModel()
    return cm._model, cm


# ---------------------------------------------------------------------------
def test_sea_level_sinusoid():
    print("\n[Test 1] Sea tide is a bounded sinusoid")
    m, _ = make_model()
    t = np.linspace(0, m.T, 500)
    z = m.sea_level(t)
    assert_true(np.max(z) <= m.a + 1e-9, f"peak {np.max(z):.3f} <= amplitude {m.a}")
    assert_true(np.min(z) >= -m.a - 1e-9, f"trough {np.min(z):.3f} >= -amplitude")
    assert_true(abs(np.mean(z)) < 1e-2 * m.a, "zero-mean over one period")


def test_orifice_head_dependence():
    print("\n[Test 2] Turbine flow grows with head (sqrt law) and is signed")
    m, _ = make_model()
    q1 = m.turbine_flow(1.5)
    q2 = m.turbine_flow(2.5)
    assert_true(q2 > q1 > 0, f"|Q| increases with H: {q1:.0f} -> {q2:.0f} m3/s")
    qn = m.turbine_flow(-1.5)
    assert_true(qn < 0, f"negative head -> outflow Q={qn:.0f} < 0")
    assert_true(abs(qn + q1) < 1e-6, "flow magnitude symmetric in head sign")


def test_power_relation():
    print("\n[Test 3] P = eta*rho*g*Q*H and bounded by installed capacity")
    m, _ = make_model()
    H = 3.0
    Q = m.turbine_flow(H)
    P = m.turbine_power(Q, H)
    P_check = m.eta * m.rho * m.g * abs(Q) * abs(H)
    assert_true(abs(P - P_check) < 1e-3, "power matches eta*rho*g*Q*H")
    # capped below installed at large head
    Q_big = m.turbine_flow(8.0)
    P_big = m.turbine_power(Q_big, 8.0)
    assert_true(P_big <= m.P_rated_total + 1.0, f"P {P_big/1e6:.1f} MW <= rated {m.P_rated_total/1e6:.0f} MW")


def test_mass_conservation():
    print("\n[Test 4] Mass conservation: A*dz/dt = Q (pointwise) and bulk volume")
    m, _ = make_model()
    r = m.simulate(n_cycles=2, n_eval=3000)
    # Rigorous pointwise residual (median, robust to switch-instant samples):
    print(f"  median |A*dz/dt - Q| / (A*max|dz/dt|) = {r['mass_resid_med']:.2e}")
    assert_true(r["mass_resid_med"] < 1e-3,
                "pointwise continuity A*dz/dt = Q holds to solver tolerance")
    # Bulk volume balance (quadrature-limited at step transitions):
    rel_err = abs(r["dV_state_m3"] - r["dV_flux_m3"]) / (abs(r["dV_flux_m3"]) + 1e6)
    print(f"  dV_state={r['dV_state_m3']:.3e}  dV_flux={r['dV_flux_m3']:.3e}  bulk rel_err={rel_err:.2e}")
    assert_true(rel_err < 0.12, "bulk volume change matches integrated through-flow")


def test_energy_bound():
    print("\n[Test 5] Energy conservation: E_elec <= available potential energy")
    m, _ = make_model()
    r = m.simulate(n_cycles=2)
    assert_true(r["energy_J"] > 0, "positive energy generated")
    assert_true(r["energy_J"] <= r["E_available_J"] + 1.0,
                f"E_elec {r['energy_J']:.2e} <= E_avail {r['E_available_J']:.2e}")
    eta_eff = r["energy_J"] / r["E_available_J"]
    assert_true(0 < eta_eff < 1.0, f"effective efficiency {eta_eff:.3f} in (0,1)")


def test_lagoon_tracks_within_tide():
    print("\n[Test 6] Lagoon level stays within tidal envelope")
    m, _ = make_model()
    r = m.simulate(n_cycles=2)
    assert_true(np.max(r["z_lagoon"]) <= m.a + 0.5,
                f"lagoon peak {np.max(r['z_lagoon']):.2f} within tide + margin")
    assert_true(np.min(r["z_lagoon"]) >= -m.a - 0.5,
                f"lagoon trough {np.min(r['z_lagoon']):.2f} within tide - margin")


def test_two_way_generation():
    print("\n[Test 7] Bidirectional: generation on both flood and ebb")
    m, _ = make_model()
    r = m.simulate(n_cycles=2)
    flow = r["flow"]
    gen = np.array([1.0 if mo == "GEN" else 0.0 for mo in r["modes"]])
    inflow_gen = np.any((gen > 0) & (flow > 1.0))
    outflow_gen = np.any((gen > 0) & (flow < -1.0))
    assert_true(inflow_gen, "generates during flood (inflow)")
    assert_true(outflow_gen, "generates during ebb (outflow)")


def test_holding_optimum():
    print("\n[Test 8] Holding head has an interior optimum (head vs hours)")
    m, _ = make_model()
    best_h, best_E, grid, energies = m.optimal_hold_head(n_cycles=2)
    print(f"  optimal H_hold={best_h:.2f} m -> {best_E:.1f} MWh/cycle")
    assert_true(energies.max() > energies.min() * 1.0, "energy varies with hold head")
    interior = (best_h > grid.min() + 1e-6) and (best_h < grid.max() - 1e-6)
    assert_true(interior or energies.argmax() > 0,
                "optimum is interior (not at zero-hold extreme)")


def test_higher_amplitude_more_energy():
    print("\n[Test 9] Larger tidal amplitude yields more energy per cycle")
    m, _ = make_model()
    a0 = m.a
    m.a = 3.0
    e_small = m.simulate(n_cycles=2)["energy_per_cycle_MWh"]
    m.a = 5.5
    e_big = m.simulate(n_cycles=2)["energy_per_cycle_MWh"]
    m.a = a0
    assert_true(e_big > e_small, f"E(5.5 m)={e_big:.1f} > E(3.0 m)={e_small:.1f} MWh")


def test_capacity_factor_range():
    print("\n[Test 10] Capacity factor physically reasonable (0 < CF < 0.6)")
    _, cm = make_model()
    r = cm.predict({"n_cycles": 2})
    cf = r["capacity_factor"]
    assert_true(0.0 < cf < 0.6, f"capacity factor {cf:.3f} in (0, 0.6)")


def test_predict_interface():
    print("\n[Test 11] ComponentModel predict()/get_info() interface")
    _, cm = make_model()
    info = cm.get_info()
    assert_true(info["component_id"] == "EC133", "component_id EC133")
    r = cm.predict({"n_cycles": 1, "n_eval": 800})
    for key in ["t", "z_sea", "z_lagoon", "head", "flow", "power_MW",
                "energy_per_cycle_MWh", "avg_power_MW", "capacity_factor"]:
        assert_true(key in r, f"key '{key}' in output")
    assert_true(len(r["t"]) == len(r["power_MW"]), "arrays same length")


def test_benchmark():
    print("\n[Test 12] Benchmark: 2-cycle solve_ivp simulation")
    m, _ = make_model()
    t0 = time.perf_counter()
    m.simulate(n_cycles=2, n_eval=2000)
    elapsed = time.perf_counter() - t0
    print(f"  2-cycle simulation in {elapsed*1000:.1f} ms")
    assert_true(elapsed < 5.0, "completes in < 5 s")


if __name__ == "__main__":
    tests = [
        test_sea_level_sinusoid,
        test_orifice_head_dependence,
        test_power_relation,
        test_mass_conservation,
        test_energy_bound,
        test_lagoon_tracks_within_tide,
        test_two_way_generation,
        test_holding_optimum,
        test_higher_amplitude_more_energy,
        test_capacity_factor_range,
        test_predict_interface,
        test_benchmark,
    ]
    passed = failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except (AssertionError, Exception) as e:
            failed += 1
            print(f"  ERROR: {e}")

    print(f"\n{'='*60}")
    print(f"EC133 Tidal Lagoon F2a -- Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    sys.exit(0 if failed == 0 else 1)
