"""
EC059 — Evacuated Tube Solar Collector — F2a Lumped-Capacitance
Test suite: energy conservation, efficiency monotonicity, night/limits, ODE, interface.
Run: python3 scripts/test_model.py   (system python3, NOT pytest)
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from model import EvacuatedTubeF2a
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
def test_zero_at_night():
    print("\n[Test 1] Q = 0 and efficiency = 0 at night (G = 0)")
    m, _ = make_model()
    r = m.simulate(0.0, T_ambient_c=10.0, T_inlet_c=40.0, dt=30.0, duration_s=3600.0)
    assert_true(np.all(r["useful_heat_w"] <= 1e-6), "useful_heat == 0 with no sun")
    assert_true(np.all(r["efficiency"] == 0.0), "efficiency == 0 with no sun")
    assert_true(r["q_absorbed_w"][-1] == 0.0, "absorbed solar == 0 at night")


def test_energy_conservation():
    print("\n[Test 2] Energy balance closes: C*dT/dt = Q_abs - Q_loss - Q_use")
    m, _ = make_model()
    r = m.simulate(800.0, T_ambient_c=20.0, T_inlet_c=40.0, dt=10.0, duration_s=2000.0)
    # pick an interior point, finite-difference dT/dt
    i = len(r["t"]) // 2
    dt = r["t"][i + 1] - r["t"][i - 1]
    dTdt = (r["T_absorber_c"][i + 1] - r["T_absorber_c"][i - 1]) / dt
    lhs = m.C * dTdt
    # use raw useful (may be clamped in output, recompute raw)
    q_use_raw = m.q_useful(r["T_absorber_c"][i], 40.0)
    rhs = r["q_absorbed_w"][i] - r["q_loss_w"][i] - q_use_raw
    rel = abs(lhs - rhs) / (abs(rhs) + 1.0)
    assert_true(rel < 0.02, f"balance residual {rel*100:.3f}% (LHS={lhs:.1f}, RHS={rhs:.1f})")


def test_steady_state_balance():
    print("\n[Test 3] At steady state Q_abs = Q_loss + Q_useful")
    m, _ = make_model()
    ss = m.steady_state(900.0, T_ambient_c=15.0, T_inlet_c=50.0)
    q_use_raw = m.q_useful(ss["T_absorber_c"], 50.0)
    resid = ss["q_absorbed_w"] - ss["q_loss_w"] - q_use_raw
    assert_true(abs(resid) < 5.0, f"SS residual {resid:.3f} W ~ 0")


def test_efficiency_decreases_with_reduced_temp():
    print("\n[Test 4] eta decreases monotonically with x=(Tm-Ta)/G")
    m, _ = make_model()
    G = 800.0
    Ta = 15.0
    etas = []
    for Tin in [20.0, 40.0, 60.0, 80.0, 100.0, 130.0]:
        eta = m.efficiency_steady(G, Tin, Ta)
        etas.append(eta)
    for k in range(1, len(etas)):
        assert_true(etas[k] <= etas[k - 1] + 1e-6,
                    f"eta at Tin step {k}: {etas[k]:.4f} <= {etas[k-1]:.4f}")
    print(f"  eta sweep: {[round(e,3) for e in etas]}")


def test_efficiency_intercept():
    print("\n[Test 5] eta -> eta_0*IAM as (Tm-Ta)/G -> 0")
    m, _ = make_model()
    # Inlet at ambient, high G => x ~ 0 => eta near optical efficiency
    eta = m.efficiency_steady(1000.0, T_inlet_c=20.0, T_ambient_c=20.0)
    assert_true(eta <= m.optical_eff + 1e-9, f"eta {eta:.4f} <= eta_0 {m.optical_eff}")
    assert_true(eta > 0.60, f"near-intercept eta {eta:.4f} close to eta_0 {m.optical_eff}")


def test_efficiency_bounds():
    print("\n[Test 6] efficiency stays in [0, eta_0]")
    m, _ = make_model()
    for Tin in [10.0, 60.0, 120.0, 180.0]:
        for G in [200.0, 600.0, 1000.0]:
            eta = m.efficiency_steady(G, Tin, 20.0)
            assert_true(0.0 <= eta <= m.optical_eff + 1e-9, f"eta({G},{Tin})={eta:.4f} bounded")


def test_radiation_dominates_loss():
    print("\n[Test 7] Loss is radiation-dominated and U_L rises with temperature")
    m, _ = make_model()
    UL_low = m.U_L(30.0, 20.0)
    UL_high = m.U_L(150.0, 20.0)
    assert_true(UL_high > UL_low, f"U_L(150C)={UL_high:.3f} > U_L(30C)={UL_low:.3f}")
    # radiative part of loss at high T exceeds residual conduction part
    T_c = 150.0
    Tk, Tak = T_c + 273.15, 20.0 + 273.15
    q_rad = m.emissivity * 5.670374419e-8 * m.A_abs * (Tk**4 - Tak**4)
    q_cond = m.U_residual * m.area * (T_c - 20.0)
    assert_true(q_rad > q_cond, f"radiation {q_rad:.1f} W > residual conduction {q_cond:.1f} W")


def test_iam_behavior():
    print("\n[Test 8] IAM = 1 at normal incidence, decreasing off-normal, 0 beyond 80deg")
    m, _ = make_model()
    assert_true(abs(m.iam(0.0) - 1.0) < 1e-9, "IAM(0deg) == 1")
    assert_true(m.iam(45.0) < 1.0, "IAM(45deg) < 1")
    assert_true(m.iam(60.0) < m.iam(45.0), "IAM decreases with angle")
    assert_true(m.iam(85.0) == 0.0, "IAM(85deg) == 0")


def test_cold_start_warms_up():
    print("\n[Test 9] Cold start: absorber warms above ambient under sun")
    m, _ = make_model()
    r = m.simulate(800.0, T_ambient_c=15.0, T_inlet_c=15.0, T0_c=15.0,
                   dt=10.0, duration_s=3600.0)
    assert_true(r["T_absorber_c"][0] <= 15.0 + 1e-6, "starts at ambient")
    assert_true(r["T_absorber_c"][-1] > r["T_absorber_c"][0] + 1.0, "warms up")
    assert_true(r["T_absorber_c"][-1] < 250.0, f"stays physical ({r['T_absorber_c'][-1]:.1f} C)")


def test_etc_outperforms_at_high_dt():
    print("\n[Test 10] Low loss: ETC retains efficiency at high (Tm-Ta)/G")
    m, _ = make_model()
    # at large deltaT / moderate G, ETC efficiency should still be > 0.3
    eta = m.efficiency_steady(600.0, T_inlet_c=90.0, T_ambient_c=10.0)
    assert_true(eta > 0.25, f"ETC eta at deltaT=80,G=600 = {eta:.4f} (vacuum keeps it high)")


def test_predict_interface():
    print("\n[Test 11] ComponentModel predict() interface + get_info()")
    _, cm = make_model()
    r = cm.predict({"irradiance": 700.0, "T_ambient_c": 18.0,
                    "T_inlet_c": 45.0, "dt": 30.0, "duration_s": 1800.0})
    for key in ["t", "T_absorber_c", "T_outlet_c", "useful_heat_w",
                "efficiency", "q_absorbed_w", "q_loss_w", "U_L_w_m2k"]:
        assert_true(key in r, f"Key '{key}' in output")
    assert_true(len(r["t"]) == len(r["useful_heat_w"]), "arrays same length")
    info = cm.get_info()
    assert_true(info["component_id"] == "EC059", "get_info id == EC059")


def test_time_varying_and_benchmark():
    print("\n[Test 12] Time-varying irradiance + benchmark timing")
    m, _ = make_model()

    def G_of_t(t):
        return 1000.0 * max(0.0, np.sin(np.pi * t / 86400.0))  # half-day sine

    t0 = time.perf_counter()
    r = m.simulate(G_of_t, T_ambient_c=12.0, T_inlet_c=40.0,
                   dt=300.0, duration_s=86400.0)
    elapsed = time.perf_counter() - t0
    print(f"  24h sim (dt=300s) in {elapsed*1000:.1f} ms")
    assert_true(elapsed < 5.0, "completes in < 5 s")
    # night portion (t near 0) has zero useful heat
    assert_true(r["useful_heat_w"][0] <= 1e-6, "no heat at t=0 (G=0)")
    assert_true(np.max(r["useful_heat_w"]) > 0.0, "produces heat midday")


if __name__ == "__main__":
    tests = [
        test_zero_at_night,
        test_energy_conservation,
        test_steady_state_balance,
        test_efficiency_decreases_with_reduced_temp,
        test_efficiency_intercept,
        test_efficiency_bounds,
        test_radiation_dominates_loss,
        test_iam_behavior,
        test_cold_start_warms_up,
        test_etc_outperforms_at_high_dt,
        test_predict_interface,
        test_time_varying_and_benchmark,
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
    print(f"EC059 ETC F2a -- Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    sys.exit(0 if failed == 0 else 1)
