"""
EC188 -- SMES -- F2a Physics-Lumped
Test suite: energy law, ODE conservation, charge/discharge dynamics, limits,
round-trip efficiency bounds, predict() interface, benchmark timing.
NO pytest -- run as:  python3 scripts/test_model.py
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from model import SMES_F2a
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
def test_energy_law():
    print("\n[Test 1] E = 0.5 L I^2 exact")
    m, _ = make_model()
    for I in [0.0, 100.0, 250.0, m.I_max]:
        E = m.energy_J(I)
        assert_true(abs(E - 0.5 * m.L * I ** 2) < 1e-6, f"E({I})={E:.3e} J")
    assert_true(abs(m.E_max_MJ - 0.5 * m.L * m.I_max ** 2 / 1e6) < 1e-9,
                f"E_max={m.E_max_MJ:.3f} MJ")


def test_current_energy_inverse():
    print("\n[Test 2] I = sqrt(2E/L) inverts E = 0.5 L I^2")
    m, _ = make_model()
    for I in [50.0, 200.0, 447.0]:
        E = m.energy_J(I)
        assert_true(abs(m.current_from_energy(E) - I) < 1e-6,
                    f"round-trip I={I} -> {m.current_from_energy(E):.4f}")


def test_charge_raises_current_and_energy():
    print("\n[Test 3] Positive chopper voltage charges the coil")
    m, _ = make_model()
    r = m.simulate(0.0, 2500.0, mode="voltage", dt=0.01, duration_s=1.0)
    assert_true(r["I_coil_A"][-1] > r["I_coil_A"][0], "current rose")
    assert_true(r["E_stored_MJ"][-1] > r["E_stored_MJ"][0], "energy rose")
    assert_true(np.all(r["P_coil_MW"][1:] > 0), "coil power positive (into coil)")


def test_discharge_lowers_current():
    print("\n[Test 4] Negative chopper voltage discharges the coil")
    m, _ = make_model()
    r = m.simulate(447.0, -2500.0, mode="voltage", dt=0.01, duration_s=1.0)
    assert_true(r["I_coil_A"][-1] < r["I_coil_A"][0], "current fell")
    assert_true(r["E_stored_MJ"][-1] < r["E_stored_MJ"][0], "energy fell")
    assert_true(np.all(r["P_coil_MW"][1:-1] < 0), "coil power negative (out of coil)")


def test_ode_matches_analytic_dIdt():
    print("\n[Test 5] dI/dt = V/L (R~0): I(t) = I0 + V/L * t")
    m, _ = make_model()
    V = 1000.0
    r = m.simulate(0.0, V, mode="voltage", dt=0.05, duration_s=1.0)
    I_analytic = V / m.L * r["t"]   # R_coil ~ 0
    err = np.max(np.abs(r["I_coil_A"] - I_analytic))
    assert_true(err < 1e-2, f"max |I_num - I_analytic| = {err:.2e} A")


def test_energy_conservation():
    print("\n[Test 6] Energy conservation: dE = integral(P_coil dt)")
    m, _ = make_model()
    r = m.simulate(50.0, 1500.0, mode="voltage", dt=0.002, duration_s=0.8)
    dE_state = (r["E_stored_MJ"][-1] - r["E_stored_MJ"][0]) * 1e6   # J
    int_P = np.trapezoid(r["P_coil_W"], r["t"])                      # J
    rel = abs(dE_state - int_P) / (abs(int_P) + 1e-9)
    assert_true(rel < 1e-3, f"rel mismatch {rel:.2e} (dE={dE_state:.1f} J, int P={int_P:.1f} J)")


def test_current_limits():
    print("\n[Test 7] Coil current clamps within [I_min, I_max]")
    m, _ = make_model()
    r = m.simulate(0.0, m.V_dc_max, mode="voltage", dt=0.05, duration_s=60.0)
    assert_true(np.all(r["I_coil_A"] <= m.I_max + 1e-6), "never exceeds I_max")
    assert_true(np.all(r["I_coil_A"] >= m.I_min - 1e-6), "never below I_min")
    r2 = m.simulate(50.0, -m.V_dc_max, mode="voltage", dt=0.05, duration_s=60.0)
    assert_true(r2["I_coil_A"][-1] <= 1e-3, "fully discharges and holds at floor")


def test_power_VI():
    print("\n[Test 8] Coil power P = V * I")
    m, _ = make_model()
    r = m.simulate(100.0, 2000.0, mode="voltage", dt=0.01, duration_s=0.5)
    err = np.max(np.abs(r["P_coil_W"] - r["V_chop_V"] * r["I_coil_A"]))
    assert_true(err < 1e-6, f"max |P - V I| = {err:.2e} W")


def test_fast_response():
    print("\n[Test 9] Fast response: reaches rated power quickly")
    m, _ = make_model()
    # Request rated DC coil power at I ~ 250 A; chopper applies V = P/I.
    r = m.simulate(250.0, m.P_rated, mode="power", dt=1e-3, duration_s=0.05)
    # SMES hallmark: sub-second to full power. Coil power should reach ~rated.
    pk = np.max(np.abs(r["P_coil_W"]))
    assert_true(pk >= 0.5 * m.P_rated, f"peak coil power {pk/1e6:.2f} MW within 50 ms")


def test_round_trip_efficiency_bounds():
    print("\n[Test 10] 0 < round-trip eff < 1 (converter + cryo losses)")
    m, _ = make_model()
    for P_MW in [0.5, 1.0, 2.0]:
        rt = m.round_trip_efficiency(P_W=P_MW * 1e6)
        eta = rt["eta_rt"]
        assert_true(0.0 < eta < 1.0, f"P={P_MW} MW: eta_rt={eta*100:.2f}%")
        # converter-only ceiling is eta_conv^2
        assert_true(eta < m.eta_conv ** 2 + 1e-9,
                    f"eta_rt {eta:.4f} <= eta_conv^2 {m.eta_conv**2:.4f}")
    # cryo penalty makes lower power less efficient (cryo fixed cost dominates)
    e_lo = m.round_trip_efficiency(P_W=0.5e6)["eta_rt"]
    e_hi = m.round_trip_efficiency(P_W=2.0e6)["eta_rt"]
    assert_true(e_hi > e_lo, f"higher power more efficient: {e_hi:.4f} > {e_lo:.4f}")


def test_predict_interface():
    print("\n[Test 11] ComponentModel predict() interface")
    _, cm = make_model()
    r = cm.predict({"I0_A": 0.0, "command": 2000.0, "mode": "voltage",
                    "dt": 0.05, "duration_s": 0.5})
    for key in ["t", "I_coil_A", "V_chop_V", "E_stored_MJ", "SOC",
                "P_coil_MW", "P_grid_MW", "P_cryo_MW"]:
        assert_true(key in r, f"key '{key}' present")
    assert_true(len(r["t"]) == len(r["I_coil_A"]), "arrays same length")
    assert_true(np.all((r["SOC"] >= -1e-9) & (r["SOC"] <= 1.0 + 1e-9)),
                "SOC in [0,1]")
    info = cm.get_info()
    assert_true(info["component_id"] == "EC188", "component_id EC188")


def test_benchmark():
    print("\n[Test 12] Benchmark: 5 s sim at dt=1 ms")
    m, _ = make_model()
    t0 = time.perf_counter()
    m.simulate(0.0, 2500.0, mode="voltage", dt=1e-3, duration_s=5.0)
    elapsed = time.perf_counter() - t0
    print(f"  5 s simulation in {elapsed*1000:.1f} ms")
    assert_true(elapsed < 5.0, "completes in < 5 s")


if __name__ == "__main__":
    tests = [
        test_energy_law,
        test_current_energy_inverse,
        test_charge_raises_current_and_energy,
        test_discharge_lowers_current,
        test_ode_matches_analytic_dIdt,
        test_energy_conservation,
        test_current_limits,
        test_power_VI,
        test_fast_response,
        test_round_trip_efficiency_bounds,
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
    print(f"EC188 SMES F2a -- Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    sys.exit(0 if failed == 0 else 1)
