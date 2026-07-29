"""
EC001 -- PEM Fuel Cell (PEMFC) -- F2a Electrochemical
Test suite: physics sanity, ODE convergence, edge cases.
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from model import PEMFC_F2a
from predict import ComponentModel

PASS = "\u2713"
FAIL = "\u2717"


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
def test_nernst_range():
    print("\n[Test 1] Nernst voltage in physical range")
    m, _ = make_model()
    E = m.nernst_voltage(353.15, 1.0, 0.21)
    assert_true(1.0 < E < 1.3, f"E_nernst={E:.4f} V in [1.0, 1.3]")
    E2 = m.nernst_voltage(353.15, 2.0, 0.5)
    assert_true(E2 > E, f"Higher pressures raise E: {E2:.4f} > {E:.4f}")


def test_voltage_below_nernst():
    print("\n[Test 2] V_cell < E_nernst for j > 0")
    m, _ = make_model()
    for j in [0.1, 0.5, 1.0]:
        V = m.cell_voltage(j, 353.15, 1.0, 0.21)
        E = m.nernst_voltage(353.15, 1.0, 0.21)
        assert_true(V < E, f"V({j})={V:.4f} < E={E:.4f}")


def test_voltage_monotone():
    print("\n[Test 3] V_cell decreases with j")
    m, _ = make_model()
    j_vals = np.linspace(0.01, 1.3, 50)
    V_prev = m.cell_voltage(j_vals[0], 353.15, 1.0, 0.21)
    for j in j_vals[1:]:
        V = m.cell_voltage(j, 353.15, 1.0, 0.21)
        assert_true(V <= V_prev + 1e-9, f"V({j:.2f})={V:.4f} <= V_prev={V_prev:.4f}")
        V_prev = V
    print("  All 49 pairs checked.")


def test_thermal_ode_heats_up():
    print("\n[Test 4] Thermal ODE: stack heats up from cold start")
    m, _ = make_model()
    r = m.simulate(0.6, 300.0, 1.0, 0.21, 0.5, 120.0)
    assert_true(r["temperature"][-1] > 300.0, f"T_final={r['temperature'][-1]:.2f} > 300 K")
    assert_true(r["temperature"][-1] < 400.0, f"T_final={r['temperature'][-1]:.2f} < 400 K (reasonable)")


def test_thermal_steady_state():
    print("\n[Test 5] Thermal reaches approximate steady state")
    m, _ = make_model()
    r = m.simulate(0.5, 343.15, 1.0, 0.21, 1.0, 600.0)
    dT = abs(r["temperature"][-1] - r["temperature"][-2])
    assert_true(dT < 0.1, f"Near SS: dT={dT:.4f} K between last two steps")


def test_efficiency_range():
    print("\n[Test 6] Efficiency in (0, 1)")
    m, _ = make_model()
    r = m.simulate(0.5, 353.15, 1.0, 0.21, 1.0, 10.0)
    for eta in r["efficiency"]:
        assert_true(0 < eta < 1.0, f"eta={eta:.4f}")


def test_overpotentials_positive():
    print("\n[Test 7] All overpotentials >= 0")
    m, _ = make_model()
    r = m.simulate(0.6, 353.15, 1.0, 0.21, 1.0, 10.0)
    for name, arr in r["overpotentials"].items():
        if name == "E_nernst":
            continue
        assert_true(np.all(arr >= -1e-9), f"{name} all >= 0")


def test_step_response():
    print("\n[Test 8] Step current response -- voltage drops")
    m, _ = make_model()
    def step_j(t):
        return 0.3 if t < 30 else 0.8
    r = m.simulate(step_j, 353.15, 1.0, 0.21, 0.5, 60.0)
    idx_before = np.argmin(np.abs(r["t"] - 29.5))
    idx_after = np.argmin(np.abs(r["t"] - 31.0))
    assert_true(r["voltage"][idx_after] < r["voltage"][idx_before],
                "Voltage drops after current step up")


def test_concentration_near_jL():
    print("\n[Test 9] Concentration loss diverges near j_L")
    m, _ = make_model()
    v1 = m.concentration_overpotential(1.0)
    v2 = m.concentration_overpotential(1.45)
    assert_true(v2 > v1 * 3, f"eta_conc(1.45)={v2:.4f} >> eta_conc(1.0)={v1:.4f}")


def test_predict_interface():
    print("\n[Test 10] ComponentModel predict() interface")
    _, cm = make_model()
    r = cm.predict({"current_density_A_cm2": 0.5, "dt": 1.0, "duration_s": 5.0})
    for key in ["t", "voltage", "power_density", "efficiency", "temperature", "overpotentials"]:
        assert_true(key in r, f"Key '{key}' in output")
    assert_true(len(r["t"]) == len(r["voltage"]), "Arrays same length")


def test_benchmark():
    print("\n[Test 11] Benchmark: 60s sim at dt=0.1")
    m, _ = make_model()
    t0 = time.perf_counter()
    m.simulate(0.5, 343.15, 1.0, 0.21, 0.1, 60.0)
    elapsed = time.perf_counter() - t0
    print(f"  60s simulation in {elapsed*1000:.1f} ms")
    assert_true(elapsed < 10.0, "Completes in < 10 s")


if __name__ == "__main__":
    tests = [
        test_nernst_range,
        test_voltage_below_nernst,
        test_voltage_monotone,
        test_thermal_ode_heats_up,
        test_thermal_steady_state,
        test_efficiency_range,
        test_overpotentials_positive,
        test_step_response,
        test_concentration_near_jL,
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
    print(f"EC001 PEMFC F2a -- Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    sys.exit(0 if failed == 0 else 1)
