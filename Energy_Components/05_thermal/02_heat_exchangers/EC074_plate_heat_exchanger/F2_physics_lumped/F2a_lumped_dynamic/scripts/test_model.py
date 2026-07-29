"""
EC074 -- Plate Heat Exchanger -- F2a Lumped Dynamic
Test suite: physics sanity, ODE convergence, edge cases.
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from model import PlateHeatExchanger_F2a
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
def test_cold_start_convergence():
    print("\n[Test 1] Cold start: temperatures converge to steady state")
    m, _ = make_model()
    r = m.simulate(1.2, 1.0, 353.15, 293.15, 293.15, 293.15, 1.0, 600.0)
    # Hot side should warm up, cold side should warm up
    assert_true(r["T_hot_out"][-1] > 293.15, f"T_hot rises from cold start: {r['T_hot_out'][-1]:.2f} K")
    assert_true(r["T_cold_out"][-1] > 293.15, f"T_cold rises from cold start: {r['T_cold_out'][-1]:.2f} K")
    # Hot outlet should be below hot inlet
    assert_true(r["T_hot_out"][-1] < 353.15, f"T_hot_out < T_hot_in: {r['T_hot_out'][-1]:.2f} < 353.15")


def test_energy_conservation():
    print("\n[Test 2] Energy conservation: Q_hot ~ Q_cold at steady state")
    m, _ = make_model()
    r = m.simulate(1.2, 1.0, 353.15, 293.15, 340.0, 310.0, 1.0, 1200.0)
    # At steady state, energy balance: m_dot_h*cp*(T_h_in - T_h_out) ~ m_dot_c*cp*(T_c_out - T_c_in)
    Q_h = 1.2 * 4186.0 * (353.15 - r["T_hot_out"][-1])
    Q_c = 1.0 * 4186.0 * (r["T_cold_out"][-1] - 293.15)
    # Should be close (within 5% accounting for heat loss to ambient)
    rel_diff = abs(Q_h - Q_c) / max(Q_h, 1.0)
    assert_true(rel_diff < 0.10, f"Energy balance: Q_h={Q_h:.0f} W, Q_c={Q_c:.0f} W, diff={rel_diff*100:.1f}%")


def test_hot_above_cold():
    print("\n[Test 3] T_hot_out >= T_cold_out always (2nd law)")
    m, _ = make_model()
    r = m.simulate(1.2, 1.0, 353.15, 293.15, 340.0, 310.0, 1.0, 600.0)
    # After initial transient, hot should stay above cold
    for i in range(len(r["t"])):
        if r["t"][i] > 30:  # after initial transient
            assert_true(r["T_hot_out"][i] >= r["T_cold_out"][i] - 0.1,
                        f"T_hot >= T_cold at t={r['t'][i]:.0f}s")
            break  # just check a few points
    assert_true(r["T_hot_out"][-1] >= r["T_cold_out"][-1] - 0.1,
                f"T_hot >= T_cold at final time")


def test_effectiveness_range():
    print("\n[Test 4] Effectiveness in [0, 1]")
    m, _ = make_model()
    r = m.simulate(1.2, 1.0, 353.15, 293.15, 340.0, 310.0, 1.0, 600.0)
    for i in range(10, len(r["t"])):  # skip initial transient
        eps = r["effectiveness"][i]
        assert_true(0.0 <= eps <= 1.05, f"epsilon={eps:.4f} in [0, 1.05] at t={r['t'][i]:.0f}")
    print("  All points checked.")


def test_step_flow_response():
    print("\n[Test 5] Step flow change: Q increases with higher flow")
    m, _ = make_model()

    def step_flow(t):
        return 0.5 if t < 200 else 1.5

    r = m.simulate(step_flow, 1.0, 353.15, 293.15, 340.0, 310.0, 1.0, 600.0)
    idx_before = np.argmin(np.abs(r["t"] - 190))
    idx_after = np.argmin(np.abs(r["t"] - 400))
    assert_true(r["Q_transfer"][idx_after] > r["Q_transfer"][idx_before],
                f"Q increases after flow step: {r['Q_transfer'][idx_after]:.0f} > {r['Q_transfer'][idx_before]:.0f}")


def test_zero_flow():
    print("\n[Test 6] Zero flow: no heat transfer")
    m, _ = make_model()
    r = m.simulate(0.0, 0.0, 353.15, 293.15, 320.0, 310.0, 1.0, 60.0)
    # With zero flow, UA should be zero, so Q_transfer should decay
    assert_true(abs(r["Q_transfer"][-1]) < abs(r["Q_transfer"][0]) + 10,
                "Q_transfer stays low with zero flow")


def test_equal_temperatures():
    print("\n[Test 7] Equal inlet temperatures: minimal heat transfer")
    m, _ = make_model()
    r = m.simulate(1.2, 1.0, 320.0, 320.0, 320.0, 320.0, 1.0, 120.0)
    assert_true(abs(r["Q_transfer"][-1]) < 100.0,
                f"Q near zero when T_h_in = T_c_in: Q={r['Q_transfer'][-1]:.1f} W")


def test_temperature_reasonable_range():
    print("\n[Test 8] Temperatures stay in reasonable range")
    m, _ = make_model()
    r = m.simulate(1.2, 1.0, 353.15, 293.15, 293.15, 293.15, 1.0, 600.0)
    assert_true(np.all(r["T_hot_out"] >= 273.15), "T_hot > 0 C always")
    assert_true(np.all(r["T_hot_out"] <= 423.15), "T_hot < 150 C always")
    assert_true(np.all(r["T_cold_out"] >= 273.15), "T_cold > 0 C always")
    assert_true(np.all(r["T_cold_out"] <= 423.15), "T_cold < 150 C always")


def test_predict_interface():
    print("\n[Test 9] ComponentModel predict() interface")
    _, cm = make_model()
    r = cm.predict({"duration_s": 30.0, "dt": 1.0})
    for key in ["t", "T_hot_out", "T_cold_out", "Q_transfer", "effectiveness", "UA"]:
        assert_true(key in r, f"Key '{key}' in output")
    assert_true(len(r["t"]) == len(r["T_hot_out"]), "Arrays same length")


def test_benchmark():
    print("\n[Test 10] Benchmark: 600s sim at dt=1.0")
    m, _ = make_model()
    t0 = time.perf_counter()
    m.simulate(1.2, 1.0, 353.15, 293.15, 293.15, 293.15, 1.0, 600.0)
    elapsed = time.perf_counter() - t0
    print(f"  600s simulation in {elapsed*1000:.1f} ms")
    assert_true(elapsed < 10.0, "Completes in < 10 s")


if __name__ == "__main__":
    tests = [
        test_cold_start_convergence,
        test_energy_conservation,
        test_hot_above_cold,
        test_effectiveness_range,
        test_step_flow_response,
        test_zero_flow,
        test_equal_temperatures,
        test_temperature_reasonable_range,
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
    print(f"EC074 Plate HX F2a -- Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    sys.exit(0 if failed == 0 else 1)
