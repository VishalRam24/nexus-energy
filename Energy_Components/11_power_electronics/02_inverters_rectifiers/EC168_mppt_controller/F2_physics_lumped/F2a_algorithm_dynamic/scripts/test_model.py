"""
EC168 -- MPPT Controller -- F2a Algorithm Dynamic
Test suite: physics sanity, MPPT convergence, edge cases.
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from model import MPPTController_F2a, PVSingleDiode
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
def test_pv_iv_monotone():
    print("\n[Test 1] PV I-V curve: current decreases with voltage")
    m, _ = make_model()
    pv = m.pv
    V_arr = np.linspace(0.5, 35.0, 50)
    I_prev = pv.current(V_arr[0], 1000.0)
    for V in V_arr[1:]:
        I = pv.current(V, 1000.0)
        assert_true(I <= I_prev + 0.01,
                    f"I({V:.1f}V)={I:.3f} <= I_prev={I_prev:.3f}")
        I_prev = I


def test_pv_power_peak():
    print("\n[Test 2] PV P-V curve has a maximum (MPP exists)")
    m, _ = make_model()
    pv = m.pv
    V_m, I_m, P_m = pv.mpp(1000.0)
    assert_true(P_m > 200.0, f"P_mpp={P_m:.1f} W > 200 W")
    assert_true(P_m < 400.0, f"P_mpp={P_m:.1f} W < 400 W (realistic for 300W panel)")
    assert_true(20.0 < V_m < 42.0, f"V_mpp={V_m:.1f} V in [20, 42]")


def test_pv_irradiance_scaling():
    print("\n[Test 3] Higher irradiance -> higher P_mpp")
    m, _ = make_model()
    pv = m.pv
    _, _, P_500 = pv.mpp(500.0)
    _, _, P_1000 = pv.mpp(1000.0)
    assert_true(P_1000 > P_500, f"P(1000)={P_1000:.1f} > P(500)={P_500:.1f}")


def test_pv_zero_irradiance():
    print("\n[Test 4] Zero irradiance -> zero power")
    m, _ = make_model()
    pv = m.pv
    _, _, P_0 = pv.mpp(0.0)
    assert_true(P_0 < 0.1, f"P(G=0)={P_0:.4f} ~ 0")


def test_mppt_convergence():
    print("\n[Test 5] MPPT converges near MPP within 1 second")
    m, _ = make_model()
    r = m.simulate(1000.0, 298.15, 0.001, 1.0)
    # Get theoretical MPP
    _, _, P_mpp = m.pv.mpp(1000.0)
    # Check last 100 ms average tracking efficiency
    idx_last = r["t"] > 0.8
    eta_avg = np.mean(r["tracking_efficiency"][idx_last])
    assert_true(eta_avg > 0.90,
                f"Avg tracking efficiency (last 200ms) = {eta_avg:.4f} > 0.90")


def test_mppt_step_response():
    print("\n[Test 6] MPPT tracks irradiance step change")
    m, _ = make_model()
    def G_step(t):
        return 1000.0 if t < 0.5 else 500.0
    r = m.simulate(G_step, 298.15, 0.001, 1.5)
    # Power should decrease after step
    idx_before = np.argmin(np.abs(r["t"] - 0.45))
    idx_after = np.argmin(np.abs(r["t"] - 1.4))
    assert_true(r["P_pv"][idx_after] < r["P_pv"][idx_before],
                "P_pv decreases after irradiance drop")
    # Should still track reasonably well
    _, _, P_mpp_500 = m.pv.mpp(500.0)
    eta_end = r["tracking_efficiency"][idx_after]
    assert_true(eta_end > 0.80,
                f"Tracking efficiency after step = {eta_end:.3f} > 0.80")


def test_duty_cycle_bounds():
    print("\n[Test 7] Duty cycle stays within [D_min, D_max]")
    m, _ = make_model()
    r = m.simulate(1000.0, 298.15, 0.001, 1.0)
    assert_true(np.all(r["duty_cycle"] >= m.D_min - 1e-6),
                f"D >= D_min={m.D_min}")
    assert_true(np.all(r["duty_cycle"] <= m.D_max + 1e-6),
                f"D <= D_max={m.D_max}")


def test_output_voltage_positive():
    print("\n[Test 8] Output voltage and current are non-negative")
    m, _ = make_model()
    r = m.simulate(1000.0, 298.15, 0.001, 0.5)
    assert_true(np.all(r["V_out"] >= -0.01), "V_out >= 0")
    assert_true(np.all(r["I_L"] >= -0.01), "I_L >= 0")


def test_predict_interface():
    print("\n[Test 9] ComponentModel predict() interface")
    _, cm = make_model()
    r = cm.predict({"irradiance": 800.0, "dt": 0.001, "duration_s": 0.1})
    for key in ["t", "V_pv", "I_pv", "P_pv", "V_ref", "duty_cycle",
                "I_L", "V_out", "P_out", "tracking_efficiency"]:
        assert_true(key in r, f"Key '{key}' in output")
    assert_true(len(r["t"]) == len(r["P_pv"]), "Arrays same length")


def test_temperature_effect():
    print("\n[Test 10] Higher temperature -> lower V_mpp (PV physics)")
    m, _ = make_model()
    pv = m.pv
    V_25, _, _ = pv.mpp(1000.0, 298.15)
    V_50, _, _ = pv.mpp(1000.0, 323.15)
    assert_true(V_50 < V_25, f"V_mpp(50C)={V_50:.2f} < V_mpp(25C)={V_25:.2f}")


def test_benchmark():
    print("\n[Test 11] Benchmark: 1s sim at dt=0.001")
    m, _ = make_model()
    t0 = time.perf_counter()
    m.simulate(1000.0, 298.15, 0.001, 1.0)
    elapsed = time.perf_counter() - t0
    print(f"  1s simulation in {elapsed*1000:.1f} ms")
    assert_true(elapsed < 30.0, "Completes in < 30 s")


if __name__ == "__main__":
    tests = [
        test_pv_iv_monotone,
        test_pv_power_peak,
        test_pv_irradiance_scaling,
        test_pv_zero_irradiance,
        test_mppt_convergence,
        test_mppt_step_response,
        test_duty_cycle_bounds,
        test_output_voltage_positive,
        test_predict_interface,
        test_temperature_effect,
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
    print(f"EC168 MPPT Controller F2a -- Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    sys.exit(0 if failed == 0 else 1)
