"""
EC122 -- Pumped Hydro Storage (PHS) -- F2a Dynamic
Test suite: physics sanity, ODE convergence, edge cases.
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from model import PumpedHydroStorage_F2a
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
def test_turbine_positive_flow():
    print("\n[Test 1] Turbine mode produces positive flow")
    m, _ = make_model()
    r = m.simulate(200e6, "turbine", dt=1.0, duration_s=60.0)
    assert_true(r["Q"][-1] > 0 or np.any(r["Q"] > 0),
                f"Q develops positive values in turbine mode (Q_final={r['Q'][-1]:.2f})")


def test_pump_negative_flow():
    print("\n[Test 2] Pump mode produces negative flow")
    m, _ = make_model()
    r = m.simulate(-200e6, "pump", dt=1.0, duration_s=60.0)
    assert_true(r["Q"][-1] < 0 or np.any(r["Q"] < 0),
                f"Q develops negative values in pump mode (Q_final={r['Q'][-1]:.2f})")


def test_reservoir_mass_balance():
    print("\n[Test 3] Reservoir mass balance (water conservation)")
    m, _ = make_model()
    r = m.simulate(100e6, "turbine", dt=1.0, duration_s=300.0)
    dH_up = r["H_upper"][0] - r["H_upper"][-1]
    dH_lo = r["H_lower"][-1] - r["H_lower"][0]
    V_up = m.A_upper * dH_up
    V_lo = m.A_lower * dH_lo
    ratio = V_up / max(abs(V_lo), 1.0) if abs(V_lo) > 1 else 1.0
    assert_true(abs(ratio - 1.0) < 0.05,
                f"Volume balance: V_up={V_up:.0f}, V_lo={V_lo:.0f}, ratio={ratio:.4f}")


def test_upper_reservoir_drops_turbine():
    print("\n[Test 4] Upper reservoir level drops in turbine mode")
    m, _ = make_model()
    r = m.simulate(200e6, "turbine", dt=1.0, duration_s=300.0)
    assert_true(r["H_upper"][-1] < r["H_upper"][0],
                f"H_up: {r['H_upper'][0]:.2f} -> {r['H_upper'][-1]:.2f}")


def test_upper_reservoir_rises_pump():
    print("\n[Test 5] Upper reservoir level rises in pump mode")
    m, _ = make_model()
    r = m.simulate(-200e6, "pump", Q0=0.0, H_up0=50.0, dt=1.0, duration_s=300.0)
    assert_true(r["H_upper"][-1] > r["H_upper"][0],
                f"H_up: {r['H_upper'][0]:.2f} -> {r['H_upper'][-1]:.2f}")


def test_soc_in_range():
    print("\n[Test 6] SOC stays in [0, 1]")
    m, _ = make_model()
    r = m.simulate(200e6, "turbine", dt=5.0, duration_s=3600.0)
    assert_true(np.all(r["SOC"] >= -0.01) and np.all(r["SOC"] <= 1.01),
                f"SOC range: [{r['SOC'].min():.4f}, {r['SOC'].max():.4f}]")


def test_efficiency_range():
    print("\n[Test 7] Efficiency in reasonable range")
    m, _ = make_model()
    r = m.simulate(200e6, "turbine", dt=1.0, duration_s=120.0)
    eff_nonzero = r["efficiency"][r["efficiency"] > 0]
    if len(eff_nonzero) > 0:
        assert_true(np.all(eff_nonzero <= 1.0),
                    f"Efficiency <= 1.0, max={eff_nonzero.max():.4f}")
        assert_true(np.mean(eff_nonzero) > 0.5,
                    f"Avg efficiency={np.mean(eff_nonzero):.4f} > 0.5")
    else:
        print(f"  {PASS}  No non-zero efficiency (idle)")


def test_idle_flow_decays():
    print("\n[Test 8] Idle mode: flow decays toward zero")
    m, _ = make_model()
    r = m.simulate(0.0, "idle", Q0=50.0, dt=1.0, duration_s=60.0)
    assert_true(abs(r["Q"][-1]) < abs(r["Q"][0]),
                f"Q decays: {r['Q'][0]:.2f} -> {r['Q'][-1]:.2f}")


def test_predict_interface():
    print("\n[Test 9] ComponentModel predict() interface")
    _, cm = make_model()
    r = cm.predict({"P_electrical_W": 100e6, "mode": "turbine",
                     "dt": 1.0, "duration_s": 10.0})
    for key in ["t", "Q", "omega", "H_upper", "H_lower", "H_net",
                "P_hydraulic", "P_electrical", "efficiency", "SOC", "E_stored"]:
        assert_true(key in r, f"Key '{key}' in output")
    assert_true(len(r["t"]) == len(r["Q"]), "Arrays same length")


def test_benchmark():
    print("\n[Test 10] Benchmark: 3600s sim at dt=1.0")
    m, _ = make_model()
    t0 = time.perf_counter()
    m.simulate(200e6, "turbine", dt=1.0, duration_s=3600.0)
    elapsed = time.perf_counter() - t0
    print(f"  3600s simulation in {elapsed*1000:.1f} ms")
    assert_true(elapsed < 30.0, "Completes in < 30 s")


if __name__ == "__main__":
    tests = [
        test_turbine_positive_flow,
        test_pump_negative_flow,
        test_reservoir_mass_balance,
        test_upper_reservoir_drops_turbine,
        test_upper_reservoir_rises_pump,
        test_soc_in_range,
        test_efficiency_range,
        test_idle_flow_decays,
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
    print(f"EC122 PHS F2a -- Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    sys.exit(0 if failed == 0 else 1)
