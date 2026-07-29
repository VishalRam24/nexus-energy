"""
EC204 -- Calcium Looping -- F2a Carbonator/Calciner Coupled ODE
Test suite: physics sanity, conservation, cyclic decay, ODE behaviour, interface.
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from model import CalciumLoopingF2a
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
def test_capacity_decays_with_cycles():
    print("\n[Test 1] Grasa-Abanades capacity decays monotonically with N")
    m, _ = make_model()
    Ns = [1, 2, 5, 10, 50, 100, 500]
    Xs = [m.carrying_capacity(N) for N in Ns]
    for a, b in zip(Xs[:-1], Xs[1:]):
        assert_true(b <= a + 1e-12, f"X_N decreasing: {b:.4f} <= {a:.4f}")
    assert_true(Xs[0] <= m.X_max1 + 1e-9, f"X_1={Xs[0]:.4f} <= X_max1={m.X_max1}")
    print(f"  X_1={Xs[0]:.3f}, X_10={Xs[3]:.3f}, X_500={Xs[-1]:.3f}")


def test_capacity_residual_limit():
    print("\n[Test 2] Capacity tends to residual X_r as N -> infinity")
    m, _ = make_model()
    X_big = m.carrying_capacity(100000)
    assert_true(abs(X_big - m.X_r) < 1e-2, f"X_inf={X_big:.4f} ~ X_r={m.X_r}")
    assert_true(X_big >= m.X_r - 1e-9, f"never below residual {X_big:.4f}")


def test_carbon_conservation():
    print("\n[Test 3] Carbon balance: F_CO2_in == captured + out at all times")
    m, _ = make_model()
    r = m.simulate(5, dt=5.0, duration_s=200.0)
    assert_true(np.max(np.abs(r["carbon_balance_residual"])) < 1e-9,
                f"max residual={np.max(np.abs(r['carbon_balance_residual'])):.2e} mol/s")
    total = r["capture_rate"] + r["co2_out"]
    assert_true(np.allclose(total, m.F_CO2_in), "captured + out == F_CO2_in")


def test_capture_rate_bounded():
    print("\n[Test 4] Capture rate in [0, F_CO2_in], efficiency in [0,1]")
    m, _ = make_model()
    r = m.simulate(1, dt=5.0, duration_s=200.0)
    assert_true(np.all(r["capture_rate"] >= -1e-12), "capture_rate >= 0")
    assert_true(np.all(r["capture_rate"] <= m.F_CO2_in + 1e-9), "capture_rate <= F_CO2_in")
    assert_true(np.all((r["capture_efficiency"] >= -1e-12) &
                       (r["capture_efficiency"] <= 1.0 + 1e-9)), "eff in [0,1]")


def test_conversion_monotone_and_capped():
    print("\n[Test 5] Conversion rises monotonically toward capacity X_N")
    m, _ = make_model()
    r = m.simulate(10, dt=5.0, duration_s=400.0)
    X = r["conversion"]
    for a, b in zip(X[:-1], X[1:]):
        assert_true(b >= a - 1e-9, f"X non-decreasing: {b:.4f} >= {a:.4f}")
    X_N = r["capacity"][-1]
    assert_true(X[-1] <= X_N + 1e-6, f"X_final={X[-1]:.4f} <= X_N={X_N:.4f}")


def test_more_cycles_less_capture():
    print("\n[Test 6] Aged sorbent captures less than fresh sorbent")
    m, _ = make_model()
    r_fresh = m.simulate(1, dt=10.0, duration_s=400.0)
    r_aged = m.simulate(200, dt=10.0, duration_s=400.0)
    assert_true(r_aged["capture_rate"][-1] < r_fresh["capture_rate"][-1],
                f"aged {r_aged['capture_rate'][-1]:.3f} < fresh {r_fresh['capture_rate'][-1]:.3f} mol/s")


def test_exothermic_heats_bed():
    print("\n[Test 7] Exothermic carbonation heats bed above setpoint (cold-bias start)")
    m, _ = make_model()
    # Start slightly below setpoint: reaction heat should push T up toward/above setpoint
    r = m.simulate(1, T0_K=m.T_carb - 20.0, dt=2.0, duration_s=300.0)
    assert_true(r["temperature"][-1] > r["temperature"][0],
                f"T rose: {r['temperature'][-1]:.2f} > {r['temperature'][0]:.2f} K")
    assert_true(r["temperature"][-1] < m.T_carb + 100.0,
                f"T bounded near setpoint: {r['temperature'][-1]:.2f} K")


def test_calciner_endothermic_duty_positive():
    print("\n[Test 8] Calciner duty positive (endothermic regeneration)")
    m, _ = make_model()
    r = m.simulate(5, dt=10.0, duration_s=300.0)
    assert_true(np.all(r["calciner_duty"] > 0), "calciner duty > 0 (endothermic)")
    # Duty must exceed pure reaction enthalpy of regenerated CO2 (also heats solids)
    R = r["capture_rate"][-1] + m.F_makeup
    Q_rxn = (-m.dH_carb) * R
    assert_true(r["calciner_duty"][-1] > Q_rxn,
                f"duty {r['calciner_duty'][-1]/1e3:.1f} kW > rxn {Q_rxn/1e3:.1f} kW (incl. sensible)")


def test_makeup_raises_average_capacity():
    print("\n[Test 9] Higher fresh make-up raises population-average capacity")
    m, _ = make_model()
    X_low = m.average_capacity(0.01)
    X_high = m.average_capacity(0.20)
    assert_true(X_high > X_low, f"X_ave(f=0.20)={X_high:.4f} > X_ave(f=0.01)={X_low:.4f}")
    assert_true(m.X_r <= X_low <= m.X_max1, "average within [X_r, X_max1]")


def test_predict_interface():
    print("\n[Test 10] ComponentModel predict() interface + array consistency")
    _, cm = make_model()
    r = cm.predict({"cycle_number": 10, "dt": 10.0, "duration_s": 100.0})
    for key in ["t", "conversion", "capacity", "capture_rate", "co2_out",
                "capture_efficiency", "temperature", "calciner_duty",
                "carbon_balance_residual"]:
        assert_true(key in r, f"Key '{key}' in output")
    n = len(r["t"])
    assert_true(all(len(r[k]) == n for k in r), "All series same length")
    info = cm.get_info()
    assert_true(info["component_id"] == "EC204", "component_id == EC204")


def test_benchmark():
    print("\n[Test 11] Benchmark: 300s coupled ODE at dt=1.0")
    m, _ = make_model()
    t0 = time.perf_counter()
    m.simulate(10, dt=1.0, duration_s=300.0)
    elapsed = time.perf_counter() - t0
    print(f"  300s coupled simulation in {elapsed*1000:.1f} ms")
    assert_true(elapsed < 5.0, "Completes in < 5 s")


if __name__ == "__main__":
    tests = [
        test_capacity_decays_with_cycles,
        test_capacity_residual_limit,
        test_carbon_conservation,
        test_capture_rate_bounded,
        test_conversion_monotone_and_capped,
        test_more_cycles_less_capture,
        test_exothermic_heats_bed,
        test_calciner_endothermic_duty_positive,
        test_makeup_raises_average_capacity,
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

    print(f"\n{'='*62}")
    print(f"EC204 Calcium Looping F2a -- Results: {passed} passed, {failed} failed")
    print(f"{'='*62}")
    sys.exit(0 if failed == 0 else 1)
