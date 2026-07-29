"""
EC099 — Stirling Engine — F1a — Test Suite
Physics sanity checks, edge cases, and benchmark.
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
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
    return ComponentModel()


def test_predict_returns_dict():
    print("\n[Test 1] predict() returns expected keys")
    m = make_model()
    r = m.predict({"PLR": 0.8})
    for key in ["efficiency", "power_output", "heat_input",
                "heat_rejected", "carnot_efficiency"]:
        assert_true(key in r, f"Key '{key}' present")


def test_get_info():
    print("\n[Test 2] get_info() returns correct metadata")
    m = make_model()
    info = m.get_info()
    assert_true(info["ec_id"] == "EC099", "EC ID is EC099")
    assert_true(info["fidelity"] == "F1a", "Fidelity is F1a")


def test_rated_efficiency_realistic():
    print("\n[Test 3] Rated efficiency in realistic Stirling range")
    m = make_model()
    r = m.predict({"PLR": 1.0})
    eta = float(r["efficiency"])
    # f_carnot=0.5, eta_carnot(650C/50C) ~ 0.65 -> ~0.32
    assert_true(0.20 <= eta <= 0.45, f"eta={eta:.4f} in [0.20, 0.45]")


def test_efficiency_decreases_at_part_load():
    print("\n[Test 4] Efficiency decreases at part-load")
    m = make_model()
    plrs = np.array([0.3, 0.5, 0.7, 0.9, 1.0])
    r = m.predict({"PLR": plrs})
    eta = np.asarray(r["efficiency"])
    assert_true(np.all(np.diff(eta) >= -1e-12),
                f"Efficiency monotone non-decreasing in PLR: {eta}")


def test_carnot_limit():
    print("\n[Test 5] Efficiency never exceeds Carnot limit")
    m = make_model()
    plrs = np.linspace(0.2, 1.0, 50)
    r = m.predict({"PLR": plrs})
    carnot = float(r["carnot_efficiency"])
    assert_true(np.all(np.asarray(r["efficiency"]) <= carnot + 1e-10),
                f"All eta <= Carnot ({carnot:.4f})")


def test_zero_load():
    print("\n[Test 6] Zero load gives zero output")
    m = make_model()
    r = m.predict({"PLR": 0.0})
    assert_true(float(r["power_output"]) == 0.0, "Power = 0 at PLR=0")
    assert_true(float(r["efficiency"]) == 0.0, "Efficiency = 0 at PLR=0")


def test_below_minimum_load():
    print("\n[Test 7] Below minimum PLR gives zero efficiency")
    m = make_model()
    r = m.predict({"PLR": 0.1})
    assert_true(float(r["efficiency"]) == 0.0, "Efficiency = 0 below PLR_min")


def test_q_in_greater_than_p_out():
    print("\n[Test 8] Heat input > power output (2nd law)")
    m = make_model()
    r = m.predict({"PLR": 0.8})
    P = float(r["power_output"])
    Q = float(r["heat_input"])
    assert_true(Q > P, f"Q_in={Q/1e3:.2f} kW > P_out={P/1e3:.2f} kW")


def test_heat_rejected_positive():
    print("\n[Test 9] Heat rejected > 0 above min load")
    m = make_model()
    r = m.predict({"PLR": 0.8})
    Q_rej = float(r["heat_rejected"])
    assert_true(Q_rej > 0.0, f"Q_rejected = {Q_rej/1e3:.2f} kW > 0")


def test_hot_side_temperature_effect():
    print("\n[Test 10] Higher T_h -> higher Carnot")
    m = make_model()
    r1 = m.predict({"PLR": 1.0, "T_h": 500.0})
    r2 = m.predict({"PLR": 1.0, "T_h": 750.0})
    assert_true(float(r2["carnot_efficiency"]) > float(r1["carnot_efficiency"]),
                "Higher T_h -> higher Carnot")
    assert_true(float(r2["efficiency"]) >= float(r1["efficiency"]),
                "Higher T_h -> higher cycle eta")


def test_cold_side_temperature_effect():
    print("\n[Test 11] Lower T_c -> higher Carnot")
    m = make_model()
    r1 = m.predict({"PLR": 1.0, "T_c": 60.0})
    r2 = m.predict({"PLR": 1.0, "T_c": 10.0})
    assert_true(float(r2["carnot_efficiency"]) > float(r1["carnot_efficiency"]),
                "Lower T_c -> higher Carnot")


def test_array_inputs():
    print("\n[Test 12] Array inputs work")
    m = make_model()
    plrs = np.array([0.3, 0.5, 0.8, 1.0])
    r = m.predict({"PLR": plrs})
    assert_true(np.asarray(r["efficiency"]).shape == (4,),
                f"Output shape = {np.asarray(r['efficiency']).shape}")


def test_benchmark():
    print("\n[Test 13] Benchmark: 1000 predictions")
    m = make_model()
    plrs = np.random.uniform(0.2, 1.0, 1000)
    start = time.perf_counter()
    m.predict({"PLR": plrs})
    elapsed = time.perf_counter() - start
    print(f"         1000 predictions in {elapsed*1000:.2f} ms")
    assert_true(elapsed < 1.0, f"Completed in {elapsed:.3f}s < 1s")


if __name__ == "__main__":
    tests = [
        test_predict_returns_dict,
        test_get_info,
        test_rated_efficiency_realistic,
        test_efficiency_decreases_at_part_load,
        test_carnot_limit,
        test_zero_load,
        test_below_minimum_load,
        test_q_in_greater_than_p_out,
        test_heat_rejected_positive,
        test_hot_side_temperature_effect,
        test_cold_side_temperature_effect,
        test_array_inputs,
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
    print(f"EC099 Stirling Engine F1a — Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    sys.exit(0 if failed == 0 else 1)
