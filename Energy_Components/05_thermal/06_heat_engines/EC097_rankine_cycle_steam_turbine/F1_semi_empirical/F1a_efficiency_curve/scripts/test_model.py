"""
EC097 — Rankine Cycle (Steam Turbine) — F1a — Test Suite
Physics sanity checks, edge cases, and performance benchmark.
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
    for key in ["efficiency", "power_output", "heat_input", "steam_flow", "carnot_efficiency"]:
        assert_true(key in r, f"Key '{key}' present in output")


def test_get_info():
    print("\n[Test 2] get_info() returns correct metadata")
    m = make_model()
    info = m.get_info()
    assert_true(info["ec_id"] == "EC097", "EC ID is EC097")
    assert_true(info["fidelity"] == "F1a", "Fidelity is F1a")


def test_rated_efficiency():
    print("\n[Test 3] Rated efficiency at PLR=1.0")
    m = make_model()
    r = m.predict({"PLR": 1.0})
    eta = float(r["efficiency"])
    assert_true(0.35 <= eta <= 0.42, f"eta={eta:.4f} in range [0.35, 0.42]")


def test_efficiency_decreases_at_part_load():
    print("\n[Test 4] Efficiency decreases at part-load")
    m = make_model()
    plrs = np.array([0.3, 0.5, 0.7, 0.9, 1.0])
    r = m.predict({"PLR": plrs})
    eta = r["efficiency"]
    assert_true(np.all(np.diff(eta) > 0), f"Efficiency increases with PLR: {eta}")


def test_carnot_limit():
    print("\n[Test 5] Efficiency never exceeds Carnot limit")
    m = make_model()
    plrs = np.linspace(0.2, 1.0, 50)
    r = m.predict({"PLR": plrs})
    carnot = float(r["carnot_efficiency"])
    assert_true(np.all(r["efficiency"] <= carnot + 1e-10),
                f"All efficiencies <= Carnot ({carnot:.4f})")


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


def test_power_output_scaling():
    print("\n[Test 8] Power output scales linearly with PLR")
    m = make_model()
    P_rated = m.params["turbine"]["P_rated"]["value"]
    r = m.predict({"PLR": 0.5})
    assert_true(abs(float(r["power_output"]) - 0.5 * P_rated) < 1.0,
                f"P at PLR=0.5 = {float(r['power_output'])/1e6:.1f} MW")


def test_heat_input_consistent():
    print("\n[Test 9] Heat input = Power / Efficiency")
    m = make_model()
    r = m.predict({"PLR": 0.8})
    P = float(r["power_output"])
    eta = float(r["efficiency"])
    Q = float(r["heat_input"])
    expected = P / eta
    assert_true(abs(Q - expected) / expected < 1e-6, f"Q_in={Q/1e6:.1f} MW = P/eta={expected/1e6:.1f} MW")


def test_steam_temperature_effect():
    print("\n[Test 10] Higher steam temperature increases Carnot limit")
    m = make_model()
    r1 = m.predict({"PLR": 1.0, "T_steam": 450.0})
    r2 = m.predict({"PLR": 1.0, "T_steam": 600.0})
    assert_true(float(r2["carnot_efficiency"]) > float(r1["carnot_efficiency"]),
                "Higher T_steam -> higher Carnot efficiency")


def test_array_inputs():
    print("\n[Test 11] Array inputs work correctly")
    m = make_model()
    plrs = np.array([0.3, 0.5, 0.8, 1.0])
    r = m.predict({"PLR": plrs})
    assert_true(r["efficiency"].shape == (4,), f"Output shape = {r['efficiency'].shape}")


def test_benchmark():
    print("\n[Test 12] Benchmark: 1000 predictions")
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
        test_rated_efficiency,
        test_efficiency_decreases_at_part_load,
        test_carnot_limit,
        test_zero_load,
        test_below_minimum_load,
        test_power_output_scaling,
        test_heat_input_consistent,
        test_steam_temperature_effect,
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
    print(f"EC097 Rankine Cycle F1a -- Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    sys.exit(0 if failed == 0 else 1)
