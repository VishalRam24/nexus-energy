"""EC064 — Small Wind Turbine — F1a — Test Suite (no pytest)"""
import sys, time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel

PASS = 0
FAIL = 0


def assert_true(condition, msg):
    global PASS, FAIL
    if condition:
        print(f"  ✓ {msg}")
        PASS += 1
    else:
        print(f"  ✗ {msg}")
        FAIL += 1


def test_physics_sanity():
    """P=0 below cut-in and above cut-out; P=P_rated at v_rated."""
    m = ComponentModel()
    r_low = m.predict({"v": 1.0})   # below cut-in
    r_high = m.predict({"v": 30.0}) # above cut-out
    r_rated = m.predict({"v": 15.0}) # above v_rated, capped
    assert_true(float(r_low["P_kW"]) == 0.0, "P=0 below cut-in speed")
    assert_true(float(r_high["P_kW"]) == 0.0, "P=0 above cut-out speed")
    assert_true(abs(float(r_rated["P_kW"]) - 10.0) < 0.01, f"P=10kW at rated+ (got {float(r_rated['P_kW']):.2f})")


def test_limits():
    """P <= P_rated; P >= 0."""
    m = ComponentModel()
    vs = np.linspace(0, 30, 100)
    r = m.predict({"v": vs})
    ps = r["P_kW"]
    assert_true(float(np.max(ps)) <= 10.001, "P <= P_rated always")
    assert_true(float(np.min(ps)) >= 0.0, "P >= 0 always")


def test_monotonicity():
    """P increases with v between cut-in and rated."""
    m = ComponentModel()
    vs = np.arange(3.0, 12.0, 0.5)
    ps = [float(m.predict({"v": v})["P_kW"]) for v in vs]
    assert_true(all(ps[i] <= ps[i+1] for i in range(len(ps)-1)), "P increases from cut-in to rated")


def test_predict_interface():
    """predict() returns required keys; get_info() correct."""
    m = ComponentModel()
    r = m.predict({"v": 8.0})
    keys = ["P_kW", "Cp_actual", "P_aero_kW"]
    assert_true(all(k in r for k in keys), "All output keys present")
    info = m.get_info()
    assert_true(info["ec_id"] == "EC064", "ec_id == EC064")


def test_benchmark():
    """1000 predictions in < 1 s."""
    m = ComponentModel()
    N = 1000
    t0 = time.perf_counter()
    m.predict({"v": np.random.uniform(0, 30, N)})
    elapsed = time.perf_counter() - t0
    assert_true(elapsed < 1.0, f"Benchmark: {N} predictions in {elapsed*1000:.1f} ms")


if __name__ == "__main__":
    print("EC064 Small Wind Turbine — F1a Tests")
    test_physics_sanity()
    test_limits()
    test_monotonicity()
    test_predict_interface()
    test_benchmark()
    print(f"\nResults: {PASS} passed, {FAIL} failed")
    sys.exit(0 if FAIL == 0 else 1)
