"""EC067 — AWE — F1a — Test Suite (no pytest)"""
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
    """P=0 below cut-in and above cut-out; P capped at P_rated."""
    m = ComponentModel()
    r_low = m.predict({"v": 2.0})
    r_high = m.predict({"v": 35.0})
    r_rated = m.predict({"v": 20.0})
    assert_true(float(r_low["P_kW"]) == 0.0, "P=0 below cut-in (v=2)")
    assert_true(float(r_high["P_kW"]) == 0.0, "P=0 above cut-out (v=35)")
    assert_true(abs(float(r_rated["P_kW"]) - 100.0) < 0.01, f"P=100kW at rated+ (got {float(r_rated['P_kW']):.2f})")


def test_limits():
    """P <= P_rated; capacity_factor in [0,1]."""
    m = ComponentModel()
    vs = np.linspace(0, 35, 100)
    r = m.predict({"v": vs})
    assert_true(float(np.max(r["P_kW"])) <= 100.001, "P <= P_rated")
    assert_true(float(np.min(r["capacity_factor"])) >= 0.0, "CF >= 0")
    assert_true(float(np.max(r["capacity_factor"])) <= 1.001, "CF <= 1")


def test_monotonicity():
    """P increases from cut-in to rated."""
    m = ComponentModel()
    vs = np.arange(4.0, 15.0, 0.5)
    ps = [float(m.predict({"v": v})["P_kW"]) for v in vs]
    assert_true(all(ps[i] <= ps[i+1] for i in range(len(ps)-1)), "P increases from cut-in to rated")


def test_predict_interface():
    """predict() returns required keys; get_info() correct."""
    m = ComponentModel()
    r = m.predict({"v": 10.0})
    keys = ["P_kW", "P_rated_kW", "capacity_factor"]
    assert_true(all(k in r for k in keys), "All output keys present")
    info = m.get_info()
    assert_true(info["ec_id"] == "EC067", "ec_id == EC067")


def test_benchmark():
    """1000 predictions in < 1 s."""
    m = ComponentModel()
    N = 1000
    t0 = time.perf_counter()
    m.predict({"v": np.random.uniform(0, 35, N)})
    elapsed = time.perf_counter() - t0
    assert_true(elapsed < 1.0, f"Benchmark: {N} predictions in {elapsed*1000:.1f} ms")


if __name__ == "__main__":
    print("EC067 AWE — F1a Tests")
    test_physics_sanity()
    test_limits()
    test_monotonicity()
    test_predict_interface()
    test_benchmark()
    print(f"\nResults: {PASS} passed, {FAIL} failed")
    sys.exit(0 if FAIL == 0 else 1)
