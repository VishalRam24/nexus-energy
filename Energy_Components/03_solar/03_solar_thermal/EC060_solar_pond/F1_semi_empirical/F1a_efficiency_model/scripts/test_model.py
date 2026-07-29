"""EC060 — Solar Pond — F1a — Test Suite (no pytest)"""
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
    """At G=700, Tm=80, Ta=20: Q = 10000*(0.05*700 - 0.5*60 - 0.001*3600)."""
    m = ComponentModel()
    r = m.predict({"G": 700.0, "Tm": 80.0, "Ta": 20.0})
    expected_kW = 10000.0 * (0.05 * 700.0 - 0.5 * 60.0 - 0.001 * 3600.0) / 1000.0
    expected_kW = max(expected_kW, 0.0)
    assert_true(abs(float(r["Q_kW"]) - expected_kW) < 0.1, f"Q≈{expected_kW:.1f} kW (got {float(r['Q_kW']):.1f})")


def test_limits():
    """Q=0 at G=0 or when losses exceed gains; eta in [0,1]."""
    m = ComponentModel()
    r0 = m.predict({"G": 0.0, "Tm": 80.0, "Ta": 20.0})
    assert_true(float(r0["Q_kW"]) == 0.0, "Q=0 at G=0")
    r1 = m.predict({"G": 500.0, "Tm": 30.0, "Ta": 25.0})
    assert_true(0.0 <= float(r1["eta"]) <= 1.0, "eta in [0,1]")


def test_monotonicity():
    """Q increases with G (at fixed Tm, Ta)."""
    m = ComponentModel()
    gs = [100, 300, 500, 700, 900]
    qs = [float(m.predict({"G": g, "Tm": 60.0, "Ta": 20.0})["Q_kW"]) for g in gs]
    assert_true(all(qs[i] <= qs[i+1] for i in range(len(qs)-1)), "Q increases with G")


def test_predict_interface():
    """predict() returns required keys; get_info() correct."""
    m = ComponentModel()
    r = m.predict({"G": 600.0, "Tm": 70.0, "Ta": 25.0})
    keys = ["Q_kW", "eta", "dT"]
    assert_true(all(k in r for k in keys), "All output keys present")
    info = m.get_info()
    assert_true(info["ec_id"] == "EC060", "ec_id == EC060")


def test_benchmark():
    """1000 predictions in < 1 s."""
    m = ComponentModel()
    N = 1000
    t0 = time.perf_counter()
    m.predict({"G": np.random.uniform(0, 1100, N), "Tm": np.random.uniform(50, 90, N), "Ta": np.random.uniform(10, 35, N)})
    elapsed = time.perf_counter() - t0
    assert_true(elapsed < 1.0, f"Benchmark: {N} predictions in {elapsed*1000:.1f} ms")


if __name__ == "__main__":
    print("EC060 Solar Pond — F1a Tests")
    test_physics_sanity()
    test_limits()
    test_monotonicity()
    test_predict_interface()
    test_benchmark()
    print(f"\nResults: {PASS} passed, {FAIL} failed")
    sys.exit(0 if FAIL == 0 else 1)
