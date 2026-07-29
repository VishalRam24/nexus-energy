"""EC061 — Unglazed Collector — F1a — Test Suite (no pytest)"""
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
    """At G=700, Tm=28, Ta=20: Q = 4*(0.90*700 - 15*8) = 4*(630-120)=2040 W."""
    m = ComponentModel()
    r = m.predict({"G": 700.0, "Tm": 28.0, "Ta": 20.0})
    expected = 4.0 * (0.90 * 700.0 - 15.0 * 8.0)
    assert_true(abs(float(r["Q_W"]) - expected) < 1.0, f"Q={expected:.0f} W (got {float(r['Q_W']):.1f})")


def test_limits():
    """Q=0 at G=0; also Q=0 when Tm close to Ta and G small."""
    m = ComponentModel()
    r0 = m.predict({"G": 0.0, "Tm": 28.0, "Ta": 20.0})
    assert_true(float(r0["Q_W"]) == 0.0, "Q=0 at G=0")
    r1 = m.predict({"G": 100.0, "Tm": 40.0, "Ta": 10.0})  # large dT → Q may be 0
    assert_true(float(r1["Q_W"]) >= 0.0, "Q >= 0 (clipped at zero)")


def test_monotonicity():
    """Q increases with G; decreases with Tm-Ta."""
    m = ComponentModel()
    gs = [200, 400, 600, 800]
    qs = [float(m.predict({"G": g, "Tm": 25.0, "Ta": 20.0})["Q_W"]) for g in gs]
    assert_true(all(qs[i] < qs[i+1] for i in range(len(qs)-1)), "Q increases with G")
    Tms = [21, 25, 30, 35]
    qs_t = [float(m.predict({"G": 700.0, "Tm": t, "Ta": 20.0})["Q_W"]) for t in Tms]
    assert_true(all(qs_t[i] >= qs_t[i+1] for i in range(len(qs_t)-1)), "Q decreases as Tm increases")


def test_predict_interface():
    """predict() returns required keys; get_info() correct."""
    m = ComponentModel()
    r = m.predict({"G": 600.0})
    keys = ["Q_W", "eta", "dT"]
    assert_true(all(k in r for k in keys), "All output keys present")
    info = m.get_info()
    assert_true(info["ec_id"] == "EC061", "ec_id == EC061")


def test_benchmark():
    """1000 predictions in < 1 s."""
    m = ComponentModel()
    N = 1000
    t0 = time.perf_counter()
    m.predict({"G": np.random.uniform(0, 1100, N), "Tm": np.random.uniform(20, 35, N), "Ta": np.random.uniform(10, 25, N)})
    elapsed = time.perf_counter() - t0
    assert_true(elapsed < 1.0, f"Benchmark: {N} predictions in {elapsed*1000:.1f} ms")


if __name__ == "__main__":
    print("EC061 Unglazed Collector — F1a Tests")
    test_physics_sanity()
    test_limits()
    test_monotonicity()
    test_predict_interface()
    test_benchmark()
    print(f"\nResults: {PASS} passed, {FAIL} failed")
    sys.exit(0 if FAIL == 0 else 1)
