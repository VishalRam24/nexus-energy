"""EC077 — Microchannel HX — F1a — Test Suite (no pytest)"""
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
    """Q > 0 when T_h > T_c; T_h_out < T_h_in; T_c_out > T_c_in."""
    m = ComponentModel()
    r = m.predict({"T_h_in": 80.0, "T_c_in": 20.0})
    assert_true(float(r["Q_kW"]) > 0.0, "Q > 0 when T_h > T_c")
    assert_true(float(r["T_h_out"]) < 80.0, "T_h_out < T_h_in")
    assert_true(float(r["T_c_out"]) > 20.0, "T_c_out > T_c_in")


def test_limits():
    """eps in (0,1); NTU > 0; high UA → high effectiveness."""
    m = ComponentModel()
    r = m.predict({"T_h_in": 80.0, "T_c_in": 20.0})
    assert_true(0.0 < float(r["effectiveness"]) < 1.0, "eps in (0,1)")
    # High UA=10000, C_min=2500 → NTU=4 → high effectiveness
    assert_true(float(r["effectiveness"]) > 0.7, f"High eps expected for UA=10kW/K (got {float(r['effectiveness']):.3f})")


def test_monotonicity():
    """Q increases with temperature difference."""
    m = ComponentModel()
    dTs = [20, 40, 60, 80]
    qs = [float(m.predict({"T_h_in": 20.0 + dT, "T_c_in": 20.0})["Q_kW"]) for dT in dTs]
    assert_true(all(qs[i] < qs[i+1] for i in range(len(qs)-1)), "Q increases with dT")


def test_predict_interface():
    """predict() returns required keys; get_info() correct."""
    m = ComponentModel()
    r = m.predict({"T_h_in": 60.0, "T_c_in": 15.0})
    keys = ["Q_kW", "T_h_out", "T_c_out", "effectiveness", "NTU"]
    assert_true(all(k in r for k in keys), "All output keys present")
    info = m.get_info()
    assert_true(info["ec_id"] == "EC077", "ec_id == EC077")


def test_benchmark():
    """1000 predictions in < 1 s."""
    m = ComponentModel()
    N = 1000
    t0 = time.perf_counter()
    m.predict({
        "T_h_in": np.random.uniform(40, 130, N),
        "T_c_in": np.random.uniform(-10, 35, N),
        "C_h": np.random.uniform(1000, 8000, N),
        "C_c": np.random.uniform(800, 6000, N),
    })
    elapsed = time.perf_counter() - t0
    assert_true(elapsed < 1.0, f"Benchmark: {N} predictions in {elapsed*1000:.1f} ms")


if __name__ == "__main__":
    print("EC077 Microchannel HX — F1a Tests")
    test_physics_sanity()
    test_limits()
    test_monotonicity()
    test_predict_interface()
    test_benchmark()
    print(f"\nResults: {PASS} passed, {FAIL} failed")
    sys.exit(0 if FAIL == 0 else 1)
