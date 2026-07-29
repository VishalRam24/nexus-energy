"""EC076 — Regenerative HX — F1a — Test Suite (no pytest)"""
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
    """Q > 0 when T_h > T_c; eps < 1 always."""
    m = ComponentModel()
    r = m.predict({"T_h_in": 200.0, "T_c_in": 20.0})
    assert_true(float(r["Q_kW"]) > 0.0, "Q > 0 when T_h > T_c")
    assert_true(0.0 < float(r["effectiveness"]) < 1.0, "eps in (0,1)")
    assert_true(float(r["T_h_out"]) < 200.0, "T_h_out < T_h_in")
    assert_true(float(r["T_c_out"]) > 20.0, "T_c_out > T_c_in")


def test_limits():
    """Balanced flow (C_h=C_c) → eps = NTU/(NTU+1)."""
    m = ComponentModel()
    r = m.predict({"T_h_in": 300.0, "T_c_in": 20.0, "C_h": 2000.0, "C_c": 2000.0})
    NTU = 5000.0 / 2000.0
    eps_expected = NTU / (NTU + 1.0)
    assert_true(abs(float(r["effectiveness"]) - eps_expected) < 1e-4,
                f"Balanced eps={eps_expected:.4f} (got {float(r['effectiveness']):.4f})")


def test_monotonicity():
    """Q increases with temperature difference."""
    m = ComponentModel()
    dTs = [50, 100, 150, 200, 300]
    qs = [float(m.predict({"T_h_in": 20.0 + dT, "T_c_in": 20.0})["Q_kW"]) for dT in dTs]
    assert_true(all(qs[i] < qs[i+1] for i in range(len(qs)-1)), "Q increases with temperature difference")


def test_predict_interface():
    """predict() returns required keys; get_info() correct."""
    m = ComponentModel()
    r = m.predict({"T_h_in": 150.0, "T_c_in": 30.0})
    keys = ["Q_kW", "T_h_out", "T_c_out", "effectiveness", "NTU"]
    assert_true(all(k in r for k in keys), "All output keys present")
    info = m.get_info()
    assert_true(info["ec_id"] == "EC076", "ec_id == EC076")


def test_benchmark():
    """1000 predictions in < 1 s."""
    m = ComponentModel()
    N = 1000
    t0 = time.perf_counter()
    m.predict({
        "T_h_in": np.random.uniform(100, 400, N),
        "T_c_in": np.random.uniform(0, 50, N),
        "C_h": np.random.uniform(1000, 5000, N),
        "C_c": np.random.uniform(1000, 4000, N),
    })
    elapsed = time.perf_counter() - t0
    assert_true(elapsed < 1.0, f"Benchmark: {N} predictions in {elapsed*1000:.1f} ms")


if __name__ == "__main__":
    print("EC076 Regenerative HX — F1a Tests")
    test_physics_sanity()
    test_limits()
    test_monotonicity()
    test_predict_interface()
    test_benchmark()
    print(f"\nResults: {PASS} passed, {FAIL} failed")
    sys.exit(0 if FAIL == 0 else 1)
