"""EC057 — Stirling Dish CSP — F1a — Test Suite (no pytest)"""
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
    """At DNI=900, theta=0: P = 0.85*0.35*900*88 = 23.6 kW."""
    m = ComponentModel()
    r = m.predict({"DNI": 900.0, "theta": 0.0})
    expected = 0.85 * 0.35 * 900.0 * 88.0 / 1000.0
    assert_true(abs(float(r["P_kW"]) - expected) < 0.1, f"P≈{expected:.1f} kW (got {float(r['P_kW']):.2f})")
    assert_true(float(r["P_kW"]) < float(r["Q_focal_kW"]), "P_electric < Q_focal (energy chain)")


def test_limits():
    """P=0 at DNI=0."""
    m = ComponentModel()
    r0 = m.predict({"DNI": 0.0})
    assert_true(float(r0["P_kW"]) == 0.0, "P=0 at DNI=0")


def test_monotonicity():
    """P increases with DNI; decreases with theta."""
    m = ComponentModel()
    dnis = [200, 400, 600, 800, 1000]
    ps = [float(m.predict({"DNI": d})["P_kW"]) for d in dnis]
    assert_true(all(ps[i] < ps[i+1] for i in range(len(ps)-1)), "P increases with DNI")
    thetas = [0, 10, 20, 30, 40]
    ps_t = [float(m.predict({"DNI": 800.0, "theta": t})["P_kW"]) for t in thetas]
    assert_true(all(ps_t[i] >= ps_t[i+1] for i in range(len(ps_t)-1)), "P decreases with theta")


def test_predict_interface():
    """predict() returns required keys; get_info() correct."""
    m = ComponentModel()
    r = m.predict({"DNI": 800.0})
    keys = ["P_kW", "Q_focal_kW", "eta_optical", "eta_system"]
    assert_true(all(k in r for k in keys), "All output keys present")
    info = m.get_info()
    assert_true(info["ec_id"] == "EC057", "ec_id == EC057")


def test_benchmark():
    """1000 predictions in < 1 s."""
    m = ComponentModel()
    N = 1000
    t0 = time.perf_counter()
    m.predict({"DNI": np.random.uniform(0, 1100, N), "theta": np.random.uniform(0, 40, N)})
    elapsed = time.perf_counter() - t0
    assert_true(elapsed < 1.0, f"Benchmark: {N} predictions in {elapsed*1000:.1f} ms")


if __name__ == "__main__":
    print("EC057 Stirling Dish CSP — F1a Tests")
    test_physics_sanity()
    test_limits()
    test_monotonicity()
    test_predict_interface()
    test_benchmark()
    print(f"\nResults: {PASS} passed, {FAIL} failed")
    sys.exit(0 if FAIL == 0 else 1)
