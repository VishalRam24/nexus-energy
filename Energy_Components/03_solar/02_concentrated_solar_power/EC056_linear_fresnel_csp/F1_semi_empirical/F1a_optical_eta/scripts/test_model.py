"""EC056 — Linear Fresnel CSP — F1a — Test Suite (no pytest)"""
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
    """Q at theta=0 = eta_opt * eta_th * DNI * A."""
    m = ComponentModel()
    r = m.predict({"DNI": 800.0, "theta": 0.0})
    # eta_opt=0.60, eta_th=0.70, DNI=800, A=5000 → Q=0.60*0.70*800*5000=1.68MW
    expected_Q = 0.60 * 0.70 * 800.0 * 5000.0 / 1e6
    assert_true(abs(float(r["Q_MW"]) - expected_Q) < 0.01, f"Q={expected_Q:.2f} MW (got {float(r['Q_MW']):.2f})")
    assert_true(float(r["P_MW"]) < float(r["Q_MW"]), "P_electric < Q_thermal")


def test_limits():
    """Q=0 at DNI=0; Q>=0 always."""
    m = ComponentModel()
    r0 = m.predict({"DNI": 0.0})
    assert_true(float(r0["Q_MW"]) == 0.0, "Q=0 at DNI=0")
    r1 = m.predict({"DNI": 900.0, "theta": 70.0})  # high angle → low output
    assert_true(float(r1["Q_MW"]) >= 0.0, "Q>=0 at high incidence angle")


def test_monotonicity():
    """Q increases with DNI; decreases with theta."""
    m = ComponentModel()
    dnis = [200, 400, 600, 800, 1000]
    qs = [float(m.predict({"DNI": d})["Q_MW"]) for d in dnis]
    assert_true(all(qs[i] < qs[i+1] for i in range(len(qs)-1)), "Q monotonically increases with DNI")
    thetas = [0, 15, 30, 45, 60]
    qs_t = [float(m.predict({"DNI": 800.0, "theta": t})["Q_MW"]) for t in thetas]
    assert_true(all(qs_t[i] >= qs_t[i+1] for i in range(len(qs_t)-1)), "Q decreases with theta")


def test_predict_interface():
    """predict() returns required keys; get_info() correct."""
    m = ComponentModel()
    r = m.predict({"DNI": 700.0})
    keys = ["Q_MW", "P_MW", "eta_optical", "eta_system"]
    assert_true(all(k in r for k in keys), "All output keys present")
    info = m.get_info()
    assert_true(info["ec_id"] == "EC056", "ec_id == EC056")


def test_benchmark():
    """1000 predictions in < 1 s."""
    m = ComponentModel()
    N = 1000
    t0 = time.perf_counter()
    m.predict({"DNI": np.random.uniform(0, 1100, N), "theta": np.random.uniform(0, 60, N)})
    elapsed = time.perf_counter() - t0
    assert_true(elapsed < 1.0, f"Benchmark: {N} predictions in {elapsed*1000:.1f} ms")


if __name__ == "__main__":
    print("EC056 Linear Fresnel CSP — F1a Tests")
    test_physics_sanity()
    test_limits()
    test_monotonicity()
    test_predict_interface()
    test_benchmark()
    print(f"\nResults: {PASS} passed, {FAIL} failed")
    sys.exit(0 if FAIL == 0 else 1)
