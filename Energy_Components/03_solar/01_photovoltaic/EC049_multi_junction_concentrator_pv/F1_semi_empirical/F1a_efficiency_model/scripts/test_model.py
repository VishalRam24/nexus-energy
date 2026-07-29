"""EC049 — Multi-Junction CPV — F1a — Test Suite (no pytest)"""
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
    """P=0 at DNI=0; P>0 at standard conditions."""
    m = ComponentModel()
    r0 = m.predict({"DNI": 0.0})
    assert_true(float(r0["P_W"]) == 0.0, "P=0 when DNI=0")
    r1 = m.predict({"DNI": 900.0, "T_cell": 25.0})
    # eta=0.40, C=500, A=1e-4, DNI=900 → P=900*500*1e-4*0.40=18 W
    assert_true(abs(float(r1["P_W"]) - 18.0) < 0.1, f"P≈18W at DNI=900, T=25°C (got {float(r1['P_W']):.2f})")


def test_limits():
    """eta_eff in (0, 1); P >= 0."""
    m = ComponentModel()
    r = m.predict({"DNI": 1000.0, "T_cell": 80.0})
    assert_true(0.0 < float(r["eta_eff"]) <= 1.0, "eta_eff in (0,1]")
    assert_true(float(r["P_W"]) >= 0.0, "P >= 0")


def test_monotonicity():
    """P increases with DNI."""
    m = ComponentModel()
    dnis = [100, 300, 500, 700, 900]
    ps = [float(m.predict({"DNI": d})["P_W"]) for d in dnis]
    assert_true(all(ps[i] < ps[i+1] for i in range(len(ps)-1)), "P monotonically increases with DNI")


def test_predict_interface():
    """predict() returns required keys; get_info() correct."""
    m = ComponentModel()
    r = m.predict({"DNI": 800.0, "T_cell": 30.0, "theta_incidence": 10.0})
    keys = ["P_W", "eta_eff", "irr_concentrated", "P_max_ref"]
    assert_true(all(k in r for k in keys), "All output keys present")
    info = m.get_info()
    assert_true(info["ec_id"] == "EC049", "ec_id == EC049")


def test_benchmark():
    """1000 predictions in < 1 s."""
    m = ComponentModel()
    N = 1000
    t0 = time.perf_counter()
    m.predict({"DNI": np.random.uniform(0, 1100, N), "T_cell": np.random.uniform(20, 80, N)})
    elapsed = time.perf_counter() - t0
    assert_true(elapsed < 1.0, f"Benchmark: {N} predictions in {elapsed*1000:.1f} ms")


if __name__ == "__main__":
    print("EC049 Multi-Junction CPV — F1a Tests")
    test_physics_sanity()
    test_limits()
    test_monotonicity()
    test_predict_interface()
    test_benchmark()
    print(f"\nResults: {PASS} passed, {FAIL} failed")
    sys.exit(0 if FAIL == 0 else 1)
