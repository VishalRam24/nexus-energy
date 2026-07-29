"""EC050 — OPV — F1a — Test Suite (no pytest)"""
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
    """Voc ~ 0.8V; FF > 0.4; Pmp > 0 at STC."""
    m = ComponentModel()
    r = m.predict({"G": 1000.0})
    assert_true(0.6 < float(r["Voc_V"]) < 1.1, f"Voc ~ 0.8V (got {float(r['Voc_V']):.3f})")
    assert_true(float(r["FF"]) > 0.40, f"FF > 0.40 (got {float(r['FF']):.3f})")
    assert_true(float(r["Pmp_W"]) > 0.0, "Pmp > 0 at STC")


def test_limits():
    """Pmp=0 at G=0; FF in (0,1)."""
    m = ComponentModel()
    r0 = m.predict({"G": 0.0})
    assert_true(float(r0["Pmp_W"]) == 0.0, "Pmp=0 at G=0")
    r1 = m.predict({"G": 1000.0})
    assert_true(0.0 < float(r1["FF"]) < 1.0, "FF in (0,1)")


def test_monotonicity():
    """Pmp increases with irradiance G."""
    m = ComponentModel()
    gs = [100, 300, 500, 700, 900, 1000]
    ps = [float(m.predict({"G": g})["Pmp_W"]) for g in gs]
    assert_true(all(ps[i] < ps[i+1] for i in range(len(ps)-1)), "Pmp monotonically increases with G")


def test_predict_interface():
    """predict() returns required keys; get_info() correct."""
    m = ComponentModel()
    r = m.predict({"G": 800.0})
    keys = ["Voc_V", "Isc_A", "Vmp_V", "Imp_A", "Pmp_W", "FF", "eta"]
    assert_true(all(k in r for k in keys), "All output keys present")
    info = m.get_info()
    assert_true(info["ec_id"] == "EC050", "ec_id == EC050")


def test_benchmark():
    """10 predictions (each solves IV curve) in < 5 s."""
    m = ComponentModel()
    N = 10
    gs = np.random.uniform(100, 1100, N)
    t0 = time.perf_counter()
    m.predict({"G": gs})
    elapsed = time.perf_counter() - t0
    assert_true(elapsed < 5.0, f"Benchmark: {N} predictions in {elapsed*1000:.1f} ms")


if __name__ == "__main__":
    print("EC050 OPV — F1a Tests")
    test_physics_sanity()
    test_limits()
    test_monotonicity()
    test_predict_interface()
    test_benchmark()
    print(f"\nResults: {PASS} passed, {FAIL} failed")
    sys.exit(0 if FAIL == 0 else 1)
