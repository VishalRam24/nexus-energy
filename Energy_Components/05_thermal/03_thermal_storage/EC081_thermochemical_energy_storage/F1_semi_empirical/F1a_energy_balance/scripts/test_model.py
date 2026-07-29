"""EC081 — Thermochemical Storage — F1a — Test Suite (no pytest)"""
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
    """At x=1: E_max = 1000*400e3 J = 111.1 kWh; E_usable = 77.8 kWh."""
    m = ComponentModel()
    r = m.predict({"x": 1.0})
    E_max_expected = 1000.0 * 400e3 / 3.6e6  # kWh
    E_usable_expected = E_max_expected * 0.70
    assert_true(abs(float(r["E_stored_kWh"]) - E_max_expected) < 0.1,
                f"E_max={E_max_expected:.1f} kWh (got {float(r['E_stored_kWh']):.1f})")
    assert_true(abs(float(r["E_usable_kWh"]) - E_usable_expected) < 0.1,
                f"E_usable={E_usable_expected:.1f} kWh (got {float(r['E_usable_kWh']):.1f})")


def test_limits():
    """x=0 → E_stored=0; x=1 → SOC=1."""
    m = ComponentModel()
    r0 = m.predict({"x": 0.0})
    assert_true(float(r0["E_stored_kWh"]) == 0.0, "E_stored=0 at x=0")
    assert_true(float(r0["SOC"]) == 0.0, "SOC=0 at x=0")
    r1 = m.predict({"x": 1.0})
    assert_true(abs(float(r1["SOC"]) - 1.0) < 1e-9, "SOC=1 at x=1")


def test_monotonicity():
    """E_stored increases with x."""
    m = ComponentModel()
    xs = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    es = [float(m.predict({"x": xi})["E_stored_kWh"]) for xi in xs]
    assert_true(all(es[i] <= es[i+1] for i in range(len(es)-1)), "E_stored monotonically increases with x")


def test_predict_interface():
    """predict() returns required keys; get_info() correct."""
    m = ComponentModel()
    r = m.predict({"x": 0.5})
    keys = ["E_stored_kWh", "E_usable_kWh", "SOC", "E_max_kWh", "eta_rt", "P_1h_kW"]
    assert_true(all(k in r for k in keys), "All output keys present")
    info = m.get_info()
    assert_true(info["ec_id"] == "EC081", "ec_id == EC081")


def test_benchmark():
    """1000 predictions in < 1 s."""
    m = ComponentModel()
    N = 1000
    t0 = time.perf_counter()
    m.predict({"x": np.random.uniform(0, 1, N)})
    elapsed = time.perf_counter() - t0
    assert_true(elapsed < 1.0, f"Benchmark: {N} predictions in {elapsed*1000:.1f} ms")


if __name__ == "__main__":
    print("EC081 Thermochemical Storage — F1a Tests")
    test_physics_sanity()
    test_limits()
    test_monotonicity()
    test_predict_interface()
    test_benchmark()
    print(f"\nResults: {PASS} passed, {FAIL} failed")
    sys.exit(0 if FAIL == 0 else 1)
