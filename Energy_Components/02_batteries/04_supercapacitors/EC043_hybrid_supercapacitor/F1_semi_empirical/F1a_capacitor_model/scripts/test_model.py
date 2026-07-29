"""EC043 — Hybrid Supercapacitor — F1a — Test Suite (no pytest)"""
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
    """V = Q/C at I=0; V_oc at 50% SOC = V_max/2."""
    m = ComponentModel()
    r = m.predict({"Q": 380.0, "I": 0.0})  # half charge, no current
    V_expected = 380.0 / 200.0  # 1.9 V
    assert_true(abs(float(r["V_oc"]) - V_expected) < 1e-6, f"V_oc at Q=380C → {V_expected:.3f} V")
    assert_true(abs(float(r["V_terminal"]) - V_expected) < 1e-6, "V_terminal = V_oc when I=0")


def test_limits():
    """V clamped to [0, V_max]; SOC in [0,1]."""
    m = ComponentModel()
    r_full = m.predict({"Q": 760.0, "I": 0.0})
    r_empty = m.predict({"Q": 0.0, "I": 0.0})
    assert_true(abs(float(r_full["SOC"]) - 1.0) < 1e-9, "SOC=1 at full charge")
    assert_true(abs(float(r_empty["SOC"])) < 1e-9, "SOC=0 at empty")
    assert_true(float(r_full["V_oc"]) <= 3.8 + 1e-9, "V_oc <= V_max")


def test_monotonicity():
    """Higher Q → higher V_oc (monotonic)."""
    m = ComponentModel()
    Qs = [0, 100, 200, 380, 600, 760]
    vocs = [float(m.predict({"Q": q, "I": 0.0})["V_oc"]) for q in Qs]
    assert_true(all(vocs[i] <= vocs[i+1] for i in range(len(vocs)-1)), "V_oc monotonically increases with Q")


def test_predict_interface():
    """predict() returns required keys with correct types."""
    m = ComponentModel()
    r = m.predict({"Q": 400.0, "I": 50.0})
    keys = ["V_terminal", "V_oc", "SOC", "P_W", "E_Wh", "V_drop_esr"]
    assert_true(all(k in r for k in keys), "All output keys present")
    info = m.get_info()
    assert_true(info["ec_id"] == "EC043", "ec_id == EC043")
    assert_true(info["fidelity"] == "F1a", "fidelity == F1a")


def test_benchmark():
    """1000 predictions in < 1 s."""
    m = ComponentModel()
    N = 1000
    Qs = np.random.uniform(0, 760, N)
    Is = np.random.uniform(-200, 200, N)
    t0 = time.perf_counter()
    m.predict({"Q": Qs, "I": Is})
    elapsed = time.perf_counter() - t0
    assert_true(elapsed < 1.0, f"Benchmark: {N} predictions in {elapsed*1000:.1f} ms")


if __name__ == "__main__":
    print("EC043 Hybrid Supercapacitor — F1a Tests")
    test_physics_sanity()
    test_limits()
    test_monotonicity()
    test_predict_interface()
    test_benchmark()
    print(f"\nResults: {PASS} passed, {FAIL} failed")
    sys.exit(0 if FAIL == 0 else 1)
