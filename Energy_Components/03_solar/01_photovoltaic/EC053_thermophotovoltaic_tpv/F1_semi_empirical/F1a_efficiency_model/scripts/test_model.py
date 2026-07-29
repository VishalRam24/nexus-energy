"""EC053 — TPV — F1a — Test Suite (no pytest)"""
import sys, time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel

PASS = 0
FAIL = 0
SIGMA = 5.67e-8


def assert_true(condition, msg):
    global PASS, FAIL
    if condition:
        print(f"  ✓ {msg}")
        PASS += 1
    else:
        print(f"  ✗ {msg}")
        FAIL += 1


def test_physics_sanity():
    """P proportional to T^4; verify at T=1500K."""
    m = ComponentModel()
    r = m.predict({"T_emitter": 1500.0, "F_view": 0.5})
    # eta=0.25, eps=0.9, sigma*1500^4=28.7e6, A=1e-4, F=0.5 → P_inc=287W/m²*1e-4m²*0.5=0.01435W → P=0.00359W
    expected_irr = 0.9 * SIGMA * 1500.0**4
    assert_true(abs(float(r["irradiance_Wm2"]) - expected_irr) / expected_irr < 0.001,
                f"Irradiance = eps*sigma*T^4 (got {float(r['irradiance_Wm2']):.0f} W/m²)")
    assert_true(float(r["P_W"]) > 0.0, "P > 0 at T=1500K")


def test_limits():
    """P=0 at T=0; P increases with T."""
    m = ComponentModel()
    r0 = m.predict({"T_emitter": 0.0})
    assert_true(float(r0["P_W"]) == 0.0, "P=0 at T_emitter=0")
    r1 = m.predict({"T_emitter": 1000.0})
    r2 = m.predict({"T_emitter": 2000.0})
    assert_true(float(r2["P_W"]) > float(r1["P_W"]), "P increases with T_emitter")


def test_monotonicity():
    """P ~ T^4 (monotonic with T)."""
    m = ComponentModel()
    temps = [800, 1000, 1200, 1500, 1800, 2000]
    ps = [float(m.predict({"T_emitter": t})["P_W"]) for t in temps]
    assert_true(all(ps[i] < ps[i+1] for i in range(len(ps)-1)), "P monotonically increases with T_emitter")


def test_predict_interface():
    """predict() returns required keys; get_info() correct."""
    m = ComponentModel()
    r = m.predict({"T_emitter": 1500.0})
    keys = ["P_W", "P_incident_W", "irradiance_Wm2", "eta_sys"]
    assert_true(all(k in r for k in keys), "All output keys present")
    info = m.get_info()
    assert_true(info["ec_id"] == "EC053", "ec_id == EC053")


def test_benchmark():
    """1000 predictions in < 1 s."""
    m = ComponentModel()
    N = 1000
    t0 = time.perf_counter()
    m.predict({"T_emitter": np.random.uniform(800, 2000, N)})
    elapsed = time.perf_counter() - t0
    assert_true(elapsed < 1.0, f"Benchmark: {N} predictions in {elapsed*1000:.1f} ms")


if __name__ == "__main__":
    print("EC053 TPV — F1a Tests")
    test_physics_sanity()
    test_limits()
    test_monotonicity()
    test_predict_interface()
    test_benchmark()
    print(f"\nResults: {PASS} passed, {FAIL} failed")
    sys.exit(0 if FAIL == 0 else 1)
