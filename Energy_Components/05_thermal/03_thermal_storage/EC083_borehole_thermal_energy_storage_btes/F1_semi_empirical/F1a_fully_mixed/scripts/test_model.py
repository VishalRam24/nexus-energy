"""EC083 — BTES — F1a — Test Suite (no pytest)"""
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
    """At T_store=10 (=T_initial): E=0, SOC=0; Q_loss > 0 when T>T_amb."""
    m = ComponentModel()
    r0 = m.predict({"T_store": 10.0, "T_amb": 10.0})
    assert_true(float(r0["E_stored_MWh"]) == 0.0, "E=0 at T_initial")
    assert_true(float(r0["SOC"]) == 0.0, "SOC=0 at T_initial")
    r1 = m.predict({"T_store": 60.0, "T_amb": 10.0})
    # Q_loss = 50 * (60-10) = 2500 W = 2.5 kW
    assert_true(abs(float(r1["Q_loss_kW"]) - 2.5) < 0.01, f"Q_loss=2.5kW (got {float(r1['Q_loss_kW']):.2f})")


def test_limits():
    """SOC in [0,1]; E_stored >= 0."""
    m = ComponentModel()
    r = m.predict({"T_store": 90.0, "T_amb": 10.0})
    assert_true(0.0 <= float(r["SOC"]) <= 1.0, "SOC in [0,1]")
    assert_true(float(r["E_stored_MWh"]) >= 0.0, "E_stored >= 0")


def test_monotonicity():
    """E_stored increases with T_store."""
    m = ComponentModel()
    temps = [10, 20, 40, 60, 80, 90]
    es = [float(m.predict({"T_store": t})["E_stored_MWh"]) for t in temps]
    assert_true(all(es[i] <= es[i+1] for i in range(len(es)-1)), "E_stored increases with T_store")


def test_predict_interface():
    """predict() returns required keys; get_info() correct."""
    m = ComponentModel()
    r = m.predict({"T_store": 50.0})
    keys = ["E_stored_MWh", "Q_loss_kW", "SOC", "Q_net_kW", "E_max_MWh"]
    assert_true(all(k in r for k in keys), "All output keys present")
    info = m.get_info()
    assert_true(info["ec_id"] == "EC083", "ec_id == EC083")


def test_benchmark():
    """1000 predictions in < 1 s."""
    m = ComponentModel()
    N = 1000
    t0 = time.perf_counter()
    m.predict({"T_store": np.random.uniform(10, 90, N), "T_amb": np.random.uniform(5, 15, N)})
    elapsed = time.perf_counter() - t0
    assert_true(elapsed < 1.0, f"Benchmark: {N} predictions in {elapsed*1000:.1f} ms")


if __name__ == "__main__":
    print("EC083 BTES — F1a Tests")
    test_physics_sanity()
    test_limits()
    test_monotonicity()
    test_predict_interface()
    test_benchmark()
    print(f"\nResults: {PASS} passed, {FAIL} failed")
    sys.exit(0 if FAIL == 0 else 1)
