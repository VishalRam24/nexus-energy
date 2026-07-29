"""EC170 -- Solid State Transformer (SST) -- F1a -- Test Suite"""
import sys, os, time
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
from predict import ComponentModel

PASS = "\u2713"
FAIL = "\u2717"


def assert_true(condition, msg):
    if condition:
        print(f"  {PASS} {msg}")
    else:
        print(f"  {FAIL} {msg}")
        raise AssertionError(msg)


def main():
    print("EC170 Solid State Transformer F1a -- Test Suite")
    model = ComponentModel()

    # Test 1: keys
    r = model.predict({"v_in": 10000.0, "p_in": 80000.0})
    assert_true(all(k in r for k in ["v_out", "p_out_w", "p_loss_w", "efficiency"]),
                "predict returns required keys")

    # Test 2: get_info
    info = model.get_info()
    assert_true(info["ec_id"] == "EC170", "ec_id == EC170")
    assert_true(info["fidelity"] == "F1a", "fidelity == F1a")

    # Test 3: V_out = N * V_in
    N = model._model.N  # 0.04
    V_in = 10000.0
    r = model.predict({"v_in": V_in, "p_in": 80000.0})
    expected = N * V_in
    assert_true(abs(float(r["v_out"]) - expected) < 1e-9,
                f"V_out = N*V_in = {expected:.0f} V")

    # Test 4: forward efficiency -- P_out = eta * P_in
    eta = model._model.eta
    p_in = 80000.0
    r2 = model.predict({"v_in": 10000.0, "p_in": p_in})
    assert_true(abs(float(r2["p_out_w"]) - eta * p_in) < 1e-6,
                f"P_out = eta*P_in = {eta*p_in:.0f} W (forward)")

    # Test 5: losses > 0 for non-zero power
    assert_true(float(r2["p_loss_w"]) > 0, "Losses > 0")

    # Test 6: reverse power (negative P_in)
    p_in_rev = -80000.0
    r3 = model.predict({"v_in": 10000.0, "p_in": p_in_rev})
    # Reverse: more LV input needed -> P_out = p_in/eta = -80000/0.96
    expected_rev = p_in_rev / eta
    assert_true(abs(float(r3["p_out_w"]) - expected_rev) < 1e-6,
                "P_out = P_in/eta for reverse flow (LV->MV)")

    # Test 7: losses always >= 0
    p_arr = np.array([-80000.0, -40000.0, 0.0, 40000.0, 80000.0])
    r4 = model.predict({"v_in": 10000.0, "p_in": p_arr})
    assert_true(np.all(r4["p_loss_w"] >= 0), "Losses >= 0 in both directions")

    # Test 8: V_out scales with V_in
    v_in_arr = np.array([5000.0, 8000.0, 10000.0, 12000.0])
    r5 = model.predict({"v_in": v_in_arr, "p_in": 50000.0})
    assert_true(np.all(np.diff(r5["v_out"]) > 0), "V_out increases with V_in")

    # Test 9: efficiency fixed
    assert_true(abs(float(r2["efficiency"]) - eta) < 1e-9, f"Efficiency fixed at {eta}")

    # Test 10: zero power gives zero loss
    r6 = model.predict({"v_in": 10000.0, "p_in": 0.0})
    assert_true(float(r6["p_loss_w"]) == 0.0, "Zero loss at P_in=0")

    # Test 11: benchmark
    p_bench = np.random.uniform(-100000.0, 100000.0, 1000)
    t0 = time.perf_counter()
    model.predict({"v_in": 10000.0, "p_in": p_bench})
    elapsed = time.perf_counter() - t0
    assert_true(elapsed < 1.0, f"Benchmark: 1000 predictions in {elapsed*1000:.2f} ms")

    print(f"\nAll tests passed.")


if __name__ == "__main__":
    main()
