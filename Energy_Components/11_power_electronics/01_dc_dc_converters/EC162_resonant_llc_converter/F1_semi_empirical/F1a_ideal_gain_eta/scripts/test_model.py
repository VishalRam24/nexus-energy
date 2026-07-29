"""EC162 -- Resonant LLC Converter -- F1a -- Test Suite"""
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
    print("EC162 LLC Resonant Converter F1a -- Test Suite")
    model = ComponentModel()

    # Test 1: keys
    r = model.predict({"v_in": 400.0, "fn": 1.0, "p_in": 80.0})
    assert_true(all(k in r for k in ["v_out", "gain_M", "p_out_w", "p_loss_w", "efficiency"]),
                "predict returns required keys")

    # Test 2: get_info
    info = model.get_info()
    assert_true(info["ec_id"] == "EC162", "ec_id == EC162")
    assert_true(info["fidelity"] == "F1a", "fidelity == F1a")

    # Test 3: gain M ~ 1 at resonance (fn=1)
    r_res = model.predict({"v_in": 400.0, "fn": 1.0, "p_in": 80.0})
    M = float(r_res["gain_M"])
    assert_true(abs(M - 1.0) < 0.01, f"M ~ 1 at resonance fn=1 (got {M:.4f})")

    # Test 4: V_out = N * V_in * M at resonance
    N = model._model.N
    V_in = 400.0
    r2 = model.predict({"v_in": V_in, "fn": 1.0, "p_in": 80.0})
    expected = N * V_in * float(r2["gain_M"])
    assert_true(abs(float(r2["v_out"]) - expected) < 1e-9,
                f"V_out = N*V_in*M = {expected:.2f} V")

    # Test 5: M is symmetric: M(fn > 1) < 1 and M(fn < 1) > 1 (step down/up)
    r_above = model.predict({"v_in": 400.0, "fn": 1.3, "p_in": 50.0})
    r_below = model.predict({"v_in": 400.0, "fn": 0.7, "p_in": 50.0})
    assert_true(float(r_above["gain_M"]) < 1.0, "M < 1 for fn > 1")
    assert_true(float(r_below["gain_M"]) > 1.0, "M > 1 for fn < 1")

    # Test 6: P_out = eta * P_in
    p_in = 80.0
    r3 = model.predict({"v_in": 400.0, "fn": 1.0, "p_in": p_in})
    assert_true(abs(float(r3["p_out_w"]) - model._model.eta * p_in) < 1e-9,
                f"P_out = eta*P_in = {model._model.eta*p_in:.2f} W")

    # Test 7: efficiency fixed
    assert_true(abs(float(r3["efficiency"]) - model._model.eta) < 1e-9,
                f"Efficiency fixed at {model._model.eta}")

    # Test 8: losses > 0
    assert_true(float(r3["p_loss_w"]) > 0, "Losses > 0")

    # Test 9: V_out scales linearly with V_in
    v_in_arr = np.array([200.0, 300.0, 400.0, 500.0])
    r4 = model.predict({"v_in": v_in_arr, "fn": 1.0, "p_in": 50.0})
    assert_true(np.all(np.diff(r4["v_out"]) > 0), "V_out increases linearly with V_in")

    # Test 10: gain M is positive everywhere
    fn_range = np.linspace(0.5, 2.0, 100)
    r5 = model.predict({"v_in": 400.0, "fn": fn_range, "p_in": 50.0})
    assert_true(np.all(r5["gain_M"] > 0), "Gain M > 0 for all fn")

    # Test 11: benchmark
    fn_bench = np.random.uniform(0.5, 2.0, 1000)
    t0 = time.perf_counter()
    model.predict({"v_in": 400.0, "fn": fn_bench, "p_in": 50.0})
    elapsed = time.perf_counter() - t0
    assert_true(elapsed < 1.0, f"Benchmark: 1000 predictions in {elapsed*1000:.2f} ms")

    print(f"\nAll tests passed.")


if __name__ == "__main__":
    main()
