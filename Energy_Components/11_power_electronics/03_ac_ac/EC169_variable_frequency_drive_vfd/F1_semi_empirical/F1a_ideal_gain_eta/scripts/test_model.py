"""EC169 -- Variable Frequency Drive (VFD) -- F1a -- Test Suite"""
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
    print("EC169 VFD F1a -- Test Suite")
    model = ComponentModel()

    # Test 1: keys
    r = model.predict({"f_out": 50.0, "p_out": 10000.0})
    assert_true(all(k in r for k in ["v_out_ll", "v_hz_ratio", "p_in_w", "p_loss_w", "efficiency"]),
                "predict returns required keys")

    # Test 2: get_info
    info = model.get_info()
    assert_true(info["ec_id"] == "EC169", "ec_id == EC169")
    assert_true(info["fidelity"] == "F1a", "fidelity == F1a")

    # Test 3: V_out = V_rated at f_rated
    V_rated = model._model.V_rated
    f_rated = model._model.f_rated
    r = model.predict({"f_out": f_rated, "p_out": 10000.0})
    assert_true(abs(float(r["v_out_ll"]) - V_rated) < 1e-9,
                f"V_out = V_rated = {V_rated}V at f_rated")

    # Test 4: V_out = V_rated * f/f_rated for f < f_rated
    f = 25.0
    r2 = model.predict({"f_out": f, "p_out": 5000.0})
    expected = V_rated * f / f_rated
    assert_true(abs(float(r2["v_out_ll"]) - expected) < 1e-9,
                f"V_out = V_rated*f/f_rated = {expected:.1f}V at f=25Hz")

    # Test 5: V_out capped at V_rated for f > f_rated (field weakening)
    r3 = model.predict({"f_out": 100.0, "p_out": 10000.0})
    assert_true(float(r3["v_out_ll"]) == V_rated, "V_out capped at V_rated above f_rated")

    # Test 6: V_out = 0 at f = 0
    r4 = model.predict({"f_out": 0.0, "p_out": 0.0})
    assert_true(float(r4["v_out_ll"]) == 0.0, "V_out = 0 at f_out = 0")

    # Test 7: V_out increases with f (below rated)
    f_arr = np.linspace(5.0, 50.0, 20)
    r5 = model.predict({"f_out": f_arr, "p_out": 5000.0})
    assert_true(np.all(np.diff(r5["v_out_ll"]) > 0), "V_out increases with f (constant V/Hz)")

    # Test 8: V/Hz ratio is constant below f_rated
    r6 = model.predict({"f_out": np.array([10.0, 25.0, 50.0]), "p_out": 5000.0})
    vh = r6["v_hz_ratio"]
    assert_true(np.max(np.abs(np.diff(vh))) < 1e-6, "V/Hz ratio is constant below f_rated")

    # Test 9: P_in = P_out / eta
    eta = model._model.eta
    p_out = 10000.0
    r7 = model.predict({"f_out": 50.0, "p_out": p_out})
    assert_true(abs(float(r7["p_in_w"]) - p_out / eta) < 1e-6, "P_in = P_out / eta")

    # Test 10: efficiency fixed
    assert_true(abs(float(r7["efficiency"]) - eta) < 1e-9, f"Efficiency fixed at {eta}")

    # Test 11: benchmark
    f_bench = np.random.uniform(0.0, 120.0, 1000)
    p_bench = np.random.uniform(0.0, 15000.0, 1000)
    t0 = time.perf_counter()
    model.predict({"f_out": f_bench, "p_out": p_bench})
    elapsed = time.perf_counter() - t0
    assert_true(elapsed < 1.0, f"Benchmark: 1000 predictions in {elapsed*1000:.2f} ms")

    print(f"\nAll tests passed.")


if __name__ == "__main__":
    main()
