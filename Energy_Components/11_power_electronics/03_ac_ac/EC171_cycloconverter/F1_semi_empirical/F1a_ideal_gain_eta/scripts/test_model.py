"""EC171 -- Cycloconverter -- F1a -- Test Suite"""
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
    print("EC171 Cycloconverter F1a -- Test Suite")
    model = ComponentModel()

    # Test 1: keys
    r = model.predict({"v_in_rms": 6000.0, "alpha_rad": 0.3, "p_out": 500000.0})
    assert_true(all(k in r for k in ["v_out_rms", "power_factor", "p_in_w", "p_loss_w", "efficiency"]),
                "predict returns required keys")

    # Test 2: get_info
    info = model.get_info()
    assert_true(info["ec_id"] == "EC171", "ec_id == EC171")
    assert_true(info["fidelity"] == "F1a", "fidelity == F1a")

    # Test 3: V_out = V_in * cos(alpha)
    v_in = 6000.0
    alpha = 0.5
    r = model.predict({"v_in_rms": v_in, "alpha_rad": alpha, "p_out": 500000.0})
    expected = v_in * np.cos(alpha)
    assert_true(abs(float(r["v_out_rms"]) - expected) < 1e-6,
                f"V_out = V_in*cos(alpha) = {expected:.2f} V")

    # Test 4: V_out = V_in at alpha=0 (maximum)
    r2 = model.predict({"v_in_rms": 6000.0, "alpha_rad": 0.0, "p_out": 500000.0})
    assert_true(abs(float(r2["v_out_rms"]) - 6000.0) < 1e-9, "V_out = V_in at alpha=0")

    # Test 5: V_out << V_in at alpha ~ pi/2 (near-zero output at max firing angle)
    r3 = model.predict({"v_in_rms": 6000.0, "alpha_rad": np.pi/2, "p_out": 0.0})
    assert_true(float(r3["v_out_rms"]) < 0.01 * 6000.0,
                "V_out << V_in at alpha=pi/2 (< 1% of input)")

    # Test 6: V_out decreases as alpha increases
    alpha_arr = np.linspace(0.0, np.pi/2, 50)
    r4 = model.predict({"v_in_rms": 6000.0, "alpha_rad": alpha_arr, "p_out": 100000.0})
    assert_true(np.all(np.diff(r4["v_out_rms"]) <= 0), "V_out decreases with alpha")

    # Test 7: PF = cos(alpha)
    alpha = 0.5
    r5 = model.predict({"v_in_rms": 6000.0, "alpha_rad": alpha, "p_out": 500000.0})
    assert_true(abs(float(r5["power_factor"]) - np.cos(alpha)) < 1e-9,
                "PF = cos(alpha)")

    # Test 8: PF in (0, 1]
    r6 = model.predict({"v_in_rms": 6000.0, "alpha_rad": alpha_arr, "p_out": 100000.0})
    assert_true(np.all(r6["power_factor"] >= 0) and np.all(r6["power_factor"] <= 1.0),
                "PF in [0, 1]")

    # Test 9: P_in = P_out / eta
    eta = model._model.eta
    p_out = 500000.0
    r7 = model.predict({"v_in_rms": 6000.0, "alpha_rad": 0.3, "p_out": p_out})
    assert_true(abs(float(r7["p_in_w"]) - p_out / eta) < 1e-6, "P_in = P_out / eta")

    # Test 10: efficiency fixed
    assert_true(abs(float(r7["efficiency"]) - eta) < 1e-9, f"Efficiency fixed at {eta}")

    # Test 11: benchmark
    alpha_bench = np.random.uniform(0.0, np.pi/2, 1000)
    p_bench = np.random.uniform(0.0, 1000000.0, 1000)
    t0 = time.perf_counter()
    model.predict({"v_in_rms": 6000.0, "alpha_rad": alpha_bench, "p_out": p_bench})
    elapsed = time.perf_counter() - t0
    assert_true(elapsed < 1.0, f"Benchmark: 1000 predictions in {elapsed*1000:.2f} ms")

    print(f"\nAll tests passed.")


if __name__ == "__main__":
    main()
