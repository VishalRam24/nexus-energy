"""EC161 -- Dual Active Bridge (DAB) -- F1a -- Test Suite"""
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
    print("EC161 DAB DC-DC F1a -- Test Suite")
    model = ComponentModel()

    # Test 1: keys
    r = model.predict({"v1": 400.0, "v2": 400.0, "phi_rad": 0.3})
    assert_true(all(k in r for k in ["v_out_ideal", "p_transfer_w", "p_out_w", "p_loss_w", "efficiency"]),
                "predict returns required keys")

    # Test 2: get_info
    info = model.get_info()
    assert_true(info["ec_id"] == "EC161", "ec_id == EC161")
    assert_true(info["fidelity"] == "F1a", "fidelity == F1a")

    # Test 3: ideal voltage gain
    N = model._model.N  # 1.0
    V1 = 400.0
    r = model.predict({"v1": V1, "v2": 400.0, "phi_rad": 0.3})
    assert_true(abs(float(r["v_out_ideal"]) - N * V1) < 1e-9,
                f"V_out_ideal = N*V1 = {N*V1:.1f} V")

    # Test 4: zero phase shift gives zero power
    r_zero = model.predict({"v1": 400.0, "v2": 400.0, "phi_rad": 0.0})
    assert_true(abs(float(r_zero["p_transfer_w"])) < 1e-9,
                "P_transfer = 0 at phi=0")

    # Test 5: bidirectional -- negative phi gives negative power
    r_neg = model.predict({"v1": 400.0, "v2": 400.0, "phi_rad": -0.3})
    assert_true(float(r_neg["p_transfer_w"]) < 0,
                "Negative phi gives negative (reverse) power transfer")

    # Test 6: power increases with phi up to pi/2
    phi_arr = np.linspace(0.01, np.pi/2 * 0.99, 50)
    r_arr = model.predict({"v1": 400.0, "v2": 400.0, "phi_rad": phi_arr})
    assert_true(np.all(np.diff(r_arr["p_transfer_w"]) > 0),
                "Power increases monotonically with phi (0 to pi/2)")

    # Test 7: efficiency is fixed
    r2 = model.predict({"v1": 400.0, "v2": 400.0, "phi_rad": 0.5})
    assert_true(abs(float(r2["efficiency"]) - model._model.eta) < 1e-9,
                f"Efficiency fixed at {model._model.eta}")

    # Test 8: losses > 0 when power flows
    assert_true(float(r2["p_loss_w"]) > 0, "Losses > 0 when power is transferred")

    # Test 9: P_out = eta * P_transfer for forward flow
    p_t = float(r2["p_transfer_w"])
    p_o = float(r2["p_out_w"])
    assert_true(abs(p_o - model._model.eta * p_t) < 1e-6,
                "P_out = eta * P_transfer (forward)")

    # Test 10: efficiency in (0, 1)
    phi_range = np.linspace(0.1, 1.5, 20)
    r3 = model.predict({"v1": 400.0, "v2": 400.0, "phi_rad": phi_range})
    assert_true(np.all(r3["efficiency"] > 0) and np.all(r3["efficiency"] < 1),
                "Efficiency in (0, 1)")

    # Test 11: benchmark
    phi_bench = np.random.uniform(-np.pi/2, np.pi/2, 1000)
    t0 = time.perf_counter()
    model.predict({"v1": 400.0, "v2": 400.0, "phi_rad": phi_bench})
    elapsed = time.perf_counter() - t0
    assert_true(elapsed < 1.0, f"Benchmark: 1000 predictions in {elapsed*1000:.2f} ms")

    print(f"\nAll tests passed.")


if __name__ == "__main__":
    main()
