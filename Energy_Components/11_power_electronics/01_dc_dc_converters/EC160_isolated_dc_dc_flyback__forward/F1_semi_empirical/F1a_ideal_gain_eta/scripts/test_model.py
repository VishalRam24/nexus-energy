"""EC160 -- Isolated DC-DC Converter (Flyback/Forward) -- F1a -- Test Suite"""
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
    print("EC160 Isolated DC-DC F1a -- Test Suite")
    model = ComponentModel()

    # Test 1: keys
    r = model.predict({"v_in": 400.0, "duty_cycle": 0.3, "p_in": 80.0})
    assert_true(all(k in r for k in ["v_out", "p_out_w", "p_loss_w", "efficiency"]),
                "predict returns required keys")

    # Test 2: get_info
    info = model.get_info()
    assert_true(info["ec_id"] == "EC160", "ec_id == EC160")
    assert_true(info["fidelity"] == "F1a", "fidelity == F1a")

    # Test 3: V_out = N*D*V_in
    N = model._model.N  # 0.1
    D = 0.3
    V_in = 400.0
    r = model.predict({"v_in": V_in, "duty_cycle": D, "p_in": 80.0})
    expected_vout = N * D * V_in
    assert_true(abs(float(r["v_out"]) - expected_vout) < 1e-9,
                f"V_out = N*D*V_in = {expected_vout:.3f} V")

    # Test 4: P_out = eta * P_in
    p_in = 80.0
    r = model.predict({"v_in": 400.0, "duty_cycle": 0.3, "p_in": p_in})
    expected_pout = model._model.eta * p_in
    assert_true(abs(float(r["p_out_w"]) - expected_pout) < 1e-9,
                f"P_out = eta*P_in = {expected_pout:.2f} W")

    # Test 5: P_loss = (1-eta)*P_in
    expected_loss = (1 - model._model.eta) * p_in
    assert_true(abs(float(r["p_loss_w"]) - expected_loss) < 1e-9,
                f"P_loss = (1-eta)*P_in = {expected_loss:.2f} W")

    # Test 6: efficiency is fixed
    eta = float(r["efficiency"])
    assert_true(abs(eta - model._model.eta) < 1e-9, f"Efficiency fixed at {model._model.eta}")

    # Test 7: efficiency in (0, 1)
    D_range = np.linspace(0.05, 0.5, 20)
    r2 = model.predict({"v_in": 400.0, "duty_cycle": D_range, "p_in": 50.0})
    assert_true(np.all(r2["efficiency"] > 0) and np.all(r2["efficiency"] < 1),
                "Efficiency in (0, 1) for range of duty cycles")

    # Test 8: V_out scales with V_in
    v_in_arr = np.array([200.0, 300.0, 400.0, 500.0])
    r3 = model.predict({"v_in": v_in_arr, "duty_cycle": 0.3, "p_in": 50.0})
    assert_true(np.all(np.diff(r3["v_out"]) > 0), "V_out increases with V_in")

    # Test 9: V_out scales with D
    D_arr = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    r4 = model.predict({"v_in": 400.0, "duty_cycle": D_arr, "p_in": 50.0})
    assert_true(np.all(np.diff(r4["v_out"]) > 0), "V_out increases with D")

    # Test 10: zero D gives zero V_out (clipped to 0.05, so check at d=0 clips)
    r5 = model.predict({"v_in": 400.0, "duty_cycle": 0.0, "p_in": 0.0})
    assert_true(float(r5["v_out"]) >= 0.0, "V_out >= 0 at D=0")

    # Test 11: benchmark
    D_bench = np.random.uniform(0.05, 0.5, 1000)
    p_bench = np.random.uniform(1.0, 100.0, 1000)
    t0 = time.perf_counter()
    model.predict({"v_in": 400.0, "duty_cycle": D_bench, "p_in": p_bench})
    elapsed = time.perf_counter() - t0
    assert_true(elapsed < 1.0, f"Benchmark: 1000 predictions in {elapsed*1000:.2f} ms")

    print(f"\nAll tests passed.")


if __name__ == "__main__":
    main()
