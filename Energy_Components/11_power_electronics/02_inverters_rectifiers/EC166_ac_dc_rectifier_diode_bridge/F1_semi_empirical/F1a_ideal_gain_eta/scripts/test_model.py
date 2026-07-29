"""EC166 -- Diode Bridge Rectifier (3-Phase) -- F1a -- Test Suite"""
import sys, os, time
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
from predict import ComponentModel, _K_3PHASE

PASS = "\u2713"
FAIL = "\u2717"


def assert_true(condition, msg):
    if condition:
        print(f"  {PASS} {msg}")
    else:
        print(f"  {FAIL} {msg}")
        raise AssertionError(msg)


def main():
    print("EC166 Diode Bridge Rectifier F1a -- Test Suite")
    model = ComponentModel()

    # Test 1: keys
    r = model.predict({"v_ll": 400.0, "p_out": 30000.0})
    assert_true(all(k in r for k in ["v_dc", "i_dc", "p_in_w", "p_loss_w", "efficiency"]),
                "predict returns required keys")

    # Test 2: get_info
    info = model.get_info()
    assert_true(info["ec_id"] == "EC166", "ec_id == EC166")
    assert_true(info["fidelity"] == "F1a", "fidelity == F1a")

    # Test 3: V_dc = 3*sqrt(2)/pi * V_LL
    v_ll = 400.0
    r = model.predict({"v_ll": v_ll, "p_out": 30000.0})
    expected_vdc = _K_3PHASE * v_ll
    assert_true(abs(float(r["v_dc"]) - expected_vdc) < 1e-6,
                f"V_dc = {_K_3PHASE:.4f}*V_LL = {expected_vdc:.2f} V")

    # Test 4: V_dc scales linearly with V_LL
    v_ll_arr = np.array([200.0, 300.0, 400.0, 500.0, 600.0])
    r2 = model.predict({"v_ll": v_ll_arr, "p_out": 10000.0})
    expected = _K_3PHASE * v_ll_arr
    np.testing.assert_allclose(r2["v_dc"], expected, rtol=1e-9)
    assert_true(True, "V_dc linear with V_LL (vectorized)")

    # Test 5: I_dc = P_out / V_dc
    p_out = 30000.0
    r3 = model.predict({"v_ll": 400.0, "p_out": p_out})
    i_dc = float(r3["i_dc"])
    v_dc = float(r3["v_dc"])
    assert_true(abs(i_dc - p_out / v_dc) < 1e-6, "I_dc = P_out / V_dc")

    # Test 6: P_in = P_out / eta
    p_in = float(r3["p_in_w"])
    eta = model._model.eta
    assert_true(abs(p_in - p_out / eta) < 1e-6, "P_in = P_out / eta")

    # Test 7: losses = P_in - P_out
    p_loss = float(r3["p_loss_w"])
    assert_true(abs(p_loss - (p_in - p_out)) < 1e-6, "P_loss = P_in - P_out")

    # Test 8: efficiency fixed
    assert_true(abs(float(r3["efficiency"]) - eta) < 1e-9, f"Efficiency fixed at {eta}")

    # Test 9: V_dc > V_LL (3-phase rectifier characteristic)
    assert_true(float(r3["v_dc"]) > 400.0, "V_dc > V_LL (3-phase rectifier boosts DC)")

    # Test 10: zero output gives zero current
    r4 = model.predict({"v_ll": 400.0, "p_out": 0.0})
    assert_true(float(r4["i_dc"]) == 0.0, "I_dc = 0 at P_out = 0")

    # Test 11: benchmark
    v_bench = np.random.uniform(100.0, 700.0, 1000)
    p_bench = np.random.uniform(0.0, 50000.0, 1000)
    t0 = time.perf_counter()
    model.predict({"v_ll": v_bench, "p_out": p_bench})
    elapsed = time.perf_counter() - t0
    assert_true(elapsed < 1.0, f"Benchmark: 1000 predictions in {elapsed*1000:.2f} ms")

    print(f"\nAll tests passed.")


if __name__ == "__main__":
    main()
