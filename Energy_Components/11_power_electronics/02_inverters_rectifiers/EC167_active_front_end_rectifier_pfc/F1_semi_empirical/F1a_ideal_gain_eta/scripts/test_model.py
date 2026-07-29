"""EC167 -- Active Front End Rectifier PFC -- F1a -- Test Suite"""
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
    print("EC167 Active Front End Rectifier PFC F1a -- Test Suite")
    model = ComponentModel()

    # Test 1: keys
    r = model.predict({"v_ll": 400.0, "v_dc_set": 700.0, "p_out": 30000.0})
    assert_true(all(k in r for k in ["v_dc", "i_dc", "p_in_w", "p_loss_w",
                                      "i_ac_rms", "power_factor", "efficiency"]),
                "predict returns required keys")

    # Test 2: get_info
    info = model.get_info()
    assert_true(info["ec_id"] == "EC167", "ec_id == EC167")
    assert_true(info["fidelity"] == "F1a", "fidelity == F1a")

    # Test 3: V_dc equals setpoint
    v_dc_set = 700.0
    r = model.predict({"v_ll": 400.0, "v_dc_set": v_dc_set, "p_out": 30000.0})
    assert_true(abs(float(r["v_dc"]) - v_dc_set) < 1e-9, "V_dc = V_dc_setpoint (controlled)")

    # Test 4: I_dc = P_out / V_dc
    p_out = 30000.0
    i_dc = float(r["i_dc"])
    assert_true(abs(i_dc - p_out / v_dc_set) < 1e-6, "I_dc = P_out / V_dc")

    # Test 5: P_in = P_out / eta
    eta = model._model.eta
    p_in = float(r["p_in_w"])
    assert_true(abs(p_in - p_out / eta) < 1e-6, "P_in = P_out / eta")

    # Test 6: I_ac = P_in / (sqrt(3) * V_LL)
    v_ll = 400.0
    i_ac_expected = p_in / (np.sqrt(3.0) * v_ll)
    assert_true(abs(float(r["i_ac_rms"]) - i_ac_expected) < 1e-6,
                "I_ac = P_in / (sqrt(3)*V_LL)")

    # Test 7: PF = 1
    assert_true(abs(float(r["power_factor"]) - 1.0) < 1e-9, "Power factor = 1.0")

    # Test 8: efficiency fixed
    assert_true(abs(float(r["efficiency"]) - eta) < 1e-9, f"Efficiency fixed at {eta}")

    # Test 9: losses > 0
    assert_true(float(r["p_loss_w"]) > 0, "Losses > 0")

    # Test 10: V_dc independent of V_LL (setpoint-controlled)
    v_ll_arr = np.array([350.0, 400.0, 450.0])
    r2 = model.predict({"v_ll": v_ll_arr, "v_dc_set": 700.0, "p_out": 20000.0})
    assert_true(np.all(r2["v_dc"] == 700.0), "V_dc is constant regardless of V_LL")

    # Test 11: I_ac increases with P_out
    p_out_arr = np.array([10000.0, 20000.0, 30000.0, 40000.0])
    r3 = model.predict({"v_ll": 400.0, "v_dc_set": 700.0, "p_out": p_out_arr})
    assert_true(np.all(np.diff(r3["i_ac_rms"]) > 0), "I_ac increases with P_out")

    # Test 12: zero output
    r4 = model.predict({"v_ll": 400.0, "v_dc_set": 700.0, "p_out": 0.0})
    assert_true(float(r4["i_dc"]) == 0.0, "I_dc = 0 at P_out = 0")

    # Test 13: benchmark
    p_bench = np.random.uniform(0.0, 50000.0, 1000)
    t0 = time.perf_counter()
    model.predict({"v_ll": 400.0, "v_dc_set": 700.0, "p_out": p_bench})
    elapsed = time.perf_counter() - t0
    assert_true(elapsed < 1.0, f"Benchmark: 1000 predictions in {elapsed*1000:.2f} ms")

    print(f"\nAll tests passed.")


if __name__ == "__main__":
    main()
