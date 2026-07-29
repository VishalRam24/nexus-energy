"""EC177 -- BLDC Motor -- F1a -- Test Suite"""
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
    print("EC177 BLDC Motor F1a -- Test Suite")
    model = ComponentModel()
    m = model._model

    # Test 1: keys
    r = model.predict({"load_fraction": 1.0})
    assert_true(all(k in r for k in ["efficiency", "torque_nm", "current_a",
                                      "output_power_kw", "input_power_kw", "losses_kw"]),
                "predict returns required keys")

    # Test 2: get_info
    info = model.get_info()
    assert_true(info["ec_id"] == "EC177", "ec_id == EC177")
    assert_true(info["fidelity"] == "F1a", "fidelity == F1a")

    # Test 3: efficiency < 1
    plr = np.linspace(0.05, 1.2, 100)
    r2 = model.predict({"load_fraction": plr})
    assert_true(np.all(r2["efficiency"] < 1.0), "Efficiency < 1 everywhere")

    # Test 4: efficiency > 0
    assert_true(np.all(r2["efficiency"] > 0.0), "Efficiency > 0 everywhere")

    # Test 5: peak efficiency near sqrt(c0/c2)
    peak_plr = np.sqrt(m.c0 / m.c2)
    plr_range = np.linspace(0.01, 1.2, 1000)
    r3 = model.predict({"load_fraction": plr_range})
    peak_idx = np.argmax(r3["efficiency"])
    assert_true(abs(plr_range[peak_idx] - peak_plr) < 0.02,
                f"Peak efficiency near PLR={peak_plr:.3f}")

    # Test 6: eta at rated ~ 0.92
    r4 = model.predict({"load_fraction": 1.0})
    # At PLR=1: eta = 1/(1+c0+c2) = 1/(1 + (1/eta_rated - 1)) = eta_rated
    assert_true(abs(float(r4["efficiency"]) - m.eta_rated) < 0.0001,
                f"Eta at PLR=1 = {float(r4['efficiency']):.4f} ~= {m.eta_rated}")

    # Test 7: torque scales linearly with PLR
    plr_arr = np.array([0.25, 0.5, 0.75, 1.0])
    r5 = model.predict({"load_fraction": plr_arr})
    expected_T = plr_arr * m.T_rated
    np.testing.assert_allclose(r5["torque_nm"], expected_T, rtol=1e-9)
    assert_true(True, "Torque scales linearly with PLR")

    # Test 8: current = torque / Kt
    r6 = model.predict({"load_fraction": 1.0})
    expected_I = float(r6["torque_nm"]) / m.Kt
    assert_true(abs(float(r6["current_a"]) - expected_I) < 1e-9, "I = T / Kt")

    # Test 9: power balance P_in = P_out + P_loss
    plr_range2 = np.linspace(0.1, 1.2, 50)
    r7 = model.predict({"load_fraction": plr_range2})
    diff = np.abs(r7["input_power_kw"] - r7["output_power_kw"] - r7["losses_kw"])
    assert_true(np.all(diff < 1e-9), "Power balance: P_in = P_out + P_loss")

    # Test 10: output power = PLR * P_rated
    r8 = model.predict({"load_fraction": np.array([0.5, 1.0])})
    assert_true(abs(float(r8["output_power_kw"][0]) - 0.5 * m.P_rated) < 1e-9,
                "P_out = PLR * P_rated")

    # Test 11: losses > 0
    r9 = model.predict({"load_fraction": np.linspace(0.05, 1.2, 20)})
    assert_true(np.all(r9["losses_kw"] > 0), "Losses > 0")

    # Test 12: benchmark
    plr_bench = np.random.uniform(0.05, 1.2, 1000)
    t0 = time.perf_counter()
    model.predict({"load_fraction": plr_bench})
    elapsed = time.perf_counter() - t0
    assert_true(elapsed < 1.0, f"Benchmark: 1000 predictions in {elapsed*1000:.2f} ms")

    print(f"\nAll tests passed.")


if __name__ == "__main__":
    main()
