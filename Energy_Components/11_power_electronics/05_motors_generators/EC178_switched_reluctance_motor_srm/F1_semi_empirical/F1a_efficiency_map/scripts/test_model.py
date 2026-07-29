"""EC178 -- Switched Reluctance Motor (SRM) -- F1a -- Test Suite"""
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
    print("EC178 SRM F1a -- Test Suite")
    model = ComponentModel()
    m = model._model

    # Test 1: keys
    r = model.predict({"load_fraction": 1.0})
    assert_true(all(k in r for k in ["efficiency", "torque_avg_nm", "torque_ripple_nm",
                                      "output_power_kw", "input_power_kw", "losses_kw"]),
                "predict returns required keys")

    # Test 2: get_info
    info = model.get_info()
    assert_true(info["ec_id"] == "EC178", "ec_id == EC178")
    assert_true(info["fidelity"] == "F1a", "fidelity == F1a")

    # Test 3: efficiency < 1
    plr = np.linspace(0.05, 1.2, 100)
    r2 = model.predict({"load_fraction": plr})
    assert_true(np.all(r2["efficiency"] < 1.0), "Efficiency < 1 everywhere")

    # Test 4: efficiency > 0
    assert_true(np.all(r2["efficiency"] > 0.0), "Efficiency > 0 everywhere")

    # Test 5: eta at rated = eta_rated = 0.88
    r3 = model.predict({"load_fraction": 1.0})
    assert_true(abs(float(r3["efficiency"]) - m.eta_rated) < 0.0001,
                f"Eta at PLR=1 = {float(r3['efficiency']):.4f} ~= {m.eta_rated}")

    # Test 6: SRM eta_rated < BLDC (0.88 < 0.92)
    assert_true(m.eta_rated < 0.92, "SRM eta < BLDC (0.88 < 0.92) -- higher losses")

    # Test 7: torque scales linearly
    plr_arr = np.array([0.25, 0.5, 0.75, 1.0])
    r4 = model.predict({"load_fraction": plr_arr})
    expected_T = plr_arr * m.T_rated
    np.testing.assert_allclose(r4["torque_avg_nm"], expected_T, rtol=1e-9)
    assert_true(True, "Torque scales linearly with PLR")

    # Test 8: torque ripple = ripple_factor * T_avg
    r5 = model.predict({"load_fraction": 1.0})
    T_avg = float(r5["torque_avg_nm"])
    T_rip = float(r5["torque_ripple_nm"])
    assert_true(abs(T_rip - m.ripple_factor * T_avg) < 1e-9,
                f"Torque ripple = {m.ripple_factor} * T_avg = {m.ripple_factor*T_avg:.3f} Nm")

    # Test 9: power balance
    plr_range = np.linspace(0.1, 1.2, 50)
    r6 = model.predict({"load_fraction": plr_range})
    diff = np.abs(r6["input_power_kw"] - r6["output_power_kw"] - r6["losses_kw"])
    assert_true(np.all(diff < 1e-9), "Power balance: P_in = P_out + P_loss")

    # Test 10: losses > 0
    assert_true(np.all(r6["losses_kw"] > 0), "Losses > 0")

    # Test 11: peak efficiency near sqrt(c0/c2)
    peak_plr = np.sqrt(m.c0 / m.c2)
    plr_fine = np.linspace(0.01, 1.2, 1000)
    r7 = model.predict({"load_fraction": plr_fine})
    peak_idx = np.argmax(r7["efficiency"])
    assert_true(abs(plr_fine[peak_idx] - peak_plr) < 0.02,
                f"Peak efficiency near PLR={peak_plr:.3f}")

    # Test 12: benchmark
    plr_bench = np.random.uniform(0.05, 1.2, 1000)
    t0 = time.perf_counter()
    model.predict({"load_fraction": plr_bench})
    elapsed = time.perf_counter() - t0
    assert_true(elapsed < 1.0, f"Benchmark: 1000 predictions in {elapsed*1000:.2f} ms")

    print(f"\nAll tests passed.")


if __name__ == "__main__":
    main()
