"""EC165 -- Multilevel Inverter (3-Level NPC) -- F1a -- Test Suite"""
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
    print("EC165 Multilevel Inverter (3-Level NPC) F1a -- Test Suite")
    model = ComponentModel()

    # Test 1: keys
    r = model.predict({"v_dc": 800.0, "ma": 1.0, "p_in": 80000.0})
    assert_true(all(k in r for k in ["v_ac_phase_rms", "v_ac_line_rms", "thd_approx",
                                      "p_out_w", "p_loss_w", "efficiency"]),
                "predict returns required keys")

    # Test 2: get_info
    info = model.get_info()
    assert_true(info["ec_id"] == "EC165", "ec_id == EC165")
    assert_true(info["fidelity"] == "F1a", "fidelity == F1a")

    # Test 3: V_ac_phase_rms = ma * V_dc / (2*sqrt(2))
    v_dc, ma = 800.0, 1.0
    r = model.predict({"v_dc": v_dc, "ma": ma, "p_in": 80000.0})
    expected = ma * v_dc / (2.0 * np.sqrt(2.0))
    assert_true(abs(float(r["v_ac_phase_rms"]) - expected) < 1e-6,
                f"V_ac_phase_rms = {expected:.2f} V")

    # Test 4: V_LL = sqrt(3) * V_phase
    v_phase = float(r["v_ac_phase_rms"])
    v_ll = float(r["v_ac_line_rms"])
    assert_true(abs(v_ll - np.sqrt(3) * v_phase) < 1e-6,
                f"V_LL = sqrt(3)*V_phase")

    # Test 5: V_ac scales with ma
    ma_arr = np.array([0.3, 0.6, 0.9, 1.0])
    r2 = model.predict({"v_dc": 800.0, "ma": ma_arr, "p_in": 50000.0})
    assert_true(np.all(np.diff(r2["v_ac_line_rms"]) > 0), "V_ac_LL increases with ma")

    # Test 6: THD decreases with ma
    assert_true(np.all(np.diff(r2["thd_approx"]) < 0), "THD decreases as ma increases")

    # Test 7: THD > 0
    assert_true(np.all(r2["thd_approx"] > 0), "THD > 0")

    # Test 8: P_out = eta * P_in
    p_in = 80000.0
    r3 = model.predict({"v_dc": 800.0, "ma": 1.0, "p_in": p_in})
    assert_true(abs(float(r3["p_out_w"]) - model._model.eta * p_in) < 1e-6,
                "P_out = eta * P_in")

    # Test 9: efficiency fixed
    assert_true(abs(float(r3["efficiency"]) - model._model.eta) < 1e-9,
                f"Efficiency fixed at {model._model.eta}")

    # Test 10: losses > 0
    assert_true(float(r3["p_loss_w"]) > 0, "Losses > 0")

    # Test 11: V_ac = 0 at ma = 0
    r4 = model.predict({"v_dc": 800.0, "ma": 0.0, "p_in": 0.0})
    assert_true(float(r4["v_ac_line_rms"]) == 0.0, "V_ac = 0 at ma = 0")

    # Test 12: benchmark
    ma_bench = np.random.uniform(0.3, 1.15, 1000)
    t0 = time.perf_counter()
    model.predict({"v_dc": 800.0, "ma": ma_bench, "p_in": 50000.0})
    elapsed = time.perf_counter() - t0
    assert_true(elapsed < 1.0, f"Benchmark: 1000 predictions in {elapsed*1000:.2f} ms")

    print(f"\nAll tests passed.")


if __name__ == "__main__":
    main()
