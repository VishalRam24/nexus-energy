"""EC179 -- Wound Rotor Synchronous Generator -- F1a -- Test Suite"""
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
    print("EC179 Wound Rotor Synchronous Generator F1a -- Test Suite")
    model = ComponentModel()
    m = model._model

    # Test 1: keys
    r = model.predict({"load_fraction": 1.0})
    assert_true(all(k in r for k in ["efficiency", "p_elec_out_kw", "p_mech_in_kw",
                                      "losses_kw", "terminal_current_ka", "sync_speed_rpm"]),
                "predict returns required keys")

    # Test 2: get_info
    info = model.get_info()
    assert_true(info["ec_id"] == "EC179", "ec_id == EC179")
    assert_true(info["fidelity"] == "F1a", "fidelity == F1a")

    # Test 3: efficiency < 1
    plr = np.linspace(0.1, 1.1, 100)
    r2 = model.predict({"load_fraction": plr})
    assert_true(np.all(r2["efficiency"] < 1.0), "Efficiency < 1 everywhere")

    # Test 4: efficiency > 0
    assert_true(np.all(r2["efficiency"] > 0.0), "Efficiency > 0 everywhere")

    # Test 5: eta at rated = 0.97
    r3 = model.predict({"load_fraction": 1.0})
    assert_true(abs(float(r3["efficiency"]) - m.eta_rated) < 0.0001,
                f"Eta at PLR=1 = {float(r3['efficiency']):.4f} ~= {m.eta_rated}")

    # Test 6: P_elec = PLR * P_rated
    plr_arr = np.array([0.5, 0.75, 1.0])
    r4 = model.predict({"load_fraction": plr_arr})
    expected = plr_arr * m.P_rated
    np.testing.assert_allclose(r4["p_elec_out_kw"], expected, rtol=1e-9)
    assert_true(True, "P_elec = PLR * P_rated")

    # Test 7: P_mech > P_elec (mechanical input > electrical output due to losses)
    r5 = model.predict({"load_fraction": 1.0})
    assert_true(float(r5["p_mech_in_kw"]) > float(r5["p_elec_out_kw"]),
                "P_mech_in > P_elec_out (generator convention)")

    # Test 8: power balance P_mech = P_elec + P_loss
    diff = np.abs(r4["p_mech_in_kw"] - r4["p_elec_out_kw"] - r4["losses_kw"])
    assert_true(np.all(diff < 1e-6), "Power balance: P_mech = P_elec + P_loss")

    # Test 9: losses > 0
    assert_true(np.all(r2["losses_kw"] > 0), "Losses > 0")

    # Test 10: synchronous speed = 120*f/poles = 3000 rpm
    r6 = model.predict({"load_fraction": 1.0})
    expected_rpm = 120.0 * m.f / m.poles
    assert_true(abs(float(r6["sync_speed_rpm"]) - expected_rpm) < 1e-6,
                f"Sync speed = {expected_rpm:.0f} rpm")

    # Test 11: terminal current increases with load
    plr_range = np.array([0.25, 0.5, 0.75, 1.0])
    r7 = model.predict({"load_fraction": plr_range})
    assert_true(np.all(np.diff(r7["terminal_current_ka"]) > 0),
                "Terminal current increases with load")

    # Test 12: terminal current at rated load
    r8 = model.predict({"load_fraction": 1.0, "power_factor": 0.85})
    p_rated_w = m.P_rated * 1000.0  # W
    i_expected = p_rated_w / (np.sqrt(3.0) * m.V_terminal * 0.85) / 1000.0  # kA
    assert_true(abs(float(r8["terminal_current_ka"]) - i_expected) < 1e-6,
                f"I_rated = {i_expected:.4f} kA")

    # Test 13: benchmark
    plr_bench = np.random.uniform(0.1, 1.1, 1000)
    t0 = time.perf_counter()
    model.predict({"load_fraction": plr_bench})
    elapsed = time.perf_counter() - t0
    assert_true(elapsed < 1.0, f"Benchmark: 1000 predictions in {elapsed*1000:.2f} ms")

    print(f"\nAll tests passed.")


if __name__ == "__main__":
    main()
