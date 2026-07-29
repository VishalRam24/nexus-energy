"""EC180 — DFIG — F1a Efficiency Map — Test Suite"""
import sys, os, numpy as np
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


def run_tests():
    print("EC180 DFIG F1a Efficiency Map — Tests")
    print("=" * 50)
    model = ComponentModel()

    # Test: predict returns required keys
    r = model.predict({"load_fraction": 1.0, "slip": -0.25})
    for key in ["efficiency", "output_power_w", "input_power_w", "losses_w", "rotor_speed_rpm"]:
        assert_true(key in r, f"Output key '{key}' present")

    # Test: get_info returns correct component
    info = model.get_info()
    assert_true(info["ec_id"] == "EC180", "ec_id == EC180")
    assert_true(info["fidelity"] == "F1a", "fidelity == F1a")

    # Test: rated efficiency ~0.95 at full load zero slip
    r = model.predict({"load_fraction": 1.0, "slip": 0.0})
    assert_true(abs(float(r["efficiency"]) - 0.95) < 1e-9, "eta=0.95 at PLR=1 slip=0")

    # Test: efficiency bounded [0, 1]
    plr = np.linspace(0.05, 1.2, 50)
    r = model.predict({"load_fraction": plr, "slip": 0.0})
    assert_true(np.all(r["efficiency"] >= 0.0) and np.all(r["efficiency"] <= 1.0),
                "Efficiency bounded [0, 1]")

    # Test: power balance P_in = P_out + losses
    r = model.predict({"load_fraction": np.linspace(0.1, 1.2, 20), "slip": 0.0})
    diff = np.abs(r["input_power_w"] - r["output_power_w"] - r["losses_w"])
    assert_true(np.all(diff < 1e-3), "Power balance: P_in = P_out + losses")

    # Test: losses positive
    r = model.predict({"load_fraction": 1.0, "slip": 0.0})
    assert_true(float(r["losses_w"]) > 0.0, "Losses > 0 at rated operation")

    # Test: efficiency decreases at part load
    r_full = model.predict({"load_fraction": 1.0, "slip": 0.0})
    r_part = model.predict({"load_fraction": 0.3, "slip": 0.0})
    assert_true(float(r_full["efficiency"]) >= float(r_part["efficiency"]),
                "Efficiency at full load >= part load")

    # Test: slip penalty reduces efficiency
    r_zero = model.predict({"load_fraction": 1.0, "slip": 0.0})
    r_slip = model.predict({"load_fraction": 1.0, "slip": 0.30})
    assert_true(float(r_zero["efficiency"]) > float(r_slip["efficiency"]),
                "Zero slip has higher efficiency than slip=0.30")

    # Test: super-sync and sub-sync symmetric slip penalty
    r_sup = model.predict({"load_fraction": 1.0, "slip": -0.25})
    r_sub = model.predict({"load_fraction": 1.0, "slip": 0.25})
    assert_true(abs(float(r_sup["efficiency"]) - float(r_sub["efficiency"])) < 1e-9,
                "Efficiency symmetric around slip=0")

    # Test: rotor speed at rated slip -0.25
    r = model.predict({"load_fraction": 1.0, "slip": -0.25})
    assert_true(abs(r["rotor_speed_rpm"] - 1875.0) < 1e-6,
                "Rotor speed = 1875 rpm at slip=-0.25")

    # Test: vectorized
    plr = np.linspace(0.1, 1.2, 100)
    r = model.predict({"load_fraction": plr, "slip": 0.0})
    assert_true(len(r["efficiency"]) == 100, "Vectorized: 100 outputs for 100 inputs")

    # Test: rated power 2 MW
    r = model.predict({"load_fraction": 1.0, "slip": 0.0})
    assert_true(abs(float(r["output_power_w"]) - 2e6) < 1e-3, "P_out = 2 MW at PLR=1")

    print()
    print("All tests passed.")


if __name__ == "__main__":
    run_tests()
