"""EC204 — Calcium Looping — F1a — Test Suite"""
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
    print("EC204 Calcium Looping F1a — Tests")
    print("=" * 50)
    model = ComponentModel()

    r = model.predict({"capacity_fraction": 1.0})
    for key in ["capture_rate", "co2_captured_tCO2_h", "Q_thermal_GJ_h",
                "W_elec_GJ_h", "SEC_thermal_GJ_tCO2", "SEC_elec_GJ_tCO2", "SEC_total_GJ_tCO2"]:
        assert_true(key in r, f"Output key '{key}' present")

    info = model.get_info()
    assert_true(info["ec_id"] == "EC204", "ec_id == EC204")
    assert_true(info["fidelity"] == "F1a", "fidelity == F1a")

    # Fresh sorbent: cycle 1
    r1 = model.predict({"capacity_fraction": 1.0, "cycle_number": 1})
    assert_true(abs(r1["capture_rate"] - 0.90 * (1 - 0.002 * 1)) < 1e-9,
                "capture_rate at cycle 1 = 0.90 * (1 - 0.002)")
    assert_true(abs(r1["SEC_thermal_GJ_tCO2"] - 3.2) < 1e-9, "SEC_thermal = 3.2 GJ/tCO2")

    # Capture rate decreases with cycling
    r100 = model.predict({"capacity_fraction": 1.0, "cycle_number": 100})
    assert_true(r100["capture_rate"] < r1["capture_rate"],
                "Capture rate decreases with cycle number")

    # Capture rate never below floor (0.50)
    r500 = model.predict({"capacity_fraction": 1.0, "cycle_number": 500})
    assert_true(r500["capture_rate"] >= 0.50, "Capture rate >= floor 0.50")

    # Positive values
    assert_true(float(r1["co2_captured_tCO2_h"]) > 0, "CO2 captured > 0")
    assert_true(float(r1["Q_thermal_GJ_h"]) > 0, "Thermal energy > 0")
    assert_true(float(r1["W_elec_GJ_h"]) > 0, "Electric energy > 0")

    # Linear scaling with capacity
    r_full = model.predict({"capacity_fraction": 1.0, "cycle_number": 1})
    r_half = model.predict({"capacity_fraction": 0.5, "cycle_number": 1})
    assert_true(abs(float(r_half["co2_captured_tCO2_h"]) - 0.5 * float(r_full["co2_captured_tCO2_h"])) < 1e-9,
                "CO2 captured scales linearly with capacity")

    # SEC constant
    assert_true(abs(r100["SEC_thermal_GJ_tCO2"] - 3.2) < 1e-9,
                "SEC_thermal constant regardless of cycle number")

    # Vectorized capacity
    cf = np.linspace(0.1, 1.0, 10)
    r_vec = model.predict({"capacity_fraction": cf, "cycle_number": 1})
    assert_true(len(r_vec["co2_captured_tCO2_h"]) == 10, "Vectorized: 10 outputs")

    print()
    print("All tests passed.")


if __name__ == "__main__":
    run_tests()
