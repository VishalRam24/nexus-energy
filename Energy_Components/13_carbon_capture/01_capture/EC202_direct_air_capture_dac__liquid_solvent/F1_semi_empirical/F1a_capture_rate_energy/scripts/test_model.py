"""EC202 — DAC Liquid Solvent — F1a Capture Rate & Energy — Test Suite"""
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
    print("EC202 DAC Liquid Solvent F1a — Tests")
    print("=" * 50)
    model = ComponentModel()

    r = model.predict({"capacity_fraction": 1.0})
    for key in ["capture_rate", "co2_captured_tCO2_h", "Q_thermal_GJ_h",
                "W_elec_GJ_h", "SEC_thermal_GJ_tCO2", "SEC_elec_GJ_tCO2", "SEC_total_GJ_tCO2"]:
        assert_true(key in r, f"Output key '{key}' present")

    info = model.get_info()
    assert_true(info["ec_id"] == "EC202", "ec_id == EC202")
    assert_true(info["fidelity"] == "F1a", "fidelity == F1a")

    r = model.predict({"capacity_fraction": 1.0})
    assert_true(abs(r["capture_rate"] - 0.90) < 1e-9, "capture_rate = 0.90")
    assert_true(abs(r["SEC_thermal_GJ_tCO2"] - 6.0) < 1e-9, "SEC_thermal = 6.0 GJ/tCO2")
    assert_true(abs(r["SEC_total_GJ_tCO2"] - (6.0 + 1.8)) < 1e-9, "SEC_total = 7.8 GJ/tCO2")

    # CO2 at half capacity = 0.5 * full
    r_full = model.predict({"capacity_fraction": 1.0})
    r_half = model.predict({"capacity_fraction": 0.5})
    assert_true(abs(float(r_half["co2_captured_tCO2_h"]) - 0.5 * float(r_full["co2_captured_tCO2_h"])) < 1e-9,
                "CO2 captured scales linearly with capacity")

    # Thermal energy scales linearly
    assert_true(abs(float(r_half["Q_thermal_GJ_h"]) - 0.5 * float(r_full["Q_thermal_GJ_h"])) < 1e-9,
                "Thermal energy scales linearly with capacity")

    # SEC constant regardless of capacity
    r25 = model.predict({"capacity_fraction": 0.25})
    assert_true(abs(r25["SEC_thermal_GJ_tCO2"] - 6.0) < 1e-9,
                "SEC_thermal constant at part load")

    # All values positive
    assert_true(float(r["co2_captured_tCO2_h"]) > 0, "CO2 captured > 0")
    assert_true(float(r["Q_thermal_GJ_h"]) > 0, "Thermal energy > 0")
    assert_true(float(r["W_elec_GJ_h"]) > 0, "Electric energy > 0")

    # Vectorized
    cf = np.linspace(0.1, 1.0, 20)
    r_vec = model.predict({"capacity_fraction": cf})
    assert_true(len(r_vec["co2_captured_tCO2_h"]) == 20, "Vectorized: 20 outputs")

    print()
    print("All tests passed.")


if __name__ == "__main__":
    run_tests()
