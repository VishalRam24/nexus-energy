"""EC203 — Membrane CO2 Separation — F1a — Test Suite"""
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
    print("EC203 Membrane CO2 Separation F1a — Tests")
    print("=" * 50)
    model = ComponentModel()

    r = model.predict({"capacity_fraction": 1.0})
    for key in ["CO2_recovery", "CO2_purity", "co2_captured_tCO2_h", "W_elec_GJ_h", "SEC_MJ_kgCO2"]:
        assert_true(key in r, f"Output key '{key}' present")

    info = model.get_info()
    assert_true(info["ec_id"] == "EC203", "ec_id == EC203")
    assert_true(info["fidelity"] == "F1a", "fidelity == F1a")

    assert_true(abs(r["CO2_recovery"] - 0.80) < 1e-9, "CO2_recovery = 0.80")
    assert_true(abs(r["CO2_purity"] - 0.95) < 1e-9, "CO2_purity = 0.95")

    # SEC at design PR = 10 should equal SEC_ref = 0.75
    r_ref = model.predict({"capacity_fraction": 1.0, "pressure_ratio": 10.0})
    assert_true(abs(r_ref["SEC_MJ_kgCO2"] - 0.75) < 1e-9, "SEC at PR=10 = 0.75 MJ/kgCO2")

    # SEC clamped to min at low PR
    r_low = model.predict({"capacity_fraction": 1.0, "pressure_ratio": 5.0})
    assert_true(r_low["SEC_MJ_kgCO2"] >= 0.5, "SEC >= 0.5 MJ/kgCO2 (min clamp)")

    # SEC clamped to max at high PR
    r_high = model.predict({"capacity_fraction": 1.0, "pressure_ratio": 20.0})
    assert_true(r_high["SEC_MJ_kgCO2"] <= 1.0, "SEC <= 1.0 MJ/kgCO2 (max clamp)")

    # SEC increases with pressure ratio
    r10 = model.predict({"capacity_fraction": 1.0, "pressure_ratio": 10.0})
    r15 = model.predict({"capacity_fraction": 1.0, "pressure_ratio": 15.0})
    assert_true(r15["SEC_MJ_kgCO2"] >= r10["SEC_MJ_kgCO2"], "SEC increases with PR")

    # Linear scaling with capacity
    r_full = model.predict({"capacity_fraction": 1.0})
    r_half = model.predict({"capacity_fraction": 0.5})
    assert_true(abs(float(r_half["co2_captured_tCO2_h"]) - 0.5 * float(r_full["co2_captured_tCO2_h"])) < 1e-9,
                "CO2 captured scales linearly")

    # Positive outputs
    assert_true(float(r["co2_captured_tCO2_h"]) > 0, "CO2 captured > 0")
    assert_true(float(r["W_elec_GJ_h"]) > 0, "Electric energy > 0")

    # Vectorized
    cf = np.linspace(0.1, 1.0, 15)
    r_vec = model.predict({"capacity_fraction": cf})
    assert_true(len(r_vec["co2_captured_tCO2_h"]) == 15, "Vectorized: 15 outputs")

    print()
    print("All tests passed.")


if __name__ == "__main__":
    run_tests()
