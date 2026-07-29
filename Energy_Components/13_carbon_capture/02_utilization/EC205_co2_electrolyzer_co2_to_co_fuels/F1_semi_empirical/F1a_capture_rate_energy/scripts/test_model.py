"""EC205 — CO2 Electrolyzer — F1a — Test Suite"""
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
    print("EC205 CO2 Electrolyzer F1a — Tests")
    print("=" * 50)
    model = ComponentModel()

    r = model.predict({"capacity_fraction": 1.0})
    for key in ["faradaic_efficiency", "co2_converted_kg_h", "co_produced_kg_h",
                "W_elec_kWh_h", "SEC_kWh_kgCO2"]:
        assert_true(key in r, f"Output key '{key}' present")

    info = model.get_info()
    assert_true(info["ec_id"] == "EC205", "ec_id == EC205")
    assert_true(info["fidelity"] == "F1a", "fidelity == F1a")

    assert_true(abs(r["faradaic_efficiency"] - 0.85) < 1e-9, "FE = 0.85")

    # SEC at reference j=200 mA/cm2
    r_ref = model.predict({"capacity_fraction": 1.0, "current_density_mA_cm2": 200.0})
    assert_true(abs(r_ref["SEC_kWh_kgCO2"] - 8.0) < 1e-9, "SEC = 8.0 kWh/kgCO2 at j=200")

    # SEC increases away from reference
    r_high = model.predict({"capacity_fraction": 1.0, "current_density_mA_cm2": 300.0})
    assert_true(r_high["SEC_kWh_kgCO2"] > r_ref["SEC_kWh_kgCO2"], "SEC increases at j=300")

    # CO product has correct mass ratio to CO2
    co2 = float(r["co2_converted_kg_h"])
    co = float(r["co_produced_kg_h"])
    expected_ratio = 28.01 / 44.01
    assert_true(abs(co / co2 - expected_ratio) < 1e-3, "CO/CO2 mass ratio = 28.01/44.01")

    # Linear scaling with capacity
    r_full = model.predict({"capacity_fraction": 1.0})
    r_half = model.predict({"capacity_fraction": 0.5})
    assert_true(abs(float(r_half["co2_converted_kg_h"]) - 0.5 * float(r_full["co2_converted_kg_h"])) < 1e-6,
                "CO2 converted scales linearly")

    # Positive outputs
    assert_true(float(r["co2_converted_kg_h"]) > 0, "CO2 converted > 0")
    assert_true(float(r["co_produced_kg_h"]) > 0, "CO produced > 0")
    assert_true(float(r["W_elec_kWh_h"]) > 0, "Electric power > 0")

    # Vectorized
    cf = np.linspace(0.1, 1.0, 12)
    r_vec = model.predict({"capacity_fraction": cf})
    assert_true(len(r_vec["co2_converted_kg_h"]) == 12, "Vectorized: 12 outputs")

    print()
    print("All tests passed.")


if __name__ == "__main__":
    run_tests()
