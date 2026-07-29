"""EC212 — MSF — F1a — Test Suite"""
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
    print("EC212 MSF F1a — Tests")
    print("=" * 50)
    model = ComponentModel()

    r = model.predict({"capacity_fraction": 1.0})
    for key in ["GOR", "recovery", "SEC_thermal_kJ_kg", "SEC_elec_kWh_m3",
                "distillate_flow_m3_h", "Q_thermal_GJ_h", "W_elec_kWh_h", "steam_consumption_kg_h"]:
        assert_true(key in r, f"Output key '{key}' present")

    info = model.get_info()
    assert_true(info["ec_id"] == "EC212", "ec_id == EC212")
    assert_true(info["fidelity"] == "F1a", "fidelity == F1a")

    # Reference conditions
    r_ref = model.predict({"capacity_fraction": 1.0, "T_top_brine_C": 110.0})
    assert_true(abs(r_ref["GOR"] - 8.0) < 1e-9, "GOR = 8.0 at T_top=110 C")
    assert_true(abs(r_ref["SEC_thermal_kJ_kg"] - 250.0) < 1e-9, "SEC_thermal = 250 kJ/kg")
    assert_true(abs(r_ref["SEC_elec_kWh_m3"] - 3.5) < 1e-9, "SEC_elec = 3.5 kWh/m3")

    # GOR increases with T_top
    r_high = model.predict({"capacity_fraction": 1.0, "T_top_brine_C": 120.0})
    assert_true(r_high["GOR"] > r_ref["GOR"], "GOR increases with T_top")

    # Thermal energy check: 1000 m3/h * 1000 kg/m3 * 250 kJ/kg / 1e6 = 250 GJ/h
    assert_true(abs(float(r_ref["Q_thermal_GJ_h"]) - 250.0) < 1e-6, "Q_thermal = 250 GJ/h at CF=1")

    # Electric power: 1000 m3/h * 3.5 kWh/m3 = 3500 kWh/h
    assert_true(abs(float(r_ref["W_elec_kWh_h"]) - 3500.0) < 1e-6, "W_elec = 3500 kWh/h at CF=1")

    # Linear scaling
    r_half = model.predict({"capacity_fraction": 0.5})
    assert_true(abs(float(r_half["distillate_flow_m3_h"]) - 500.0) < 1e-6,
                "Distillate = 500 m3/h at CF=0.5")

    # All positive
    assert_true(float(r_ref["Q_thermal_GJ_h"]) > 0, "Thermal energy > 0")
    assert_true(float(r_ref["W_elec_kWh_h"]) > 0, "Electric power > 0")
    assert_true(float(r_ref["steam_consumption_kg_h"]) > 0, "Steam consumption > 0")

    # Vectorized
    cf = np.linspace(0.1, 1.0, 10)
    r_vec = model.predict({"capacity_fraction": cf})
    assert_true(len(r_vec["distillate_flow_m3_h"]) == 10, "Vectorized: 10 outputs")

    print()
    print("All tests passed.")


if __name__ == "__main__":
    run_tests()
