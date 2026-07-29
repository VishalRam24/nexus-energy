"""EC206 — CO2 Mineralization — F1a — Test Suite"""
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
    print("EC206 CO2 Mineralization F1a — Tests")
    print("=" * 50)
    model = ComponentModel()

    r = model.predict({"capacity_fraction": 1.0})
    for key in ["conversion", "co2_stored_tCO2_h", "W_elec_GJ_h",
                "SEC_GJ_tCO2", "carbonate_produced_t_h"]:
        assert_true(key in r, f"Output key '{key}' present")

    info = model.get_info()
    assert_true(info["ec_id"] == "EC206", "ec_id == EC206")
    assert_true(info["fidelity"] == "F1a", "fidelity == F1a")

    assert_true(abs(r["conversion"] - 0.80) < 1e-9, "conversion = 0.80")
    assert_true(abs(r["SEC_GJ_tCO2"] - 0.5) < 1e-9, "SEC = 0.5 GJ/tCO2")

    # CO2 stored = capacity * conversion = 1.0 * 1.0 * 0.80 = 0.80 tCO2/h
    assert_true(abs(float(r["co2_stored_tCO2_h"]) - 0.80) < 1e-9,
                "CO2 stored = 0.80 tCO2/h at CF=1")

    # SEC check: W = CO2_stored * SEC
    expected_W = 0.80 * 0.5
    assert_true(abs(float(r["W_elec_GJ_h"]) - expected_W) < 1e-9,
                "W_elec = CO2_stored * SEC")

    # Carbonate produced check
    carbonate = float(r["carbonate_produced_t_h"])
    assert_true(carbonate > float(r["co2_stored_tCO2_h"]), "Carbonate mass > CO2 stored mass")

    # Linear scaling
    r_full = model.predict({"capacity_fraction": 1.0})
    r_half = model.predict({"capacity_fraction": 0.5})
    assert_true(abs(float(r_half["co2_stored_tCO2_h"]) - 0.5 * float(r_full["co2_stored_tCO2_h"])) < 1e-9,
                "CO2 stored scales linearly")

    # All positive
    assert_true(float(r["co2_stored_tCO2_h"]) > 0, "CO2 stored > 0")
    assert_true(float(r["W_elec_GJ_h"]) > 0, "Electric energy > 0")
    assert_true(float(r["carbonate_produced_t_h"]) > 0, "Carbonate produced > 0")

    # Vectorized
    cf = np.linspace(0.1, 1.0, 8)
    r_vec = model.predict({"capacity_fraction": cf})
    assert_true(len(r_vec["co2_stored_tCO2_h"]) == 8, "Vectorized: 8 outputs")

    print()
    print("All tests passed.")


if __name__ == "__main__":
    run_tests()
