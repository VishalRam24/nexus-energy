"""EC215 — Solar Still / HDH — F1a — Test Suite"""
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
    print("EC215 Solar Still / HDH F1a — Tests")
    print("=" * 50)
    model = ComponentModel()

    r = model.predict({"capacity_fraction": 1.0})
    for key in ["mode", "GOR", "yield_L_h", "yield_m3_h", "solar_power_W", "SEC_solar_kWh_m3"]:
        assert_true(key in r, f"Output key '{key}' present")

    info = model.get_info()
    assert_true(info["ec_id"] == "EC215", "ec_id == EC215")
    assert_true(info["fidelity"] == "F1a", "fidelity == F1a")

    # GOR at reference irradiance = GOR_ref = 2.0
    r_ref = model.predict({"capacity_fraction": 1.0, "solar_irradiance_W_m2": 800.0})
    assert_true(abs(r_ref["GOR"] - 2.0) < 1e-9, "GOR = 2.0 at G=800 W/m2")

    # GOR clamped to [1, 3]
    r_low = model.predict({"capacity_fraction": 1.0, "solar_irradiance_W_m2": 400.0})
    r_high = model.predict({"capacity_fraction": 1.0, "solar_irradiance_W_m2": 1200.0})
    assert_true(r_low["GOR"] >= 1.0, "GOR >= 1 (lower bound)")
    assert_true(r_high["GOR"] <= 3.0, "GOR <= 3 (upper bound)")

    # GOR increases with irradiance
    r_mid = model.predict({"capacity_fraction": 1.0, "solar_irradiance_W_m2": 600.0})
    assert_true(r_ref["GOR"] >= r_mid["GOR"], "GOR increases with irradiance")

    # Yield positive and scales with irradiance
    assert_true(float(r_ref["yield_L_h"]) > 0, "Yield > 0 at design conditions")
    assert_true(float(r_high["yield_L_h"]) > float(r_low["yield_L_h"]),
                "Yield increases with irradiance")

    # Yield scales with capacity
    r_full = model.predict({"capacity_fraction": 1.0})
    r_half = model.predict({"capacity_fraction": 0.5})
    assert_true(abs(float(r_half["yield_L_h"]) - 0.5 * float(r_full["yield_L_h"])) < 1e-9,
                "Yield scales linearly with capacity")

    # Solar power scales with capacity
    assert_true(float(r_ref["solar_power_W"]) > 0, "Solar power > 0")
    sp_half = float(model.predict({"capacity_fraction": 0.5, "solar_irradiance_W_m2": 800.0})["solar_power_W"])
    sp_full = float(model.predict({"capacity_fraction": 1.0, "solar_irradiance_W_m2": 800.0})["solar_power_W"])
    assert_true(abs(sp_half - 0.5 * sp_full) < 1e-6, "Solar power scales linearly")

    # Yield consistency: yield_m3_h = yield_L_h / 1000
    r = model.predict({"capacity_fraction": 1.0})
    assert_true(abs(float(r["yield_m3_h"]) - float(r["yield_L_h"]) / 1000.0) < 1e-12,
                "yield_m3_h = yield_L_h / 1000")

    # Vectorized
    cf = np.linspace(0.1, 1.0, 10)
    r_vec = model.predict({"capacity_fraction": cf})
    assert_true(len(r_vec["yield_L_h"]) == 10, "Vectorized: 10 outputs")

    print()
    print("All tests passed.")


if __name__ == "__main__":
    run_tests()
