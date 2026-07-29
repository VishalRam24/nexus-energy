"""EC211 — Forward Osmosis — F1a — Test Suite"""
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
    print("EC211 Forward Osmosis F1a — Tests")
    print("=" * 50)
    model = ComponentModel()

    r = model.predict({"capacity_fraction": 1.0})
    for key in ["recovery", "rejection", "SEC_membrane_kWh_m3", "SEC_regen_kWh_m3",
                "SEC_total_kWh_m3", "permeate_flow_m3_h", "concentrate_flow_m3_h", "W_elec_kWh_h"]:
        assert_true(key in r, f"Output key '{key}' present")

    info = model.get_info()
    assert_true(info["ec_id"] == "EC211", "ec_id == EC211")
    assert_true(info["fidelity"] == "F1a", "fidelity == F1a")

    assert_true(abs(r["recovery"] - 0.80) < 1e-9, "recovery = 0.80")
    assert_true(abs(r["rejection"] - 0.95) < 1e-9, "rejection = 0.95")
    assert_true(abs(r["SEC_membrane_kWh_m3"] - 0.5) < 1e-9, "SEC_membrane = 0.5 kWh/m3")

    # With regen: total = membrane + regen = 0.5 + 3.0 = 3.5
    r_with = model.predict({"capacity_fraction": 1.0, "include_regen": True})
    assert_true(abs(r_with["SEC_total_kWh_m3"] - 3.5) < 1e-9, "SEC_total = 3.5 kWh/m3 with regen")

    # Without regen: total = 0.5
    r_no = model.predict({"capacity_fraction": 1.0, "include_regen": False})
    assert_true(abs(r_no["SEC_total_kWh_m3"] - 0.5) < 1e-9, "SEC_total = 0.5 kWh/m3 without regen")

    # Mass balance: permeate + concentrate = feed
    r = model.predict({"capacity_fraction": 1.0})
    feed = 100.0
    assert_true(abs(float(r["permeate_flow_m3_h"]) + float(r["concentrate_flow_m3_h"]) - feed) < 1e-6,
                "permeate + concentrate = feed")

    # Linear scaling
    r_full = model.predict({"capacity_fraction": 1.0})
    r_half = model.predict({"capacity_fraction": 0.5})
    assert_true(abs(float(r_half["permeate_flow_m3_h"]) - 0.5 * float(r_full["permeate_flow_m3_h"])) < 1e-9,
                "Permeate scales linearly")

    # Positive outputs
    assert_true(float(r["permeate_flow_m3_h"]) > 0, "Permeate > 0")
    assert_true(float(r["W_elec_kWh_h"]) > 0, "Electric power > 0")

    # Vectorized
    cf = np.linspace(0.1, 1.0, 10)
    r_vec = model.predict({"capacity_fraction": cf})
    assert_true(len(r_vec["permeate_flow_m3_h"]) == 10, "Vectorized: 10 outputs")

    print()
    print("All tests passed.")


if __name__ == "__main__":
    run_tests()
