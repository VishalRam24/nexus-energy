"""EC210 — Electrodialysis — F1a — Test Suite"""
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
    print("EC210 Electrodialysis F1a — Tests")
    print("=" * 50)
    model = ComponentModel()

    r = model.predict({"capacity_fraction": 1.0})
    for key in ["recovery", "rejection", "SEC_kWh_m3", "permeate_flow_m3_h",
                "concentrate_flow_m3_h", "W_elec_kWh_h"]:
        assert_true(key in r, f"Output key '{key}' present")

    info = model.get_info()
    assert_true(info["ec_id"] == "EC210", "ec_id == EC210")
    assert_true(info["fidelity"] == "F1a", "fidelity == F1a")

    # Design point: 4000 ppm -> SEC = 1.0 kWh/m3
    r_ref = model.predict({"capacity_fraction": 1.0, "feed_salinity_ppm": 4000.0})
    assert_true(abs(r_ref["SEC_kWh_m3"] - 1.0) < 1e-9, "SEC = 1.0 kWh/m3 at 4000 ppm")
    assert_true(abs(r_ref["recovery"] - 0.85) < 1e-9, "recovery = 0.85")
    assert_true(abs(r_ref["rejection"] - 0.95) < 1e-9, "rejection = 0.95")

    # SEC increases with salinity
    r_low = model.predict({"capacity_fraction": 1.0, "feed_salinity_ppm": 1000.0})
    r_high = model.predict({"capacity_fraction": 1.0, "feed_salinity_ppm": 8000.0})
    assert_true(r_high["SEC_kWh_m3"] > r_ref["SEC_kWh_m3"], "SEC increases with salinity")
    assert_true(r_low["SEC_kWh_m3"] < r_ref["SEC_kWh_m3"], "SEC decreases at lower salinity")

    # Permeate + concentrate = feed
    r = model.predict({"capacity_fraction": 1.0})
    feed = 100.0  # capacity_m3_h
    perm = float(r["permeate_flow_m3_h"])
    conc = float(r["concentrate_flow_m3_h"])
    assert_true(abs(perm + conc - feed) < 1e-6, "permeate + concentrate = feed flow")

    # Linear scaling with capacity
    r_full = model.predict({"capacity_fraction": 1.0})
    r_half = model.predict({"capacity_fraction": 0.5})
    assert_true(abs(float(r_half["permeate_flow_m3_h"]) - 0.5 * float(r_full["permeate_flow_m3_h"])) < 1e-9,
                "Permeate scales linearly with capacity")

    # All positive
    assert_true(float(r["permeate_flow_m3_h"]) > 0, "Permeate flow > 0")
    assert_true(float(r["W_elec_kWh_h"]) > 0, "Electric power > 0")

    # Vectorized
    cf = np.linspace(0.1, 1.0, 10)
    r_vec = model.predict({"capacity_fraction": cf})
    assert_true(len(r_vec["permeate_flow_m3_h"]) == 10, "Vectorized: 10 outputs")

    print()
    print("All tests passed.")


if __name__ == "__main__":
    run_tests()
