"""EC214 — MVC — F1a — Test Suite"""
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
    print("EC214 MVC F1a — Tests")
    print("=" * 50)
    model = ComponentModel()

    r = model.predict({"capacity_fraction": 1.0})
    for key in ["recovery", "SEC_kWh_m3", "distillate_flow_m3_h",
                "concentrate_flow_m3_h", "W_elec_kWh_h", "GOR_equiv"]:
        assert_true(key in r, f"Output key '{key}' present")

    info = model.get_info()
    assert_true(info["ec_id"] == "EC214", "ec_id == EC214")
    assert_true(info["fidelity"] == "F1a", "fidelity == F1a")

    # Reference SEC at design CR=1.2
    r_ref = model.predict({"capacity_fraction": 1.0, "compression_ratio": 1.2})
    assert_true(abs(r_ref["SEC_kWh_m3"] - 10.0) < 1e-9, "SEC = 10.0 kWh/m3 at CR=1.2")
    assert_true(abs(r_ref["recovery"] - 0.50) < 1e-9, "recovery = 0.50")

    # SEC in range [8, 12]
    r_lo = model.predict({"capacity_fraction": 1.0, "compression_ratio": 1.05})
    r_hi = model.predict({"capacity_fraction": 1.0, "compression_ratio": 1.5})
    assert_true(r_lo["SEC_kWh_m3"] >= 8.0, "SEC >= 8 kWh/m3 (min clamp)")
    assert_true(r_hi["SEC_kWh_m3"] <= 12.0, "SEC <= 12 kWh/m3 (max clamp)")

    # SEC increases with compression ratio
    assert_true(r_hi["SEC_kWh_m3"] > r_ref["SEC_kWh_m3"], "SEC increases with CR")

    # No thermal input (all-electric)
    assert_true("W_elec_kWh_h" in r_ref, "Electric power output present")
    assert_true("Q_thermal_GJ_h" not in r_ref, "No thermal input output (all-electric)")

    # Positive outputs
    assert_true(float(r_ref["distillate_flow_m3_h"]) > 0, "Distillate > 0")
    assert_true(float(r_ref["W_elec_kWh_h"]) > 0, "Electric power > 0")
    assert_true(float(r_ref["concentrate_flow_m3_h"]) > 0, "Concentrate > 0")

    # Linear scaling
    r_full = model.predict({"capacity_fraction": 1.0})
    r_half = model.predict({"capacity_fraction": 0.5})
    assert_true(abs(float(r_half["distillate_flow_m3_h"]) - 0.5 * float(r_full["distillate_flow_m3_h"])) < 1e-9,
                "Distillate scales linearly with capacity")

    # Vectorized
    cf = np.linspace(0.1, 1.0, 10)
    r_vec = model.predict({"capacity_fraction": cf})
    assert_true(len(r_vec["distillate_flow_m3_h"]) == 10, "Vectorized: 10 outputs")

    print()
    print("All tests passed.")


if __name__ == "__main__":
    run_tests()
