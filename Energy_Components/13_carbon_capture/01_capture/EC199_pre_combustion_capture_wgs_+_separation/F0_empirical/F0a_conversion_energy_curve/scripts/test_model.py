"""F0a tests for EC199 Pre-Combustion Capture (WGS + Separation) (no pytest)."""
import os, sys, time

sys.path.insert(0, os.path.dirname(__file__))
from predict import ComponentModel  # noqa: E402

_passed = 0
_failed = 0


def assert_true(cond, msg):
    global _passed, _failed
    if cond:
        _passed += 1
        print("  \u2713 " + msg)
    else:
        _failed += 1
        print("  \u2717 " + msg)


def main():
    m = ComponentModel()
    info = m.get_info()
    assert_true(info["component_id"] == "EC199", "get_info reports EC199")
    out = m.predict({"T_WGS_C": 250.0, "co_flow_kg_s": 10.0})
    assert_true(abs(out["wgs_conversion"] - 0.95) < 1e-6, "rated 250 C conversion == 0.95 (datasheet peak)")
    assert_true(0.0 < out["overall_capture_rate"] < 1.0, "overall capture rate in (0,1)")
    assert_true(abs(out["total_specific_energy_GJ_tCO2"] - 0.40) < 1e-6, "SEC = sep+comp = 0.40 GJ/tCO2")
    assert_true(out["co2_captured_kg_s"] > 0, "captured CO2 positive")
    # peak: conversion at 250 >= at 180 and at 420
    peak = m.predict({"T_WGS_C": 250.0})["wgs_conversion"]
    lo = m.predict({"T_WGS_C": 180.0})["wgs_conversion"]
    hi = m.predict({"T_WGS_C": 420.0})["wgs_conversion"]
    assert_true(peak >= lo and peak >= hi, "conversion peaks near 250 C")
    assert_true(abs(m.predict({"T_WGS_C": 420.0})["wgs_conversion"] - 0.62) < 1e-6, "endpoint 420 C -> 0.62")
    t0 = time.time(); [m.predict({"T_WGS_C": 250.0}) for _ in range(1000)]
    assert_true((time.time() - t0) < 1.0, "1000 predictions < 1 s")
    print("\n{} passed, {} failed".format(_passed, _failed))
    return 0 if _failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
