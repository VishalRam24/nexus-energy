"""F0a tests for EC202 Direct Air Capture (DAC) -- Liquid Solvent (no pytest)."""
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
    assert_true(info["component_id"] == "EC202", "get_info reports EC202")
    out = m.predict({"co2_rated_kg_s": 1.0, "capacity_fraction": 1.0})
    assert_true(abs(out["SEC_thermal_GJ_tCO2"] - 6.0) < 1e-6, "full-load thermal SEC == 6.0 GJ/tCO2 (datasheet)")
    assert_true(abs(out["SEC_elec_GJ_tCO2"] - 1.8) < 1e-6, "full-load elec SEC == 1.8 GJ/tCO2")
    assert_true(out["capture_rate"] == 0.90, "capture rate 0.90")
    assert_true(7.0 < out["SEC_total_GJ_tCO2"] < 9.0, "full-load total SEC ~7.8 GJ/tCO2")
    # part load raises specific energy
    pl = m.predict({"capacity_fraction": 0.1})["SEC_total_GJ_tCO2"]
    fl = m.predict({"capacity_fraction": 1.0})["SEC_total_GJ_tCO2"]
    assert_true(pl > fl, "part-load SEC higher than full load")
    assert_true(m.predict({"capacity_fraction": 0.5})["co2_captured_kg_s"] < out["co2_captured_kg_s"], "less CO2 at part load")
    t0 = time.time(); [m.predict({"capacity_fraction": 1.0}) for _ in range(1000)]
    assert_true((time.time() - t0) < 1.0, "1000 predictions < 1 s")
    print("\n{} passed, {} failed".format(_passed, _failed))
    return 0 if _failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
