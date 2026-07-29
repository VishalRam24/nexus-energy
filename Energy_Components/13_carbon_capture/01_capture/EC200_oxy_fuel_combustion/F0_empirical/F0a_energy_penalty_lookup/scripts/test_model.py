"""F0a tests for EC200 Oxy-Fuel Combustion Capture (no pytest)."""
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
    assert_true(info["component_id"] == "EC200", "get_info reports EC200")
    out = m.predict({"fuel_rate": 10.0, "load": 1.0})
    assert_true(abs(out["capture_rate"] - 0.95) < 1e-6, "full-load capture == 0.95 (datasheet)")
    assert_true(out["o2_demand_kg_s"] > out["co2_generated_kg_s"] * 0.0, "O2 demand positive")
    assert_true(out["o2_demand_kg_s"] > 0 and out["co2_captured_kg_s"] > 0, "flows positive")
    # specific penalty in literature oxy band (~200-350 kWh/tCO2 incl ASU+comp)
    assert_true(150.0 < out["specific_penalty_kWh_tCO2"] < 500.0, "specific penalty in oxy band")
    # capture rate monotonic with load
    a = m.predict({"load": 0.4})["capture_rate"]; b = m.predict({"load": 1.0})["capture_rate"]
    assert_true(b >= a, "capture rate non-decreasing with load")
    # co2 scales with fuel
    lo = m.predict({"fuel_rate": 1.0})["co2_generated_kg_s"]; hi = m.predict({"fuel_rate": 100.0})["co2_generated_kg_s"]
    assert_true(hi > lo * 90, "CO2 scales with fuel rate")
    t0 = time.time(); [m.predict({"fuel_rate": 10.0}) for _ in range(1000)]
    assert_true((time.time() - t0) < 1.0, "1000 predictions < 1 s")
    print("\n{} passed, {} failed".format(_passed, _failed))
    return 0 if _failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
