"""F0a tests for EC198 Post-Combustion Capture (Amine Scrubbing) (no pytest)."""
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
    assert_true(info["component_id"] == "EC198", "get_info reports EC198")
    out = m.predict({"flue_gas_rate": 500.0, "co2_fraction": 0.12, "capture_rate": 0.90})
    assert_true(abs(out["reboiler_duty_GJ_tCO2"] - 3.20) < 1e-6, "rated 90% duty == 3.20 GJ/tCO2 (datasheet)")
    assert_true(2.5 < out["total_specific_energy_GJ_tCO2"] < 5.0, "total SEC in plausible MEA band")
    assert_true(out["co2_captured_kg_s"] > 0, "captured CO2 positive")
    # monotonic: higher capture -> higher reboiler duty
    a = m.predict({"capture_rate": 0.85})["reboiler_duty_GJ_tCO2"]
    b = m.predict({"capture_rate": 0.95})["reboiler_duty_GJ_tCO2"]
    assert_true(b > a, "reboiler duty monotonic increasing in capture rate")
    # endpoint matches table
    assert_true(abs(m.predict({"capture_rate": 0.80})["reboiler_duty_GJ_tCO2"] - 2.95) < 1e-6, "endpoint 0.80 -> 2.95")
    # capture mass scales with flow
    lo = m.predict({"flue_gas_rate": 100.0})["co2_captured_kg_s"]
    hi = m.predict({"flue_gas_rate": 1000.0})["co2_captured_kg_s"]
    assert_true(hi > lo * 9, "captured CO2 scales ~linearly with flue flow")
    t0 = time.time(); [m.predict({"capture_rate": 0.9}) for _ in range(1000)]
    assert_true((time.time() - t0) < 1.0, "1000 predictions < 1 s")
    print("\n{} passed, {} failed".format(_passed, _failed))
    return 0 if _failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
