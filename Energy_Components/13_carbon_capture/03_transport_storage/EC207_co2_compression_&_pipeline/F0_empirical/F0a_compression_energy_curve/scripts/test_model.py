"""F0a tests for EC207 CO2 Compression & Pipeline (no pytest)."""
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
    assert_true(info["component_id"] == "EC207", "get_info reports EC207")
    out = m.predict({"mass_flow": 100.0, "P_outlet": 150.0, "pipeline_length_km": 100.0})
    assert_true(abs(out["SEC_kWh_tCO2"] - 100.0) < 1e-6, "150 bar SEC == 100 kWh/tCO2 (datasheet)")
    assert_true(80.0 <= out["SEC_kWh_tCO2"] <= 132.0, "SEC in IPCC band")
    assert_true(out["compression_power_MW"] > 0, "compression power positive")
    assert_true(abs(out["pipeline_pressure_drop_bar"] - 4.0) < 1e-6, "100 km * 0.04 bar/km == 4 bar")
    # SEC monotonic in outlet pressure
    a = m.predict({"P_outlet": 100.0})["SEC_kWh_tCO2"]; b = m.predict({"P_outlet": 250.0})["SEC_kWh_tCO2"]
    assert_true(b > a, "SEC monotonic increasing in outlet pressure")
    # power scales with mass flow
    lo = m.predict({"mass_flow": 10.0})["compression_power_MW"]; hi = m.predict({"mass_flow": 1000.0})["compression_power_MW"]
    assert_true(hi > lo * 90, "power scales with mass flow")
    t0 = time.time(); [m.predict({"P_outlet": 150.0}) for _ in range(1000)]
    assert_true((time.time() - t0) < 1.0, "1000 predictions < 1 s")
    print("\n{} passed, {} failed".format(_passed, _failed))
    return 0 if _failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
