"""F0a tests for EC201 Direct Air Capture (DAC) -- Solid Sorbent (no pytest)."""
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
    assert_true(info["component_id"] == "EC201", "get_info reports EC201")
    out = m.predict({"air_flow_m3h": 1.0e6, "ambient_temp": 25.0})
    assert_true(abs(out["E_thermal_kWh_tCO2"] - 1500.0) < 1e-6, "rated 25 C thermal == 1500 kWh/tCO2 (datasheet)")
    assert_true(abs(out["E_electric_kWh_tCO2"] - 250.0) < 1e-6, "electric == 250 kWh/tCO2")
    assert_true(out["co2_captured_kg_h"] > 0, "captured CO2 positive")
    assert_true(1500.0 < out["total_energy_kWh_tCO2"] < 2200.0, "total energy in DAC band")
    # colder air -> higher thermal demand
    cold = m.predict({"ambient_temp": -10.0})["E_thermal_kWh_tCO2"]
    warm = m.predict({"ambient_temp": 45.0})["E_thermal_kWh_tCO2"]
    assert_true(cold > warm, "thermal demand higher in cold air")
    assert_true(abs(m.predict({"ambient_temp": -10.0})["E_thermal_kWh_tCO2"] - 1650.0) < 1e-6, "endpoint -10 C -> 1650")
    t0 = time.time(); [m.predict({"ambient_temp": 25.0}) for _ in range(1000)]
    assert_true((time.time() - t0) < 1.0, "1000 predictions < 1 s")
    print("\n{} passed, {} failed".format(_passed, _failed))
    return 0 if _failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
