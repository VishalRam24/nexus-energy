"""F0a tests for EC208 CO2 Geological Sequestration (no pytest)."""
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
    assert_true(info["component_id"] == "EC208", "get_info reports EC208")
    out = m.predict({"P_wellhead_bar": 150.0, "area_km2": 100.0})
    # 150 wellhead vs 200 reservoir -> overpressure clamped to 0 -> no injection
    assert_true(out["overpressure_bar"] == 0.0, "wellhead below reservoir P -> zero overpressure")
    assert_true(out["injection_rate_kg_s"] == 0.0, "no injection without overpressure")
    # capacity: 100 km2 * 100 m * 0.15 * 0.02 * 700 / 1e9 = 21 Mt
    assert_true(abs(out["storage_capacity_Mt"] - 21.0) < 1e-6, "static storage capacity == 21 Mt (pore-volume datasheet)")
    # with overpressure injection > 0 and monotonic
    inj = m.predict({"P_wellhead_bar": 240.0})
    assert_true(inj["overpressure_bar"] == 40.0, "240-200 -> 40 bar overpressure")
    assert_true(abs(inj["injection_rate_kg_s"] - 110.0) < 1e-6, "40 bar overpressure -> 110 kg/s (curve)")
    a = m.predict({"P_wellhead_bar": 210.0})["injection_rate_kg_s"]; b = m.predict({"P_wellhead_bar": 260.0})["injection_rate_kg_s"]
    assert_true(b > a, "injection monotonic in wellhead pressure")
    # capacity scales with area
    assert_true(m.predict({"area_km2": 200.0})["storage_capacity_Mt"] > out["storage_capacity_Mt"], "capacity scales with area")
    t0 = time.time(); [m.predict({"P_wellhead_bar": 240.0}) for _ in range(1000)]
    assert_true((time.time() - t0) < 1.0, "1000 predictions < 1 s")
    print("\n{} passed, {} failed".format(_passed, _failed))
    return 0 if _failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
