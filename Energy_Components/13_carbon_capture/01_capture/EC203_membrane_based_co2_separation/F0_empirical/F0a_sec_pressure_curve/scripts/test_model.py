"""F0a tests for EC203 Membrane-Based CO2 Separation (no pytest)."""
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
    assert_true(info["component_id"] == "EC203", "get_info reports EC203")
    out = m.predict({"co2_feed_kg_s": 10.0, "pressure_ratio": 10.0})
    assert_true(abs(out["SEC_MJ_kgCO2"] - 0.75) < 1e-6, "rated PR=10 SEC == 0.75 MJ/kgCO2 (datasheet)")
    assert_true(out["permeate_purity"] == 0.95, "permeate purity 0.95")
    assert_true(abs(out["co2_captured_kg_s"] - 8.0) < 1e-6, "recovery 0.80 -> 8 kg/s from 10 feed")
    assert_true(0.5 <= out["SEC_MJ_kgCO2"] <= 1.0, "SEC in physical band 0.5-1.0")
    # SEC monotonic in pressure ratio
    a = m.predict({"pressure_ratio": 5.0})["SEC_MJ_kgCO2"]; b = m.predict({"pressure_ratio": 20.0})["SEC_MJ_kgCO2"]
    assert_true(b > a, "SEC monotonic increasing in pressure ratio")
    assert_true(abs(m.predict({"pressure_ratio": 20.0})["SEC_MJ_kgCO2"] - 1.00) < 1e-6, "endpoint PR=20 -> 1.00")
    t0 = time.time(); [m.predict({"pressure_ratio": 10.0}) for _ in range(1000)]
    assert_true((time.time() - t0) < 1.0, "1000 predictions < 1 s")
    print("\n{} passed, {} failed".format(_passed, _failed))
    return 0 if _failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
