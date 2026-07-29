"""F0a tests for EC206 CO2 Mineralization (no pytest)."""
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
    assert_true(info["component_id"] == "EC206", "get_info reports EC206")
    out = m.predict({"co2_in_kg_s": 1.0, "temperature_C": 100.0})
    assert_true(abs(out["conversion"] - 0.80) < 1e-6, "design 100 C conversion == 0.80 (datasheet)")
    assert_true(abs(out["SEC_GJ_tCO2"] - 0.5) < 1e-6, "SEC == 0.5 GJ/tCO2")
    assert_true(0.0 < out["conversion"] < 1.0, "conversion in (0,1)")
    assert_true(out["carbonate_produced_kg_s"] > out["co2_mineralized_kg_s"], "carbonate mass > CO2 mass (mass gain)")
    # conversion improves with temperature (low end)
    lo = m.predict({"temperature_C": 25.0})["conversion"]
    mid = m.predict({"temperature_C": 100.0})["conversion"]
    assert_true(mid > lo, "conversion improves from 25 C to 100 C")
    assert_true(abs(m.predict({"temperature_C": 25.0})["conversion"] - 0.55) < 1e-6, "endpoint 25 C -> 0.55")
    t0 = time.time(); [m.predict({"temperature_C": 100.0}) for _ in range(1000)]
    assert_true((time.time() - t0) < 1.0, "1000 predictions < 1 s")
    print("\n{} passed, {} failed".format(_passed, _failed))
    return 0 if _failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
