"""F0a tests for EC204 Calcium Looping (no pytest)."""
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
    assert_true(info["component_id"] == "EC204", "get_info reports EC204")
    out = m.predict({"co2_in_kg_s": 10.0, "cycle_number": 1})
    assert_true(abs(out["capture_rate"] - 0.90) < 1e-6, "cycle 1 capture == 0.90 (datasheet)")
    assert_true(abs(out["SEC_thermal_GJ_tCO2"] - 3.2) < 1e-6, "thermal SEC == 3.2 GJ/tCO2")
    assert_true(0.0 < out["capture_rate"] <= 1.0, "capture rate in (0,1]")
    assert_true(3.0 < out["SEC_total_GJ_tCO2"] < 4.0, "total SEC ~3.5 GJ/tCO2")
    # capture decays with cycles
    early = m.predict({"cycle_number": 1})["capture_rate"]
    late = m.predict({"cycle_number": 500})["capture_rate"]
    assert_true(late < early, "capture rate decays with cycle number")
    assert_true(abs(m.predict({"cycle_number": 500})["capture_rate"] - 0.40) < 1e-6, "residual conversion 0.40 at 500 cycles")
    t0 = time.time(); [m.predict({"cycle_number": 1}) for _ in range(1000)]
    assert_true((time.time() - t0) < 1.0, "1000 predictions < 1 s")
    print("\n{} passed, {} failed".format(_passed, _failed))
    return 0 if _failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
