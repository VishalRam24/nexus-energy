"""F0a tests for EC094 Evaporative Cooler (Direct). Custom harness, no pytest. NumPy only."""
import os, sys, time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
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

    r = m.predict({"T_db_C": 35.0, "T_wb_C": 20.0, "m_air_kg_s": 1.0})
    assert_true(20.0 <= r["T_out_C"] <= 35.0, "T_wb <= T_out <= T_db")
    assert_true(abs(r["effectiveness"] - 0.85) < 1e-9, "effectiveness=0.85 (datasheet)")
    expected = 35.0 - 0.85 * (35.0 - 20.0)
    assert_true(abs(r["T_out_C"] - expected) < 1e-6, "T_out = T_db - eps*(T_db-T_wb)")
    assert_true(r["Q_cool_W"] > 0.0, "Q_cool>0 for warm dry air")
    rs = m.predict({"T_db_C": 25.0, "T_wb_C": 25.0, "m_air_kg_s": 1.0})
    assert_true(abs(rs["Q_cool_W"]) < 1e-9, "Q_cool=0 at saturation")
    assert_true(r["COP"] > 1.0, "COP>1 (low fan power)")
    r2 = m.predict({"T_db_C": 35.0, "T_wb_C": 20.0, "m_air_kg_s": 2.0})
    assert_true(abs(r2["Q_cool_W"] - 2.0 * r["Q_cool_W"]) < 1e-6, "Q_cool scales with m_air")
    info = m.get_info()
    assert_true(info["component_id"] == "EC094", "get_info() id correct")

    t0 = time.time()
    for _ in range(1000):
        m.predict({"T_db_C": 35.0, "T_wb_C": 20.0})
    dt = (time.time() - t0) * 1000.0
    assert_true(dt < 2000.0, "1000 predictions fast (%.1f ms)" % dt)

    print("\n%d passed, %d failed" % (_passed, _failed))


if __name__ == "__main__":
    main()
    sys.exit(0 if _failed == 0 else 1)
