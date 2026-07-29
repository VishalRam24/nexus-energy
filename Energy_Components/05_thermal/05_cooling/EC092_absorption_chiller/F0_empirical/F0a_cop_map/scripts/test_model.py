"""F0a tests for EC092 Single-Effect LiBr-H2O Absorption Chiller. Custom harness, no pytest. NumPy only."""
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

    xkey = "T_gen_C"
    for x in [70.0, 80.0, 90.0, 100.0, 120.0]:
        r = m.predict({xkey: x})
        assert_true(r["COP"] > 0.0, "COP>0 at %s=%.2f (COP=%.3f)" % (xkey, x, r["COP"]))
    rr = m.predict({xkey: 90.0})
    assert_true(abs(rr["COP"] - 0.7) < 1e-6,
                "COP at rated %s equals COP_rated (0.7)" % xkey)
    cmin, cmax = min([0.55, 0.65, 0.7, 0.73, 0.75]), max([0.55, 0.65, 0.7, 0.73, 0.75])
    rmid = m.predict({xkey: 0.5 * ([70.0, 80.0, 90.0, 100.0, 120.0][0] + [70.0, 80.0, 90.0, 100.0, 120.0][-1])})
    assert_true(cmin - 1e-9 <= rmid["COP"] <= cmax + 1e-9, "interp COP within bounds")
    rp = m.predict({xkey: 90.0, "part_load_ratio": 0.5})
    assert_true(abs(rp["Q_cool_kW"] - 0.5 * 500.0) < 1e-6, "Q_cool = PLR*Q_rated")
    assert_true(abs(rp["W_in_kW"] - rp["Q_cool_kW"] / rp["COP"]) < 1e-6, "W_in = Q/COP")
    rlo = m.predict({xkey: [70.0, 80.0, 90.0, 100.0, 120.0][0] - 100.0})
    assert_true(rlo["COP"] > 0.0, "edge low input gives valid COP")
    info = m.get_info()
    assert_true(info["component_id"] == "EC092", "get_info() id correct")

    t0 = time.time()
    for _ in range(1000):
        m.predict({"T_gen_C": 90.0})
    dt = (time.time() - t0) * 1000.0
    assert_true(dt < 2000.0, "1000 predictions fast (%.1f ms)" % dt)

    print("\n%d passed, %d failed" % (_passed, _failed))


if __name__ == "__main__":
    main()
    sys.exit(0 if _failed == 0 else 1)
