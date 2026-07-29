"""F0a tests for EC088 Oil-Fired Boiler. Custom harness, no pytest. NumPy only."""
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

    for plr in (0.0, 0.1, 0.3, 0.5, 0.75, 1.0):
        r = m.predict({"part_load_ratio": plr})
        assert_true(0.0 < r["efficiency"] <= 1.0,
                    "0<eta<=1 at PLR=%.2f (eta=%.3f)" % (plr, r["efficiency"]))
    r1 = m.predict({"part_load_ratio": 1.0})
    assert_true(abs(r1["efficiency"] - 0.87) < 1e-6,
                "eta at PLR=1.0 equals eta_nom (0.87)")
    rh = m.predict({"part_load_ratio": 0.5})
    assert_true(abs(rh["Q_out_kW"] - 0.5 * 500.0) < 1e-6, "Q_out = PLR*Q_rated")
    assert_true(r1["Q_in_kW"] >= r1["Q_out_kW"] - 1e-9, "Q_in >= Q_out")
    rlow = m.predict({"part_load_ratio": 0.0})
    assert_true(0.0 < rlow["efficiency"] <= 1.0, "below PLR_min clamps to valid eta")
    assert_true(set(["efficiency", "Q_out_kW", "Q_in_kW"]).issubset(r1.keys()),
                "predict() returns standard keys")
    info = m.get_info()
    assert_true(info["component_id"] == "EC088" and info["fidelity"].startswith("F0a"),
                "get_info() reports id/fidelity")

    t0 = time.time()
    for _ in range(1000):
        m.predict({"part_load_ratio": 0.6})
    dt = (time.time() - t0) * 1000.0
    assert_true(dt < 2000.0, "1000 predictions fast (%.1f ms)" % dt)

    print("\n%d passed, %d failed" % (_passed, _failed))


if __name__ == "__main__":
    main()
    sys.exit(0 if _failed == 0 else 1)
