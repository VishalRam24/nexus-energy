"""F0a tests for EC090 Solar Water Heater Combi System. Custom harness, no pytest. NumPy only."""
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

    for G in (0.0, 200.0, 600.0, 1000.0):
        r = m.predict({"irradiance_W_m2": G})
        assert_true(0.0 <= r["f_solar"] <= 1.0, "0<=f_solar<=1 at G=%.0f" % G)
    r0 = m.predict({"irradiance_W_m2": 0.0})
    assert_true(abs(r0["f_solar"]) < 1e-9, "f_solar=0 at night")
    assert_true(abs(r0["Q_solar_W"]) < 1e-9, "Q_solar=0 at night")
    rlo = m.predict({"irradiance_W_m2": 200.0})
    rhi = m.predict({"irradiance_W_m2": 800.0})
    assert_true(rhi["f_solar"] > rlo["f_solar"], "f_solar increases with irradiance")
    rr = m.predict({"irradiance_W_m2": 600.0, "Q_demand_W": 10000.0})
    covered = rr["Q_solar_W"] + rr["Q_aux_input_W"] * 0.90
    assert_true(abs(covered - 10000.0) < 1e-6, "solar + aux*eta = demand")
    assert_true(abs(rr["f_solar"] - 0.36) < 1e-6, "f_solar=0.36 at G=600 (datasheet)")
    info = m.get_info()
    assert_true(info["component_id"] == "EC090", "get_info() id correct")

    t0 = time.time()
    for _ in range(1000):
        m.predict({"irradiance_W_m2": 600.0})
    dt = (time.time() - t0) * 1000.0
    assert_true(dt < 2000.0, "1000 predictions fast (%.1f ms)" % dt)

    print("\n%d passed, %d failed" % (_passed, _failed))


if __name__ == "__main__":
    main()
    sys.exit(0 if _failed == 0 else 1)
