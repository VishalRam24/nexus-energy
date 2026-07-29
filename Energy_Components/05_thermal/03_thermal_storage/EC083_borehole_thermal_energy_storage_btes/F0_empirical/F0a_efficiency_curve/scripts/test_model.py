"""F0a tests for EC083 Borehole Thermal Energy Storage (BTES). Custom harness, no pytest. NumPy only."""
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

    xkey = "T_store_C"
    for x in [30.0, 50.0, 70.0, 90.0]:
        r = m.predict({xkey: x, "E_charge_kWh": 1000.0})
        assert_true(0.0 < r["efficiency"] < 1.0, "0<eta<1 at %s=%.1f" % (xkey, x))
        assert_true(r["E_discharge_kWh"] <= r["E_charge_kWh"] + 1e-9,
                    "discharge <= charge at %s=%.1f" % (xkey, x))
    rr = m.predict({xkey: 70.0, "E_charge_kWh": 1000.0})
    assert_true(abs(rr["efficiency"] - 0.65) < 1e-6,
                "eta at rated %s equals eta_rated (0.65)" % xkey)
    assert_true(abs(rr["E_discharge_kWh"] - 0.65 * 1000.0) < 1e-6,
                "E_discharge = eta*E_charge")
    cmin, cmax = min([0.55, 0.6, 0.65, 0.7]), max([0.55, 0.6, 0.65, 0.7])
    rmid = m.predict({xkey: 0.5 * ([30.0, 50.0, 70.0, 90.0][0] + [30.0, 50.0, 70.0, 90.0][-1])})
    assert_true(cmin - 1e-9 <= rmid["efficiency"] <= cmax + 1e-9, "interp within bounds")
    info = m.get_info()
    assert_true(info["component_id"] == "EC083", "get_info() id correct")

    t0 = time.time()
    for _ in range(1000):
        m.predict({"T_store_C": 70.0, "E_charge_kWh": 1000.0})
    dt = (time.time() - t0) * 1000.0
    assert_true(dt < 2000.0, "1000 predictions fast (%.1f ms)" % dt)

    print("\n%d passed, %d failed" % (_passed, _failed))


if __name__ == "__main__":
    main()
    sys.exit(0 if _failed == 0 else 1)
