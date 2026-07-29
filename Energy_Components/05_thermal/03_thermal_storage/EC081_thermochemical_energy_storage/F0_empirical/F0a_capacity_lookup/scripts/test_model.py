"""F0a tests for Thermochemical Storage (CaO/Ca(OH)2) (EC081). NumPy-only, no pytest."""
import os, sys, time
sys.path.insert(0, os.path.dirname(__file__))
from predict import ComponentModel

_p = 0
_f = 0
def assert_true(cond, msg):
    global _p, _f
    if cond:
        _p += 1; print("  \u2713", msg)
    else:
        _f += 1; print("  \u2717", msg)

def run():
    m = ComponentModel()
    lut = m.lut
    E = m.E_cap

    assert_true(abs(lut.energy_at(0.0)) < 1e-6, "empty store holds 0 energy at SOC=0")
    assert_true(abs(lut.energy_at(1.0) - E) < 1e-3,
                "full store holds rated capacity %.2f kWh at SOC=1" % E)

    assert_true(lut.energy_at(0.25) < lut.energy_at(0.75),
                "stored energy increases monotonically with SOC")
    assert_true(abs(lut.energy_at(0.5) - 0.5 * E) < 1e-3, "linear midpoint = 0.5 * capacity")

    assert_true(abs(lut.energy_at(2.0) - E) < 1e-3, "SOC>1 clips to full capacity")

    assert_true(0.0 < m.eta_rt <= 1.0, "round-trip efficiency in (0,1]")

    out = m.predict({"soc": 0.5})
    assert_true(set(["energy_stored_kWh", "energy_deliverable_kWh", "round_trip_efficiency"]).issubset(out),
                "predict() returns required keys")
    assert_true(out["energy_deliverable_kWh"] <= out["energy_stored_kWh"] + 1e-9,
                "deliverable energy <= stored (losses applied)")

    info = m.get_info()
    assert_true(info["component_id"] == "EC081" and "valid_ranges" in info,
                "get_info() returns id + valid_ranges")

    t0 = time.time()
    for _ in range(1000):
        lut.energy_at(0.5)
    dt = time.time() - t0
    assert_true(dt < 1.0, "1000 lookups fast (%.1f ms)" % (dt * 1e3))

if __name__ == "__main__":
    print("== EC081 Thermochemical Storage (CaO/Ca(OH)2) F0a ==")
    run()
    print("\n%d passed, %d failed" % (_p, _f))
    sys.exit(0 if _f == 0 else 1)
