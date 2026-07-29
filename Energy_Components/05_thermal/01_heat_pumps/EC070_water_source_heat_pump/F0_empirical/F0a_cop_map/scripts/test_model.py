"""F0a tests for Water-Source Heat Pump (EC070). NumPy-only, no pytest."""
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
    rated = m.p["rated"]
    rs = rated["rated_T_source"]["value"]; rk = rated["rated_T_sink"]["value"]
    rated_cop = rated["rated_cop"]["value"]

    cop = m.map.cop(rs, rk)
    assert_true(cop > 1.0, "COP at rated point > 1 (thermodynamically valid)")
    assert_true(abs(cop - rated_cop) / rated_cop < 0.12,
                "COP at rated point %.2f within 12%% of datasheet %.2f" % (cop, rated_cop))

    sink = m.map.sink
    lo = m.map.cop(m.map.src[0], sink[len(sink)//2])
    hi = m.map.cop(m.map.src[-1], sink[len(sink)//2])
    assert_true(hi > lo, "COP increases with warmer source temperature (monotonic)")

    c_hotsink = m.map.cop(m.map.src[len(m.map.src)//2], sink[-1])
    c_coolsink = m.map.cop(m.map.src[len(m.map.src)//2], sink[0])
    assert_true(c_coolsink > c_hotsink, "COP decreases with higher sink temperature (monotonic)")

    mid = sink[len(sink)//2]
    e1 = m.map.cop(-999, mid); e2 = m.map.cop(m.map.src[0], mid)
    assert_true(abs(e1 - e2) < 1e-9, "out-of-range source clips to nearest breakpoint")

    out = m.predict({"T_source": rs, "T_sink": rk, "part_load_ratio": 1.0})
    assert_true(set(["COP", "Q_thermal_kW", "P_input_kW"]).issubset(out),
                "predict() returns required keys")
    assert_true(out["P_input_kW"] > 0 and out["Q_thermal_kW"] > 0,
                "predict() positive thermal + input power")

    info = m.get_info()
    assert_true(info["component_id"] == "EC070" and "valid_ranges" in info,
                "get_info() returns id + valid_ranges")

    t0 = time.time()
    for _ in range(1000):
        m.map.cop(rs, rk)
    dt = time.time() - t0
    assert_true(dt < 1.0, "1000 lookups fast (%.1f ms)" % (dt * 1e3))

if __name__ == "__main__":
    print("== EC070 Water-Source Heat Pump F0a ==")
    run()
    print("\n%d passed, %d failed" % (_p, _f))
    sys.exit(0 if _f == 0 else 1)
