"""F0a tests for Finned-Tube Heat Exchanger (EC075). NumPy-only, no pytest."""
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
    cu = m.curve

    assert_true(abs(cu.effectiveness(0.0)) < 1e-6, "effectiveness = 0 at NTU = 0")

    e_lo = cu.effectiveness(0.5); e_hi = cu.effectiveness(4.0)
    assert_true(0.0 < e_lo < e_hi <= 1.0, "effectiveness in (0,1] and rising with NTU")

    assert_true(cu.effectiveness(8.0) <= 1.0 + 1e-9, "effectiveness never exceeds 1 (2nd law)")

    rated = m.p["rated"]
    ntu_r = rated["NTU_rated"]["value"]; eps_r = rated["effectiveness_rated"]["value"]
    assert_true(abs(cu.effectiveness(ntu_r) - eps_r) < 1e-3,
                "effectiveness at rated NTU matches datasheet %.4f" % eps_r)

    out = m.predict({"T_h_in": 90.0, "T_c_in": 20.0})
    assert_true(set(["effectiveness", "NTU", "Q_W", "Q_max_W"]).issubset(out),
                "predict() returns required keys")
    assert_true(0.0 < out["Q_W"] <= out["Q_max_W"] + 1e-6,
                "0 < Q <= Q_max (heat duty bounded by max)")

    cold = m.predict({"T_h_in": 20.0, "T_c_in": 80.0})
    assert_true(cold["Q_W"] < 0, "reversed gradient gives negative duty (sign correct)")

    info = m.get_info()
    assert_true(info["component_id"] == "EC075" and "valid_ranges" in info,
                "get_info() returns id + valid_ranges")

    t0 = time.time()
    for _ in range(1000):
        cu.effectiveness(2.0)
    dt = time.time() - t0
    assert_true(dt < 1.0, "1000 lookups fast (%.1f ms)" % (dt * 1e3))

if __name__ == "__main__":
    print("== EC075 Finned-Tube Heat Exchanger F0a ==")
    run()
    print("\n%d passed, %d failed" % (_p, _f))
    sys.exit(0 if _f == 0 else 1)
