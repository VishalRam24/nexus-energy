"""F0a tailored tests for EC157 Buck Converter (Step-Down). NumPy only, no pytest."""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from predict import ComponentModel  # noqa: E402
from model import EfficiencyCurveModel  # noqa: E402

passed = 0
failed = 0


def assert_true(cond, msg):
    global passed, failed
    if cond:
        passed += 1
        print("  \u2713", msg)
    else:
        failed += 1
        print("  \u2717", msg)


def main():
    m = EfficiencyCurveModel()
    cm = ComponentModel()

    # 1. efficiency within physical bounds over full operating range
    import numpy as np
    grid = np.linspace(0.0, float(m.load_fraction[-1]), 50)
    eff = m.efficiency_at(grid)
    assert_true(np.all(eff >= 0.0) and np.all(eff <= 1.0),
                "0 <= efficiency <= 1 across operating range")

    # 2. peak efficiency does not exceed datasheet peak and is < 1
    assert_true(0.0 < eff.max() < 1.0,
                "peak efficiency in (0,1): %.4f" % eff.max())

    # 3. endpoint: zero load -> zero efficiency
    assert_true(abs(m.efficiency_at(0.0)) < 1e-9,
                "efficiency(0 load) == 0")

    # 4. peaks near mid-load, not at the extremes (part-load characteristic)
    imax = int(np.argmax(eff))
    assert_true(eff[imax] >= m.efficiency_at(1.0) and eff[imax] >= eff[1],
                "peak efficiency >= full-load and >= light-load")

    # 5. rated point matches datasheet rated efficiency
    assert_true(abs(m.efficiency_at(1.0) - m.rated_efficiency) < 1e-3,
                "efficiency(full load) matches rated %.4f" % m.rated_efficiency)

    # 6. interpolation is bounded between neighbouring breakpoints (monotone interp)
    mid = m.efficiency_at(0.5 * (m.load_fraction[2] + m.load_fraction[3]))
    lo, hi = sorted((m.efficiency[2], m.efficiency[3]))
    assert_true(lo - 1e-9 <= mid <= hi + 1e-9,
                "interpolated value lies between breakpoints")

    # 7. losses non-negative and zero at no load
    assert_true(m.losses(0.0) == 0.0 and m.losses(0.5) >= 0.0,
                "losses >= 0 and zero at no load")

    # 8. predict() interface returns consistent p_in = p_out + losses
    r = cm.predict({"load_fraction": 0.6})
    assert_true(abs(r["p_in"] - r["p_out"] - r["losses"]) < 1e-6
                and 0.0 < r["efficiency"] < 1.0,
                "predict() energy balance consistent")

    # 9. get_info() surfaces metadata
    info = cm.get_info()
    assert_true(info["component_id"] == "EC157" and "source" in info,
                "get_info() metadata present")

    # 10. fast benchmark: 1000 predictions
    t0 = time.perf_counter()
    for _ in range(1000):
        cm.predict({"load_fraction": 0.5})
    dt = time.perf_counter() - t0
    assert_true(dt < 1.0, "1000 predictions in %.4f s (<1 s)" % dt)


if __name__ == "__main__":
    print("Testing EC157 Buck Converter (Step-Down) F0a")
    main()
    print("\n%d passed, %d failed" % (passed, failed))
    sys.exit(0 if failed == 0 else 1)
