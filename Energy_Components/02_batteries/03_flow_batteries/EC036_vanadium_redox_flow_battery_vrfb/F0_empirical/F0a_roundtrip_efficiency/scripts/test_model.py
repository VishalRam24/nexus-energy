"""Tailored tests for Vanadium Redox Flow Battery (VRFB) (EC036) F0a round-trip-efficiency lookup. No pytest."""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from predict import ComponentModel

_failed = 0


def assert_true(cond, msg):
    global _failed
    if cond:
        print(f"  \u2713 {msg}")
    else:
        _failed += 1
        print(f"  \u2717 {msg}")


def main():
    m = ComponentModel()
    c = m.curve

    # 1. all tabulated efficiencies are physical
    assert_true(np.all((c.eta > 0.0) & (c.eta < 1.0)), "all efficiencies in (0,1)")

    # 2. efficiency is monotonically non-increasing with C-rate
    assert_true(np.all(np.diff(c.eta) <= 1e-9), "efficiency non-increasing vs C-rate")

    # 3. zero C-rate -> curve maximum (least ohmic loss), and physical
    assert_true(c.efficiency(0.0) >= c.efficiency(5.0), "eta(0) >= eta(5C)")
    assert_true(c.efficiency(0.0) == np.max(c.eta), "eta at rest is curve maximum")
    assert_true(0.0 < c.efficiency(0.0) < 1.0, "eta at rest in (0,1)")

    # 4. rated point matches datasheet within tolerance
    assert_true(abs(c.efficiency(c.c_rated) - c.eta_rated) < 0.02,
                f"eta at rated C ({c.c_rated}) == eta_rated ({c.eta_rated})")

    # 5. interpolation endpoints exact
    assert_true(abs(c.efficiency(c.c_rates[0]) - c.eta[0]) < 1e-9, "endpoint c_rates[0] exact")
    assert_true(abs(c.efficiency(c.c_rates[-1]) - c.eta[-1]) < 1e-9, "endpoint c_rates[-1] exact")

    # 6. interpolation between two breakpoints stays bracketed
    mid = c.efficiency(0.5 * (c.c_rates[1] + c.c_rates[2]))
    assert_true(min(c.eta[2], c.eta[1]) - 1e-9 <= mid <= max(c.eta[1], c.eta[2]) + 1e-9,
                "interpolated value bracketed by neighbours")

    # 7. predict() interface
    out = m.predict({"c_rate": c.c_rated})
    assert_true("roundtrip_efficiency" in out and 0.0 < out["roundtrip_efficiency"] < 1.0,
                "predict() returns valid efficiency")
    info = m.get_info()
    assert_true(info["component_id"] == "EC036" and info["version"] == "1.0.0",
                "get_info() id/version correct")

    # 8. fast benchmark
    t0 = time.time()
    xs = np.linspace(0, 5, 1000)
    c.efficiency(xs)
    dt = (time.time() - t0) * 1e3
    assert_true(dt < 50.0, f"1000-pt lookup fast ({dt:.2f} ms)")

    print(f"\n{'PASSED' if _failed == 0 else 'FAILED'}: {_failed} failure(s)")
    return _failed


if __name__ == "__main__":
    sys.exit(0 if main() == 0 else 1)
