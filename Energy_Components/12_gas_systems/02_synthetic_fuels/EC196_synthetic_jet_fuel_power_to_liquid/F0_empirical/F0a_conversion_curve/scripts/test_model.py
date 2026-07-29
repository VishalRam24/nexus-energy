"""Tailored tests for EC196 F0a — Synthetic Jet Fuel (Power-to-Liquid). NumPy only, no pytest."""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from model import LookupCurve  # noqa: E402
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


def _physical_bounds(metric, y):
    m = metric.lower()
    if any(k in m for k in ("conversion", "efficiency", "yield", "selectivity",
                            "recovery", "fraction")):
        return np.all(y >= 0.0) and np.all(y <= 1.0)
    if any(k in m for k in ("energy", "power", "sec", "flow", "capacity",
                            "throughput", "drop", "duty", "specific", "temperature")):
        return np.all(y >= 0.0)
    return np.all(np.isfinite(y))


def main():
    m = ComponentModel()
    info = m.get_info()
    xk, yk = info["input"], info["output"]
    lk = m.params["lookup"]
    xs = np.asarray(lk[xk]["value"], dtype=float)
    ys = np.asarray(lk[yk]["value"], dtype=float)
    order = np.argsort(xs)
    ys_ord = ys[order]

    print("Testing EC196 Synthetic Jet Fuel (Power-to-Liquid) (F0a)")

    xs_sorted = np.sort(xs)
    assert_true(np.isclose(m.predict({xk: float(xs_sorted[0])})[yk], ys_ord[0], atol=1e-9),
                "lookup at lower breakpoint matches table")
    assert_true(np.isclose(m.predict({xk: float(xs_sorted[-1])})[yk], ys_ord[-1], atol=1e-9),
                "lookup at upper breakpoint matches table")

    xq = np.linspace(float(xs.min()), float(xs.max()), 25)
    yq = np.asarray(m.predict({xk: xq.tolist()})[yk])
    assert_true(np.all(yq >= ys.min() - 1e-9) and np.all(yq <= ys.max() + 1e-9),
                "interpolated values stay within tabulated range")

    assert_true(_physical_bounds(info["metric"], yq), "metric values respect physical bounds")

    diffs = np.diff(ys_ord)
    mono = np.all(diffs >= -1e-9) or np.all(diffs <= 1e-9)
    assert_true(mono, "tabulated curve is monotonic")

    below = m.predict({xk: float(xs.min()) - 1e6})[yk]
    above = m.predict({xk: float(xs.max()) + 1e6})[yk]
    assert_true(np.isclose(below, ys_ord[0], atol=1e-9)
                and np.isclose(above, ys_ord[-1], atol=1e-9),
                "out-of-range queries clamp to endpoints")

    rated = m.params["rated"]
    rx = float(rated[xk]["value"])
    ry = float(rated[yk]["value"])
    assert_true(np.isclose(m.predict({xk: rx})[yk], ry, atol=max(1e-6, 0.02 * abs(ry) + 1e-6)),
                "rated point matches datasheet value")

    r = m.predict({xk: rx})
    assert_true(yk in r and np.ndim(r[yk]) == 0,
                "predict() returns scalar output under expected key")

    raw = LookupCurve(xs.tolist(), ys.tolist())
    assert_true(np.isclose(float(raw.lookup(rx)), m.predict({xk: rx})[yk], atol=1e-12),
                "model.lookup matches ComponentModel.predict")

    t0 = time.perf_counter()
    grid = np.linspace(float(xs.min()), float(xs.max()), 1000)
    _ = m.predict({xk: grid.tolist()})
    dt = (time.perf_counter() - t0) * 1e3
    assert_true(dt < 200.0, "1000 predictions fast (%.2f ms)" % dt)

    print("\nResults: %d passed, %d failed" % (_passed, _failed))
    return _failed


if __name__ == "__main__":
    failed = main()
    sys.exit(0 if failed == 0 else 1)
