"""Tailored tests for EC007 F0a — Reversible Fuel Cell (RFC). NumPy only, no pytest."""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from model import EfficiencyCurve  # noqa: E402
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
    info = m.get_info()
    xk, yk = info["input"], info["output"]
    lk = m.params["lookup"]
    xs = np.asarray(lk[xk]["value"], dtype=float)
    ys = np.asarray(lk[yk]["value"], dtype=float)

    print("Testing EC007 Reversible Fuel Cell (RFC) (F0a)")

    # 1. lookup endpoints reproduce table exactly
    xs_sorted = np.sort(xs)
    assert_true(np.isclose(m.predict({xk: float(xs_sorted[0])})[yk],
                           ys[np.argsort(xs)][0], atol=1e-9),
                "lookup at lower breakpoint matches table")
    assert_true(np.isclose(m.predict({xk: float(xs_sorted[-1])})[yk],
                           ys[np.argsort(xs)][-1], atol=1e-9),
                "lookup at upper breakpoint matches table")

    # 2. interpolation lands within table value bounds (physical)
    xq = np.linspace(float(xs.min()), float(xs.max()), 25)
    yq = np.asarray(m.predict({xk: xq.tolist()})[yk])
    assert_true(np.all(yq >= ys.min() - 1e-9) and np.all(yq <= ys.max() + 1e-9),
                "interpolated values stay within tabulated range")

    # 3. physical bounds on the metric
    assert_true(_physical_bounds(info["metric"], yq),
                "metric values respect physical bounds")

    # 4. clamping outside the table to endpoints
    below = m.predict({xk: float(xs.min()) - 1e6})[yk]
    above = m.predict({xk: float(xs.max()) + 1e6})[yk]
    assert_true(np.isclose(below, ys[np.argsort(xs)][0], atol=1e-9)
                and np.isclose(above, ys[np.argsort(xs)][-1], atol=1e-9),
                "out-of-range queries clamp to endpoints")

    # 5. rated point recovered from datasheet
    rated = m.params["rated"]
    rx = float(rated[xk]["value"])
    ry = float(rated[yk]["value"])
    assert_true(np.isclose(m.predict({xk: rx})[yk], ry, atol=max(1e-6, 0.02 * abs(ry) + 1e-6)),
                "rated point matches datasheet value")

    # 6. predict() interface returns the right key and scalar type
    r = m.predict({xk: rx})
    assert_true(yk in r and np.ndim(r[yk]) == 0, "predict() returns scalar output under expected key")

    # 7. model class direct lookup matches predict()
    raw = EfficiencyCurve(xs.tolist(), ys.tolist())
    assert_true(np.isclose(float(raw.lookup(rx)), m.predict({xk: rx})[yk], atol=1e-12),
                "model.lookup matches ComponentModel.predict")

    # 8. fast benchmark: 1000 predictions
    t0 = time.perf_counter()
    grid = np.linspace(float(xs.min()), float(xs.max()), 1000)
    _ = m.predict({xk: grid.tolist()})
    dt = (time.perf_counter() - t0) * 1e3
    assert_true(dt < 200.0, "1000 predictions fast (%.2f ms)" % dt)

    print("\nResults: %d passed, %d failed" % (_passed, _failed))
    return _failed


def _physical_bounds(metric, y):
    if "efficiency" in metric:
        return np.all(y >= 0.0) and np.all(y <= 1.0)
    if "recovery" in metric:
        return np.all(y >= 0.0) and np.all(y <= 1.0)
    if "boiloff" in metric:
        return np.all(y >= 0.0) and np.all(y <= 100.0)
    if "pressure" in metric or "mass" in metric or "capacity" in metric or "energy" in metric:
        return np.all(y >= 0.0)
    return np.all(np.isfinite(y))


if __name__ == "__main__":
    failed = main()
    sys.exit(0 if failed == 0 else 1)
