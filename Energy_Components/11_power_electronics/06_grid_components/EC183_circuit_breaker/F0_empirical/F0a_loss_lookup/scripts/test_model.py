"""F0a tests for EC183 — custom harness, no pytest."""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from predict import ComponentModel  # noqa: E402

passed = 0
failed = 0


def assert_true(cond, msg):
    global passed, failed
    if cond:
        passed += 1
        print(f"  \u2713 {msg}")
    else:
        failed += 1
        print(f"  \u2717 {msg}")


m = ComponentModel()

# 1. Loss fraction in physical bounds
lf = np.linspace(m.curve.x[0], m.curve.x[-1], 25)
loss = m.curve.lookup(lf)
assert_true(np.all((loss >= 0.0) & (loss < 1.0)), "0 <= loss fraction < 1 across loading range")

# 2. Loss fraction is monotone (i^2 R lines) OR unimodal (no-load-dominated
#    converters): at most one slope sign change across the loading range.
diffs = np.diff(loss)
sign_changes = np.sum(np.diff(np.sign(diffs[np.abs(diffs) > 1e-12])) != 0)
assert_true(sign_changes <= 1, "loss fraction monotone or unimodal across loading")

# 3. Rated point matches datasheet loss
assert_true(abs(m.predict({"load_fraction": 1.0})["loss_fraction"] - m.loss_rated) < 1e-9,
            "loss at full load matches datasheet rated value")

# 4. Endpoint clamping
assert_true(m.predict({"load_fraction": 9.0})["loss_fraction"] == m.curve.y[-1],
            "out-of-range loading clamps to endpoint")

# 5. Efficiency = 1 - loss and power balance
out = m.predict({"load_fraction": 0.8, "power_in": 100.0})
assert_true(abs(out["efficiency"] - (1.0 - out["loss_fraction"])) < 1e-12, "efficiency == 1 - loss")
assert_true(abs(out["power_out"] + out["power_loss"] - 100.0) < 1e-9, "power_out + power_loss == power_in")

# 6. predict() interface keys
keys = set(m.predict({"load_fraction": 0.5}).keys())
assert_true({"loss_fraction", "efficiency"}.issubset(keys), "predict() returns required keys")

# 7. get_info() metadata
info = m.get_info()
assert_true(info["component_id"] == "EC183" and info["version"] == "1.0.0", "get_info() metadata correct")

# 8. Fast benchmark
t0 = time.time()
for _ in range(1000):
    m.predict({"load_fraction": 0.7})
dt = time.time() - t0
assert_true(dt < 1.0, f"1000 predictions fast ({dt*1e3:.1f} ms)")

print(f"\n{passed} passed, {failed} failed")
sys.exit(0 if failed == 0 else 1)
