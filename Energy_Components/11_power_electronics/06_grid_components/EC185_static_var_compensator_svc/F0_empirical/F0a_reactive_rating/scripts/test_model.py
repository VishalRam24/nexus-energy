"""F0a tests for EC185 — custom harness, no pytest."""
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
qf = np.linspace(0.0, 1.0, 25)
loss = m.curve.lookup(qf)
assert_true(np.all((loss >= 0.0) & (loss < 0.1)), "0 <= loss fraction < 10% across output range")

# 2. Loss monotonic non-decreasing with output
assert_true(np.all(np.diff(loss) >= -1e-12), "loss fraction non-decreasing with |Q|")

# 3. Demand within range passes through unclamped
r = m.predict({"q_demand": m.q_max})
assert_true(abs(r["q_out"] - m.q_max) < 1e-9 and not r["q_clamped"], "rated demand passes unclamped")

# 4. Over-demand clamps to Q_max
r = m.predict({"q_demand": m.q_max * 5})
assert_true(abs(r["q_out"] - m.q_max) < 1e-9 and r["q_clamped"], "over-demand clamps to Q_max")

# 5. Inductive limit clamps to Q_min
r = m.predict({"q_demand": m.q_min * 5})
assert_true(abs(r["q_out"] - m.q_min) < 1e-9, "inductive over-demand clamps to Q_min")

# 6. Loss == loss_factor*|Q| at rated
r = m.predict({"q_demand": m.q_max})
assert_true(abs(r["loss"] - m.loss_factor * abs(m.q_max)) < 1e-6, "loss == loss_factor*|Q_out| at rated")

# 7. get_info() metadata
info = m.get_info()
assert_true(info["component_id"] == "EC185" and info["version"] == "1.0.0", "get_info() metadata correct")

# 8. Fast benchmark
t0 = time.time()
for _ in range(1000):
    m.predict({"q_demand": m.q_max * 0.5})
dt = time.time() - t0
assert_true(dt < 1.0, f"1000 predictions fast ({dt*1e3:.1f} ms)")

print(f"\n{passed} passed, {failed} failed")
sys.exit(0 if failed == 0 else 1)
