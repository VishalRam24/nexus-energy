"""EC223 RTG F0a - tests (no pytest)."""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from model import RTGPowerDecay
from predict import ComponentModel

passed = 0
failed = 0


def assert_true(cond, msg):
    global passed, failed
    if cond:
        passed += 1
        print("  ✓ " + msg)
    else:
        failed += 1
        print("  ✗ " + msg)


c = RTGPowerDecay()
m = ComponentModel()

# 1 physical: BOL power positive, hundreds of W
p0 = c.power_W(0.0)
assert_true(100.0 < p0 < 400.0, "BOL power in hundreds of W")

# 2 rated point (BOL) matches datasheet
assert_true(abs(p0 - 292.5) < 1e-2, "BOL power ~292.5 W (GPHS-RTG)")

# 3 monotonic decreasing with time
ts = [0, 5, 10, 20, 30, 45, 60, 65]
ps = [c.power_W(t) for t in ts]
assert_true(all(b < a for a, b in zip(ps, ps[1:])), "power monotonic decreasing in time")

# 4 endpoints
assert_true(c.power_W(65.0) == 0.0, "power -> 0 W at end of life (65 yr)")

# 5 Voyager-style: still significant after ~45 yr
assert_true(c.power_W(45.0) > 50.0, "still >50 W after 45 yr (Voyager-like longevity)")

# 6 interpolation between breakpoints
mid = c.power_W(15.0)
assert_true(c.power_W(20.0) < mid < c.power_W(10.0), "interp between 10 and 20 yr")

# 7 predict + get_info interface
out = m.predict({"t_years": 0.0})
info = m.get_info()
assert_true(abs(out["power_W"] - 292.5) < 1e-2 and info["component_id"] == "EC223"
            and info["fidelity"].startswith("F0a"), "predict()/get_info interface")

# 8 fast benchmark
t0 = time.time()
for _ in range(1000):
    c.power_W(25.0)
dt = time.time() - t0
assert_true(dt < 1.0, "1000 lookups under 1s (%.3fs)" % dt)

print("\n%d passed, %d failed" % (passed, failed))
sys.exit(0 if failed == 0 else 1)
