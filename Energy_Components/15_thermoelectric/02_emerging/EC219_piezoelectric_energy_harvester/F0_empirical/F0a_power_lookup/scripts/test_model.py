"""EC219 Piezoelectric Energy Harvester F0a - tests (no pytest)."""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from model import PiezoPowerLookup
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


c = PiezoPowerLookup()
m = ComponentModel()

# 1 physical: power positive and small (mW scale)
p1g = c.power_mW(9.81)
assert_true(0.0 < p1g < 100.0, "power positive, mW scale at 1g")

# 2 rated point matches datasheet
assert_true(abs(p1g - 0.5) < 1e-3, "rated 0.5 mW at 1 g")

# 3 monotonic increasing with acceleration
accs = [0.5, 1, 2, 5, 9.81, 20, 50]
ps = [c.power_mW(a) for a in accs]
assert_true(all(b > a for a, b in zip(ps, ps[1:])), "power monotonic in acceleration")

# 4 endpoints
assert_true(abs(c.power_mW(0.5) - 0.0013) < 1e-3, "low endpoint at 0.5 m/s^2")
assert_true(abs(c.power_mW(50.0) - 12.9889) < 1e-3, "high endpoint at 50 m/s^2")

# 5 roughly square-law: doubling acceleration ~4x power
assert_true(abs(c.power_mW(2.0) / c.power_mW(1.0) - 4.0) < 0.2, "square-law: 2x accel ~4x power")

# 6 interpolation between breakpoints
mid = c.power_mW(3.0)
assert_true(c.power_mW(2.0) < mid < c.power_mW(5.0), "interp between 2 and 5 m/s^2")

# 7 predict + get_info interface
out = m.predict({"acceleration": 9.81})
info = m.get_info()
assert_true(abs(out["power_mW"] - 0.5) < 1e-3 and info["component_id"] == "EC219"
            and info["fidelity"].startswith("F0a"), "predict()/get_info interface")

# 8 fast benchmark
t0 = time.time()
for _ in range(1000):
    c.power_mW(7.0)
dt = time.time() - t0
assert_true(dt < 1.0, "1000 lookups under 1s (%.3fs)" % dt)

print("\n%d passed, %d failed" % (passed, failed))
sys.exit(0 if failed == 0 else 1)
