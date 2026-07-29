"""EC220 TENG F0a - tests (no pytest)."""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from model import TENGPowerLookup
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


c = TENGPowerLookup()
m = ComponentModel()

# 1 physical: power positive, sub-mW scale
p3 = c.power_mW(3.0)
assert_true(0.0 < p3 < 1.0, "power positive, sub-mW scale at 3 Hz")

# 2 rated point matches datasheet
assert_true(abs(p3 - 0.021177) < 1e-5, "rated ~21 uW at 3 Hz")

# 3 monotonic increasing with frequency
fs = [0.1, 0.5, 1, 3, 10, 30, 100]
ps = [c.power_mW(f) for f in fs]
assert_true(all(b > a for a, b in zip(ps, ps[1:])), "power monotonic in frequency")

# 4 endpoints
assert_true(abs(c.power_mW(0.1) - 0.000706) < 1e-5, "low endpoint at 0.1 Hz")
assert_true(abs(c.power_mW(100.0) - 0.705896) < 1e-5, "high endpoint at 100 Hz")

# 5 linear in frequency: 10x freq ~10x power
assert_true(abs(c.power_mW(10.0) / c.power_mW(1.0) - 10.0) < 0.1, "linear: 10x freq ~10x power")

# 6 interpolation between breakpoints
mid = c.power_mW(2.0)
assert_true(c.power_mW(1.0) < mid < c.power_mW(3.0), "interp between 1 and 3 Hz")

# 7 predict + get_info interface
out = m.predict({"frequency": 3.0})
info = m.get_info()
assert_true(abs(out["power_mW"] - 0.021177) < 1e-5 and info["component_id"] == "EC220"
            and info["fidelity"].startswith("F0a"), "predict()/get_info interface")

# 8 fast benchmark
t0 = time.time()
for _ in range(1000):
    c.power_mW(5.0)
dt = time.time() - t0
assert_true(dt < 1.0, "1000 lookups under 1s (%.3fs)" % dt)

print("\n%d passed, %d failed" % (passed, failed))
sys.exit(0 if failed == 0 else 1)
