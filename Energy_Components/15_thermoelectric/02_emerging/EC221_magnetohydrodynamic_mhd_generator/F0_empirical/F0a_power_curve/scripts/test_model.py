"""EC221 MHD Generator F0a - tests (no pytest)."""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from model import MHDPowerCurve
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


c = MHDPowerCurve()
m = ComponentModel()

# 1 physical: power density positive
assert_true(c.power_density(800.0) > 0.0, "power density positive at 800 m/s")

# 2 rated point matches datasheet
assert_true(abs(c.power_density(800.0) - 40000000.0) < 1.0, "rated 40 MW/m^3 at 800 m/s")

# 3 monotonic increasing with velocity
us = [200, 400, 600, 800, 1000, 1500, 2000]
pds = [c.power_density(u) for u in us]
assert_true(all(b > a for a, b in zip(pds, pds[1:])), "power density monotonic in velocity")

# 4 endpoints
assert_true(abs(c.power_density(200.0) - 2500000.0) < 1.0, "low endpoint at 200 m/s")
assert_true(abs(c.power_density(2000.0) - 250000000.0) < 1.0, "high endpoint at 2000 m/s")

# 5 square-law on breakpoints: 2x velocity ~4x power
assert_true(abs(c.power_density(400.0) / c.power_density(200.0) - 4.0) < 0.01, "square-law: 2x u ~4x power")

# 6 total power = density * volume
p = c.power(800.0)
assert_true(abs(p - 40000000.0 * 1.25) < 1.0, "total power = density * 1.25 m^3")

# 7 predict + get_info interface
out = m.predict({"u": 800.0})
info = m.get_info()
assert_true(out["power_density"] > 0 and info["component_id"] == "EC221"
            and info["fidelity"].startswith("F0a"), "predict()/get_info interface")

# 8 fast benchmark
t0 = time.time()
for _ in range(1000):
    c.power_density(700.0)
dt = time.time() - t0
assert_true(dt < 1.0, "1000 lookups under 1s (%.3fs)" % dt)

print("\n%d passed, %d failed" % (passed, failed))
sys.exit(0 if failed == 0 else 1)
