"""EC218 Thermionic Converter F0a - tests (no pytest)."""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from model import ThermionicPowerCurve
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


c = ThermionicPowerCurve()
m = ComponentModel()

# 1 physical: power density positive
assert_true(c.power_density(1700.0) > 0.0, "power density positive at 1700K")

# 2 rated point matches datasheet
assert_true(abs(c.power_density(1700.0) - 2040182.3) < 1.0, "rated ~204 W/cm^2 at 1700K")

# 3 monotonic increasing with emitter temp
ts = [1200, 1400, 1600, 1700, 1800, 2000]
pds = [c.power_density(t) for t in ts]
assert_true(all(b > a for a, b in zip(pds, pds[1:])), "power density monotonic in Te")

# 4 endpoints
assert_true(abs(c.power_density(1200.0) - 3440.0) < 1.0, "low endpoint at 1200K")
assert_true(abs(c.power_density(2000.0) - 21889616.0) < 1.0, "high endpoint at 2000K")

# 5 interpolation between breakpoints
mid = c.power_density(1500.0)
assert_true(c.power_density(1400.0) < mid < c.power_density(1600.0), "interp between 1400K and 1600K")

# 6 total power = density * area
p = c.power(1700.0)
assert_true(abs(p - 2040182.3 * 1e-4) < 1e-3, "total power = density * 1cm^2 area")

# 7 predict + get_info interface
out = m.predict({"T_emitter": 1700.0})
info = m.get_info()
assert_true(out["power_density"] > 0 and info["component_id"] == "EC218"
            and info["fidelity"].startswith("F0a"), "predict()/get_info interface")

# 8 fast benchmark
t0 = time.time()
for _ in range(1000):
    c.power_density(1650.0)
dt = time.time() - t0
assert_true(dt < 1.0, "1000 lookups under 1s (%.3fs)" % dt)

print("\n%d passed, %d failed" % (passed, failed))
sys.exit(0 if failed == 0 else 1)
