"""EC222 Betavoltaic Cell F0a - tests (no pytest)."""
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from model import BetavoltaicPowerDecay
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


c = BetavoltaicPowerDecay()
m = ComponentModel()

# 1 physical: power positive, nW scale at BOL
p0 = c.power_nW(0.0)
assert_true(0.0 < p0 < 1e4, "power positive, nW scale at t=0")

# 2 rated point (BOL) matches datasheet
assert_true(abs(p0 - 181.3784) < 1e-3, "BOL power ~181 nW")

# 3 monotonic decreasing with time
ts = [0, 10, 25, 50, 100, 200, 500]
ps = [c.power_nW(t) for t in ts]
assert_true(all(b < a for a, b in zip(ps, ps[1:])), "power monotonic decreasing in time")

# 4 endpoints
assert_true(abs(c.power_nW(500.0) - 5.7074) < 1e-3, "end endpoint at 500 yr")

# 5 half-life check: power at t_half ~ half of BOL (interp approx)
p_half = c.power_nW(c.t_half)
assert_true(abs(p_half - p0 / 2.0) / (p0 / 2.0) < 0.05, "power at one half-life ~ P0/2")

# 6 interpolation between breakpoints
mid = c.power_nW(30.0)
assert_true(c.power_nW(50.0) < mid < c.power_nW(25.0), "interp between 25 and 50 yr")

# 7 predict + get_info interface
out = m.predict({"t_years": 0.0})
info = m.get_info()
assert_true(abs(out["power_nW"] - 181.3784) < 1e-3 and info["component_id"] == "EC222"
            and info["fidelity"].startswith("F0a"), "predict()/get_info interface")

# 8 fast benchmark
t0 = time.time()
for _ in range(1000):
    c.power_nW(40.0)
dt = time.time() - t0
assert_true(dt < 1.0, "1000 lookups under 1s (%.3fs)" % dt)

print("\n%d passed, %d failed" % (passed, failed))
sys.exit(0 if failed == 0 else 1)
