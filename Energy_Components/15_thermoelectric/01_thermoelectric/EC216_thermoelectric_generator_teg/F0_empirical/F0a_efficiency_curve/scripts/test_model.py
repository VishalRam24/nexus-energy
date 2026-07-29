"""EC216 TEG F0a - tests (no pytest)."""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from model import TEGEfficiencyCurve
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


c = TEGEfficiencyCurve()
m = ComponentModel()

# 1 physical bounds
e200 = c.efficiency(200.0)
assert_true(0.0 < e200 < 1.0, "efficiency at 200C in (0,1)")

# 2 rated point matches datasheet
assert_true(abs(e200 - 0.0724) < 1e-3, "rated efficiency ~7.24% at 200C")

# 3 monotonic increasing with T_hot
ts = [50, 100, 150, 200, 250, 300]
effs = [c.efficiency(t) for t in ts]
assert_true(all(b > a for a, b in zip(effs, effs[1:])), "efficiency monotonic in T_hot")

# 4 endpoints
assert_true(abs(c.efficiency(50.0) - 0.0109) < 1e-3, "low endpoint at 50C")
assert_true(abs(c.efficiency(300.0) - 0.1004) < 1e-3, "high endpoint at 300C")

# 5 interpolation between breakpoints
mid = c.efficiency(125.0)
assert_true(c.efficiency(100.0) < mid < c.efficiency(150.0), "interp between 100C and 150C")

# 6 predict interface
out = m.predict({"T_hot": 200.0, "T_cold": 30.0})
assert_true(abs(out["efficiency"] - 0.0724) < 1e-3 and out["delta_T"] == 170.0, "predict() interface")

# 7 get_info
info = m.get_info()
assert_true(info["component_id"] == "EC216" and info["fidelity"].startswith("F0a"), "get_info metadata")

# 8 fast benchmark
t0 = time.time()
for _ in range(1000):
    c.efficiency(180.0)
dt = time.time() - t0
assert_true(dt < 1.0, "1000 lookups under 1s (%.3fs)" % dt)

print("\n%d passed, %d failed" % (passed, failed))
sys.exit(0 if failed == 0 else 1)
