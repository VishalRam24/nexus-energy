"""EC217 TEC F0a - tests (no pytest)."""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from model import TECCopMap
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


c = TECCopMap()
m = ComponentModel()

# 1 physical: COP > 1 at small lift
assert_true(c.cop(10.0) > 1.0, "COP > 1 at small lift (10K)")

# 2 rated point matches datasheet
assert_true(abs(c.cop(30.0) - 0.7529) < 1e-3, "rated COP ~0.75 at 30K lift")

# 3 monotonic decreasing with deltaT
dts = [5, 10, 20, 30, 40, 50, 60, 67]
cops = [c.cop(d) for d in dts]
assert_true(all(b < a for a, b in zip(cops, cops[1:])), "COP monotonic decreasing in deltaT")

# 4 endpoints
assert_true(abs(c.cop(5.0) - 7.3471) < 1e-3, "low-lift endpoint at 5K")
assert_true(c.cop(67.0) >= 0.0 and c.cop(67.0) < 0.1, "COP -> ~0 at dT_max")

# 5 interpolation between breakpoints
mid = c.cop(25.0)
assert_true(c.cop(30.0) < mid < c.cop(20.0), "interp between 20K and 30K")

# 6 predict interface (Th/Tc form)
out = m.predict({"Th": 300.0, "Tc": 270.0})
assert_true(abs(out["COP"] - 0.7529) < 1e-3 and out["delta_T"] == 30.0, "predict() Th/Tc interface")

# 7 get_info
info = m.get_info()
assert_true(info["component_id"] == "EC217" and info["fidelity"].startswith("F0a"), "get_info metadata")

# 8 fast benchmark
t0 = time.time()
for _ in range(1000):
    c.cop(25.0)
dt = time.time() - t0
assert_true(dt < 1.0, "1000 lookups under 1s (%.3fs)" % dt)

print("\n%d passed, %d failed" % (passed, failed))
sys.exit(0 if failed == 0 else 1)
