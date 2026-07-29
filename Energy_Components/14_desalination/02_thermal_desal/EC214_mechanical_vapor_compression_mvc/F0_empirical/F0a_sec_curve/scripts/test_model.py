"""Tailored tests for the EC214 MVC F0a SEC-vs-CR curve (no pytest)."""
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from predict import ComponentModel

passed = 0
failed = 0


def assert_true(cond, msg):
    global passed, failed
    if cond:
        passed += 1
        print(f"  ✓ {msg}")
    else:
        failed += 1
        print(f"  ✗ {msg}")


m = ComponentModel()
c = m.curve
cr_bp = c.cr_bp
sec_bp = c.sec_bp

# 1. SEC within the datasheet 8-12 kWh/m3 band across CR range
crs = np.linspace(cr_bp[0], cr_bp[-1], 25)
sec = c.sec(crs)
assert_true(np.all((sec >= c.sec_min - 1e-9) & (sec <= c.sec_max + 1e-9)), "SEC within 8-12 kWh/m3 band")

# 2. SEC monotonic increasing with compression ratio
assert_true(np.all(np.diff(sec) > 0), "SEC monotonic increasing with CR")

# 3. Rated point matches datasheet
assert_true(abs(c.sec(c.cr_rated) - c.sec_rated) < 1e-6, "SEC at rated CR matches datasheet (10 kWh/m3)")

# 4. Endpoints exact
assert_true(abs(c.sec(cr_bp[0]) - sec_bp[0]) < 1e-9, "low endpoint exact (8 kWh/m3)")
assert_true(abs(c.sec(cr_bp[-1]) - sec_bp[-1]) < 1e-9, "high endpoint exact (12 kWh/m3)")

# 5. Edge clamp
assert_true(abs(c.sec(1.0) - sec_bp[0]) < 1e-9, "CR below min clamps")
assert_true(abs(c.sec(2.0) - sec_bp[-1]) < 1e-9, "CR above max clamps")

# 6. predict() interface and distillate balance
out = m.predict({"compression_ratio": 1.2, "capacity_fraction": 0.5})
assert_true({"sec_kWh_m3", "distillate_flow_m3h", "elec_power_kW", "recovery"} <= set(out), "predict returns expected keys")
assert_true(abs(out["distillate_flow_m3h"] - c.capacity_m3_h * 0.5) < 1e-6, "distillate = capacity*load")
assert_true(abs(out["elec_power_kW"] - out["sec_kWh_m3"] * out["distillate_flow_m3h"]) < 1e-6, "power = SEC*flow")

# 7. Recovery physical
assert_true(0.0 < c.recovery < 1.0, "recovery in (0,1)")

# 8. Fast benchmark
t0 = time.time()
for _ in range(1000):
    c.sec(1.3)
dt = time.time() - t0
assert_true(dt < 1.0, f"1000 lookups fast ({dt*1e3:.1f} ms)")

print(f"\n{passed} passed, {failed} failed")
sys.exit(0 if failed == 0 else 1)
