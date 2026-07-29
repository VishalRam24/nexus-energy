"""Tailored tests for the EC209 RO F0a SEC-vs-recovery curve (no pytest)."""
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
rec_bp = c.recovery_bp
sec_bp = c.sec_bp

# 1. SEC physically plausible for SWRO (thermo min ~1, practical 2-6 kWh/m3)
recs = np.linspace(rec_bp[0], rec_bp[-1], 25)
sec = c.sec(recs)
assert_true(np.all((sec > 1.0) & (sec < 7.0)), "SEC within plausible SWRO band (1-7 kWh/m3)")

# 2. SEC monotonic increasing with recovery
assert_true(np.all(np.diff(sec) > 0), "SEC monotonic increasing with recovery")

# 3. Rated point matches datasheet
assert_true(abs(c.sec(c.recovery_rated) - c.sec_rated) < 1e-6, "SEC at rated recovery matches datasheet")

# 4. Endpoints reproduced exactly
assert_true(abs(c.sec(rec_bp[0]) - sec_bp[0]) < 1e-9, "low endpoint exact")
assert_true(abs(c.sec(rec_bp[-1]) - sec_bp[-1]) < 1e-9, "high endpoint exact")

# 5. Edge inputs clamp
assert_true(abs(c.sec(0.0) - sec_bp[0]) < 1e-9, "recovery below min clamps")
assert_true(abs(c.sec(1.0) - sec_bp[-1]) < 1e-9, "recovery above max clamps")

# 6. predict() interface and mass balance
out = m.predict({"recovery": 0.45, "feed_flow_m3h": 200.0})
assert_true({"sec_kWh_m3", "permeate_flow_m3h", "power_kW", "permeate_salinity_g_L"} <= set(out), "predict returns expected keys")
assert_true(abs(out["permeate_flow_m3h"] - 200.0 * 0.45) < 1e-6, "permeate = feed*recovery")
assert_true(out["permeate_salinity_g_L"] < m.S_feed, "permeate fresher than feed")

# 7. Fast benchmark
t0 = time.time()
for _ in range(1000):
    c.sec(0.5)
dt = time.time() - t0
assert_true(dt < 1.0, f"1000 lookups fast ({dt*1e3:.1f} ms)")

print(f"\n{passed} passed, {failed} failed")
sys.exit(0 if failed == 0 else 1)
