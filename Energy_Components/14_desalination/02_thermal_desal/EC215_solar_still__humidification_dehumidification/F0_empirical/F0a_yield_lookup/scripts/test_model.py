"""Tailored tests for the EC215 solar still / HDH F0a yield lookup (no pytest)."""
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
lk = m.lookup
irr_bp = lk.irr_bp
y_bp = lk.yield_bp

# 1. Productivity plausible for solar stills (typically 1-12 L/m2/day)
irrs = np.linspace(irr_bp[0], irr_bp[-1], 25)
prod = lk.productivity(irrs)
assert_true(np.all((prod > 1.0) & (prod < 12.0)), "productivity in plausible band (1-12 L/m2/day)")

# 2. Productivity monotonic increasing with irradiance
assert_true(np.all(np.diff(prod) > 0), "productivity monotonic increasing with irradiance")

# 3. Rated point matches datasheet (5 L/m2/day at 800 W/m2)
assert_true(abs(lk.productivity(lk.irr_rated) - lk.yield_rated) < 1e-6, "productivity at rated irradiance matches datasheet")

# 4. Endpoints exact
assert_true(abs(lk.productivity(irr_bp[0]) - y_bp[0]) < 1e-9, "low endpoint exact")
assert_true(abs(lk.productivity(irr_bp[-1]) - y_bp[-1]) < 1e-9, "high endpoint exact")

# 5. Edge clamp
assert_true(abs(lk.productivity(100.0) - y_bp[0]) < 1e-9, "irradiance below min clamps")
assert_true(abs(lk.productivity(2000.0) - y_bp[-1]) < 1e-9, "irradiance above max clamps")

# 6. Daily yield scales with area
assert_true(abs(lk.daily_yield(800.0, 10.0) - lk.productivity(800.0) * 10.0) < 1e-6, "daily yield = productivity*area")
assert_true(lk.daily_yield(800.0, 20.0) > lk.daily_yield(800.0, 10.0), "more area gives more yield")

# 7. predict() interface and HDH GOR physical
out = m.predict({"solar_irradiance_W_m2": 800.0})
assert_true({"productivity_L_m2_day", "daily_yield_L_day", "gor_hdh", "sec_solar_kWh_m3"} <= set(out), "predict returns expected keys")
assert_true(lk.gor_min <= lk.gor_hdh <= lk.gor_max, "HDH GOR within 1-3 band")

# 8. Fast benchmark
t0 = time.time()
for _ in range(1000):
    lk.productivity(750.0)
dt = time.time() - t0
assert_true(dt < 1.0, f"1000 lookups fast ({dt*1e3:.1f} ms)")

print(f"\n{passed} passed, {failed} failed")
sys.exit(0 if failed == 0 else 1)
