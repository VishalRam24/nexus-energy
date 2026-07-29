"""Tailored tests for the EC210 ED F0a SEC-vs-load lookup (no pytest)."""
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
load_bp = lk.load_bp
sec_bp = lk.sec_bp

# 1. SEC plausible for brackish ED (~0.5-2 kWh/m3)
loads = np.linspace(load_bp[0], load_bp[-1], 25)
sec = lk.sec(loads)
assert_true(np.all((sec > 0.5) & (sec < 2.0)), "SEC within plausible ED band (0.5-2 kWh/m3)")

# 2. SEC monotonic decreasing with load (fixed losses spread)
assert_true(np.all(np.diff(sec) <= 1e-12), "SEC non-increasing with load")

# 3. Rated point matches datasheet at full load
assert_true(abs(lk.sec(1.0) - lk.sec_rated) < 1e-6, "SEC at full load matches rated")

# 4. Endpoints exact
assert_true(abs(lk.sec(load_bp[0]) - sec_bp[0]) < 1e-9, "low endpoint exact")
assert_true(abs(lk.sec(load_bp[-1]) - sec_bp[-1]) < 1e-9, "high endpoint exact")

# 5. Edge clamp
assert_true(abs(lk.sec(0.0) - sec_bp[0]) < 1e-9, "load below min clamps")
assert_true(abs(lk.sec(2.0) - sec_bp[-1]) < 1e-9, "load above max clamps")

# 6. Recovery and permeate balance
assert_true(0.0 < lk.recovery < 1.0, "recovery in (0,1)")
assert_true(abs(lk.permeate_flow(1.0) - lk.capacity_m3_h * lk.recovery) < 1e-6, "permeate = capacity*recovery at full load")

# 7. predict() interface
out = m.predict({"capacity_fraction": 0.5})
assert_true({"sec_kWh_m3", "permeate_flow_m3h", "power_kW", "recovery"} <= set(out), "predict returns expected keys")

# 8. Fast benchmark
t0 = time.time()
for _ in range(1000):
    lk.sec(0.7)
dt = time.time() - t0
assert_true(dt < 1.0, f"1000 lookups fast ({dt*1e3:.1f} ms)")

print(f"\n{passed} passed, {failed} failed")
sys.exit(0 if failed == 0 else 1)
