"""Tailored tests for the EC211 FO F0a SEC lookup (no pytest)."""
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

# 1. Rated total SEC matches datasheet (membrane + regen at full load)
assert_true(abs(lk.sec(1.0, True) - lk.sec_total) < 1e-6, "full-load SEC w/ regen matches rated total")

# 2. Membrane-only SEC equals datasheet membrane term at full load
assert_true(abs(lk.sec(1.0, False) - lk.sec_membrane) < 1e-6, "full-load membrane-only matches sec_membrane")

# 3. Regen adds exactly the regen term
assert_true(abs(lk.sec(1.0, True) - lk.sec(1.0, False) - lk.sec_regen) < 1e-9, "regen term adds correctly")

# 4. SEC physically positive and bounded
loads = np.linspace(lk.load_bp[0], lk.load_bp[-1], 25)
sec = lk.sec(loads, True)
assert_true(np.all((sec > 0.0) & (sec < 6.0)), "SEC in plausible band (0-6 kWh/m3)")

# 5. Membrane factor monotonic decreasing with load and endpoints exact
mf = lk.membrane_factor(loads)
assert_true(np.all(np.diff(mf) <= 1e-12), "membrane factor non-increasing with load")
assert_true(abs(lk.membrane_factor(1.0) - 1.0) < 1e-9, "membrane factor = 1 at full load")

# 6. Edge clamp
assert_true(abs(lk.membrane_factor(0.0) - lk.mem_factor_bp[0]) < 1e-9, "load below min clamps")
assert_true(abs(lk.membrane_factor(2.0) - lk.mem_factor_bp[-1]) < 1e-9, "load above max clamps")

# 7. predict() interface and recovery balance
out = m.predict({"capacity_fraction": 1.0})
assert_true({"sec_kWh_m3", "permeate_flow_m3h", "power_kW", "include_regen"} <= set(out), "predict returns expected keys")
assert_true(abs(out["permeate_flow_m3h"] - lk.capacity_m3_h * lk.recovery) < 1e-6, "permeate = capacity*recovery")

# 8. Fast benchmark
t0 = time.time()
for _ in range(1000):
    lk.sec(0.7, True)
dt = time.time() - t0
assert_true(dt < 1.0, f"1000 lookups fast ({dt*1e3:.1f} ms)")

print(f"\n{passed} passed, {failed} failed")
sys.exit(0 if failed == 0 else 1)
