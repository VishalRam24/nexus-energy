"""Tailored tests for the EC109 F0a efficiency lookup (no pytest)."""
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
u = m.params["unit"]
eta_rated = u["eta_rated"]["value"]
plr_bp = np.asarray(u["plr_breakpoints"]["value"])
eta_bp = np.asarray(u["eta_breakpoints"]["value"])

# 1. Efficiency in physical bounds across the operating range
plrs = np.linspace(plr_bp[0], plr_bp[-1], 25)
etas = m.lookup.efficiency(plrs)
assert_true(np.all((etas > 0) & (etas < 1)), "efficiency in (0,1) across range")

# 2. Rated point matches datasheet efficiency
assert_true(abs(m.lookup.efficiency(1.0) - eta_rated) < 1e-6, "eta at PLR=1 equals rated")

# 3. Monotonic increasing with PLR (part-load penalty)
assert_true(np.all(np.diff(etas) > 0), "efficiency monotonic increasing with PLR")

# 4. Endpoints of the table reproduced exactly
assert_true(abs(m.lookup.efficiency(plr_bp[0]) - eta_bp[0]) < 1e-9, "low endpoint exact")
assert_true(abs(m.lookup.efficiency(plr_bp[-1]) - eta_bp[-1]) < 1e-9, "high endpoint exact")

# 5. Edge inputs clamp (below PLR_min / above 1)
assert_true(abs(m.lookup.efficiency(0.0) - eta_bp[0]) < 1e-9, "PLR below min clamps to floor")
assert_true(abs(m.lookup.efficiency(2.0) - eta_bp[-1]) < 1e-9, "PLR above 1 clamps to rated")

# 6. Ambient derate lowers efficiency above reference temperature
assert_true(m.lookup.efficiency(1.0, 40.0) < m.lookup.efficiency(1.0, 15.0), "hot ambient derates efficiency")

# 7. predict() interface returns expected keys and consistent fuel power
out = m.predict({"part_load_ratio": 0.8, "ambient_temp_c": 15.0})
pkey = next(k for k in out if k.startswith("power_output_"))
fkey = next(k for k in out if k.startswith("fuel_power_"))
assert_true("efficiency" in out and pkey and fkey, "predict returns expected keys")
assert_true(abs(out[pkey] / out[fkey] - out["efficiency"]) < 1e-9, "fuel/power consistent with eta")

# 8. Fast benchmark
t0 = time.time()
for _ in range(1000):
    m.lookup.efficiency(0.7, 20.0)
dt = time.time() - t0
assert_true(dt < 1.0, f"1000 lookups fast ({dt*1e3:.1f} ms)")

print(f"\n{passed} passed, {failed} failed")
sys.exit(0 if failed == 0 else 1)
