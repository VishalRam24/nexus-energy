"""Tailored tests for the EC212 MSF F0a GOR lookup (no pytest)."""
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
T_bp = lk.T_top_bp
gor_bp = lk.gor_bp

# 1. GOR plausible for MSF (typically 4-12)
temps = np.linspace(T_bp[0], T_bp[-1], 25)
gor = lk.gor(temps)
assert_true(np.all((gor > 4.0) & (gor < 12.0)), "GOR in plausible MSF band (4-12)")

# 2. GOR monotonic increasing with top brine temperature
assert_true(np.all(np.diff(gor) > 0), "GOR monotonic increasing with T_top")

# 3. Rated GOR matches datasheet
assert_true(abs(lk.gor(lk.T_top_rated) - lk.gor_rated) < 1e-6, "GOR at rated T_top matches datasheet")

# 4. Endpoints exact
assert_true(abs(lk.gor(T_bp[0]) - gor_bp[0]) < 1e-9, "low endpoint exact")
assert_true(abs(lk.gor(T_bp[-1]) - gor_bp[-1]) < 1e-9, "high endpoint exact")

# 5. Edge clamp
assert_true(abs(lk.gor(50.0) - gor_bp[0]) < 1e-9, "T_top below min clamps")
assert_true(abs(lk.gor(200.0) - gor_bp[-1]) < 1e-9, "T_top above max clamps")

# 6. Thermal SEC from GOR positive and decreasing with T_top
tsec = lk.thermal_sec_from_gor(temps)
assert_true(np.all(tsec > 0) & np.all(np.diff(tsec) < 0), "thermal SEC positive and falls with T_top")

# 7. predict() interface and distillate balance
out = m.predict({"T_top_brine_C": 110.0, "capacity_fraction": 0.5})
assert_true({"gor", "thermal_sec_kJ_kg", "sec_elec_kWh_m3", "distillate_flow_m3h", "elec_power_kW"} <= set(out), "predict returns expected keys")
assert_true(abs(out["distillate_flow_m3h"] - lk.capacity_m3_h * 0.5) < 1e-6, "distillate = capacity*load")

# 8. Fast benchmark
t0 = time.time()
for _ in range(1000):
    lk.gor(108.0)
dt = time.time() - t0
assert_true(dt < 1.0, f"1000 lookups fast ({dt*1e3:.1f} ms)")

print(f"\n{passed} passed, {failed} failed")
sys.exit(0 if failed == 0 else 1)
