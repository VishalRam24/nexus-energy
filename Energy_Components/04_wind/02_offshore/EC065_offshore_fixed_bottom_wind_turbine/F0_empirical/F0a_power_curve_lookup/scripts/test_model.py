"""Tailored tests for EC065 F0a empirical power-curve lookup (no pytest)."""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from predict import ComponentModel  # noqa: E402

passed = 0
failed = 0


def assert_true(cond, msg):
    global passed, failed
    if cond:
        passed += 1
        print(f"  \u2713 {msg}")
    else:
        failed += 1
        print(f"  \u2717 {msg}")


m = ComponentModel()
info = m.get_info()
rated = info["rated_power_kw"]
ci = m.curve.cut_in
co = m.curve.cut_out
vr = m.curve.rated
pen = m.curve.motion_penalty

print(f"Testing EC065 F0a power-curve lookup")

# 1. zero below cut-in
assert_true(m.predict({"wind_speed": max(0.0, ci - 0.5)})["power_kw"] == 0.0,
            "power is zero below cut-in")

# 2. zero above cut-out
assert_true(m.predict({"wind_speed": co + 1.0})["power_kw"] == 0.0,
            "power is zero above cut-out")

# 3. power within physical bounds across the curve
ok = True
for v in [x * 0.5 for x in range(0, 61)]:
    p = m.predict({"wind_speed": v})["power_kw"]
    if p < -1e-9 or p > rated + 1e-6:
        ok = False
assert_true(ok, "power stays within [0, rated] over 0-30 m/s")

# 4. monotonic non-decreasing from cut-in up to rated speed
ws = [ci + 0.001] + [ci + 0.5 * k for k in range(1, int((vr - ci) * 2) + 1)]
ps = [m.predict({"wind_speed": v})["power_kw"] for v in ws]
mono = all(ps[i + 1] >= ps[i] - 1e-6 for i in range(len(ps) - 1))
assert_true(mono, "power is monotonic non-decreasing from cut-in to rated speed")

# 5. plateau at rated between rated speed and cut-out
expected_plateau = rated * (1.0 - pen)
p_mid = m.predict({"wind_speed": (vr + co) / 2.0})["power_kw"]
assert_true(abs(p_mid - expected_plateau) <= max(1.0, 0.02 * rated),
            "power plateaus near rated between rated speed and cut-out")

# 6. value at datasheet rated point matches rated power (within penalty)
p_rated = m.predict({"wind_speed": 14.0})["power_kw"]
assert_true(abs(p_rated - expected_plateau) <= max(1.0, 0.02 * rated),
            "power at datasheet rated point matches rated value")

# 7. capacity factor in [0,1]
cf = m.predict({"wind_speed": vr})["capacity_factor"]
assert_true(0.0 <= cf <= 1.0 + 1e-9, "capacity factor in [0,1]")

# 8. predict() interface + fast benchmark
out = m.predict({"wind_speed": 9.0})
iface = "power_kw" in out and "capacity_factor" in out
t0 = time.time()
for _ in range(1000):
    m.predict({"wind_speed": 9.0})
dt = time.time() - t0
assert_true(iface and dt < 1.0,
            f"predict() interface ok and 1000 calls fast ({dt*1e3:.1f} ms)")

print(f"\n{passed} passed, {failed} failed")
sys.exit(0 if failed == 0 else 1)
