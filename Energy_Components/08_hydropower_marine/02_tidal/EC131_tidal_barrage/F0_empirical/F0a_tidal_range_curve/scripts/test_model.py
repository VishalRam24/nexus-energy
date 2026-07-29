"""EC131 F0a — tests (no pytest)."""
import sys, time
import numpy as np
from predict import ComponentModel

passed = failed = 0


def assert_true(cond, msg):
    global passed, failed
    if cond:
        passed += 1
        print(f"  ✓ {msg}")
    else:
        failed += 1
        print(f"  ✗ {msg}")


m = ComponentModel()
info = m.get_info()

# 1. rated point ~44.33 MW at h=8
P = float(m.predict({"tidal_range_amplitude_m": 8.0})["mean_power_mw"])
assert_true(abs(P - 44.33) < 0.1, f"design power ~44.33 MW (got {P:.2f})")

# 2. power non-negative
P0 = float(m.predict({"tidal_range_amplitude_m": 0.5})["mean_power_mw"])
assert_true(P0 == 0.0, f"below h_min power=0 (got {P0:.2f})")

# 3. monotonic increasing with range
hs = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
Ps = m.predict({"tidal_range_amplitude_m": hs})["mean_power_mw"]
assert_true(np.all(np.diff(Ps) > 0), "power monotonic increasing in h")

# 4. roughly quadratic: doubling h ~4x power
P4 = float(m.predict({"tidal_range_amplitude_m": 4.0})["mean_power_mw"])
P8 = float(m.predict({"tidal_range_amplitude_m": 8.0})["mean_power_mw"])
assert_true(abs(P8 / P4 - 4.0) < 0.1, f"P(2h)/P(h)~4 (got {P8/P4:.2f})")

# 5. endpoint at table max h=12
P12 = float(m.predict({"tidal_range_amplitude_m": 12.0})["mean_power_mw"])
assert_true(abs(P12 - 99.74) < 0.1, f"endpoint h=12 -> 99.74 MW (got {P12:.2f})")

# 6. CF at design ~1
cf = float(m.predict({"tidal_range_amplitude_m": 8.0})["capacity_factor"])
assert_true(abs(cf - 1.0) < 0.01, f"CF~1 at design (got {cf:.3f})")

# 7. metadata
assert_true(info["component_id"] == "EC131" and info["version"] == "1.0.0", "get_info metadata correct")

# 8. fast benchmark
h = np.linspace(0, 12, 1000)
t0 = time.perf_counter()
m.predict({"tidal_range_amplitude_m": h})
dt = time.perf_counter() - t0
assert_true(dt < 0.5, f"1000 predictions fast ({dt*1e3:.2f} ms)")

print(f"\n{passed} passed, {failed} failed")
sys.exit(0 if failed == 0 else 1)
