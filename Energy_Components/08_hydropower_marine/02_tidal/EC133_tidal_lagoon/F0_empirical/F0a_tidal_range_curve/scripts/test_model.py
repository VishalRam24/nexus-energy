"""EC133 F0a — tests (no pytest)."""
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

# 1. design point ~13.093 MW at h=4.5
P = float(m.predict({"tidal_range_amplitude_m": 4.5})["mean_power_mw"])
assert_true(abs(P - 13.093) < 0.05, f"design power ~13.093 MW (got {P:.3f})")

# 2. below h_min (0.75) power=0
P0 = float(m.predict({"tidal_range_amplitude_m": 0.5})["mean_power_mw"])
assert_true(P0 == 0.0, f"below h_min power=0 (got {P0:.3f})")

# 3. monotonic increasing
hs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
Ps = m.predict({"tidal_range_amplitude_m": hs})["mean_power_mw"]
assert_true(np.all(np.diff(Ps) > 0), "power monotonic increasing in h")

# 4. roughly quadratic 2h->~4x
P2 = float(m.predict({"tidal_range_amplitude_m": 2.0})["mean_power_mw"])
P4 = float(m.predict({"tidal_range_amplitude_m": 4.0})["mean_power_mw"])
assert_true(abs(P4 / P2 - 4.0) < 0.1, f"P(2h)/P(h)~4 (got {P4/P2:.2f})")

# 5. endpoint h=6 -> 23.276
P6 = float(m.predict({"tidal_range_amplitude_m": 6.0})["mean_power_mw"])
assert_true(abs(P6 - 23.276) < 0.05, f"endpoint h=6 -> 23.276 MW (got {P6:.3f})")

# 6. CF at design ~1
cf = float(m.predict({"tidal_range_amplitude_m": 4.5})["capacity_factor"])
assert_true(abs(cf - 1.0) < 0.01, f"CF~1 at design (got {cf:.3f})")

# 7. metadata
assert_true(info["component_id"] == "EC133" and info["version"] == "1.0.0", "get_info metadata correct")

# 8. fast benchmark
h = np.linspace(0, 6, 1000)
t0 = time.perf_counter()
m.predict({"tidal_range_amplitude_m": h})
dt = time.perf_counter() - t0
assert_true(dt < 0.5, f"1000 predictions fast ({dt*1e3:.2f} ms)")

print(f"\n{passed} passed, {failed} failed")
sys.exit(0 if failed == 0 else 1)
