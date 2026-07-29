"""F0a tests for EC124 — custom harness, no pytest."""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from predict import ComponentModel  # noqa: E402

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
rated = m.p["rte_rated"]["value"]

# 1. RTE in physical bounds across the curve
fracs = np.linspace(0.1, 1.0, 25)
rtes = m.curve.round_trip_efficiency(fracs)
assert_true(np.all((rtes > 0.0) & (rtes < 1.0)), "0 < RTE < 1 across part-load range")

# 2. Monotonic non-decreasing toward rated
assert_true(np.all(np.diff(rtes) >= -1e-9), "RTE monotonically non-decreasing with load")

# 3. Rated point matches datasheet
assert_true(abs(m.predict({"power_fraction": 1.0})["round_trip_efficiency"] - rated) < 1e-9,
            "RTE at full load matches datasheet rated value")

# 4. Endpoints / clamping on out-of-range input
lo = m.predict({"power_fraction": 0.0})["round_trip_efficiency"]
hi = m.predict({"power_fraction": 5.0})["round_trip_efficiency"]
assert_true(lo == m.curve.rte[0] and hi == rated, "Out-of-range power fraction clamps to endpoints")

# 5. Self-discharge retains <= 1 and energy_out consistent
out = m.predict({"power_fraction": 1.0, "energy_in_kwh": 1000.0, "idle_hours": 10.0})
assert_true(out["retained_fraction"] <= 1.0 + 1e-12, "Retained fraction <= 1")
assert_true(abs(out["energy_out_kwh"] - 1000.0 * out["round_trip_efficiency"] * out["retained_fraction"]) < 1e-6,
            "energy_out = energy_in * RTE * retained")

# 6. predict() interface returns required keys
keys = set(m.predict({"power_fraction": 0.5}).keys())
assert_true({"round_trip_efficiency", "retained_fraction"}.issubset(keys), "predict() returns required keys")

# 7. get_info() interface
info = m.get_info()
assert_true(info["component_id"] == "EC124" and info["version"] == "1.0.0", "get_info() metadata correct")

# 8. Fast benchmark
t0 = time.time()
for _ in range(1000):
    m.predict({"power_fraction": 0.7})
dt = time.time() - t0
assert_true(dt < 1.0, f"1000 predictions fast ({dt*1e3:.1f} ms)")

print(f"\n{passed} passed, {failed} failed")
sys.exit(0 if failed == 0 else 1)
