"""F0a tests for EC188 — custom harness, no pytest."""
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
        print(f"  \u2713 {msg}")
    else:
        failed += 1
        print(f"  \u2717 {msg}")


m = ComponentModel()

# 1. RTE in physical bounds
pf = np.linspace(0.1, 1.0, 25)
rte = m.curve.lookup(pf)
assert_true(np.all((rte > 0.0) & (rte < 1.0)), "0 < RTE < 1 across part-load range")

# 2. RTE monotonic non-decreasing toward rated
assert_true(np.all(np.diff(rte) >= -1e-9), "RTE non-decreasing with load")

# 3. Rated RTE matches datasheet (eta_conv^2)
assert_true(abs(m.predict({"power_fraction": 1.0})["round_trip_efficiency"] - m.rte_rated) < 1e-9,
            "rated RTE matches eta_converter^2")

# 4. Stored energy at I_max ~ 10 MJ (E=0.5*L*I^2)
e = m.predict({"power_fraction": 1.0, "current_a": m.I_max})["stored_energy_mj"]
assert_true(9.5 < e < 10.5, f"E(I_max) ~ 10 MJ (got {e:.2f})")

# 5. Stored energy scales with I^2
e_half = m.stored_energy_mj(m.I_max / 2.0)
assert_true(abs(e_half - e / 4.0) < 1e-6, "stored energy scales as I^2")

# 6. Endpoint clamping
assert_true(m.predict({"power_fraction": 9.0})["round_trip_efficiency"] == m.curve.y[-1],
            "out-of-range power fraction clamps to endpoint")

# 7. get_info() metadata
info = m.get_info()
assert_true(info["component_id"] == "EC188" and info["version"] == "1.0.0", "get_info() metadata correct")

# 8. Fast benchmark
t0 = time.time()
for _ in range(1000):
    m.predict({"power_fraction": 0.7})
dt = time.time() - t0
assert_true(dt < 1.0, f"1000 predictions fast ({dt*1e3:.1f} ms)")

print(f"\n{passed} passed, {failed} failed")
sys.exit(0 if failed == 0 else 1)
