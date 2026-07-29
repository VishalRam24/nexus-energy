"""F0a tests for EC174 — custom harness, no pytest."""
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

# 1. Ratio error within class limit across rated range
lf = np.linspace(0.05, 2.0, 25)
errs = m.curve.lookup(lf)
assert_true(np.all(np.abs(errs) <= m.max_err + 1e-9), "|ratio error| <= class 0.2 limit across range")

# 2. Error positive (physical) and bounded
assert_true(np.all((errs > 0.0) & (errs < 1.0)), "0 < ratio error < 1% (physical bounds)")

# 3. Best (lowest) error near rated 100% load
assert_true(m.curve.lookup(1.0) <= m.curve.lookup(0.05),
            "error at rated load <= error at low load")

# 4. Endpoint clamping
assert_true(m.predict({"load_fraction": 10.0})["ratio_error_pct"] == m.curve.y[-1],
            "out-of-range load clamps to endpoint")

# 5. CT ratio correct
assert_true(abs(m.predict({"load_fraction": 1.0, "i_primary": 1000.0})["i_secondary"] - 10.0) < 1e-9,
            "CT: 1000A primary -> 10A secondary")

# 6. PT ratio correct
assert_true(abs(m.predict({"load_fraction": 1.0, "v_primary": 11000.0})["v_secondary"] - 110.0) < 1e-6,
            "PT: 11kV primary -> 110V secondary")

# 7. get_info() interface
info = m.get_info()
assert_true(info["component_id"] == "EC174" and info["version"] == "1.0.0", "get_info() metadata correct")

# 8. Fast benchmark
t0 = time.time()
for _ in range(1000):
    m.predict({"load_fraction": 0.7})
dt = time.time() - t0
assert_true(dt < 1.0, f"1000 predictions fast ({dt*1e3:.1f} ms)")

print(f"\n{passed} passed, {failed} failed")
sys.exit(0 if failed == 0 else 1)
