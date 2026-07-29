"""F0a tests for EC173 — custom harness, no pytest."""
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
rated = m.eta_rated

# 1. Efficiency in physical bounds across the curve
fracs = np.linspace(m.curve.load[0], m.curve.load[-1], 25)
effs = m.curve.efficiency(fracs)
assert_true(np.all((effs > 0.0) & (effs < 1.0)), "0 < efficiency < 1 across part-load range")

# 2. Part-load efficiency curve is unimodal (rises to a peak, then may dip):
#    a single sign change in the slope, peak at or below rated load.
diffs = np.diff(effs)
sign_changes = np.sum(np.diff(np.sign(diffs[np.abs(diffs) > 1e-12])) != 0)
peak_idx = int(np.argmax(effs))
assert_true(sign_changes <= 1, "efficiency curve unimodal (single peak) across load")
assert_true(effs[0] < effs[peak_idx], "efficiency rises from low load toward peak")

# 3. Rated point matches datasheet
assert_true(abs(m.predict({"load_fraction": 1.0})["efficiency"] - rated) < 1e-9,
            "efficiency at full load matches datasheet rated value")

# 4. Endpoints / clamping on out-of-range input
lo = m.predict({"load_fraction": -1.0})["efficiency"]
hi = m.predict({"load_fraction": 5.0})["efficiency"]
assert_true(lo == m.curve.eff[0] and hi == m.curve.eff[-1],
            "Out-of-range load fraction clamps to endpoints")

# 5. Loss fraction non-negative and power balance consistent
out = m.predict({"load_fraction": 0.75, "power_in": 1000.0})
assert_true(out["loss_fraction"] >= 0.0, "loss fraction non-negative")
assert_true(abs(out["power_out"] + out["power_loss"] - 1000.0) < 1e-6,
            "power_out + power_loss == power_in")

# 6. predict() interface returns required keys
keys = set(m.predict({"load_fraction": 0.5}).keys())
assert_true({"efficiency", "loss_fraction"}.issubset(keys), "predict() returns required keys")

# 7. get_info() interface
info = m.get_info()
assert_true(info["component_id"] == "EC173" and info["version"] == "1.0.0",
            "get_info() metadata correct")

# 8. Fast benchmark
t0 = time.time()
for _ in range(1000):
    m.predict({"load_fraction": 0.7})
dt = time.time() - t0
assert_true(dt < 1.0, f"1000 predictions fast ({dt*1e3:.1f} ms)")

print(f"\n{passed} passed, {failed} failed")
sys.exit(0 if failed == 0 else 1)
