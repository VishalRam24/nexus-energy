"""Tailored tests for the nuclear F0a power-map lookup (no pytest)."""
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
load_bp = np.asarray(u["load_breakpoints"]["value"])
pelec_bp = np.asarray(u["pelec_breakpoints"]["value"])
P_rated = pelec_bp[-1]
eta_rated = m.lookup.eta_rated

# 1. Electrical power non-negative and bounded by rated across range
loads = np.linspace(load_bp[0], load_bp[-1], 25)
p = m.lookup.power_elec(loads)
assert_true(np.all((p >= 0) & (p <= P_rated + 1e-6)), "P_elec within [0, rated]")

# 2. Rated electrical power matches datasheet at full load
assert_true(abs(m.lookup.power_elec(1.0) - P_rated) < 1e-6, "P_elec at load=1 equals rated")

# 3. Power monotonic increasing with load
assert_true(np.all(np.diff(p) > 0), "P_elec monotonic increasing with load")

# 4. Efficiency physically bounded (0,1) and near rated at full load
eta = m.lookup.efficiency(loads)
assert_true(np.all((eta > 0) & (eta < 1)), "efficiency in (0,1)")
assert_true(abs(m.lookup.efficiency(1.0) - eta_rated) < 1e-4, "eta at load=1 equals rated")

# 5. Endpoints reproduced exactly
assert_true(abs(m.lookup.power_elec(load_bp[0]) - pelec_bp[0]) < 1e-9, "low endpoint exact")
assert_true(abs(m.lookup.power_elec(load_bp[-1]) - pelec_bp[-1]) < 1e-9, "high endpoint exact")

# 6. Edge inputs clamp
assert_true(abs(m.lookup.power_elec(0.0) - pelec_bp[0]) < 1e-9, "load below min clamps to floor")
assert_true(abs(m.lookup.power_elec(2.0) - pelec_bp[-1]) < 1e-9, "load above 1 clamps to rated")

# 7. predict() interface
out = m.predict({"load_factor": 0.8})
assert_true({"power_output_mw", "efficiency", "thermal_power_mw"} <= set(out), "predict returns expected keys")
assert_true(abs(out["thermal_power_mw"] - m.lookup.P_thermal_mw * 0.8) < 1e-6, "thermal power = P_th*load")

# 8. Fast benchmark
t0 = time.time()
for _ in range(1000):
    m.lookup.power_elec(0.7)
dt = time.time() - t0
assert_true(dt < 1.0, f"1000 lookups fast ({dt*1e3:.1f} ms)")

print(f"\n{passed} passed, {failed} failed")
sys.exit(0 if failed == 0 else 1)
