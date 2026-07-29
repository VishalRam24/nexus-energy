"""Tailored tests for EC155 F0a delivered-heat curve model (no pytest)."""
import json
import os
import sys
import time

import numpy as np

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
params = json.load(open(os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")))
T_pts = params["curve"]["T_source_degC"]
q_pts = params["curve"]["q_specific_kW_per_kgps"]

# 1. delivered specific heat positive and physically bounded over the curve
qs = [m.predict({"T_source": T, "flow_rate_kgs": 50.0})["q_specific_kW_per_kgps"] for T in T_pts]
assert_true(all(0.0 < q < 600.0 for q in qs), "delivered specific heat within (0, 600) kW per kg/s")

# 2. monotonic increase with source temperature
assert_true(all(b > a for a, b in zip(qs, qs[1:])), "delivered heat increases with T_source")

# 3. rated point matches datasheet
rated = params["rated"]
out = m.predict({"T_source": rated["T_source_design"]["value"],
                 "flow_rate_kgs": rated["m_dot_geo_design"]["value"]})
assert_true(abs(out["Q_delivered_kW"] - rated["Q_delivered_rated"]["value"]) < 1.0,
            f"rated Q {out['Q_delivered_kW']:.1f} kW matches datasheet {rated['Q_delivered_rated']['value']}")

# 4. endpoint clamping for table interp (no explicit T_return)
lo = m.predict({"T_source": T_pts[0] - 20.0, "flow_rate_kgs": 50.0})["q_specific_kW_per_kgps"]
hi = m.predict({"T_source": T_pts[-1] + 20.0, "flow_rate_kgs": 50.0})["q_specific_kW_per_kgps"]
assert_true(abs(lo - q_pts[0]) < 1e-6 and abs(hi - q_pts[-1]) < 1e-6, "interp clamps at table endpoints")

# 5. delivered heat scales linearly with flow
q1 = m.predict({"T_source": 80, "flow_rate_kgs": 50})["Q_delivered_kW"]
q2 = m.predict({"T_source": 80, "flow_rate_kgs": 100})["Q_delivered_kW"]
assert_true(q1 > 0 and abs(q2 - 2 * q1) < 1e-6, "delivered heat linear in flow")

# 6. source == return -> zero delivered heat
z = m.predict({"T_source": 60.0, "flow_rate_kgs": 50.0, "T_return": 60.0})["Q_delivered_kW"]
assert_true(z == 0.0, "no temperature drop gives zero delivered heat")

# 7. pump power is the configured small fraction of delivered heat
out2 = m.predict({"T_source": 80, "flow_rate_kgs": 50})
frac = params["pump_power_fraction"]["value"]
assert_true(abs(out2["pump_power_kW"] - frac * out2["Q_delivered_kW"]) < 1e-9,
            "pump power equals fraction of delivered heat")

# 8. predict interface + benchmark
keys = {"q_specific_kW_per_kgps", "Q_delivered_kW", "pump_power_kW"}
assert_true(set(out2) == keys, "predict returns expected keys")
t0 = time.perf_counter()
for _ in range(1000):
    m.predict({"T_source": 90, "flow_rate_kgs": 60})
dt = time.perf_counter() - t0
assert_true(dt < 1.0, f"1000 predictions in {dt*1e3:.1f} ms (< 1 s)")

print(f"\n{passed} passed, {failed} failed")
sys.exit(0 if failed == 0 else 1)
