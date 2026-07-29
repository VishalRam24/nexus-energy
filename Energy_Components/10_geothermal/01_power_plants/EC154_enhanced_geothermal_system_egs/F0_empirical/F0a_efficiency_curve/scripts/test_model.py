"""Tailored tests for F0a geothermal-plant efficiency-curve model (no pytest).

Component-agnostic: temperatures are read from the model's own curve so the
same harness validates EC151/EC152/EC153/EC154.
"""
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
info = m.get_info()
params = json.load(open(os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")))
T_pts = params["curve"]["T_geo_degC"]
eta_pts = params["curve"]["eta_net"]

# 1. efficiency in physical bounds across the whole curve
etas = [m.predict({"T_geothermal": T, "flow_rate_kgs": 50.0})["eta_net"] for T in T_pts]
assert_true(all(0.0 < e < 0.5 for e in etas), "net efficiency within (0, 0.5)")

# 2. efficiency monotonically increases with resource temperature
assert_true(all(b >= a for a, b in zip(etas, etas[1:])) and etas[-1] > etas[0],
            "eta increases with T_geo")

# 3. rated point matches datasheet
rated = params["rated"]
eta_rated = m.predict({"T_geothermal": rated["T_geo_design"]["value"], "flow_rate_kgs": 50.0})["eta_net"]
assert_true(abs(eta_rated - rated["eta_net_rated"]["value"]) < 1e-3,
            f"rated eta {eta_rated:.4f} matches datasheet {rated['eta_net_rated']['value']}")

# 4. endpoint clamping below first / above last breakpoint
lo = m.predict({"T_geothermal": T_pts[0] - 50.0, "flow_rate_kgs": 50.0})["eta_net"]
hi = m.predict({"T_geothermal": T_pts[-1] + 50.0, "flow_rate_kgs": 50.0})["eta_net"]
assert_true(abs(lo - eta_pts[0]) < 1e-9 and abs(hi - eta_pts[-1]) < 1e-9,
            "interp clamps at table endpoints")

# 5. net power positive and scales linearly with flow
Tmid = T_pts[len(T_pts) // 2]
p1 = m.predict({"T_geothermal": Tmid, "flow_rate_kgs": 50})["net_power_kW"]
p2 = m.predict({"T_geothermal": Tmid, "flow_rate_kgs": 100})["net_power_kW"]
assert_true(p1 > 0 and abs(p2 - 2 * p1) < 1e-6, "net power positive and linear in flow")

# 6. zero thermal head -> zero power
T_rej = params["T_reject_ref"]["value"]
p0 = m.predict({"T_geothermal": T_rej, "flow_rate_kgs": 50.0, "T_rejection": T_rej})["net_power_kW"]
assert_true(p0 == 0.0, "no thermal head gives zero power")

# 7. predict interface returns expected keys
out = m.predict({"T_geothermal": Tmid, "flow_rate_kgs": 50})
assert_true(set(out) == {"eta_net", "net_power_kW"}, "predict returns eta_net and net_power_kW")

# 8. benchmark: 1000 predictions fast
t0 = time.perf_counter()
for _ in range(1000):
    m.predict({"T_geothermal": Tmid, "flow_rate_kgs": 60})
dt = time.perf_counter() - t0
assert_true(dt < 1.0, f"1000 predictions in {dt*1e3:.1f} ms (< 1 s)")

print(f"\n{passed} passed, {failed} failed")
sys.exit(0 if failed == 0 else 1)
