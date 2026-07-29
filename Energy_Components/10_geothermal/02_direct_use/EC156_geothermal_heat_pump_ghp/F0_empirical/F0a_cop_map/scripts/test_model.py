"""Tailored tests for EC156 F0a COP-map model (no pytest)."""
import json
import os
import sys
import time

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

# 1. COP > 1 everywhere on the grid (heat pump must amplify)
cops = []
for ts in params["map"]["T_source_degC"]:
    for tk in params["map"]["T_sink_degC"]:
        cops.append(m.predict({"T_source": ts, "T_sink": tk})["COP"])
assert_true(all(c > 1.0 for c in cops), "COP > 1 across the whole map")

# 2. COP increases as source temperature rises (smaller lift)
c_cold = m.predict({"T_source": 0, "T_sink": 45})["COP"]
c_warm = m.predict({"T_source": 20, "T_sink": 45})["COP"]
assert_true(c_warm > c_cold, "COP rises with warmer ground source")

# 3. COP decreases as sink temperature rises (larger lift)
c_low = m.predict({"T_source": 10, "T_sink": 35})["COP"]
c_high = m.predict({"T_source": 10, "T_sink": 65})["COP"]
assert_true(c_low > c_high, "COP falls with hotter supply temperature")

# 4. rated point matches datasheet
rated = params["rated"]
c_rated = m.predict({"T_source": rated["T_source_design"]["value"],
                     "T_sink": rated["T_sink_design"]["value"]})["COP"]
assert_true(abs(c_rated - rated["COP_rated"]["value"]) < 0.05,
            f"rated COP {c_rated:.3f} matches datasheet {rated['COP_rated']['value']}")

# 5. exact grid node reproduces the tabulated value
node = params["map"]["COP"][2][1]  # T_source=10, T_sink=45
got = m.predict({"T_source": 10, "T_sink": 45})["COP"]
assert_true(abs(got - node) < 1e-6, "lookup reproduces exact grid node")

# 6. edge clamping outside the grid
clamp = m.predict({"T_source": -10, "T_sink": 100})["COP"]
corner = params["map"]["COP"][0][-1]  # T_source min, T_sink max
assert_true(abs(clamp - corner) < 1e-6, "out-of-grid inputs clamp to corner")

# 7. electric power = Q/COP + aux; part load scales thermal output
out = m.predict({"T_source": 9, "T_sink": 45, "Q_thermal_kW": 10.0, "part_load_ratio": 0.5})
aux = params["auxiliary_power"]["value"]
expected = 10.0 * 0.5 / out["COP"] + aux
assert_true(abs(out["electric_power_kW"] - expected) < 1e-6 and out["Q_thermal_kW"] == 5.0,
            "electric power and part-load output correct")

# 8. predict interface + benchmark
assert_true(set(out) == {"COP", "Q_thermal_kW", "electric_power_kW"}, "predict returns expected keys")
t0 = time.perf_counter()
for _ in range(1000):
    m.predict({"T_source": 12, "T_sink": 50})
dt = time.perf_counter() - t0
assert_true(dt < 1.0, f"1000 predictions in {dt*1e3:.1f} ms (< 1 s)")

print(f"\n{passed} passed, {failed} failed")
sys.exit(0 if failed == 0 else 1)
