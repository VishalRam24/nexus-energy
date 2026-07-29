"""Tests for EC099 Stirling Engine (micro-CHP scale) F0a empirical lookup (no pytest)."""
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
        print(f"  \u2713 {msg}")
    else:
        failed += 1
        print(f"  \u2717 {msg}")


m = ComponentModel()
info = m.get_info()
eta_rated = info["eta_rated"]

print("Testing EC099 Stirling Engine (micro-CHP scale) (F0a)")

r = m.predict({"part_load_ratio": 1.0})
assert_true(abs(r["electrical_efficiency"] - eta_rated) < 1e-6,
            f"eta at PLR=1 == rated ({eta_rated})")

ok = all(0.0 <= m.predict({"part_load_ratio": x})["electrical_efficiency"] < 1.0
         for x in np.linspace(0.0, 1.0, 21))
assert_true(ok, "0 <= eta < 1 across full PLR range")

assert_true(m.predict({"part_load_ratio": 0.01})["electrical_efficiency"] == 0.0,
            "eta == 0 below minimum part-load")

half = m.predict({"part_load_ratio": 0.5})["power_output_W"]
full = m.predict({"part_load_ratio": 1.0})["power_output_W"]
assert_true(abs(half - 0.5 * full) < 1e-6, "power_output linear in PLR")

plr_min = m.p["rated"]["PLR_min"]["value"]
xs = np.linspace(plr_min, 1.0, 9)
etas = [m.predict({"part_load_ratio": x})["electrical_efficiency"] for x in xs]
assert_true(all(b >= a - 1e-9 for a, b in zip(etas, etas[1:])),
            "efficiency monotonic non-decreasing in operating band")

r = m.predict({"part_load_ratio": 0.8})
assert_true(r["fuel_heat_input_W"] >= r["power_output_W"] > 0,
            "fuel heat input >= electrical output")

assert_true({"electrical_efficiency", "power_output_W", "fuel_heat_input_W"}.issubset(r.keys()),
            "predict() returns standard keys")

t0 = time.perf_counter()
for _ in range(1000):
    m.predict({"part_load_ratio": 0.7})
dt = time.perf_counter() - t0
assert_true(dt < 2.0, f"1000 predictions in {dt*1e3:.1f} ms")

print(f"\n{passed} passed, {failed} failed")
sys.exit(0 if failed == 0 else 1)
