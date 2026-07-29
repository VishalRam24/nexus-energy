"""Tests for EC105 Gas Turbine CHP (with HRSG) F0a CHP lookup (no pytest)."""
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
eta_el_rated = info["eta_el_rated"]
eta_th_rated = info["eta_th_rated"]

print("Testing EC105 Gas Turbine CHP (with HRSG) (F0a CHP)")

r = m.predict({"part_load_ratio": 1.0})
assert_true(abs(r["electrical_efficiency"] - eta_el_rated) < 1e-6,
            f"eta_el at PLR=1 == rated ({eta_el_rated})")

assert_true(abs(r["thermal_efficiency"] - eta_th_rated) < 1e-6,
            f"eta_th at PLR=1 == rated ({eta_th_rated})")

ok = True
for x in np.linspace(0.0, 1.0, 21):
    rr = m.predict({"part_load_ratio": x})
    if not (0.0 <= rr["electrical_efficiency"] < 1.0 and 0.0 <= rr["thermal_efficiency"] < 1.0):
        ok = False
assert_true(ok, "0 <= eta_el, eta_th < 1 across PLR")

assert_true(abs(r["total_efficiency"] - (r["electrical_efficiency"] + r["thermal_efficiency"])) < 1e-9
            and r["total_efficiency"] < 1.0,
            "total_efficiency = eta_el + eta_th and < 1")

off = m.predict({"part_load_ratio": 0.01})
assert_true(off["electrical_efficiency"] == 0.0 and off["thermal_efficiency"] == 0.0,
            "both efficiencies == 0 below minimum part-load")

half = m.predict({"part_load_ratio": 0.5})["power_electrical_W"]
full = r["power_electrical_W"]
assert_true(abs(half - 0.5 * full) < 1e-6, "electrical power linear in PLR")

assert_true({"electrical_efficiency", "thermal_efficiency", "total_efficiency",
             "power_electrical_W"}.issubset(r.keys()),
            "predict() returns standard keys")

t0 = time.perf_counter()
for _ in range(1000):
    m.predict({"part_load_ratio": 0.7})
dt = time.perf_counter() - t0
assert_true(dt < 2.0, f"1000 predictions in {dt*1e3:.1f} ms")

print(f"\n{passed} passed, {failed} failed")
sys.exit(0 if failed == 0 else 1)
