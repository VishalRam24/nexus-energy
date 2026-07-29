"""Tailored tests for EC146 F0a torrefaction mass-yield table. No pytest."""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from predict import ComponentModel

failed = 0


def assert_true(cond, msg):
    global failed
    if cond:
        print(f"  ✓ {msg}")
    else:
        failed += 1
        print(f"  ✗ {msg}")


m = ComponentModel()
print("EC146 F0a mass-yield-table tests")

# 1. rated/datasheet point: mass yield at 280 C = 0.70, EDR = 1.22
r = m.predict({"temperature_degC": 280})
assert_true(abs(r["mass_yield"] - 0.70) < 1e-6, "mass yield at 280 C = 0.70 (datasheet)")
assert_true(abs(r["EDR"] - 1.22) < 1e-6, "EDR at 280 C = 1.22 (datasheet)")

# 2. physical bounds: mass yield in (0,1), EDR > 1
for T in (200, 230, 260, 280, 300):
    rr = m.predict({"temperature_degC": T})
    assert_true(0.0 < rr["mass_yield"] < 1.0 and rr["EDR"] >= 1.0,
                f"mass_yield in (0,1) & EDR>=1 at {T} C")

# 3. mass yield monotonically decreasing with temperature
mys = [m.predict({"temperature_degC": T})["mass_yield"] for T in (200, 230, 260, 280, 300)]
assert_true(all(mys[i] > mys[i + 1] for i in range(len(mys) - 1)), "mass yield falls with T")

# 4. EDR monotonically increasing with temperature
edrs = [m.predict({"temperature_degC": T})["EDR"] for T in (200, 230, 260, 280, 300)]
assert_true(all(edrs[i] < edrs[i + 1] for i in range(len(edrs) - 1)), "EDR rises with T")

# 5. energy yield = mass_yield * EDR and stays below 1 (energy is lost as volatiles)
assert_true(abs(r["energy_yield"] - r["mass_yield"] * r["EDR"]) < 1e-9, "energy yield = mass*EDR")
for T in (200, 250, 300):
    assert_true(m.predict({"temperature_degC": T})["energy_yield"] < 1.0,
                f"energy yield < 1 at {T} C")

# 6. torrefied LHV > raw LHV (densification)
assert_true(r["torrefied_LHV_MJ_kg"] > 18.5, "torrefied LHV > raw 18.5 MJ/kg")

# 7. predict() interface keys present
assert_true(all(k in r for k in ("mass_yield", "EDR", "energy_yield", "solid_kg")),
            "predict() keys present")

# 8. fast benchmark
t0 = time.time()
for _ in range(1000):
    m.predict({"temperature_degC": 270})
dt = time.time() - t0
assert_true(dt < 1.0, f"1000 predicts < 1s ({dt*1000:.1f} ms)")

print(f"\n{'PASS' if failed == 0 else 'FAIL'}: {failed} failed")
sys.exit(0 if failed == 0 else 1)
