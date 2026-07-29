"""Tailored tests for EC140 F0a empirical yield lookup. No pytest."""
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
print("EC140 F0a yield-table tests")

# 1. rated point matches datasheet (food_waste at optimum)
r = m.predict({"feedstock": "food_waste", "temperature_degC": 37})
assert_true(abs(r["ch4_yield_m3_kgVS"] - 0.42) < 1e-6,
            "food_waste yield at optimum = 0.42 m3/kgVS (datasheet)")

# 2. physical bounds: yields positive and within sane biogas range
for fs in ["sewage_sludge", "cattle_manure", "grass_silage", "corn_silage", "food_waste"]:
    y = m.predict({"feedstock": fs})["ch4_yield_m3_kgVS"]
    assert_true(0.0 < y < 0.6, f"{fs} CH4 yield in (0,0.6): {y:.3f}")

# 3. temperature multiplier peaks at 37 C
mult37 = m.predict({"feedstock": "cattle_manure", "temperature_degC": 37})["temp_multiplier"]
assert_true(abs(mult37 - 1.0) < 1e-9, "temp multiplier = 1.0 at 37 C")

# 4. yield monotonically rises 25->37 then falls 37->55
ys = [m.predict({"feedstock": "cattle_manure", "temperature_degC": T})["ch4_yield_m3_kgVS"]
      for T in (25, 30, 35, 37)]
assert_true(all(ys[i] < ys[i + 1] for i in range(len(ys) - 1)), "yield rises 25->37 C")
ys2 = [m.predict({"feedstock": "cattle_manure", "temperature_degC": T})["ch4_yield_m3_kgVS"]
       for T in (37, 45, 55)]
assert_true(all(ys2[i] > ys2[i + 1] for i in range(len(ys2) - 1)), "yield falls 37->55 C")

# 5. energy = ch4 * LHV (9.97)
r2 = m.predict({"feedstock": "cattle_manure"})
assert_true(abs(r2["energy_yield_kwh_kgVS"] - r2["ch4_yield_m3_kgVS"] * 9.97) < 1e-9,
            "energy = CH4 yield * LHV(9.97)")

# 6. totals scale linearly with VS fed
ra = m.predict({"feedstock": "food_waste", "vs_fed_kg": 1000})
assert_true(abs(ra["ch4_total_m3"] - 1000 * ra["ch4_yield_m3_kgVS"]) < 1e-6,
            "ch4_total scales with vs_fed_kg")

# 7. predict() interface returns required keys
assert_true(all(k in r for k in ("ch4_yield_m3_kgVS", "energy_total_kwh")),
            "predict() returns expected keys")

# 8. fast benchmark
t0 = time.time()
for _ in range(1000):
    m.predict({"feedstock": "corn_silage", "temperature_degC": 38})
dt = time.time() - t0
assert_true(dt < 1.0, f"1000 predicts < 1s ({dt*1000:.1f} ms)")

print(f"\n{'PASS' if failed == 0 else 'FAIL'}: {failed} failed")
sys.exit(0 if failed == 0 else 1)
