"""Tailored tests for EC141 F0a empirical BMP lookup. No pytest."""
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
print("EC141 F0a BMP-table tests")

# 1. rated point matches datasheet (food_waste BMP at optimum)
r = m.predict({"feedstock": "food_waste", "temperature_degC": 55})
assert_true(abs(r["bmp_L_CH4_kgVS"] - 420.0) < 1e-6,
            "food_waste BMP at optimum = 420 L/kgVS (datasheet)")

# 2. physical bounds: BMP in sane range for all feedstocks
for fs in ["sewage_sludge", "cattle_manure", "corn_silage", "grass_silage", "food_waste"]:
    b = m.predict({"feedstock": fs})["bmp_L_CH4_kgVS"]
    assert_true(100.0 < b < 500.0, f"{fs} BMP in (100,500): {b:.1f}")

# 3. temp multiplier peaks at 55 C
assert_true(abs(m.predict({"feedstock": "cattle_manure", "temperature_degC": 55})["temp_multiplier"] - 1.0) < 1e-9,
            "temp multiplier = 1.0 at 55 C")

# 4. monotonic rise 45->55 then fall 55->65
up = [m.predict({"feedstock": "cattle_manure", "temperature_degC": T})["bmp_L_CH4_kgVS"] for T in (45, 50, 53, 55)]
assert_true(all(up[i] < up[i + 1] for i in range(len(up) - 1)), "BMP rises 45->55 C")
dn = [m.predict({"feedstock": "cattle_manure", "temperature_degC": T})["bmp_L_CH4_kgVS"] for T in (55, 60, 65)]
assert_true(all(dn[i] > dn[i + 1] for i in range(len(dn) - 1)), "BMP falls 55->65 C")

# 5. energy = BMP/1000 * LHV
r2 = m.predict({"feedstock": "corn_silage"})
assert_true(abs(r2["energy_yield_kwh_kgVS"] - r2["bmp_L_CH4_kgVS"] / 1000.0 * 9.97) < 1e-9,
            "energy = BMP/1000 * LHV(9.97)")

# 6. CH4 fraction in 0.5-0.7
assert_true(0.5 < r2["methane_fraction"] < 0.7, "methane fraction in (0.5,0.7)")

# 7. predict() interface keys present
assert_true(all(k in r for k in ("bmp_L_CH4_kgVS", "energy_total_kwh")), "predict() keys present")

# 8. fast benchmark
t0 = time.time()
for _ in range(1000):
    m.predict({"feedstock": "grass_silage", "temperature_degC": 54})
dt = time.time() - t0
assert_true(dt < 1.0, f"1000 predicts < 1s ({dt*1000:.1f} ms)")

print(f"\n{'PASS' if failed == 0 else 'FAIL'}: {failed} failed")
sys.exit(0 if failed == 0 else 1)
