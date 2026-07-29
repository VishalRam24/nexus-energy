"""Tailored tests for EC147 F0a HTL bio-crude yield table. No pytest."""
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
print("EC147 F0a biocrude-yield-table tests")

# 1. datasheet point: nannochloropsis yield at 330 C = 0.48
r = m.predict({"feedstock": "microalgae_nannochloropsis", "temperature_degC": 330})
assert_true(abs(r["biocrude_yield"] - 0.48) < 1e-6,
            "nannochloropsis yield at 330 C = 0.48 (datasheet)")

# 2. physical bounds: 0 < yield < 0.6 for all feedstocks
for fs in m.table.feedstocks:
    y = m.predict({"feedstock": fs})["biocrude_yield"]
    assert_true(0.0 < y < 0.6, f"{fs} yield in (0,0.6): {y:.3f}")

# 3. temp multiplier peaks at 330 C
assert_true(abs(m.predict({"feedstock": "wood_biomass", "temperature_degC": 330})["temp_multiplier"] - 1.0) < 1e-9,
            "temp multiplier = 1.0 at 330 C")

# 4. yield rises 250->330 then falls 330->400
up = [m.predict({"feedstock": "wood_biomass", "temperature_degC": T})["biocrude_yield"] for T in (250, 290, 330)]
assert_true(all(up[i] < up[i + 1] for i in range(len(up) - 1)), "yield rises 250->330 C")
dn = [m.predict({"feedstock": "wood_biomass", "temperature_degC": T})["biocrude_yield"] for T in (330, 360, 400)]
assert_true(all(dn[i] > dn[i + 1] for i in range(len(dn) - 1)), "yield falls 330->400 C")

# 5. lipid-rich algae out-yields lignocellulosic wood
assert_true(m.predict({"feedstock": "microalgae_nannochloropsis"})["biocrude_yield"] >
            m.predict({"feedstock": "wood_biomass"})["biocrude_yield"],
            "lipid-rich algae > wood yield")

# 6. energy yield = yield * HHV(35)
assert_true(abs(r["energy_yield_MJ_kg"] - r["biocrude_yield"] * 35.0) < 1e-9, "energy = yield * HHV(35)")

# 7. predict() interface keys present
assert_true(all(k in r for k in ("biocrude_yield", "biocrude_kg", "energy_yield_MJ_kg")),
            "predict() keys present")

# 8. fast benchmark
t0 = time.time()
for _ in range(1000):
    m.predict({"feedstock": "sewage_sludge", "temperature_degC": 320})
dt = time.time() - t0
assert_true(dt < 1.0, f"1000 predicts < 1s ({dt*1000:.1f} ms)")

print(f"\n{'PASS' if failed == 0 else 'FAIL'}: {failed} failed")
sys.exit(0 if failed == 0 else 1)
