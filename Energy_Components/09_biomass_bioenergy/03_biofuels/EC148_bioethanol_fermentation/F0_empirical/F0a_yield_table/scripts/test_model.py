"""Tailored tests for EC148 F0a bioethanol yield table. No pytest."""
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
print("EC148 F0a ethanol-yield-table tests")

# 1. datasheet point: corn yield at optimum = 400 L/tonne
r = m.predict({"feedstock": "corn", "temperature_degC": 32})
assert_true(abs(r["ethanol_L_per_tonne"] - 400.0) < 1e-6, "corn yield at 32 C = 400 L/t (datasheet)")

# 2. physical bounds: yields positive & < 500 L/tonne for all feedstocks
for fs in m.table.feedstocks:
    y = m.predict({"feedstock": fs})["ethanol_L_per_tonne"]
    assert_true(0.0 < y < 500.0, f"{fs} yield in (0,500): {y:.1f}")

# 3. temp multiplier peaks at 32 C
assert_true(abs(m.predict({"feedstock": "corn", "temperature_degC": 32})["temp_multiplier"] - 1.0) < 1e-9,
            "temp multiplier = 1.0 at 32 C")

# 4. yield rises 20->32 then falls 32->45
up = [m.predict({"feedstock": "corn", "temperature_degC": T})["ethanol_L_per_tonne"] for T in (20, 28, 32)]
assert_true(all(up[i] < up[i + 1] for i in range(len(up) - 1)), "yield rises 20->32 C")
dn = [m.predict({"feedstock": "corn", "temperature_degC": T})["ethanol_L_per_tonne"] for T in (32, 38, 45)]
assert_true(all(dn[i] > dn[i + 1] for i in range(len(dn) - 1)), "yield falls 32->45 C")

# 5. starch corn out-yields cellulosic switchgrass
assert_true(m.predict({"feedstock": "corn"})["ethanol_L_per_tonne"] >
            m.predict({"feedstock": "switchgrass"})["ethanol_L_per_tonne"],
            "corn > switchgrass yield")

# 6. energy = yield * LHV(21.1)
assert_true(abs(r["energy_yield_MJ_per_tonne"] - r["ethanol_L_per_tonne"] * 21.1) < 1e-6,
            "energy = yield * LHV(21.1)")

# 7. predict() interface keys present
assert_true(all(k in r for k in ("ethanol_L_per_tonne", "ethanol_L")), "predict() keys present")

# 8. fast benchmark
t0 = time.time()
for _ in range(1000):
    m.predict({"feedstock": "sugarcane", "temperature_degC": 30})
dt = time.time() - t0
assert_true(dt < 1.0, f"1000 predicts < 1s ({dt*1000:.1f} ms)")

print(f"\n{'PASS' if failed == 0 else 'FAIL'}: {failed} failed")
sys.exit(0 if failed == 0 else 1)
