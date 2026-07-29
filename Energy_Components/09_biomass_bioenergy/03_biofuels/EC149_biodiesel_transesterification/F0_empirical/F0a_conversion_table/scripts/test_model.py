"""Tailored tests for EC149 F0a biodiesel conversion table. No pytest."""
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
print("EC149 F0a conversion-table tests")

# 1. datasheet point: low-FFA rapeseed at 60 C -> base conversion 0.98
r = m.predict({"feedstock": "rapeseed_oil", "temperature_degC": 60})
assert_true(abs(r["conversion"] - 0.98) < 1e-6, "rapeseed (low FFA) conversion = 0.98 (datasheet)")

# 2. physical bounds: conversion in (0,1) for all feedstocks
for fs in m.table.feedstocks:
    c = m.predict({"feedstock": fs})["conversion"]
    assert_true(0.0 < c < 1.0, f"{fs} conversion in (0,1): {c:.3f}")

# 3. FFA penalty: high-FFA WCO converts worse than low-FFA rapeseed
assert_true(m.predict({"feedstock": "waste_cooking_oil"})["conversion"] <
            m.predict({"feedstock": "rapeseed_oil"})["conversion"],
            "high-FFA WCO conversion < low-FFA rapeseed")

# 4. conversion monotonically decreases with FFA
convs = [m.predict({"feedstock": "soybean_oil", "ffa_pct": f})["conversion"] for f in (0, 1, 3, 6, 10)]
assert_true(all(convs[i] >= convs[i + 1] for i in range(len(convs) - 1)) and convs[0] > convs[-1],
            "conversion decreases with FFA")

# 5. temp multiplier peaks at 60 C
assert_true(abs(m.predict({"feedstock": "soybean_oil", "temperature_degC": 60})["temp_multiplier"] - 1.0) < 1e-9,
            "temp multiplier = 1.0 at 60 C")

# 6. high oil-content WCO yields more biodiesel mass per tonne than low-oil soybean
assert_true(m.predict({"feedstock": "waste_cooking_oil"})["biodiesel_frac"] >
            m.predict({"feedstock": "soybean_oil"})["biodiesel_frac"],
            "high oil-content WCO biodiesel_frac > soybean")

# 7. predict() interface keys present
assert_true(all(k in r for k in ("conversion", "biodiesel_frac", "biodiesel_tonnes")),
            "predict() keys present")

# 8. fast benchmark
t0 = time.time()
for _ in range(1000):
    m.predict({"feedstock": "palm_oil", "temperature_degC": 62})
dt = time.time() - t0
assert_true(dt < 1.0, f"1000 predicts < 1s ({dt*1000:.1f} ms)")

print(f"\n{'PASS' if failed == 0 else 'FAIL'}: {failed} failed")
sys.exit(0 if failed == 0 else 1)
