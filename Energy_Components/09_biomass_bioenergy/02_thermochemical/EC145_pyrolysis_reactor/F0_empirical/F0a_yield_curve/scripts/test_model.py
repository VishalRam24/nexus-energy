"""Tailored tests for EC145 F0a pyrolysis yield curve. No pytest."""
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
print("EC145 F0a yield-curve tests")

# 1. rated point: bio-oil peaks at 500 C (~0.72 before renorm; check it's the max)
r = m.predict({"temperature_degC": 500})
others = [m.predict({"temperature_degC": T})["bio_oil_frac"] for T in (300, 400, 600, 700)]
assert_true(all(r["bio_oil_frac"] > o for o in others), "bio-oil yield peaks at 500 C (datasheet)")

# 2. fractions sum to 1.0 at every temperature
for T in (300, 400, 500, 600, 700):
    rr = m.predict({"temperature_degC": T})
    s = rr["bio_oil_frac"] + rr["char_frac"] + rr["gas_frac"]
    assert_true(abs(s - 1.0) < 1e-9, f"yields sum to 1 at {T} C: {s:.4f}")

# 3. physical bounds: all fractions in (0,1)
for T in (300, 500, 700):
    rr = m.predict({"temperature_degC": T})
    assert_true(all(0.0 < rr[k] < 1.0 for k in ("bio_oil_frac", "char_frac", "gas_frac")),
                f"all fractions in (0,1) at {T} C")

# 4. char decreases with temperature (300 -> 700)
chars = [m.predict({"temperature_degC": T})["char_frac"] for T in (300, 400, 500)]
assert_true(chars[0] > chars[1] > chars[2], "char fraction falls 300->500 C")

# 5. gas rises at high temperature (500 -> 700)
gas_lo = m.predict({"temperature_degC": 500})["gas_frac"]
gas_hi = m.predict({"temperature_degC": 700})["gas_frac"]
assert_true(gas_hi > gas_lo, "gas fraction rises 500->700 C")

# 6. masses scale with feed
r2 = m.predict({"temperature_degC": 500, "feed_kg": 1000})
assert_true(abs(r2["bio_oil_kg"] - 1000 * r2["bio_oil_frac"]) < 1e-6, "bio_oil_kg scales with feed")

# 7. predict() interface keys present
assert_true(all(k in r for k in ("bio_oil_frac", "product_energy_MJ_kg")), "predict() keys present")

# 8. fast benchmark
t0 = time.time()
for _ in range(1000):
    m.predict({"temperature_degC": 480})
dt = time.time() - t0
assert_true(dt < 1.0, f"1000 predicts < 1s ({dt*1000:.1f} ms)")

print(f"\n{'PASS' if failed == 0 else 'FAIL'}: {failed} failed")
sys.exit(0 if failed == 0 else 1)
