"""Tailored tests for EC143 F0a CGE curve. No pytest."""
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
print("EC143 F0a CGE-curve tests")

# 1. rated point: CGE at design ER (wood) = 0.75
r = m.predict({"feedstock": "wood", "equivalence_ratio": 0.25})
assert_true(abs(r["cold_gas_efficiency"] - 0.75) < 1e-6, "wood CGE at design ER = 0.75 (datasheet)")

# 2. physical bounds: 0 < CGE < 1 across ER sweep
for ER in (0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45):
    c = m.predict({"feedstock": "wood", "equivalence_ratio": ER})["cold_gas_efficiency"]
    assert_true(0.0 < c < 1.0, f"CGE in (0,1) at ER={ER}: {c:.3f}")

# 3. CGE peaks at design ER=0.25 (greater than neighbours)
c_peak = m.predict({"equivalence_ratio": 0.25})["cold_gas_efficiency"]
c_lo = m.predict({"equivalence_ratio": 0.15})["cold_gas_efficiency"]
c_hi = m.predict({"equivalence_ratio": 0.45})["cold_gas_efficiency"]
assert_true(c_peak > c_lo and c_peak > c_hi, "CGE peaks at design ER")

# 4. moisture penalty reduces CGE
dry = m.predict({"feedstock": "wood", "moisture": 0.0})["cold_gas_efficiency"]
wet = m.predict({"feedstock": "wood", "moisture": 0.3})["cold_gas_efficiency"]
assert_true(wet < dry, "moisture reduces CGE")

# 5. higher-HHV fuel gives higher fuel factor (pine > sewage_sludge)
assert_true(m.predict({"feedstock": "pine"})["fuel_factor"] >
            m.predict({"feedstock": "sewage_sludge"})["fuel_factor"],
            "pine fuel factor > sewage_sludge")

# 6. syngas power = CGE * fuel power
r2 = m.predict({"feedstock": "wood", "fuel_power_kw": 1000})
assert_true(abs(r2["syngas_power_kw"] - 1000 * r2["cold_gas_efficiency"]) < 1e-6,
            "syngas power = CGE * fuel power")

# 7. predict() interface keys present
assert_true(all(k in r for k in ("cold_gas_efficiency", "syngas_power_kw")), "predict() keys present")

# 8. fast benchmark
t0 = time.time()
for _ in range(1000):
    m.predict({"feedstock": "corn_stover", "equivalence_ratio": 0.28})
dt = time.time() - t0
assert_true(dt < 1.0, f"1000 predicts < 1s ({dt*1000:.1f} ms)")

print(f"\n{'PASS' if failed == 0 else 'FAIL'}: {failed} failed")
sys.exit(0 if failed == 0 else 1)
