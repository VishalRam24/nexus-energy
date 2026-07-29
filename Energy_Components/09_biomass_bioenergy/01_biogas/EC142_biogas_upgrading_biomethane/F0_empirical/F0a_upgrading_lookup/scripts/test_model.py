"""Tailored tests for EC142 F0a upgrading lookup. No pytest."""
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
print("EC142 F0a upgrading-lookup tests")

# 1. recovery = 1 - slip = 0.98 (datasheet 2% slip)
r = m.predict({"feedstock": "cattle_manure", "biogas_flow_m3_h": 100})
assert_true(abs(r["ch4_recovery"] - 0.98) < 1e-9, "CH4 recovery = 0.98 (2% slip datasheet)")

# 2. product purity meets grid spec >= 96%
assert_true(r["product_CH4_pct"] >= 96.0, "product CH4 purity >= 96% (grid spec)")

# 3. biomethane per biogas ratio physically bounded 0<ratio<1 for raw<purity
for fs in ["corn_silage", "grass_silage", "cattle_manure", "food_waste", "sewage_sludge"]:
    ratio = m.predict({"feedstock": fs})["biomethane_per_biogas"]
    assert_true(0.0 < ratio < 1.0, f"{fs} biomethane/biogas ratio in (0,1): {ratio:.3f}")

# 4. higher raw CH4 -> more biomethane (monotonic)
ratios = [m.predict({"raw_CH4_pct": c})["biomethane_per_biogas"] for c in (52, 54, 60, 62, 65)]
assert_true(all(ratios[i] < ratios[i + 1] for i in range(len(ratios) - 1)),
            "biomethane yield monotonic in raw CH4")

# 5. parasitic fraction is small & positive (PSA ~ a few %)
assert_true(0.0 < r["parasitic_fraction"] < 0.15, f"parasitic fraction in (0,0.15): {r['parasitic_fraction']:.3f}")

# 6. flows scale linearly
r2 = m.predict({"feedstock": "food_waste", "biogas_flow_m3_h": 1000})
r1 = m.predict({"feedstock": "food_waste", "biogas_flow_m3_h": 500})
assert_true(abs(r2["biomethane_m3_h"] - 2 * r1["biomethane_m3_h"]) < 1e-6, "biomethane flow scales linearly")

# 7. predict() interface keys present
assert_true(all(k in r for k in ("biomethane_m3_h", "upgrading_power_kw", "energy_out_kw")),
            "predict() keys present")

# 8. fast benchmark
t0 = time.time()
for _ in range(1000):
    m.predict({"feedstock": "sewage_sludge", "biogas_flow_m3_h": 250})
dt = time.time() - t0
assert_true(dt < 1.0, f"1000 predicts < 1s ({dt*1000:.1f} ms)")

print(f"\n{'PASS' if failed == 0 else 'FAIL'}: {failed} failed")
sys.exit(0 if failed == 0 else 1)
