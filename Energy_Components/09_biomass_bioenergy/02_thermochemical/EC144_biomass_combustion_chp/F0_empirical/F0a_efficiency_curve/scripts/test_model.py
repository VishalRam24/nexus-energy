"""Tailored tests for EC144 F0a part-load efficiency curve. No pytest."""
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
print("EC144 F0a efficiency-curve tests")

# 1. rated point: at PLR=1, dry, eta_el = 0.22*0.95 (datasheet ref * full-load mult)
r = m.predict({"PLR": 1.0, "moisture": 0.0})
assert_true(abs(r["eta_electrical"] - 0.22 * 0.95) < 1e-6,
            "eta_el at PLR=1 dry = 0.22*0.95 (datasheet)")
assert_true(abs(r["eta_thermal"] - 0.55 * 0.95) < 1e-6,
            "eta_th at PLR=1 dry = 0.55*0.95 (datasheet)")

# 2. physical bounds: efficiencies in (0,1), total < 1
for plr in (0.2, 0.5, 0.8, 1.0):
    rr = m.predict({"PLR": plr})
    assert_true(0.0 < rr["eta_electrical"] < 1.0 and 0.0 < rr["eta_total"] < 1.0,
                f"efficiencies in (0,1) at PLR={plr}: tot={rr['eta_total']:.3f}")

# 3. part-load monotonic: efficiency increases with PLR
etas = [m.predict({"PLR": p})["eta_total"] for p in (0.2, 0.4, 0.6, 0.8, 1.0)]
assert_true(all(etas[i] < etas[i + 1] for i in range(len(etas) - 1)),
            "eta_total increases with PLR")

# 4. moisture reduces efficiency
dry = m.predict({"PLR": 1.0, "moisture": 0.0})["eta_total"]
wet = m.predict({"PLR": 1.0, "moisture": 0.4})["eta_total"]
assert_true(wet < dry, "moisture reduces efficiency")

# 5. PLR clamped below PLR_min (0.1 treated as 0.2)
assert_true(abs(m.predict({"PLR": 0.1})["eta_total"] - m.predict({"PLR": 0.2})["eta_total"]) < 1e-9,
            "PLR clamped to PLR_min=0.2")

# 6. power = eta * fuel power
r2 = m.predict({"PLR": 1.0, "fuel_power_kw": 5000})
assert_true(abs(r2["P_electrical_kw"] - r2["eta_electrical"] * 5000) < 1e-6,
            "P_electrical = eta_el * fuel power")

# 7. predict() interface keys present
assert_true(all(k in r for k in ("eta_electrical", "eta_thermal", "eta_total", "P_electrical_kw")),
            "predict() keys present")

# 8. fast benchmark
t0 = time.time()
for _ in range(1000):
    m.predict({"PLR": 0.7, "moisture": 0.2})
dt = time.time() - t0
assert_true(dt < 1.0, f"1000 predicts < 1s ({dt*1000:.1f} ms)")

print(f"\n{'PASS' if failed == 0 else 'FAIL'}: {failed} failed")
sys.exit(0 if failed == 0 else 1)
