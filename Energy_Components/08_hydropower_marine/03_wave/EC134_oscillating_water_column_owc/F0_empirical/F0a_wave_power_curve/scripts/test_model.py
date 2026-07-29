"""EC134 F0a — tests (no pytest)."""
import sys, time
import numpy as np
from predict import ComponentModel

passed = failed = 0


def assert_true(cond, msg):
    global passed, failed
    if cond:
        passed += 1
        print(f"  ✓ {msg}")
    else:
        failed += 1
        print(f"  ✗ {msg}")


m = ComponentModel()
info = m.get_info()
RATED = 80.0
DESIGN_HS = 3.0
DESIGN_P = 43.87

# 1. datasheet point at Hs=3 m
P = float(m.predict({"Hs_m": DESIGN_HS})["power_kw"])
assert_true(abs(P - DESIGN_P) < 0.05, f"power at Hs=3 m ~{DESIGN_P} kW (got {P:.2f})")

# 2. calm sea -> zero
P0 = float(m.predict({"Hs_m": 0.0})["power_kw"])
assert_true(P0 == 0.0, f"calm sea power=0 (got {P0:.2f})")

# 3. monotonic increasing
hs = np.array([0.5, 1.0, 2.0, 3.0])
Ps = m.predict({"Hs_m": hs})["power_kw"]
assert_true(np.all(np.diff(Ps) > 0), "power monotonic increasing in Hs")

# 4. roughly quadratic for small Hs: 2x Hs ~ 4x power
P1 = float(m.predict({"Hs_m": 1.0})["power_kw"])
P2 = float(m.predict({"Hs_m": 2.0})["power_kw"])
assert_true(abs(P2 / P1 - 4.0) < 0.1, f"P(2Hs)/P(Hs)~4 (got {P2/P1:.2f})")

# 5. power capped at rating
Pmax = float(m.predict({"Hs_m": 8.0})["power_kw"])
assert_true(abs(Pmax - RATED) < 1e-6, f"power capped at {RATED} kW (got {Pmax:.2f})")

# 6. CF in [0,1]
cf = float(m.predict({"Hs_m": DESIGN_HS})["capacity_factor"])
assert_true(0.0 <= cf <= 1.0, f"CF in [0,1] (got {cf:.3f})")

# 7. metadata
assert_true(info["component_id"] == "EC134" and info["version"] == "1.0.0", "get_info metadata correct")

# 8. fast benchmark
Hs = np.linspace(0, 8, 1000)
t0 = time.perf_counter()
m.predict({"Hs_m": Hs})
dt = time.perf_counter() - t0
assert_true(dt < 0.5, f"1000 predictions fast ({dt*1e3:.2f} ms)")

print(f"\n{passed} passed, {failed} failed")
sys.exit(0 if failed == 0 else 1)
