"""EC128 F0a — tests (no pytest)."""
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

# 1. rated point matches datasheet (~27 MW)
P = float(m.predict({"flow_rate_m3s": 30.0, "head_m": 100.0})["power_kw"])
assert_true(abs(P - 27000.0) / 27000.0 < 0.02, f"rated power ~27000 kW (got {P:.0f})")

# 2. efficiency in physical bounds
eta = float(m.predict({"flow_rate_m3s": 30.0})["overall_efficiency"])
assert_true(0.0 < eta < 1.0, f"0<eta<1 (got {eta:.3f})")

# 3. monotonic-ish: efficiency rises from low flow to design
e_lo = float(m.predict({"flow_rate_m3s": 9.0})["overall_efficiency"])
e_hi = float(m.predict({"flow_rate_m3s": 27.0})["overall_efficiency"])
assert_true(e_lo < e_hi, f"efficiency increases toward design ({e_lo:.3f}<{e_hi:.3f})")

# 4. endpoint/clamp: below table low end returns first value
e_clamp = float(m.predict({"flow_rate_m3s": 0.0})["overall_efficiency"])
assert_true(abs(e_clamp - 0.700) < 1e-9, f"low-flow clamps to 0.700 (got {e_clamp:.3f})")

# 5. capacity factor at design ~1
cf = float(m.predict({"flow_rate_m3s": 30.0, "head_m": 100.0})["capacity_factor"])
assert_true(abs(cf - 1.0) < 0.02, f"CF~1 at design (got {cf:.3f})")

# 6. power increases with head
P1 = float(m.predict({"flow_rate_m3s": 20.0, "head_m": 80.0})["power_kw"])
P2 = float(m.predict({"flow_rate_m3s": 20.0, "head_m": 120.0})["power_kw"])
assert_true(P2 > P1, f"power rises with head ({P1:.0f}<{P2:.0f})")

# 7. predict interface metadata
assert_true(info["component_id"] == "EC128" and info["version"] == "1.0.0", "get_info metadata correct")

# 8. fast benchmark
Q = np.linspace(0, 33, 1000)
t0 = time.perf_counter()
m.predict({"flow_rate_m3s": Q, "head_m": 100.0})
dt = time.perf_counter() - t0
assert_true(dt < 0.5, f"1000 predictions fast ({dt*1e3:.2f} ms)")

print(f"\n{passed} passed, {failed} failed")
sys.exit(0 if failed == 0 else 1)
