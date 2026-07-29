"""EC130 F0a — tests (no pytest)."""
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

# 1. rated point ~497.7 kW (design-point power)
P = float(m.predict({"flow_rate_m3s": 1.5, "head_m": 40.0})["power_kw"])
assert_true(abs(P - 497.7) / 497.7 < 0.02, f"rated power ~497.7 kW (got {P:.1f})")

# 2. efficiency bounds
eta = float(m.predict({"flow_rate_m3s": 1.5})["overall_efficiency"])
assert_true(0.0 < eta < 1.0, f"0<eta<1 (got {eta:.3f})")

# 3. rated eta = 0.8455
assert_true(abs(eta - 0.8455) < 1e-3, f"rated eta=0.8455 (got {eta:.4f})")

# 4. monotonic toward design
e_lo = float(m.predict({"flow_rate_m3s": 0.45})["overall_efficiency"])
assert_true(e_lo < eta, f"efficiency increases toward design ({e_lo:.3f}<{eta:.3f})")

# 5. low-flow clamp 0.660
e_clamp = float(m.predict({"flow_rate_m3s": 0.0})["overall_efficiency"])
assert_true(abs(e_clamp - 0.660) < 1e-9, f"low-flow clamps to 0.660 (got {e_clamp:.3f})")

# 6. power rises with head
P1 = float(m.predict({"flow_rate_m3s": 1.2, "head_m": 30.0})["power_kw"])
P2 = float(m.predict({"flow_rate_m3s": 1.2, "head_m": 60.0})["power_kw"])
assert_true(P2 > P1, f"power rises with head ({P1:.0f}<{P2:.0f})")

# 7. metadata
assert_true(info["component_id"] == "EC130" and info["version"] == "1.0.0", "get_info metadata correct")

# 8. fast benchmark
Q = np.linspace(0, 1.65, 1000)
t0 = time.perf_counter()
m.predict({"flow_rate_m3s": Q, "head_m": 40.0})
dt = time.perf_counter() - t0
assert_true(dt < 0.5, f"1000 predictions fast ({dt*1e3:.2f} ms)")

print(f"\n{passed} passed, {failed} failed")
sys.exit(0 if failed == 0 else 1)
