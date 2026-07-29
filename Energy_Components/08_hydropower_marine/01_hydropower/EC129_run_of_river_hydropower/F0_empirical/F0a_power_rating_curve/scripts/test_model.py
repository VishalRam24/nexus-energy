"""EC129 F0a — tests (no pytest)."""
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

# 1. rated point ~3200 kW
P = float(m.predict({"flow_rate_m3s": 50.0, "head_m": 8.0})["power_kw"])
assert_true(abs(P - 3200.0) / 3200.0 < 0.06, f"rated power ~3200 kW (got {P:.0f})")

# 2. efficiency in bounds
eta = float(m.predict({"flow_rate_m3s": 50.0})["overall_efficiency"])
assert_true(0.0 < eta < 1.0, f"0<eta<1 (got {eta:.3f})")

# 3. rated eta matches datasheet 0.8448
assert_true(abs(eta - 0.8448) < 1e-3, f"rated eta=0.8448 (got {eta:.4f})")

# 4. monotonic toward design
e_lo = float(m.predict({"flow_rate_m3s": 12.5})["overall_efficiency"])
assert_true(e_lo < eta, f"efficiency increases toward design ({e_lo:.3f}<{eta:.3f})")

# 5. low-flow clamp to 0.640
e_clamp = float(m.predict({"flow_rate_m3s": 0.0})["overall_efficiency"])
assert_true(abs(e_clamp - 0.640) < 1e-9, f"low-flow clamps to 0.640 (got {e_clamp:.3f})")

# 6. power rises with head
P1 = float(m.predict({"flow_rate_m3s": 40.0, "head_m": 6.0})["power_kw"])
P2 = float(m.predict({"flow_rate_m3s": 40.0, "head_m": 12.0})["power_kw"])
assert_true(P2 > P1, f"power rises with head ({P1:.0f}<{P2:.0f})")

# 7. metadata
assert_true(info["component_id"] == "EC129" and info["version"] == "1.0.0", "get_info metadata correct")

# 8. fast benchmark
Q = np.linspace(0, 57.5, 1000)
t0 = time.perf_counter()
m.predict({"flow_rate_m3s": Q, "head_m": 8.0})
dt = time.perf_counter() - t0
assert_true(dt < 0.5, f"1000 predictions fast ({dt*1e3:.2f} ms)")

print(f"\n{passed} passed, {failed} failed")
sys.exit(0 if failed == 0 else 1)
