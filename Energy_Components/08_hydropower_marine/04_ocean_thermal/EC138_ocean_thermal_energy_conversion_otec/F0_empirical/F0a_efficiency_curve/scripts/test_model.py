"""EC138 F0a — tests (no pytest)."""
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

# 1. design point dT=21 C -> eta_net ~3.44%
eta = float(m.predict({"dT_c": 21.0})["net_efficiency"])
assert_true(abs(eta - 0.0344) < 1e-4, f"net eff ~3.44% at dT=21 (got {eta*100:.2f}%)")

# 2. efficiency small but positive and well below Carnot (physical for OTEC)
assert_true(0.0 < eta < 0.10, f"0<eta<0.10 OTEC range (got {eta:.4f})")

# 3. below dT_min -> zero
e0 = float(m.predict({"dT_c": 8.0})["net_efficiency"])
assert_true(e0 == 0.0, f"below dT_min eff=0 (got {e0:.4f})")

# 4. monotonic increasing with dT
dTs = np.array([12.0, 16.0, 20.0, 24.0])
es = m.predict({"dT_c": dTs})["net_efficiency"]
assert_true(np.all(np.diff(es) > 0), "efficiency monotonic increasing in dT")

# 5. accepts T_warm/T_cold form
r2 = m.predict({"T_warm_c": 26.0, "T_cold_c": 5.0})
assert_true(abs(float(r2["net_efficiency"]) - 0.0344) < 1e-4, "Tw/Tc input form matches dT form")

# 6. net power at design ~34.4 kW
P = float(m.predict({"dT_c": 21.0})["net_power_kw"])
assert_true(abs(P - 34.4) < 0.2, f"net power ~34.4 kW at design (got {P:.1f})")

# 7. metadata
assert_true(info["component_id"] == "EC138" and info["version"] == "1.0.0", "get_info metadata correct")

# 8. fast benchmark
dT = np.linspace(10, 28, 1000)
t0 = time.perf_counter()
m.predict({"dT_c": dT})
dt = time.perf_counter() - t0
assert_true(dt < 0.5, f"1000 predictions fast ({dt*1e3:.2f} ms)")

print(f"\n{passed} passed, {failed} failed")
sys.exit(0 if failed == 0 else 1)
