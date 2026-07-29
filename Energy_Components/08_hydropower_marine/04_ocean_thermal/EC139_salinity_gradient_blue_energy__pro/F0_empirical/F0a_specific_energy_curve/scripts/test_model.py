"""EC139 F0a — tests (no pytest)."""
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

# 1. design point Csw=35 -> SE ~0.3658 kWh/m3
SE = float(m.predict({"C_seawater_g_per_L": 35.0})["specific_energy_kWh_per_m3"])
assert_true(abs(SE - 0.3658) < 1e-3, f"SE ~0.3658 kWh/m3 at Csw=35 (got {SE:.4f})")

# 2. SE in plausible PRO range (0.1-0.8 kWh/m3)
assert_true(0.1 < SE < 0.8, f"SE in PRO range (got {SE:.4f})")

# 3. net power at design ~212.7 kW
P = float(m.predict({"C_seawater_g_per_L": 35.0})["net_power_kw"])
assert_true(abs(P - 212.70) < 0.5, f"net power ~212.7 kW at design (got {P:.1f})")

# 4. monotonic increasing with concentration
Cs = np.array([25.0, 30.0, 35.0, 40.0])
SEs = m.predict({"C_seawater_g_per_L": Cs})["specific_energy_kWh_per_m3"]
assert_true(np.all(np.diff(SEs) > 0), "specific energy monotonic increasing in Csw")

# 5. endpoint low Csw=25 -> 0.2598
SE_lo = float(m.predict({"C_seawater_g_per_L": 25.0})["specific_energy_kWh_per_m3"])
assert_true(abs(SE_lo - 0.2598) < 1e-3, f"endpoint Csw=25 -> 0.2598 (got {SE_lo:.4f})")

# 6. power scales with SE (ratio consistent)
P_lo = float(m.predict({"C_seawater_g_per_L": 25.0})["net_power_kw"])
assert_true(abs(P_lo / SE_lo - P / SE) < 1.0, "power/SE ratio consistent (linear coupling)")

# 7. metadata
assert_true(info["component_id"] == "EC139" and info["version"] == "1.0.0", "get_info metadata correct")

# 8. fast benchmark
C = np.linspace(25, 40, 1000)
t0 = time.perf_counter()
m.predict({"C_seawater_g_per_L": C})
dt = time.perf_counter() - t0
assert_true(dt < 0.5, f"1000 predictions fast ({dt*1e3:.2f} ms)")

print(f"\n{passed} passed, {failed} failed")
sys.exit(0 if failed == 0 else 1)
