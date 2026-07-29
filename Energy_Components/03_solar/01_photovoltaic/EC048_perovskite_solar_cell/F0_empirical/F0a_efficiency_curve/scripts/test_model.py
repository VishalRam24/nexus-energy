"""Tests for EC048 Perovskite Solar Cell F0a empirical model. No pytest."""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from predict import ComponentModel  # noqa: E402

_passed = 0
_failed = 0


def assert_true(cond, msg):
    global _passed, _failed
    if cond:
        _passed += 1
        print(f"  \u2713 {msg}")
    else:
        _failed += 1
        print(f"  \u2717 {msg}")


m = ComponentModel()
P_STC = m.p_stc

# 1. STC power matches nameplate
p = m.predict({"irradiance": 1000.0, "cell_temperature": 25.0})["power"]
assert_true(abs(p - P_STC) / P_STC < 0.01, f"STC power == nameplate ({p:.3f} ~ {P_STC})")

# 2. efficiency in physical bounds
e = m.predict({"irradiance": 1000.0, "cell_temperature": 25.0})["efficiency"]
assert_true(0.0 < e < 1.0, f"0 < eff < 1 ({e:.4f})")

# 3. zero irradiance -> zero power
assert_true(m.predict({"irradiance": 0.0, "cell_temperature": 25.0})["power"] == 0.0,
            "P(G=0) == 0")

# 4. monotonic increase in power with irradiance
ps = [m.predict({"irradiance": g, "cell_temperature": 25.0})["power"]
      for g in (100, 300, 600, 900, 1100)]
assert_true(all(b > a for a, b in zip(ps, ps[1:])), "power monotonic in irradiance")

# 5. hotter cell -> lower power (negative gamma_pmp)
p25 = m.predict({"irradiance": 1000.0, "cell_temperature": 25.0})["power"]
p55 = m.predict({"irradiance": 1000.0, "cell_temperature": 55.0})["power"]
assert_true(p55 < p25, f"hotter cell lowers power ({p55:.2f} < {p25:.2f})")

# 6. low-light relative efficiency below STC efficiency
e_low = m.predict({"irradiance": 100.0, "cell_temperature": 25.0})["efficiency"]
assert_true(e_low < e, f"low-light eff < STC eff ({e_low:.4f} < {e:.4f})")

# 7. get_info interface
info = m.get_info()
assert_true(info["component_id"] == "EC048" and info["fidelity"].startswith("F0a"),
            "get_info id/fidelity correct")

# 8. benchmark
t0 = time.time()
for _ in range(1000):
    m.predict({"irradiance": 800.0, "cell_temperature": 40.0})
dt = (time.time() - t0) * 1e3
assert_true(dt < 2000, f"1000 predicts fast ({dt:.1f} ms)")

print(f"\n{_passed} passed, {_failed} failed")
sys.exit(0 if _failed == 0 else 1)
