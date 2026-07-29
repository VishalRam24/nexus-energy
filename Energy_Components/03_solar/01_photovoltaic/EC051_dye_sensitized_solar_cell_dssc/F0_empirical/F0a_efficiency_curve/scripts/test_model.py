"""Tests for EC051 Dye-Sensitized Solar Cell (DSSC) F0a empirical model. No pytest."""
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
K = m.irr_key
ETA = m.eta_stc

# 1. STC efficiency matches rated eta
e = m.predict({K: 1000.0, "cell_temperature": 25.0})["efficiency"]
assert_true(abs(e - ETA) / ETA < 0.01, f"STC eff == rated eta ({e:.4f} ~ {ETA})")

# 2. STC power density == eta*1000
pd = m.predict({K: 1000.0, "cell_temperature": 25.0})["power_density"]
assert_true(abs(pd - ETA * 1000.0) < 1.0, f"STC power density == eta*1000 ({pd:.2f})")

# 3. eff physical bounds
assert_true(0.0 < e < 1.0, f"0 < eff < 1 ({e:.4f})")

# 4. zero irradiance -> zero power
assert_true(m.predict({K: 0.0, "cell_temperature": 25.0})["power_density"] == 0.0,
            "P(G=0) == 0")

# 5. monotonic power density in irradiance
ps = [m.predict({K: g, "cell_temperature": 25.0})["power_density"]
      for g in (100, 300, 600, 900, 1100)]
assert_true(all(b > a for a, b in zip(ps, ps[1:])), "power density monotonic in irradiance")

# 6. hotter cell -> lower efficiency
e55 = m.predict({K: 1000.0, "cell_temperature": 55.0})["efficiency"]
assert_true(e55 < e, f"hotter cell lowers eff ({e55:.4f} < {e:.4f})")

# 7. get_info interface
info = m.get_info()
assert_true(info["component_id"] == "EC051" and info["fidelity"].startswith("F0a"),
            "get_info id/fidelity correct")

# 8. benchmark
t0 = time.time()
for _ in range(1000):
    m.predict({K: 800.0, "cell_temperature": 40.0})
dt = (time.time() - t0) * 1e3
assert_true(dt < 2000, f"1000 predicts fast ({dt:.1f} ms)")

print(f"\n{_passed} passed, {_failed} failed")
sys.exit(0 if _failed == 0 else 1)
