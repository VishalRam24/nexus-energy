"""Tests for EC059 Evacuated Tube Solar Collector F0a empirical model. No pytest."""
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
ETA0 = m.eta0

# 1. peak efficiency (dT=0) == eta0
e0 = m.predict({K: 1000.0, "delta_T": 0.0})["efficiency"]
assert_true(abs(e0 - ETA0) < 1e-6, f"eff(dT=0) == eta0 ({e0:.4f} ~ {ETA0})")

# 2. efficiency physical bounds
assert_true(0.0 <= e0 <= 1.0, f"0 <= eff <= 1 ({e0:.4f})")

# 3. efficiency drops as dT rises (heat loss), unless device has no loss term
has_loss = (m.curve.a1 != 0.0) or (m.curve.a2 != 0.0)
e_hot = m.predict({K: 1000.0, "delta_T": 100.0})["efficiency"]
if has_loss:
    assert_true(e_hot < e0, f"eff falls with dT ({e_hot:.4f} < {e0:.4f})")
else:
    assert_true(abs(e_hot - e0) < 1e-9, f"no-loss device: eff constant with dT ({e_hot:.4f})")

# 4. efficiency never negative even at extreme dT
e_ext = m.predict({K: 1000.0, "delta_T": 500.0})["efficiency"]
assert_true(e_ext >= 0.0, f"eff clipped >= 0 at extreme dT ({e_ext:.4f})")

# 5. zero irradiance -> zero output
assert_true(m.predict({K: 0.0, "delta_T": 50.0})["power_density"] == 0.0,
            "P(G=0) == 0")

# 6. monotonic output power with irradiance at dT=0 (no clipping artifacts)
ps = [m.predict({K: g, "delta_T": 0.0})["power_density"] for g in (200, 400, 700, 1000)]
assert_true(all(b > a for a, b in zip(ps, ps[1:])), "output power monotonic in irradiance")

# 7. T_mean/T_ambient path equals delta_T path
a = m.predict({K: 900.0, "T_mean": 90.0, "T_ambient": 25.0})["efficiency"]
b = m.predict({K: 900.0, "delta_T": 65.0})["efficiency"]
assert_true(abs(a - b) < 1e-9, "T_mean/T_amb path == delta_T path")

# 8. get_info + benchmark
info = m.get_info()
assert_true(info["component_id"] == "EC059" and info["fidelity"].startswith("F0a"),
            "get_info id/fidelity correct")
t0 = time.time()
for _ in range(1000):
    m.predict({K: 800.0, "delta_T": 50.0})
dt = (time.time() - t0) * 1e3
assert_true(dt < 2000, f"1000 predicts fast ({dt:.1f} ms)")

print(f"\n{_passed} passed, {_failed} failed")
sys.exit(0 if _failed == 0 else 1)
