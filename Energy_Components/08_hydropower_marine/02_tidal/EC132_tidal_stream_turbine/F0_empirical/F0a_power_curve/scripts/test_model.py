"""EC132 F0a — tests (no pytest)."""
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

# 1. rated power reached at rated speed
P = float(m.predict({"current_speed_ms": 2.524})["power_kw"])
assert_true(abs(P - 1000.0) < 1.0, f"rated power 1000 kW at 2.524 m/s (got {P:.1f})")

# 2. below cut-in -> zero
P0 = float(m.predict({"current_speed_ms": 0.5})["power_kw"])
assert_true(P0 == 0.0, f"below cut-in power=0 (got {P0:.1f})")

# 3. above cut-out -> zero
Pco = float(m.predict({"current_speed_ms": 4.3})["power_kw"])
assert_true(Pco == 0.0, f"above cut-out power=0 (got {Pco:.1f})")

# 4. monotonic in cube-law region
vs = np.array([1.0, 1.5, 2.0, 2.5])
Ps = m.predict({"current_speed_ms": vs})["power_kw"]
assert_true(np.all(np.diff(Ps) > 0), "power monotonic in cube-law region")

# 5. flat at rated between rated and cut-out
P3 = float(m.predict({"current_speed_ms": 3.5})["power_kw"])
assert_true(abs(P3 - 1000.0) < 1.0, f"flat at rated 3.5 m/s (got {P3:.1f})")

# 6. roughly cubic: P at 2v ~ 8x P at v (1.0 -> 2.0, both below rated)
P1 = float(m.predict({"current_speed_ms": 1.0})["power_kw"])
P2 = float(m.predict({"current_speed_ms": 2.0})["power_kw"])
assert_true(6.0 < P2 / P1 < 10.0, f"cube-law P(2v)/P(v)~8 (got {P2/P1:.1f})")

# 7. metadata + power bounded by rated
assert_true(info["component_id"] == "EC132" and P3 <= 1000.0 + 1e-6, "metadata + power<=rated")

# 8. fast benchmark
v = np.linspace(0, 4.5, 1000)
t0 = time.perf_counter()
m.predict({"current_speed_ms": v})
dt = time.perf_counter() - t0
assert_true(dt < 0.5, f"1000 predictions fast ({dt*1e3:.2f} ms)")

print(f"\n{passed} passed, {failed} failed")
sys.exit(0 if failed == 0 else 1)
