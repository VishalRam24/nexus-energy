"""Tailored tests for EC150 F0a FT alpha curve. No pytest."""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from predict import ComponentModel

failed = 0


def assert_true(cond, msg):
    global failed
    if cond:
        print(f"  ✓ {msg}")
    else:
        failed += 1
        print(f"  ✗ {msg}")


m = ComponentModel()
print("EC150 F0a alpha-curve tests")

# 1. datasheet point: alpha at 230 C = 0.88, CO conv = 0.85
r = m.predict({"temperature_degC": 230})
assert_true(abs(r["alpha"] - 0.88) < 1e-6, "alpha at 230 C = 0.88 (datasheet)")
assert_true(abs(r["CO_conversion"] - 0.85) < 1e-6, "CO conversion at 230 C = 0.85 (datasheet)")

# 2. physical bounds: 0<alpha<1, 0<CO conv<1, 0<diesel sel<1
for T in (180, 230, 280, 350):
    rr = m.predict({"temperature_degC": T})
    assert_true(0.0 < rr["alpha"] < 1.0 and 0.0 < rr["CO_conversion"] < 1.0
                and 0.0 < rr["diesel_selectivity"] < 1.0,
                f"alpha/CO/diesel in (0,1) at {T} C")

# 3. alpha decreases with temperature (LTFT -> HTFT)
alphas = [m.predict({"temperature_degC": T})["alpha"] for T in (180, 210, 230, 280, 350)]
assert_true(all(alphas[i] > alphas[i + 1] for i in range(len(alphas) - 1)), "alpha falls with T")

# 4. ASF diesel selectivity is a unimodal function of alpha (peaks at intermediate alpha ~0.85-0.9)
#    -> diesel selectivity at very high alpha (low T, 180C, wax-heavy) is lower than at 230 C
sel_180 = m.predict({"temperature_degC": 180})["diesel_selectivity"]
sel_230 = m.predict({"temperature_degC": 230})["diesel_selectivity"]
assert_true(sel_230 > sel_180, "C10-C20 diesel cut higher at 230 C than wax-heavy 180 C")

# 5. ASF mass distribution sums to ~1 over all carbon numbers (sanity on the formula)
import numpy as np
a = m.curve.alpha(230)
n = np.arange(1, 200)
w = n * (1 - a) ** 2 * a ** (n - 1)
assert_true(abs(w.sum() - 1.0) < 0.02, f"ASF mass distribution sums to ~1: {w.sum():.3f}")

# 6. diesel yield proxy = CO conv * diesel selectivity
assert_true(abs(r["diesel_yield_proxy"] - r["CO_conversion"] * r["diesel_selectivity"]) < 1e-9,
            "diesel yield proxy = CO conv * diesel sel")

# 7. predict() interface keys present
assert_true(all(k in r for k in ("alpha", "CO_conversion", "diesel_selectivity")),
            "predict() keys present")

# 8. fast benchmark
t0 = time.time()
for _ in range(1000):
    m.predict({"temperature_degC": 240})
dt = time.time() - t0
assert_true(dt < 2.0, f"1000 predicts < 2s ({dt*1000:.1f} ms)")

print(f"\n{'PASS' if failed == 0 else 'FAIL'}: {failed} failed")
sys.exit(0 if failed == 0 else 1)
