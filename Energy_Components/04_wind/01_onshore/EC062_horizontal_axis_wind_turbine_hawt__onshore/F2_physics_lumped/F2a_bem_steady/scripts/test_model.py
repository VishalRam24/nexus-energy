"""
EC062 -- HAWT Onshore -- F2a BEM Steady -- Test suite.
"""
import sys, os, time
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from model import HAWT_BEM_F2a
from predict import ComponentModel

PASS, FAIL = "\u2713", "\u2717"
def assert_true(c, m):
    if c: print(f"  {PASS}  {m}")
    else: print(f"  {FAIL}  FAILED: {m}"); raise AssertionError(m)

def make():
    cm = ComponentModel()
    return cm._model, cm

def test_betz_limit():
    print("\n[Test 1] Cp <= Betz limit (16/27 ~ 0.593)")
    m, _ = make()
    for V in [6, 8, 10, 12, 14]:
        r = m.solve(V)
        assert_true(r["Cp"] <= 0.60, f"Cp({V}m/s)={r['Cp']:.3f} <= 0.60")

def test_power_positive():
    print("\n[Test 2] Power > 0 for operational wind")
    m, _ = make()
    for V in [5, 8, 10, 12]:
        r = m.solve(V)
        assert_true(r["power_kw"] > 0, f"P({V}m/s)={r['power_kw']:.1f} kW > 0")

def test_power_increases_with_wind():
    print("\n[Test 3] Power increases with wind (below rated)")
    m, _ = make()
    P_prev = m.solve(5.0)["power_kw"]
    for V in [7, 9, 11]:
        P = m.solve(V)["power_kw"]
        assert_true(P > P_prev, f"P({V})={P:.0f} > P_prev={P_prev:.0f}")
        P_prev = P

def test_thrust_positive():
    print("\n[Test 4] Thrust > 0")
    m, _ = make()
    r = m.solve(10.0)
    assert_true(r["thrust_kN"] > 0, f"Thrust={r['thrust_kN']:.1f} kN > 0")

def test_blade_loads():
    print("\n[Test 5] Blade loads array has N_elements entries")
    m, _ = make()
    r = m.solve(10.0)
    assert_true(len(r["blade_loads"]) == m.N_el, f"N_loads={len(r['blade_loads'])} == {m.N_el}")

def test_induction_factors():
    print("\n[Test 6] Induction factors in valid range")
    m, _ = make()
    r = m.solve(10.0)
    for bl in r["blade_loads"]:
        assert_true(0 <= bl["a"] <= 0.95, f"a={bl['a']:.3f}")

def test_pitch_effect():
    print("\n[Test 7] Pitch reduces power")
    m, _ = make()
    r0 = m.solve(10.0, pitch_deg=0)
    r5 = m.solve(10.0, pitch_deg=10)
    assert_true(r0["power_kw"] > r5["power_kw"],
                f"P(pitch=0)={r0['power_kw']:.0f} > P(pitch=10)={r5['power_kw']:.0f}")

def test_predict_interface():
    print("\n[Test 8] ComponentModel predict()")
    _, cm = make()
    r = cm.predict({"wind_speed_m_s": 10.0})
    for k in ["power_kw", "thrust_kN", "Cp", "Ct", "blade_loads"]:
        assert_true(k in r, f"Key '{k}'")

def test_benchmark():
    print("\n[Test 9] Benchmark: single BEM solve")
    m, _ = make()
    t0 = time.perf_counter()
    for _ in range(100):
        m.solve(10.0)
    elapsed = (time.perf_counter() - t0) / 100
    print(f"  Single solve in {elapsed*1000:.1f} ms")
    assert_true(elapsed < 1.0, "< 1 s per solve")

if __name__ == "__main__":
    tests = [test_betz_limit, test_power_positive, test_power_increases_with_wind,
             test_thrust_positive, test_blade_loads, test_induction_factors,
             test_pitch_effect, test_predict_interface, test_benchmark]
    p = f = 0
    for t in tests:
        try: t(); p += 1
        except Exception as e: f += 1; print(f"  ERROR: {e}")
    print(f"\n{'='*60}\nEC062 HAWT F2a BEM -- {p} passed, {f} failed\n{'='*60}")
    sys.exit(0 if f == 0 else 1)
