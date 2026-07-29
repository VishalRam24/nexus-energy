"""
EC002 -- SOFC -- F2a Electrochemical -- Test suite.
"""
import sys, os, time
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from model import SOFC_F2a
from predict import ComponentModel

PASS, FAIL = "\u2713", "\u2717"

def assert_true(cond, msg):
    if cond: print(f"  {PASS}  {msg}")
    else:
        print(f"  {FAIL}  FAILED: {msg}")
        raise AssertionError(msg)

def make_model():
    cm = ComponentModel()
    return cm._model, cm

def test_nernst():
    print("\n[Test 1] Nernst voltage range")
    m, _ = make_model()
    E = m.nernst_voltage(1073.15)
    assert_true(0.8 < E < 1.2, f"E={E:.4f} in [0.8, 1.2]")

def test_voltage_below_nernst():
    print("\n[Test 2] V < E_nernst for j>0")
    m, _ = make_model()
    for j in [0.1, 0.5, 1.0]:
        V = m.cell_voltage(j, 1073.15)
        E = m.nernst_voltage(1073.15)
        assert_true(V < E, f"V({j})={V:.4f} < E={E:.4f}")

def test_voltage_monotone():
    print("\n[Test 3] V decreases with j")
    m, _ = make_model()
    j_vals = np.linspace(0.01, 2.0, 40)
    V_prev = m.cell_voltage(j_vals[0], 1073.15)
    for j in j_vals[1:]:
        V = m.cell_voltage(j, 1073.15)
        assert_true(V <= V_prev + 1e-6, f"V({j:.2f}) <= V_prev")
        V_prev = V

def test_thermal_dynamics():
    print("\n[Test 4] Thermal ODE converges")
    m, _ = make_model()
    r = m.simulate(0.5, 1073.15, dt=5.0, duration_s=3000.0)
    assert_true(r["temperature"][-1] > 1050, f"T_final={r['temperature'][-1]:.1f} > 1050")
    assert_true(r["temperature"][-1] < 1300, f"T_final < 1300 K")

def test_efficiency():
    print("\n[Test 5] Efficiency in (0,1)")
    m, _ = make_model()
    r = m.simulate(0.5, 1073.15, dt=1, duration_s=5)
    for eta in r["efficiency"]:
        assert_true(0 < eta < 1.0, f"eta={eta:.4f}")

def test_ysz_conductivity():
    print("\n[Test 6] YSZ conductivity increases with T")
    m, _ = make_model()
    s1 = m.ysz_conductivity(973.15)
    s2 = m.ysz_conductivity(1173.15)
    assert_true(s2 > s1, f"sigma(1173)={s2:.4f} > sigma(973)={s1:.4f}")

def test_predict_interface():
    print("\n[Test 7] ComponentModel predict()")
    _, cm = make_model()
    r = cm.predict({"current_density": 0.5, "dt": 1, "duration_s": 5})
    for k in ["t","voltage","power_density","efficiency","temperature","fuel_utilization"]:
        assert_true(k in r, f"Key '{k}' present")

def test_benchmark():
    print("\n[Test 8] Benchmark: 600s sim")
    m, _ = make_model()
    t0 = time.perf_counter()
    m.simulate(0.5, 1073.15, dt=1, duration_s=600)
    elapsed = time.perf_counter() - t0
    print(f"  600s sim in {elapsed*1000:.0f} ms")
    assert_true(elapsed < 30, "< 30 s")

if __name__ == "__main__":
    tests = [test_nernst, test_voltage_below_nernst, test_voltage_monotone,
             test_thermal_dynamics, test_efficiency, test_ysz_conductivity,
             test_predict_interface, test_benchmark]
    passed = failed = 0
    for t in tests:
        try: t(); passed += 1
        except Exception as e: failed += 1; print(f"  ERROR: {e}")
    print(f"\n{'='*60}\nEC002 SOFC F2a -- {passed} passed, {failed} failed\n{'='*60}")
    sys.exit(0 if failed == 0 else 1)
