"""
EC008 -- PEMEL -- F2a Electrochemical -- Test suite.
"""
import sys, os, time
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from model import PEMEL_F2a
from predict import ComponentModel

PASS, FAIL = "\u2713", "\u2717"
def assert_true(c, m):
    if c: print(f"  {PASS}  {m}")
    else: print(f"  {FAIL}  FAILED: {m}"); raise AssertionError(m)

def make():
    cm = ComponentModel()
    return cm._model, cm

def test_reversible_voltage():
    print("\n[Test 1] Reversible voltage range")
    m, _ = make()
    E = m.reversible_voltage(353.15, 1.0)
    assert_true(1.1 < E < 1.35, f"E_rev={E:.4f}")

def test_voltage_above_rev():
    print("\n[Test 2] V_cell > E_rev (electrolyser)")
    m, _ = make()
    for j in [0.5, 1.0, 2.0]:
        V = m.cell_voltage(j, 353.15)
        E = m.reversible_voltage(353.15)
        assert_true(V > E, f"V({j})={V:.4f} > E_rev={E:.4f}")

def test_voltage_increases_with_j():
    print("\n[Test 3] V increases with j (electrolyser)")
    m, _ = make()
    V_prev = m.cell_voltage(0.1, 353.15)
    for j in np.linspace(0.2, 2.5, 20):
        V = m.cell_voltage(j, 353.15)
        assert_true(V >= V_prev - 1e-6, f"V({j:.1f}) >= V_prev")
        V_prev = V

def test_h2_production():
    print("\n[Test 4] H2 production > 0")
    m, _ = make()
    h2 = m.h2_production_rate(1.0, 353.15)
    assert_true(h2 > 0, f"H2 rate={h2:.6e} mol/s > 0")

def test_faradaic_efficiency():
    print("\n[Test 5] Faradaic efficiency in (0,1)")
    m, _ = make()
    for j in [0.1, 0.5, 1.0, 2.0]:
        eta = m.faradaic_efficiency(j)
        assert_true(0 < eta <= 1.0, f"eta_F({j})={eta:.4f}")

def test_thermal_heats():
    print("\n[Test 6] Stack heats up under load")
    m, _ = make()
    r = m.simulate(1.5, 333.15, 30.0, 0.5, 120.0)
    assert_true(r["temperature"][-1] > 333.15, "T increases under load")

def test_efficiency_range():
    print("\n[Test 7] Efficiency in reasonable range")
    m, _ = make()
    r = m.simulate(1.0, 353.15, 30.0, 1.0, 10.0)
    for eta in r["efficiency"]:
        assert_true(0 < eta < 1.2, f"eta={eta:.4f}")

def test_predict():
    print("\n[Test 8] predict() interface")
    _, cm = make()
    r = cm.predict({"current_density": 1.0, "dt": 1, "duration_s": 5})
    for k in ["t","voltage","h2_production_kg_s","efficiency","temperature"]:
        assert_true(k in r, f"Key '{k}'")

def test_benchmark():
    print("\n[Test 9] Benchmark")
    m, _ = make()
    t0 = time.perf_counter()
    m.simulate(1.0, 353.15, 30, 0.1, 60)
    print(f"  60s sim in {(time.perf_counter()-t0)*1000:.0f} ms")

if __name__ == "__main__":
    tests = [test_reversible_voltage, test_voltage_above_rev, test_voltage_increases_with_j,
             test_h2_production, test_faradaic_efficiency, test_thermal_heats,
             test_efficiency_range, test_predict, test_benchmark]
    p = f = 0
    for t in tests:
        try: t(); p += 1
        except Exception as e: f += 1; print(f"  ERROR: {e}")
    print(f"\n{'='*60}\nEC008 PEMEL F2a -- {p} passed, {f} failed\n{'='*60}")
    sys.exit(0 if f == 0 else 1)
