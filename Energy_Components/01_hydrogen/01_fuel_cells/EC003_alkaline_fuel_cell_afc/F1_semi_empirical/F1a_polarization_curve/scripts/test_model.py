"""
EC003 -- Alkaline Fuel Cell (AFC) -- F1a Polarization Curve
Test suite: physics sanity checks, edge cases, benchmark.
"""

import sys, os, time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from model import AFCModel
from predict import ComponentModel

PASS = "\u2713"
FAIL = "\u2717"


def assert_true(cond, msg):
    if cond:
        print(f"  {PASS}  {msg}")
    else:
        print(f"  {FAIL}  FAILED: {msg}")
        raise AssertionError(msg)


def make_model():
    return AFCModel({
        "T": 343.15, "N_cells": 40, "A_cell": 200.0,
        "pH2": 1.0, "pO2": 0.21, "j_L": 1.0,
        "i0": 0.005, "alpha": 0.5, "R_ohm": 0.25, "B_conc": 0.010,
    })


def test_nernst_range():
    print("\n[Test 1] Nernst voltage in physical range")
    m = make_model()
    E = m.nernst_voltage()
    assert_true(0.9 < E < 1.3, f"E_Nernst={E:.4f} V in [0.9, 1.3]")
    E_hot = m.nernst_voltage(T=363.15)
    E_cold = m.nernst_voltage(T=323.15)
    assert_true(E_cold > E_hot, "E_Nernst decreases with temperature")


def test_voltage_monotonically_decreasing():
    print("\n[Test 2] V_cell decreases monotonically with j")
    m = make_model()
    j_vals = np.linspace(0.01, 0.9, 50)
    V_vals = [m.cell_voltage(j) for j in j_vals]
    for i in range(1, len(V_vals)):
        assert_true(V_vals[i] <= V_vals[i-1] + 1e-9,
                    f"V({j_vals[i]:.3f}) <= V({j_vals[i-1]:.3f})")
    print("  All consecutive pairs checked.")


def test_voltage_below_nernst():
    print("\n[Test 3] V_cell < E_Nernst for j > 0")
    m = make_model()
    E = m.nernst_voltage()
    for j in [0.01, 0.1, 0.3, 0.5, 0.8]:
        V = m.cell_voltage(j)
        assert_true(V < E, f"V({j})={V:.4f} < E_Nernst={E:.4f}")


def test_efficiency_below_unity():
    print("\n[Test 4] Efficiency in (0, 1)")
    m = make_model()
    for j in [0.01, 0.1, 0.3, 0.5, 0.8]:
        eta = m.efficiency(j)
        assert_true(0.0 < eta < 1.0, f"eta={eta:.4f} at j={j}")


def test_predict_interface():
    print("\n[Test 5] ComponentModel predict() interface")
    cm = ComponentModel()
    out = cm.predict({"current_density": 0.3, "temperature": 70.0})
    required = ["cell_voltage_V", "stack_voltage_V", "power_density_W_cm2",
                "stack_power_W", "efficiency", "E_Nernst_V", "V_act_V",
                "V_ohm_V", "V_conc_V"]
    for k in required:
        assert_true(k in out, f"Output key '{k}' present")
    assert_true(out["cell_voltage_V"] > 0, "cell_voltage > 0")
    assert_true(out["cell_voltage_V"] < out["E_Nernst_V"], "cell_voltage < E_Nernst")
    assert_true(0 < out["efficiency"] < 1.0, "efficiency in (0,1)")


def test_get_info():
    print("\n[Test 6] get_info() completeness")
    cm = ComponentModel()
    info = cm.get_info()
    for k in ["component_id", "component_name", "fidelity", "inputs", "outputs", "source"]:
        assert_true(k in info, f"'{k}' in get_info()")


def test_benchmark():
    print("\n[Test 7] Benchmark: 10,000 evaluate() calls")
    m = make_model()
    j_vals = np.linspace(0.01, 0.9, 10000)
    t0 = time.perf_counter()
    for j in j_vals:
        m.evaluate(j, 70.0)
    elapsed = time.perf_counter() - t0
    print(f"  10,000 evaluations in {elapsed*1000:.1f} ms ({elapsed/10000*1e6:.2f} us/call)")
    assert_true(elapsed < 5.0, "10,000 calls < 5 s")


if __name__ == "__main__":
    tests = [test_nernst_range, test_voltage_monotonically_decreasing,
             test_voltage_below_nernst, test_efficiency_below_unity,
             test_predict_interface, test_get_info, test_benchmark]
    passed = failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except AssertionError:
            failed += 1
        except Exception as e:
            failed += 1
            print(f"  UNEXPECTED ERROR in {t.__name__}: {e}")

    print(f"\n{'='*50}")
    print(f"EC003 AFC F1a -- Results: {passed} passed, {failed} failed")
    print(f"{'='*50}")
    sys.exit(0 if failed == 0 else 1)
