"""
EC002 -- Solid Oxide Fuel Cell (SOFC) -- F1b Thermal
Test suite: temperature-dependent physics sanity checks.
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from model import SOFCThermalModel
from predict import ComponentModel

DEFAULT_PARAMS = {
    "T_ref": 1073.0,
    "N_cells": 40,
    "A_cell": 100.0,
    "pH2": 0.97,
    "pO2": 0.21,
    "pH2O": 0.03,
    "j_L": 2.0,
    "A_sigma": 33400.0,
    "E_act_ion": 80000.0,
    "thickness_electrolyte": 0.001,
    "i0_anode_ref": 0.5,
    "E_act_anode": 100000.0,
    "i0_cathode_ref": 0.2,
    "E_act_cathode": 120000.0,
    "alpha": 0.5,
    "fuel_utilization": 0.7,
}

PASS = "\u2713"
FAIL = "\u2717"


def assert_true(cond, msg):
    if cond:
        print(f"  {PASS}  {msg}")
    else:
        print(f"  {FAIL}  FAILED: {msg}")
        raise AssertionError(msg)


def make_model():
    return SOFCThermalModel(DEFAULT_PARAMS)


def test_higher_temp_higher_voltage():
    """Higher T -> lower losses -> higher V_cell for fuel cell at moderate j."""
    print("\n[Test 1] Higher T -> higher V_cell at j=0.5 (fuel cell)")
    m = make_model()
    V_cold = float(m.cell_voltage(0.5, 973.0))
    V_hot = float(m.cell_voltage(0.5, 1173.0))
    assert_true(V_hot > V_cold,
                f"V(1173K)={V_hot:.4f} > V(973K)={V_cold:.4f}")


def test_ionic_conductivity_increases():
    print("\n[Test 2] YSZ ionic conductivity increases with T")
    m = make_model()
    s_cold = float(m.ionic_conductivity(973.0))
    s_hot = float(m.ionic_conductivity(1273.0))
    assert_true(s_hot > s_cold,
                f"sigma(1273K)={s_hot:.4f} > sigma(973K)={s_cold:.4f}")


def test_ohmic_asr_decreases():
    print("\n[Test 3] Ohmic ASR decreases with T")
    m = make_model()
    r_cold = float(m.ohmic_asr(973.0))
    r_hot = float(m.ohmic_asr(1273.0))
    assert_true(r_hot < r_cold,
                f"ASR(1273K)={r_hot:.4f} < ASR(973K)={r_cold:.4f}")


def test_voltage_bounds():
    print("\n[Test 4] Cell voltage within physical bounds")
    m = make_model()
    for T in [973.0, 1073.0, 1273.0]:
        for j in [0.01, 0.5, 1.0, 1.5]:
            V = float(m.cell_voltage(j, T))
            E = float(m.nernst_voltage(T))
            assert_true(0.0 <= V <= E + 0.001,
                        f"0 <= V({j},{T:.0f})={V:.4f} <= E={E:.4f}")


def test_heat_positive():
    print("\n[Test 5] Heat generation positive for j > 0")
    m = make_model()
    for T in [973.0, 1073.0, 1273.0]:
        for j in [0.1, 0.5, 1.0]:
            Q = float(m.heat_generation(j, T))
            assert_true(Q >= 0, f"Q({j},{T:.0f})={Q:.4f} >= 0")


def test_efficiency_range():
    print("\n[Test 6] Efficiency in valid range")
    m = make_model()
    for T in [973.0, 1073.0, 1273.0]:
        for j in [0.1, 0.5, 1.0]:
            eta = float(m.efficiency(j, T))
            assert_true(0.0 < eta <= 1.2,
                        f"eta({j},{T:.0f})={eta:.4f} in valid range")


def test_voltage_monotonic():
    print("\n[Test 7] V_cell decreases monotonically with j")
    m = make_model()
    j_vals = np.linspace(0.01, 1.7, 100)
    for T in [973.0, 1073.0]:
        V_prev = float(m.cell_voltage(j_vals[0], T))
        for j in j_vals[1:]:
            V = float(m.cell_voltage(j, T))
            assert_true(V <= V_prev + 1e-9,
                        f"V({float(j):.3f},{T:.0f}) <= V_prev")
            V_prev = V


def test_predict_interface():
    print("\n[Test 8] ComponentModel predict() interface")
    cm = ComponentModel()
    out = cm.predict({"current_density": 0.5, "temperature": 1073.0})
    required = ["cell_voltage_V", "power_density_W_cm2", "efficiency",
                "asr_ohm_cm2", "heat_generation_W_cm2"]
    for k in required:
        assert_true(k in out, f"Output key '{k}' present")
    assert_true(out["cell_voltage_V"] > 0, "cell_voltage > 0")


def test_benchmark():
    print("\n[Test 9] Benchmark: 10,000 evaluations")
    m = make_model()
    j_arr = np.linspace(0.01, 1.5, 10000)
    t0 = time.perf_counter()
    m.evaluate(j_arr, 1073.0)
    elapsed = time.perf_counter() - t0
    print(f"  10,000 evaluations (vectorized) in {elapsed*1000:.1f} ms")
    assert_true(elapsed < 2.0, "Completes in < 2 s")


if __name__ == "__main__":
    tests = [
        test_higher_temp_higher_voltage,
        test_ionic_conductivity_increases,
        test_ohmic_asr_decreases,
        test_voltage_bounds,
        test_heat_positive,
        test_efficiency_range,
        test_voltage_monotonic,
        test_predict_interface,
        test_benchmark,
    ]
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

    print(f"\n{'='*60}")
    print(f"EC002 SOFC F1b Thermal -- Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    sys.exit(0 if failed == 0 else 1)
