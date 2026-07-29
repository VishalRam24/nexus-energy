"""
EC001 -- PEM Fuel Cell (PEMFC) -- F1b Thermal
Test suite: temperature-dependent physics sanity checks.
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from model import PEMFCThermalModel
from predict import ComponentModel

DEFAULT_PARAMS = {
    "T_ref": 353.15,
    "N_cells": 50,
    "A_cell": 100.0,
    "pH2": 1.0,
    "pO2": 0.21,
    "j_L": 2.0,
    "i0_ref": 1e-4,
    "E_act": 66000.0,
    "sigma_ref": 0.1,
    "membrane_thickness": 0.0183,
    "lambda_mem": 14.0,
    "alpha": 0.5,
    "B_conc": 0.016,
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
    return PEMFCThermalModel(DEFAULT_PARAMS)


# ---------------------------------------------------------------------------
# Test 1 -- Higher temperature -> higher cell voltage at moderate current
# (kinetics improve faster than Nernst drops for PEMFC in 333-363 K range)
# ---------------------------------------------------------------------------

def test_temperature_improves_voltage():
    print("\n[Test 1] Higher T -> higher V_cell at j=0.5 (fuel cell)")
    m = make_model()
    V_cold = float(m.cell_voltage(0.5, 333.0))
    V_hot = float(m.cell_voltage(0.5, 363.0))
    assert_true(V_hot > V_cold,
                f"V(363K)={V_hot:.4f} > V(333K)={V_cold:.4f} at j=0.5")


# ---------------------------------------------------------------------------
# Test 2 -- Membrane resistance decreases with temperature
# ---------------------------------------------------------------------------

def test_membrane_resistance_decreases():
    print("\n[Test 2] Membrane resistance decreases with temperature")
    m = make_model()
    R_cold = float(m.membrane_resistance(333.0))
    R_hot = float(m.membrane_resistance(363.0))
    assert_true(R_hot < R_cold,
                f"R_mem(363K)={R_hot:.4f} < R_mem(333K)={R_cold:.4f}")


# ---------------------------------------------------------------------------
# Test 3 -- Exchange current density increases with temperature
# ---------------------------------------------------------------------------

def test_exchange_current_increases():
    print("\n[Test 3] Exchange current density increases with temperature")
    m = make_model()
    i0_cold = float(m.exchange_current_density(333.0))
    i0_hot = float(m.exchange_current_density(363.0))
    assert_true(i0_hot > i0_cold,
                f"i0(363K)={i0_hot:.6f} > i0(333K)={i0_cold:.6f}")


# ---------------------------------------------------------------------------
# Test 4 -- Cell voltage within physical bounds
# ---------------------------------------------------------------------------

def test_voltage_bounds():
    print("\n[Test 4] Cell voltage within physical bounds")
    m = make_model()
    for T in [333.0, 343.0, 353.0, 363.0]:
        for j in [0.01, 0.5, 1.0, 1.5]:
            V = float(m.cell_voltage(j, T))
            E = float(m.nernst_voltage(T))
            assert_true(0.0 <= V <= E + 0.001,
                        f"0 <= V({j},{T})={V:.4f} <= E_nernst={E:.4f}")


# ---------------------------------------------------------------------------
# Test 5 -- Heat generation is positive
# ---------------------------------------------------------------------------

def test_heat_positive():
    print("\n[Test 5] Heat generation positive for j > 0")
    m = make_model()
    for T in [333.0, 353.0, 363.0]:
        for j in [0.1, 0.5, 1.0]:
            Q = float(m.heat_generation(j, T))
            assert_true(Q > 0, f"Q({j},{T})={Q:.4f} > 0")


# ---------------------------------------------------------------------------
# Test 6 -- Efficiency < 1 (below HHV limit)
# ---------------------------------------------------------------------------

def test_efficiency_below_unity():
    print("\n[Test 6] Efficiency < 1.0 for all operating points")
    m = make_model()
    for T in [333.0, 353.0, 363.0]:
        for j in [0.01, 0.5, 1.0, 1.5]:
            eta = float(m.efficiency(j, T))
            assert_true(0.0 < eta < 1.0,
                        f"eta({j},{T})={eta:.4f} in (0,1)")


# ---------------------------------------------------------------------------
# Test 7 -- Voltage decreases monotonically with current density
# ---------------------------------------------------------------------------

def test_voltage_monotonic():
    print("\n[Test 7] V_cell decreases monotonically with j")
    m = make_model()
    j_vals = np.linspace(0.01, 1.8, 100)
    for T in [333.0, 353.0]:
        V_prev = float(m.cell_voltage(j_vals[0], T))
        for j in j_vals[1:]:
            V = float(m.cell_voltage(j, T))
            assert_true(V <= V_prev + 1e-9,
                        f"V({float(j):.3f},{T}) <= V_prev")
            V_prev = V


# ---------------------------------------------------------------------------
# Test 8 -- ComponentModel predict() interface
# ---------------------------------------------------------------------------

def test_predict_interface():
    print("\n[Test 8] ComponentModel predict() interface")
    cm = ComponentModel()
    out = cm.predict({"current_density": 0.6, "temperature": 353.15})
    required = [
        "cell_voltage_V", "power_density_W_cm2", "efficiency",
        "heat_generation_W_cm2", "membrane_resistance_ohm_cm2",
    ]
    for k in required:
        assert_true(k in out, f"Output key '{k}' present")
    assert_true(out["cell_voltage_V"] > 0, "cell_voltage > 0")
    assert_true(out["heat_generation_W_cm2"] > 0, "heat_generation > 0")


# ---------------------------------------------------------------------------
# Test 9 -- Benchmark
# ---------------------------------------------------------------------------

def test_benchmark():
    print("\n[Test 9] Benchmark: 10,000 evaluations")
    m = make_model()
    j_arr = np.linspace(0.01, 1.5, 10000)
    t0 = time.perf_counter()
    m.evaluate(j_arr, 353.0)
    elapsed = time.perf_counter() - t0
    print(f"  10,000 evaluations (vectorized) in {elapsed*1000:.1f} ms")
    assert_true(elapsed < 2.0, "Completes in < 2 s")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        test_temperature_improves_voltage,
        test_membrane_resistance_decreases,
        test_exchange_current_increases,
        test_voltage_bounds,
        test_heat_positive,
        test_efficiency_below_unity,
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
    print(f"EC001 PEMFC F1b Thermal -- Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    sys.exit(0 if failed == 0 else 1)
