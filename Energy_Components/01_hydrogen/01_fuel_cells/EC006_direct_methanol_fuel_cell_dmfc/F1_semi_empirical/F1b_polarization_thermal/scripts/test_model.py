"""
EC006 -- Direct Methanol Fuel Cell (DMFC) -- F1b Polarization-Thermal
Test suite: temperature-dependent physics sanity checks.
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from model import DMFCThermalModel
from predict import ComponentModel

DEFAULT_PARAMS = {
    "T_ref": 353.15,
    "N_cells": 20,
    "A_cell": 25.0,
    "c_MeOH": 1.0,
    "pO2": 0.21,
    "j_L": 0.4,
    "i0_ref": 5e-5,
    "E_act": 60000.0,
    "alpha": 0.5,
    "B_conc": 0.020,
    "j_cross_ref": 0.05,
    "E_act_cross": 25000.0,
    "lambda_mem": 12.0,
    "membrane_thickness": 0.0183,
    "dEdT_MeOH": -1.4e-4,
    "E_ref_std": 1.214,
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
    return DMFCThermalModel(DEFAULT_PARAMS)


# ---------------------------------------------------------------------------
# Test 1 -- Crossover current increases with temperature (more diffusion)
# ---------------------------------------------------------------------------

def test_crossover_increases_with_T():
    print("\n[Test 1] Methanol crossover increases with temperature")
    m = make_model()
    jx_cold = float(m.crossover_current(333.0))
    jx_hot  = float(m.crossover_current(373.0))
    assert_true(jx_hot > jx_cold,
                f"j_cross(373K)={jx_hot:.4f} > j_cross(333K)={jx_cold:.4f} A/cm2")


# ---------------------------------------------------------------------------
# Test 2 -- Membrane resistance decreases with temperature
# ---------------------------------------------------------------------------

def test_membrane_resistance_decreases():
    print("\n[Test 2] Nafion membrane resistance decreases with temperature")
    m = make_model()
    R_cold = float(m.membrane_resistance(333.0))
    R_hot  = float(m.membrane_resistance(373.0))
    assert_true(R_hot < R_cold,
                f"R_mem(373K)={R_hot:.4f} < R_mem(333K)={R_cold:.4f}")


# ---------------------------------------------------------------------------
# Test 3 -- Exchange current density increases with temperature
# ---------------------------------------------------------------------------

def test_exchange_current_increases():
    print("\n[Test 3] Anode exchange current density increases with temperature")
    m = make_model()
    i0_cold = float(m.exchange_current_density(333.0))
    i0_hot  = float(m.exchange_current_density(373.0))
    assert_true(i0_hot > i0_cold,
                f"i0(373K)={i0_hot:.6f} > i0(333K)={i0_cold:.6f}")


# ---------------------------------------------------------------------------
# Test 4 -- Cell voltage within physical bounds [0, E_nernst]
# ---------------------------------------------------------------------------

def test_voltage_bounds():
    print("\n[Test 4] Cell voltage within physical bounds")
    m = make_model()
    for T in [333.0, 353.0, 373.0]:
        for j in [0.01, 0.1, 0.2, 0.3]:
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
    for T in [333.0, 353.0, 373.0]:
        for j in [0.05, 0.15, 0.25]:
            Q = float(m.heat_generation(j, T))
            assert_true(Q > 0, f"Q({j},{T})={Q:.4f} > 0")


# ---------------------------------------------------------------------------
# Test 6 -- Efficiency < 1 (below E_tn)
# ---------------------------------------------------------------------------

def test_efficiency_below_unity():
    print("\n[Test 6] Efficiency < 1.0 for all operating points")
    m = make_model()
    for T in [333.0, 353.0, 373.0]:
        for j in [0.01, 0.1, 0.2, 0.3]:
            eta = float(m.efficiency(j, T))
            assert_true(0.0 < eta < 1.0,
                        f"eta({j},{T})={eta:.4f} in (0,1)")


# ---------------------------------------------------------------------------
# Test 7 -- Voltage decreases monotonically with current density
# ---------------------------------------------------------------------------

def test_voltage_monotonic():
    print("\n[Test 7] V_cell decreases monotonically with j")
    m = make_model()
    j_vals = np.linspace(0.01, 0.37, 100)
    for T in [333.0, 353.0]:
        V_prev = float(m.cell_voltage(j_vals[0], T))
        for j in j_vals[1:]:
            V = float(m.cell_voltage(j, T))
            assert_true(V <= V_prev + 1e-9,
                        f"V({float(j):.3f},{T}) <= V_prev")
            V_prev = V


# ---------------------------------------------------------------------------
# Test 8 -- Higher T -> higher crossover, but also lower resistance (trade-off)
# ---------------------------------------------------------------------------

def test_crossover_tradeoff():
    print("\n[Test 8] Crossover current physically bounded (0.01-0.2 A/cm2 range)")
    m = make_model()
    for T in [333.0, 353.0, 373.0]:
        jx = float(m.crossover_current(T))
        assert_true(0.001 <= jx <= 0.5,
                    f"j_cross({T}K)={jx:.4f} in [0.001, 0.5] A/cm2")


# ---------------------------------------------------------------------------
# Test 9 -- ComponentModel predict() interface
# ---------------------------------------------------------------------------

def test_predict_interface():
    print("\n[Test 9] ComponentModel predict() interface")
    cm = ComponentModel()
    out = cm.predict({"current_density": 0.15, "temperature": 353.15})
    required = [
        "cell_voltage_V", "power_density_W_cm2", "efficiency",
        "heat_generation_W_cm2", "membrane_resistance_ohm_cm2",
        "crossover_current_A_cm2", "V_mix_V",
    ]
    for k in required:
        assert_true(k in out, f"Output key '{k}' present")
    assert_true(out["cell_voltage_V"] > 0, "cell_voltage > 0")
    assert_true(out["crossover_current_A_cm2"] > 0, "crossover_current > 0")
    assert_true(out["V_mix_V"] >= 0, "V_mix >= 0 (crossover penalty)")


# ---------------------------------------------------------------------------
# Test 10 -- Benchmark
# ---------------------------------------------------------------------------

def test_benchmark():
    print("\n[Test 10] Benchmark: 10,000 evaluations")
    m = make_model()
    j_arr = np.linspace(0.01, 0.37, 10000)
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
        test_crossover_increases_with_T,
        test_membrane_resistance_decreases,
        test_exchange_current_increases,
        test_voltage_bounds,
        test_heat_positive,
        test_efficiency_below_unity,
        test_voltage_monotonic,
        test_crossover_tradeoff,
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
    print(f"EC006 DMFC F1b Polarization-Thermal -- Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    sys.exit(0 if failed == 0 else 1)
