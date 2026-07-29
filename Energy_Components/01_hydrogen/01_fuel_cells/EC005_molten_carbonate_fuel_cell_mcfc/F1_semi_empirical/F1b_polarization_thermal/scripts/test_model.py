"""
EC005 -- Molten Carbonate Fuel Cell (MCFC) -- F1b Polarization-Thermal
Test suite: temperature-dependent physics sanity checks.
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from model import MCFCThermalModel
from predict import ComponentModel

DEFAULT_PARAMS = {
    "T_ref": 923.15,
    "N_cells": 300,
    "A_cell": 1000.0,
    "pH2": 0.7,
    "pO2": 0.15,
    "pH2O": 0.20,
    "pCO2_cathode": 0.15,
    "pCO2_anode": 0.10,
    "j_L": 0.6,
    "i0_anode_ref": 0.02,
    "E_act_anode": 50000.0,
    "i0_cathode_ref": 0.005,
    "E_act_cathode": 60000.0,
    "alpha": 0.5,
    "B_conc": 0.008,
    "A_mc": 200000.0,
    "E_act_mc": 28000.0,
    "t_mc": 0.06,
    "E_tn": 1.21,
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
    return MCFCThermalModel(DEFAULT_PARAMS)


# ---------------------------------------------------------------------------
# Test 1 -- Molten carbonate conductivity increases with temperature
# ---------------------------------------------------------------------------

def test_carbonate_conductivity_increases():
    print("\n[Test 1] Molten carbonate conductivity increases with temperature")
    m = make_model()
    sigma_low  = float(m.carbonate_conductivity(873.0))
    sigma_high = float(m.carbonate_conductivity(973.0))
    assert_true(sigma_high > sigma_low,
                f"sigma(973K)={sigma_high:.3f} > sigma(873K)={sigma_low:.3f} S/cm")


# ---------------------------------------------------------------------------
# Test 2 -- Carbonate resistance decreases with temperature
# ---------------------------------------------------------------------------

def test_carbonate_resistance_decreases():
    print("\n[Test 2] Carbonate resistance decreases with temperature")
    m = make_model()
    R_low  = float(m.carbonate_resistance(873.0))
    R_high = float(m.carbonate_resistance(973.0))
    assert_true(R_high < R_low,
                f"R_mc(973K)={R_high:.5f} < R_mc(873K)={R_low:.5f} ohm cm2")


# ---------------------------------------------------------------------------
# Test 3 -- Exchange current densities increase with temperature
# ---------------------------------------------------------------------------

def test_exchange_currents_increase():
    print("\n[Test 3] Exchange current densities increase with temperature")
    m = make_model()
    i0a_low  = float(m.i0_anode(873.0))
    i0a_high = float(m.i0_anode(973.0))
    i0c_low  = float(m.i0_cathode(873.0))
    i0c_high = float(m.i0_cathode(973.0))
    assert_true(i0a_high > i0a_low, f"i0_an(973K)={i0a_high:.4f} > i0_an(873K)={i0a_low:.4f}")
    assert_true(i0c_high > i0c_low, f"i0_cat(973K)={i0c_high:.4f} > i0_cat(873K)={i0c_low:.4f}")


# ---------------------------------------------------------------------------
# Test 4 -- Nernst voltage physically reasonable at MCFC temperatures
# ---------------------------------------------------------------------------

def test_nernst_voltage_range():
    print("\n[Test 4] Nernst voltage physically reasonable (0.9-1.2 V at 600-700C)")
    m = make_model()
    for T in [873.0, 923.0, 973.0]:
        E = float(m.nernst_voltage(T))
        assert_true(0.8 <= E <= 1.3,
                    f"E_nernst({T})={E:.4f} in [0.8, 1.3] V")


# ---------------------------------------------------------------------------
# Test 5 -- Cell voltage within physical bounds [0, E_nernst]
# ---------------------------------------------------------------------------

def test_voltage_bounds():
    print("\n[Test 5] Cell voltage within physical bounds")
    m = make_model()
    for T in [873.0, 923.0, 973.0]:
        for j in [0.01, 0.15, 0.3, 0.5]:
            V = float(m.cell_voltage(j, T))
            E = float(m.nernst_voltage(T))
            assert_true(0.0 <= V <= E + 0.001,
                        f"0 <= V({j},{T})={V:.4f} <= E_nernst={E:.4f}")


# ---------------------------------------------------------------------------
# Test 6 -- Heat generation is positive
# ---------------------------------------------------------------------------

def test_heat_positive():
    print("\n[Test 6] Heat generation positive for j > 0")
    m = make_model()
    for T in [873.0, 923.0, 973.0]:
        for j in [0.05, 0.2, 0.4]:
            Q = float(m.heat_generation(j, T))
            assert_true(Q > 0, f"Q({j},{T})={Q:.4f} > 0")


# ---------------------------------------------------------------------------
# Test 7 -- Efficiency < 1.0
# ---------------------------------------------------------------------------

def test_efficiency_below_unity():
    print("\n[Test 7] Efficiency < 1.0 for all operating points")
    m = make_model()
    for T in [873.0, 923.0, 973.0]:
        for j in [0.01, 0.15, 0.3, 0.5]:
            eta = float(m.efficiency(j, T))
            assert_true(0.0 < eta < 1.0,
                        f"eta({j},{T})={eta:.4f} in (0,1)")


# ---------------------------------------------------------------------------
# Test 8 -- Voltage decreases monotonically with current density
# ---------------------------------------------------------------------------

def test_voltage_monotonic():
    print("\n[Test 8] V_cell decreases monotonically with j")
    m = make_model()
    j_vals = np.linspace(0.01, 0.55, 100)
    for T in [873.0, 923.0]:
        V_prev = float(m.cell_voltage(j_vals[0], T))
        for j in j_vals[1:]:
            V = float(m.cell_voltage(j, T))
            assert_true(V <= V_prev + 1e-9,
                        f"V({float(j):.3f},{T}) <= V_prev")
            V_prev = V


# ---------------------------------------------------------------------------
# Test 9 -- Carbonate conductivity physically reasonable (>1 S/cm at 650C)
# ---------------------------------------------------------------------------

def test_carbonate_conductivity_reasonable():
    print("\n[Test 9] Carbonate conductivity physically reasonable at 650C")
    m = make_model()
    sigma = float(m.carbonate_conductivity(923.0))
    # Literature: Li2CO3/K2CO3 eutectic at 650C ~ 1.5-3.5 S/cm (Uchida 1983)
    assert_true(sigma > 1.0,
                f"sigma_mc(923K) = {sigma:.3f} S/cm > 1.0 S/cm")
    assert_true(sigma < 10.0,
                f"sigma_mc(923K) = {sigma:.3f} S/cm < 10.0 S/cm (physical upper)")


# ---------------------------------------------------------------------------
# Test 10 -- ComponentModel predict() interface
# ---------------------------------------------------------------------------

def test_predict_interface():
    print("\n[Test 10] ComponentModel predict() interface")
    cm = ComponentModel()
    out = cm.predict({"current_density": 0.2, "temperature": 923.15})
    required = [
        "cell_voltage_V", "power_density_W_cm2", "efficiency",
        "heat_generation_W_cm2", "carbonate_resistance_ohm_cm2",
        "carbonate_conductivity_S_cm",
    ]
    for k in required:
        assert_true(k in out, f"Output key '{k}' present")
    assert_true(out["cell_voltage_V"] > 0, "cell_voltage > 0")
    assert_true(out["heat_generation_W_cm2"] > 0, "heat_generation > 0")


# ---------------------------------------------------------------------------
# Test 11 -- Benchmark
# ---------------------------------------------------------------------------

def test_benchmark():
    print("\n[Test 11] Benchmark: 10,000 evaluations")
    m = make_model()
    j_arr = np.linspace(0.01, 0.55, 10000)
    t0 = time.perf_counter()
    m.evaluate(j_arr, 923.0)
    elapsed = time.perf_counter() - t0
    print(f"  10,000 evaluations (vectorized) in {elapsed*1000:.1f} ms")
    assert_true(elapsed < 2.0, "Completes in < 2 s")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        test_carbonate_conductivity_increases,
        test_carbonate_resistance_decreases,
        test_exchange_currents_increase,
        test_nernst_voltage_range,
        test_voltage_bounds,
        test_heat_positive,
        test_efficiency_below_unity,
        test_voltage_monotonic,
        test_carbonate_conductivity_reasonable,
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
    print(f"EC005 MCFC F1b Polarization-Thermal -- Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    sys.exit(0 if failed == 0 else 1)
