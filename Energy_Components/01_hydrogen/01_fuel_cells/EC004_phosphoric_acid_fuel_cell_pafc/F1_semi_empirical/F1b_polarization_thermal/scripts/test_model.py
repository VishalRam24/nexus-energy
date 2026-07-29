"""
EC004 -- Phosphoric Acid Fuel Cell (PAFC) -- F1b Polarization-Thermal
Test suite: temperature-dependent physics sanity checks.
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from model import PAFCThermalModel
from predict import ComponentModel

DEFAULT_PARAMS = {
    "T_ref": 453.15,
    "N_cells": 100,
    "A_cell": 400.0,
    "pH2": 1.0,
    "pO2": 0.21,
    "j_L": 0.7,
    "i0_ref": 5e-4,
    "E_act": 70000.0,
    "alpha": 0.5,
    "B_conc": 0.012,
    "sigma_ref_H3PO4": 0.15,
    "E_act_sigma": 20000.0,
    "t_acid": 0.03,
    "E_tn_ref": 1.481,
    "k_tn": 0.000126,
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
    return PAFCThermalModel(DEFAULT_PARAMS)


# ---------------------------------------------------------------------------
# Test 1 -- H3PO4 conductivity increases with temperature
# ---------------------------------------------------------------------------

def test_acid_conductivity_increases_with_T():
    print("\n[Test 1] H3PO4 conductivity increases with temperature")
    m = make_model()
    sigma_low = float(m.acid_conductivity(423.0))
    sigma_high = float(m.acid_conductivity(483.0))
    assert_true(sigma_high > sigma_low,
                f"sigma(483K)={sigma_high:.4f} > sigma(423K)={sigma_low:.4f} S/cm")


# ---------------------------------------------------------------------------
# Test 2 -- Acid resistance decreases with temperature
# ---------------------------------------------------------------------------

def test_acid_resistance_decreases():
    print("\n[Test 2] H3PO4 resistance decreases with temperature")
    m = make_model()
    R_low  = float(m.acid_resistance(423.0))
    R_high = float(m.acid_resistance(483.0))
    assert_true(R_high < R_low,
                f"R(483K)={R_high:.5f} < R(423K)={R_low:.5f} ohm cm2")


# ---------------------------------------------------------------------------
# Test 3 -- Exchange current density increases with temperature (Arrhenius)
# ---------------------------------------------------------------------------

def test_exchange_current_increases():
    print("\n[Test 3] Cathode exchange current density increases with temperature")
    m = make_model()
    i0_low  = float(m.exchange_current_density(423.0))
    i0_high = float(m.exchange_current_density(483.0))
    assert_true(i0_high > i0_low,
                f"i0(483K)={i0_high:.6f} > i0(423K)={i0_low:.6f}")


# ---------------------------------------------------------------------------
# Test 4 -- Thermoneutral voltage decreases with temperature
# ---------------------------------------------------------------------------

def test_thermoneutral_decreases():
    print("\n[Test 4] Thermoneutral voltage decreases with temperature")
    m = make_model()
    E_tn_low  = float(m.thermoneutral_voltage(423.0))
    E_tn_high = float(m.thermoneutral_voltage(483.0))
    assert_true(E_tn_high < E_tn_low,
                f"E_tn(483K)={E_tn_high:.4f} < E_tn(423K)={E_tn_low:.4f}")


# ---------------------------------------------------------------------------
# Test 5 -- Cell voltage within physical bounds [0, E_nernst]
# ---------------------------------------------------------------------------

def test_voltage_bounds():
    print("\n[Test 5] Cell voltage within physical bounds")
    m = make_model()
    for T in [423.0, 453.0, 483.0]:
        for j in [0.01, 0.2, 0.4, 0.6]:
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
    for T in [423.0, 453.0, 483.0]:
        for j in [0.1, 0.3, 0.5]:
            Q = float(m.heat_generation(j, T))
            assert_true(Q > 0, f"Q({j},{T})={Q:.4f} > 0")


# ---------------------------------------------------------------------------
# Test 7 -- Efficiency < 1 (below thermoneutral voltage)
# ---------------------------------------------------------------------------

def test_efficiency_below_unity():
    print("\n[Test 7] Efficiency < 1.0 for all operating points")
    m = make_model()
    for T in [423.0, 453.0, 483.0]:
        for j in [0.01, 0.2, 0.4, 0.6]:
            eta = float(m.efficiency(j, T))
            assert_true(0.0 < eta < 1.0,
                        f"eta({j},{T})={eta:.4f} in (0,1)")


# ---------------------------------------------------------------------------
# Test 8 -- Voltage decreases monotonically with current density
# ---------------------------------------------------------------------------

def test_voltage_monotonic():
    print("\n[Test 8] V_cell decreases monotonically with j")
    m = make_model()
    j_vals = np.linspace(0.01, 0.65, 100)
    for T in [423.0, 453.0, 483.0]:
        V_prev = float(m.cell_voltage(j_vals[0], T))
        for j in j_vals[1:]:
            V = float(m.cell_voltage(j, T))
            assert_true(V <= V_prev + 1e-9,
                        f"V({float(j):.3f},{T}) <= V_prev")
            V_prev = V


# ---------------------------------------------------------------------------
# Test 9 -- ComponentModel predict() interface
# ---------------------------------------------------------------------------

def test_predict_interface():
    print("\n[Test 9] ComponentModel predict() interface")
    cm = ComponentModel()
    out = cm.predict({"current_density": 0.3, "temperature": 453.15})
    required = [
        "cell_voltage_V", "power_density_W_cm2", "efficiency",
        "heat_generation_W_cm2", "acid_resistance_ohm_cm2",
        "acid_conductivity_S_cm", "thermoneutral_voltage_V",
    ]
    for k in required:
        assert_true(k in out, f"Output key '{k}' present")
    assert_true(out["cell_voltage_V"] > 0, "cell_voltage > 0")
    assert_true(out["heat_generation_W_cm2"] > 0, "heat_generation > 0")
    assert_true(out["thermoneutral_voltage_V"] > out["cell_voltage_V"],
                "E_tn > V_cell (required for Q > 0)")


# ---------------------------------------------------------------------------
# Test 10 -- Benchmark
# ---------------------------------------------------------------------------

def test_benchmark():
    print("\n[Test 10] Benchmark: 10,000 evaluations")
    m = make_model()
    j_arr = np.linspace(0.01, 0.65, 10000)
    t0 = time.perf_counter()
    m.evaluate(j_arr, 453.0)
    elapsed = time.perf_counter() - t0
    print(f"  10,000 evaluations (vectorized) in {elapsed*1000:.1f} ms")
    assert_true(elapsed < 2.0, "Completes in < 2 s")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        test_acid_conductivity_increases_with_T,
        test_acid_resistance_decreases,
        test_exchange_current_increases,
        test_thermoneutral_decreases,
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
    print(f"EC004 PAFC F1b Polarization-Thermal -- Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    sys.exit(0 if failed == 0 else 1)
