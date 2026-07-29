"""
EC003 -- Alkaline Fuel Cell (AFC) -- F1b Polarization-Thermal
Test suite: temperature-dependent physics sanity checks.
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from model import AFCThermalModel
from predict import ComponentModel

DEFAULT_PARAMS = {
    "T_ref": 353.15,
    "N_cells": 50,
    "A_cell": 200.0,
    "pH2": 1.0,
    "pO2": 0.21,
    "j_L": 1.0,
    "i0_ref": 1e-3,
    "E_act": 55000.0,
    "alpha": 0.5,
    "B_conc": 0.010,
    "c_KOH": 6.0,
    "L_electrolyte": 0.05,
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
    return AFCThermalModel(DEFAULT_PARAMS)


# ---------------------------------------------------------------------------
# Test 1 -- KOH conductivity increases with temperature
# ---------------------------------------------------------------------------

def test_koh_conductivity_increases_with_T():
    print("\n[Test 1] KOH conductivity increases with temperature")
    m = make_model()
    sigma_cold = float(m.koh_conductivity(333.0))
    sigma_hot  = float(m.koh_conductivity(363.0))
    assert_true(sigma_hot > sigma_cold,
                f"sigma_KOH(363K)={sigma_hot:.4f} > sigma_KOH(333K)={sigma_cold:.4f} S/cm")


# ---------------------------------------------------------------------------
# Test 2 -- Electrolyte resistance decreases with temperature
# ---------------------------------------------------------------------------

def test_electrolyte_resistance_decreases():
    print("\n[Test 2] Electrolyte resistance decreases with temperature")
    m = make_model()
    R_cold = float(m.electrolyte_resistance(333.0))
    R_hot  = float(m.electrolyte_resistance(363.0))
    assert_true(R_hot < R_cold,
                f"R_elec(363K)={R_hot:.5f} < R_elec(333K)={R_cold:.5f} ohm cm2")


# ---------------------------------------------------------------------------
# Test 3 -- Exchange current density increases with temperature (Arrhenius)
# ---------------------------------------------------------------------------

def test_exchange_current_increases():
    print("\n[Test 3] Exchange current density increases with temperature")
    m = make_model()
    i0_cold = float(m.exchange_current_density(333.0))
    i0_hot  = float(m.exchange_current_density(363.0))
    assert_true(i0_hot > i0_cold,
                f"i0(363K)={i0_hot:.6f} > i0(333K)={i0_cold:.6f}")


# ---------------------------------------------------------------------------
# Test 4 -- Cell voltage within physical bounds [0, E_nernst]
# ---------------------------------------------------------------------------

def test_voltage_bounds():
    print("\n[Test 4] Cell voltage within physical bounds")
    m = make_model()
    for T in [333.0, 343.0, 353.0, 363.0]:
        for j in [0.01, 0.2, 0.5, 0.8]:
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
        for j in [0.1, 0.4, 0.7]:
            Q = float(m.heat_generation(j, T))
            assert_true(Q > 0, f"Q({j},{T})={Q:.4f} > 0")


# ---------------------------------------------------------------------------
# Test 6 -- Efficiency < 1 (below HHV limit)
# ---------------------------------------------------------------------------

def test_efficiency_below_unity():
    print("\n[Test 6] Efficiency < 1.0 for all operating points")
    m = make_model()
    for T in [333.0, 353.0, 363.0]:
        for j in [0.01, 0.3, 0.6, 0.9]:
            eta = float(m.efficiency(j, T))
            assert_true(0.0 < eta < 1.0,
                        f"eta({j},{T})={eta:.4f} in (0,1)")


# ---------------------------------------------------------------------------
# Test 7 -- Voltage decreases monotonically with current density
# ---------------------------------------------------------------------------

def test_voltage_monotonic():
    print("\n[Test 7] V_cell decreases monotonically with j")
    m = make_model()
    j_vals = np.linspace(0.01, 0.95, 100)
    for T in [333.0, 353.0]:
        V_prev = float(m.cell_voltage(j_vals[0], T))
        for j in j_vals[1:]:
            V = float(m.cell_voltage(j, T))
            assert_true(V <= V_prev + 1e-9,
                        f"V({float(j):.3f},{T}) <= V_prev")
            V_prev = V


# ---------------------------------------------------------------------------
# Test 8 -- KOH conductivity physically reasonable (> 0.3 S/cm at 60C for 6M)
# ---------------------------------------------------------------------------

def test_koh_conductivity_reasonable():
    print("\n[Test 8] KOH conductivity physically reasonable")
    m = make_model()
    # 6 mol/L KOH at 60C (~333K): literature ~0.4-0.6 S/cm
    sigma = float(m.koh_conductivity(333.0))
    assert_true(sigma > 0.2,
                f"sigma_KOH(333K, 6M) = {sigma:.4f} S/cm > 0.2 S/cm")
    assert_true(sigma < 2.0,
                f"sigma_KOH(333K, 6M) = {sigma:.4f} S/cm < 2.0 S/cm (physical upper bound)")


# ---------------------------------------------------------------------------
# Test 9 -- ComponentModel predict() interface
# ---------------------------------------------------------------------------

def test_predict_interface():
    print("\n[Test 9] ComponentModel predict() interface")
    cm = ComponentModel()
    out = cm.predict({"current_density": 0.4, "temperature": 353.15})
    required = [
        "cell_voltage_V", "power_density_W_cm2", "efficiency",
        "heat_generation_W_cm2", "electrolyte_resistance_ohm_cm2",
        "koh_conductivity_S_cm",
    ]
    for k in required:
        assert_true(k in out, f"Output key '{k}' present")
    assert_true(out["cell_voltage_V"] > 0, "cell_voltage > 0")
    assert_true(out["heat_generation_W_cm2"] > 0, "heat_generation > 0")
    assert_true(out["koh_conductivity_S_cm"] > 0, "koh_conductivity > 0")


# ---------------------------------------------------------------------------
# Test 10 -- Benchmark
# ---------------------------------------------------------------------------

def test_benchmark():
    print("\n[Test 10] Benchmark: 10,000 evaluations")
    m = make_model()
    j_arr = np.linspace(0.01, 0.95, 10000)
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
        test_koh_conductivity_increases_with_T,
        test_electrolyte_resistance_decreases,
        test_exchange_current_increases,
        test_voltage_bounds,
        test_heat_positive,
        test_efficiency_below_unity,
        test_voltage_monotonic,
        test_koh_conductivity_reasonable,
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
    print(f"EC003 AFC F1b Polarization-Thermal -- Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    sys.exit(0 if failed == 0 else 1)
