"""
EC011 -- AEM Electrolyser -- F1b V-I Thermal
Test suite: temperature-dependent physics sanity checks.

Critical test rule: tests must FAIL the model, not accommodate it.
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from model import AEMThermalModel
from predict import ComponentModel

DEFAULT_PARAMS = {
    "component": "EC011",
    "fidelity": "F1b",
    "description": "test",
    "source": "test",
    "unit": {
        "name": "test stack",
        "N_cells": {"value": 10},
        "electrode_area": {"value": 0.01},
        "T_ref": {"value": 333.15},
        "T_coolant": {"value": 333.15},
        "j0_anode": {"value": 1.0e-4},
        "j0_cathode": {"value": 1.0e-2},
        "Ea_anode": {"value": 52000.0},
        "Ea_cathode": {"value": 30000.0},
        "alpha_anode": {"value": 0.5},
        "alpha_cathode": {"value": 0.5},
        "r_membrane_ref": {"value": 0.25},
        "r_temp_coeff": {"value": -0.003},
        "eta_F": {"value": 0.97},
        "thermal_mass": {"value": 500.0},
        "UA_cool": {"value": 20.0},
    },
    "constants": {
        "F": {"value": 96485.0},
        "R": {"value": 8.314},
        "E_rev_ref": {"value": 1.229},
        "E_rev_T_coeff": {"value": 0.0009},
        "E_tn": {"value": 1.481},
    },
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
    return AEMThermalModel(DEFAULT_PARAMS)


# ---------------------------------------------------------------------------
# Test 1 -- Cell voltage DECREASES with temperature at moderate-to-high j
# Physics: E_rev drops ~0.09 mV/K; at mid-to-high current density the net
# effect in AEM (Ni-based OER, Ea=52 kJ/mol) is that V_cell decreases with T.
# Ref: Vincent & Bessarabov (2018), Fig. 4 -- voltage curves fall with temperature.
# ---------------------------------------------------------------------------

def test_voltage_decreases_with_temperature():
    print("\n[Test 1] V_cell decreases with T at mid j (AEM electrolyser)")
    m = make_model()
    j = 5000.0   # A/m2 = 0.5 A/cm2, mid-range
    V_cold = float(m.cell_voltage(j, 313.15))
    V_hot  = float(m.cell_voltage(j, 353.15))
    # RATIONALE: In AEM electrolysers E_rev(T) decreases at ~0.9 mV/K;
    # Arrhenius kinetics improve but not enough to overcome E_rev drop +
    # the already-low ASR at mild temperatures.  Net: V_cell(hot) < V_cell(cold).
    # See Vincent & Bessarabov (2018) Fig. 4.
    assert_true(V_hot < V_cold,
                f"V_cell(353K)={V_hot:.4f} < V_cell(313K)={V_cold:.4f} at j=5000 A/m2")


# ---------------------------------------------------------------------------
# Test 2 -- ASR decreases with temperature (negative r_T coefficient)
# ---------------------------------------------------------------------------

def test_asr_decreases_with_temperature():
    print("\n[Test 2] ASR decreases with temperature")
    m = make_model()
    ASR_cold = float(m.asr(313.15))
    ASR_hot  = float(m.asr(353.15))
    assert_true(ASR_hot < ASR_cold,
                f"ASR(353K)={ASR_hot:.4f} < ASR(313K)={ASR_cold:.4f}")


# ---------------------------------------------------------------------------
# Test 3 -- Exchange current density increases with temperature (Arrhenius)
# ---------------------------------------------------------------------------

def test_exchange_current_increases():
    print("\n[Test 3] Arrhenius: j0 increases with temperature")
    m = make_model()
    j0_a_cold, j0_c_cold = m.exchange_current_density(313.15)
    j0_a_hot,  j0_c_hot  = m.exchange_current_density(353.15)
    assert_true(float(j0_a_hot) > float(j0_a_cold),
                f"j0_a(353K)={float(j0_a_hot):.2e} > j0_a(313K)={float(j0_a_cold):.2e}")
    assert_true(float(j0_c_hot) > float(j0_c_cold),
                f"j0_c(353K)={float(j0_c_hot):.2e} > j0_c(313K)={float(j0_c_cold):.2e}")


# ---------------------------------------------------------------------------
# Test 4 -- Cell voltage > E_rev at all operating points (electrolysis requires
#           extra driving force above reversible potential)
# ---------------------------------------------------------------------------

def test_voltage_above_e_rev():
    print("\n[Test 4] V_cell > E_rev for all j > 0")
    m = make_model()
    for T in [313.15, 333.15, 353.15]:
        for j in [100.0, 1000.0, 5000.0, 10000.0]:
            V = float(m.cell_voltage(j, T))
            Erev = float(m.e_rev(T))
            assert_true(V > Erev,
                        f"V({j},{T})={V:.4f} > E_rev={Erev:.4f}")


# ---------------------------------------------------------------------------
# Test 5 -- Heat generation is positive for all j > 0
#           Q = N*I*(V_cell - E_tn); V_cell > 1.481 V in electrolysis
# ---------------------------------------------------------------------------

def test_heat_positive():
    print("\n[Test 5] Heat generation positive for j > 0")
    m = make_model()
    for T in [313.15, 333.15, 353.15]:
        for j in [1000.0, 5000.0, 10000.0]:
            Q = float(m.heat_generation(j, T))
            assert_true(Q > 0, f"Q_gen({j},{T})={Q:.2f} W > 0")


# ---------------------------------------------------------------------------
# Test 6 -- Steady-state temperature rises with current density
#           More current -> more heat -> higher T_stack
# ---------------------------------------------------------------------------

def test_thermal_ss_rises_with_current():
    print("\n[Test 6] Steady-state T_stack increases with j")
    m = make_model()
    T_low  = m.steady_state_temperature(1000.0, 333.15)
    T_high = m.steady_state_temperature(10000.0, 333.15)
    assert_true(T_high > T_low,
                f"T_ss(j=10000)={T_high:.2f} K > T_ss(j=1000)={T_low:.2f} K")


# ---------------------------------------------------------------------------
# Test 7 -- Steady-state T >= T_coolant (heat source, not sink)
# ---------------------------------------------------------------------------

def test_thermal_ss_above_coolant():
    print("\n[Test 7] T_stack >= T_coolant at steady state")
    m = make_model()
    T_cool = 333.15
    for j in [500.0, 2000.0, 8000.0]:
        T_ss = m.steady_state_temperature(j, T_cool)
        assert_true(T_ss >= T_cool - 0.01,
                    f"T_ss({j})={T_ss:.2f} K >= T_coolant={T_cool} K")


# ---------------------------------------------------------------------------
# Test 8 -- Transient: temperature converges to steady-state
# ---------------------------------------------------------------------------

def test_transient_converges():
    print("\n[Test 8] Transient temperature converges to steady-state")
    m = make_model()
    j = 5000.0
    T_cool = 333.15
    T0 = T_cool  # start at coolant temp
    T_ss = m.steady_state_temperature(j, T_cool)
    t_arr, T_arr = m.transient_temperature(j, T_cool, T0, (0, 3000), n_steps=500)
    T_final = float(T_arr[-1])
    # Final transient value should match SS within 2 K
    assert_true(abs(T_final - T_ss) < 2.0,
                f"|T_final({T_final:.2f}) - T_ss({T_ss:.2f})| < 2 K")


# ---------------------------------------------------------------------------
# Test 9 -- Hydrogen production proportional to current (Faraday)
# ---------------------------------------------------------------------------

def test_h2_faraday():
    print("\n[Test 9] Hydrogen rate proportional to current (Faraday's law)")
    m = make_model()
    j1, j2 = 2000.0, 4000.0
    n1 = float(m.hydrogen_rate(j1))
    n2 = float(m.hydrogen_rate(j2))
    ratio = n2 / n1
    assert_true(abs(ratio - 2.0) < 1e-9, f"n_H2(j2)/n_H2(j1) = {ratio:.6f} = 2.0")


# ---------------------------------------------------------------------------
# Test 10 -- LHV efficiency in physically plausible range (50 – 90 %)
# ---------------------------------------------------------------------------

def test_efficiency_range():
    print("\n[Test 10] LHV efficiency in [0.5, 0.9] range")
    m = make_model()
    for T in [313.15, 333.15, 353.15]:
        for j in [1000.0, 5000.0]:
            eta = float(m.efficiency_lhv(j, T))
            assert_true(0.50 <= eta <= 0.90,
                        f"eta_lhv({j},{T})={eta:.3f} in [0.5, 0.9]")


# ---------------------------------------------------------------------------
# Test 11 -- ComponentModel predict() interface
# ---------------------------------------------------------------------------

def test_predict_interface():
    print("\n[Test 11] ComponentModel predict() interface")
    cm = ComponentModel()
    out = cm.predict({"current_density": 5000.0, "temperature": 333.15})
    required = [
        "cell_voltage_V", "stack_voltage_V", "power_kW",
        "heat_generation_W", "ASR_ohm_cm2", "hydrogen_rate_mol_s",
        "efficiency_lhv", "temperature_K",
    ]
    for k in required:
        assert_true(k in out, f"Output key '{k}' present")
    assert_true(out["cell_voltage_V"] > 1.229, "V_cell > E_rev_ref (electrolysis)")
    assert_true(out["heat_generation_W"] > 0, "Q_gen > 0")

    # solve_thermal mode
    out2 = cm.predict({"current_density": 5000.0, "solve_thermal": True})
    assert_true(out2["temperature_K"] > 0, "Thermal solve returns valid T")


# ---------------------------------------------------------------------------
# Test 12 -- Benchmark: 1000 vectorized evaluations
# ---------------------------------------------------------------------------

def test_benchmark():
    print("\n[Test 12] Benchmark: 10 000 evaluations")
    m = make_model()
    j_arr = np.linspace(100.0, 18000.0, 10000)
    t0 = time.perf_counter()
    m.evaluate(j_arr, 333.15)
    elapsed = time.perf_counter() - t0
    print(f"  10 000 evaluations (vectorized) in {elapsed*1000:.1f} ms")
    assert_true(elapsed < 5.0, "Completes in < 5 s")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        test_voltage_decreases_with_temperature,
        test_asr_decreases_with_temperature,
        test_exchange_current_increases,
        test_voltage_above_e_rev,
        test_heat_positive,
        test_thermal_ss_rises_with_current,
        test_thermal_ss_above_coolant,
        test_transient_converges,
        test_h2_faraday,
        test_efficiency_range,
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
    print(f"EC011 AEM F1b Thermal -- Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    sys.exit(0 if failed == 0 else 1)
