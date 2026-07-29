"""
EC008 -- PEM Electrolyser (PEMEL) -- F1b Thermal
Test suite: temperature-dependent physics sanity checks.
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from model import PEMELThermalModel
from predict import ComponentModel

DEFAULT_PARAMS = {
    "T_ref": 353.15,
    "N_cells": 20,
    "A_cell": 100.0,
    "i0_anode_ref": 1e-7,
    "E_act_anode": 76000.0,
    "i0_cathode_ref": 1e-3,
    "E_act_cathode": 18000.0,
    "alpha_a": 0.5,
    "alpha_c": 0.5,
    "sigma_ref": 0.1,
    "membrane_thickness": 0.0183,
    "lambda_mem": 14.0,
    "pressure": 1.0,
    "faradaic_efficiency": 0.99,
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
    return PEMELThermalModel(DEFAULT_PARAMS)


def test_higher_temp_lower_voltage():
    """Electrolyser: higher T -> lower overpotentials -> lower cell voltage at same j."""
    print("\n[Test 1] Higher T -> lower V_cell at j=1.0 (electrolyser)")
    m = make_model()
    V_cold = float(m.cell_voltage(1.0, 323.0))
    V_hot = float(m.cell_voltage(1.0, 363.0))
    assert_true(V_hot < V_cold,
                f"V(363K)={V_hot:.4f} < V(323K)={V_cold:.4f}")


def test_efficiency_increases_with_temp():
    """Electrolyser efficiency should increase with temperature."""
    print("\n[Test 2] Efficiency increases with temperature")
    m = make_model()
    eta_cold = float(m.efficiency_voltage(1.0, 323.0))
    eta_hot = float(m.efficiency_voltage(1.0, 363.0))
    assert_true(eta_hot > eta_cold,
                f"eta(363K)={eta_hot:.4f} > eta(323K)={eta_cold:.4f}")


def test_voltage_above_reversible():
    """For electrolyser, V_cell must be >= E_rev."""
    print("\n[Test 3] V_cell >= E_rev for all operating points")
    m = make_model()
    for T in [323.0, 343.0, 363.0]:
        for j in [0.1, 0.5, 1.0, 2.0]:
            V = float(m.cell_voltage(j, T))
            E = float(m.reversible_voltage(T))
            assert_true(V >= E - 0.001,
                        f"V({j},{T:.0f})={V:.4f} >= E_rev={E:.4f}")


def test_voltage_in_physical_bounds():
    """Cell voltage should be reasonable (1.0 - 3.0 V typical)."""
    print("\n[Test 4] Cell voltage in physical bounds")
    m = make_model()
    for T in [323.0, 343.0, 363.0]:
        for j in [0.01, 0.5, 1.0, 2.0]:
            V = float(m.cell_voltage(j, T))
            assert_true(1.0 <= V <= 4.0,
                        f"V({j},{T:.0f})={V:.4f} in [1.0, 4.0]")


def test_heat_generation_sign():
    """At high current, V > E_tn, so heat generation should be positive."""
    print("\n[Test 5] Heat generation positive at high current density")
    m = make_model()
    # At moderate-high j, V_cell > E_tn -> positive heat
    for T in [323.0, 353.0]:
        Q = float(m.heat_generation(2.0, T))
        assert_true(Q > 0, f"Q(2.0, {T:.0f})={Q:.4f} > 0")


def test_membrane_resistance_decreases():
    print("\n[Test 6] Membrane resistance decreases with T")
    m = make_model()
    R_cold = float(m.membrane_resistance(323.0))
    R_hot = float(m.membrane_resistance(363.0))
    assert_true(R_hot < R_cold,
                f"R_mem(363K)={R_hot:.4f} < R_mem(323K)={R_cold:.4f}")


def test_h2_production_positive():
    print("\n[Test 7] H2 production rate positive for j > 0")
    m = make_model()
    rate = float(m.h2_production_rate(1.0, 353.0))
    assert_true(rate > 0, f"H2_rate(1.0, 353)={rate:.8f} > 0")


def test_predict_interface():
    print("\n[Test 8] ComponentModel predict() interface")
    cm = ComponentModel()
    out = cm.predict({"current_density": 1.0, "temperature": 353.15})
    required = ["cell_voltage_V", "power_consumption_W_cm2", "efficiency_voltage",
                "efficiency_faradaic", "h2_production_rate_mol_s_cm2", "heat_generation_W_cm2"]
    for k in required:
        assert_true(k in out, f"Output key '{k}' present")
    assert_true(out["cell_voltage_V"] > 1.0, "cell_voltage > 1.0 V")


def test_benchmark():
    print("\n[Test 9] Benchmark: 10,000 evaluations")
    m = make_model()
    j_arr = np.linspace(0.01, 2.0, 10000)
    t0 = time.perf_counter()
    m.evaluate(j_arr, 353.0)
    elapsed = time.perf_counter() - t0
    print(f"  10,000 evaluations (vectorized) in {elapsed*1000:.1f} ms")
    assert_true(elapsed < 2.0, "Completes in < 2 s")


if __name__ == "__main__":
    tests = [
        test_higher_temp_lower_voltage,
        test_efficiency_increases_with_temp,
        test_voltage_above_reversible,
        test_voltage_in_physical_bounds,
        test_heat_generation_sign,
        test_membrane_resistance_decreases,
        test_h2_production_positive,
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
    print(f"EC008 PEMEL F1b Thermal -- Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    sys.exit(0 if failed == 0 else 1)
