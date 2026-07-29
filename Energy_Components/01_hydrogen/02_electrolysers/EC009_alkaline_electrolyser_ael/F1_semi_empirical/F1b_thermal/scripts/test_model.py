"""
EC009 -- Alkaline Electrolyser (AEL) -- F1b Thermal
Test suite: temperature-dependent physics sanity checks.
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from model import AELThermalModel
from predict import ComponentModel

DEFAULT_PARAMS = {
    "T_ref": 343.0,
    "N_cells": 20,
    "A_cell": 0.25,
    "koh_concentration": 30.0,
    "sigma_KOH_ref": 0.5,
    "E_act_koh": 15000.0,
    "electrode_gap": 0.002,
    "i0_ref": 0.001,
    "E_act_electrode": 40000.0,
    "alpha": 0.3,
    "j_L": 5000.0,
    "bubble_coeff": 0.3,
    "faradaic_f1": 250.0,
    "faradaic_f2": 0.98,
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
    return AELThermalModel(DEFAULT_PARAMS)


def test_higher_temp_lower_voltage():
    """Electrolyser: higher T -> lower V_cell at same j."""
    print("\n[Test 1] Higher T -> lower V_cell at j=2000 A/m2 (electrolyser)")
    m = make_model()
    V_cold = float(m.cell_voltage(2000.0, 333.0))
    V_hot = float(m.cell_voltage(2000.0, 373.0))
    assert_true(V_hot < V_cold,
                f"V(373K)={V_hot:.4f} < V(333K)={V_cold:.4f}")


def test_efficiency_increases_with_temp():
    print("\n[Test 2] Efficiency increases with temperature")
    m = make_model()
    eta_cold = float(m.efficiency(2000.0, 333.0))
    eta_hot = float(m.efficiency(2000.0, 373.0))
    assert_true(eta_hot > eta_cold,
                f"eta(373K)={eta_hot:.4f} > eta(333K)={eta_cold:.4f}")


def test_koh_conductivity_increases():
    print("\n[Test 3] KOH conductivity increases with temperature")
    m = make_model()
    s_cold = float(m.koh_conductivity(333.0))
    s_hot = float(m.koh_conductivity(373.0))
    assert_true(s_hot > s_cold,
                f"sigma(373K)={s_hot:.4f} > sigma(333K)={s_cold:.4f}")


def test_voltage_above_reversible():
    print("\n[Test 4] V_cell >= E_rev for all points")
    m = make_model()
    for T in [333.0, 353.0, 373.0]:
        for j in [500.0, 1000.0, 2000.0, 3000.0]:
            V = float(m.cell_voltage(j, T))
            E = float(m.reversible_voltage(T))
            assert_true(V >= E - 0.001,
                        f"V({j},{T:.0f})={V:.4f} >= E_rev={E:.4f}")


def test_voltage_in_bounds():
    print("\n[Test 5] Cell voltage in physical bounds")
    m = make_model()
    for T in [333.0, 353.0, 373.0]:
        for j in [500.0, 2000.0, 4000.0]:
            V = float(m.cell_voltage(j, T))
            assert_true(1.0 <= V <= 3.5,
                        f"V({j},{T:.0f})={V:.4f} in [1.0, 3.5]")


def test_bubble_coverage_decreases_with_temp():
    print("\n[Test 6] Bubble coverage decreases at higher T (same j)")
    m = make_model()
    theta_cold = float(m.bubble_coverage(2000.0, 333.0))
    theta_hot = float(m.bubble_coverage(2000.0, 373.0))
    assert_true(theta_hot < theta_cold,
                f"theta(373K)={theta_hot:.4f} < theta(333K)={theta_cold:.4f}")


def test_h2_production_positive():
    print("\n[Test 7] H2 production rate positive for j > 0")
    m = make_model()
    rate = float(m.h2_production_rate(2000.0, 353.0))
    assert_true(rate > 0, f"H2_rate(2000, 353)={rate:.6f} > 0")


def test_predict_interface():
    print("\n[Test 8] ComponentModel predict() interface")
    cm = ComponentModel()
    out = cm.predict({"current_density": 2000.0, "temperature": 353.0})
    required = ["cell_voltage_V", "power_consumption_kW", "efficiency",
                "h2_production_rate_mol_s"]
    for k in required:
        assert_true(k in out, f"Output key '{k}' present")
    assert_true(out["cell_voltage_V"] > 1.0, "cell_voltage > 1.0 V")


def test_benchmark():
    print("\n[Test 9] Benchmark: 10,000 evaluations")
    m = make_model()
    j_arr = np.linspace(100, 4000, 10000)
    t0 = time.perf_counter()
    m.evaluate(j_arr, 353.0)
    elapsed = time.perf_counter() - t0
    print(f"  10,000 evaluations (vectorized) in {elapsed*1000:.1f} ms")
    assert_true(elapsed < 2.0, "Completes in < 2 s")


if __name__ == "__main__":
    tests = [
        test_higher_temp_lower_voltage,
        test_efficiency_increases_with_temp,
        test_koh_conductivity_increases,
        test_voltage_above_reversible,
        test_voltage_in_bounds,
        test_bubble_coverage_decreases_with_temp,
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
    print(f"EC009 AEL F1b Thermal -- Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    sys.exit(0 if failed == 0 else 1)
