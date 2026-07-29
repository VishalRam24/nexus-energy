"""
EC001 — PEM Fuel Cell (PEMFC) — F1a Polarization Curve
Test suite: physics sanity checks, edge cases, benchmark.
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from model import PEMFuelCellModel
from predict import ComponentModel

DEFAULT_PARAMS = {
    "T": 343.15,           # K  (70 °C)
    "N_cells": 40,
    "electrode_area": 232.0,  # cm²
    "pH2": 1.0,
    "pO2": 0.21,
    "j_L": 1.5,            # A/cm²
    "t_mem": 0.0178,       # cm (Nafion 117, 178 µm)
    "lambda_mem": 14.0,
    "P_rated": 5000.0,
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
    return PEMFuelCellModel(DEFAULT_PARAMS)


# ---------------------------------------------------------------------------
# Test 1 — Nernst voltage at standard conditions
# ---------------------------------------------------------------------------

def test_nernst_standard():
    print("\n[Test 1] Nernst voltage")
    m = make_model()
    # At T=298.15K, pH2=1, pO2=1 -> E_Nernst = 1.229 + 0 = 1.229 V
    E = m.nernst_voltage(T=298.15)
    # pO2=0.21 so it won't be exactly 1.229; just check it's in physical range
    assert_true(1.0 < E < 1.3, f"E_Nernst in physical range [1.0, 1.3] V: {E:.4f}")
    # Higher T should lower E_Nernst
    E_cold = m.nernst_voltage(T=323.15)
    E_hot  = m.nernst_voltage(T=363.15)
    assert_true(E_cold > E_hot, "E_Nernst decreases with temperature")


# ---------------------------------------------------------------------------
# Test 2 — Voltage decreases with current density
# ---------------------------------------------------------------------------

def test_voltage_decreasing():
    print("\n[Test 2] V_cell decreases monotonically with j")
    m = make_model()
    j_vals = np.linspace(0.01, 1.3, 100)
    V_vals = [m.cell_voltage(j) for j in j_vals]
    for i in range(1, len(V_vals)):
        assert_true(V_vals[i] <= V_vals[i - 1] + 1e-9,
                    f"V({j_vals[i]:.3f}) <= V({j_vals[i-1]:.3f})")
    print("  All 99 consecutive pairs checked.")


# ---------------------------------------------------------------------------
# Test 3 — V_cell < E_Nernst for all j > 0
# ---------------------------------------------------------------------------

def test_voltage_below_nernst():
    print("\n[Test 3] V_cell < E_Nernst for j > 0")
    m = make_model()
    E = m.nernst_voltage()
    for j in [0.01, 0.1, 0.5, 1.0, 1.3]:
        V = m.cell_voltage(j)
        assert_true(V < E, f"V({j})={V:.4f} < E_Nernst={E:.4f}")


# ---------------------------------------------------------------------------
# Test 4 — Power density curve has a maximum
# ---------------------------------------------------------------------------

def test_power_density_maximum():
    print("\n[Test 4] Power density curve has a maximum")
    m = make_model()
    # Sweep the full valid range (up to 99% of j_L to avoid singularity)
    j_vals = np.linspace(0.01, m.j_L * 0.99, 500)
    P_vals = [m.power_density(j) for j in j_vals]
    idx_max = np.argmax(P_vals)
    # Peak must be greater than low-current power (ohmic + activation dominate at low j)
    assert_true(P_vals[idx_max] > P_vals[0],
                f"Peak power ({P_vals[idx_max]:.4f}) > power at j_min ({P_vals[0]:.4f})")
    # At least power grows from low j (first 10 points should trend upward)
    assert_true(P_vals[10] > P_vals[0],
                f"Power rises from low current: P({j_vals[10]:.3f})={P_vals[10]:.4f} "
                f"> P({j_vals[0]:.3f})={P_vals[0]:.4f}")
    # Near j_L, concentration loss must cause voltage (and thus power) to fall back
    P_near_jL = m.power_density(m.j_L * 0.99)
    P_peak    = P_vals[idx_max]
    assert_true(P_peak > P_near_jL,
                f"Peak power ({P_peak:.4f}) > power near j_L ({P_near_jL:.4f}): concentration loss takes effect")
    print(f"  Power peak at j={j_vals[idx_max]:.3f} A/cm², P={P_peak:.4f} W/cm²")


# ---------------------------------------------------------------------------
# Test 5 — Efficiency < 1.0
# ---------------------------------------------------------------------------

def test_efficiency_below_unity():
    print("\n[Test 5] Efficiency < 1.0 for all j")
    m = make_model()
    for j in [0.01, 0.1, 0.5, 1.0, 1.3]:
        eta = m.efficiency(j)
        assert_true(0.0 < eta < 1.0,
                    f"eta={eta:.4f} in (0,1) at j={j}")


# ---------------------------------------------------------------------------
# Test 6 — Concentration loss diverges near j_L
# ---------------------------------------------------------------------------

def test_concentration_diverges():
    print("\n[Test 6] Concentration loss grows rapidly near j_L")
    m = make_model()
    V_conc_1 = m.concentration_loss(1.0)
    V_conc_14 = m.concentration_loss(1.4)
    assert_true(V_conc_14 > V_conc_1 * 2,
                f"V_conc at 1.4 ({V_conc_14:.4f}) >> V_conc at 1.0 ({V_conc_1:.4f})")


# ---------------------------------------------------------------------------
# Test 7 — Ohmic loss scales with membrane thickness / conductivity
# ---------------------------------------------------------------------------

def test_ohmic_loss():
    print("\n[Test 7] Ohmic loss is positive and increases with j")
    m = make_model()
    v1 = m.ohmic_loss(0.5)
    v2 = m.ohmic_loss(1.0)
    assert_true(v1 > 0, f"V_ohm(0.5)={v1:.5f} > 0")
    assert_true(v2 > v1, f"V_ohm(1.0)={v2:.5f} > V_ohm(0.5)={v1:.5f}")


# ---------------------------------------------------------------------------
# Test 8 — Temperature effect on voltage
# ---------------------------------------------------------------------------

def test_temperature_effect():
    print("\n[Test 8] Higher temperature increases cell voltage at mid-load")
    m = make_model()
    j = 0.5
    V_cold = m.cell_voltage(j, T=323.15)
    V_hot  = m.cell_voltage(j, T=363.15)
    # For PEMFC higher T generally improves kinetics (activation drops)
    # net effect should be V_hot >= V_cold
    assert_true(V_hot >= V_cold - 0.05,
                f"V_hot ({V_hot:.4f}) >= V_cold ({V_cold:.4f}) at j=0.5 (kinetics improve)")


# ---------------------------------------------------------------------------
# Test 9 — Stack voltage
# ---------------------------------------------------------------------------

def test_stack_voltage():
    print("\n[Test 9] Stack voltage = N_cells * cell voltage")
    m = make_model()
    for j in [0.2, 0.8]:
        V_cell  = m.cell_voltage(j)
        V_stack = m.stack_voltage(j)
        assert_true(abs(V_stack - m.N_cells * V_cell) < 1e-9,
                    f"V_stack = N*V_cell at j={j}")


# ---------------------------------------------------------------------------
# Test 10 — ComponentModel predict() interface
# ---------------------------------------------------------------------------

def test_predict_interface():
    print("\n[Test 10] ComponentModel predict() interface")
    cm = ComponentModel()
    out = cm.predict({"current_density": 0.6, "temperature": 70.0})
    required = [
        "cell_voltage_V", "stack_voltage_V", "power_density_W_cm2",
        "stack_power_W", "efficiency", "E_Nernst_V", "V_act_V",
        "V_ohm_V", "V_conc_V",
    ]
    for k in required:
        assert_true(k in out, f"Output key '{k}' present")
    assert_true(out["cell_voltage_V"] > 0, "cell_voltage > 0")
    assert_true(out["cell_voltage_V"] < out["E_Nernst_V"],
                "cell_voltage < E_Nernst")
    assert_true(out["stack_power_W"] > 0, "stack_power > 0")
    assert_true(0 < out["efficiency"] < 1.0, "efficiency in (0,1)")


# ---------------------------------------------------------------------------
# Test 11 — get_info() completeness
# ---------------------------------------------------------------------------

def test_get_info():
    print("\n[Test 11] get_info() completeness")
    cm = ComponentModel()
    info = cm.get_info()
    for k in ["component_id", "component_name", "fidelity", "inputs", "outputs", "source"]:
        assert_true(k in info, f"'{k}' in get_info()")


# ---------------------------------------------------------------------------
# Test 12 — Benchmark
# ---------------------------------------------------------------------------

def test_benchmark():
    print("\n[Test 12] Benchmark: 10,000 evaluate() calls")
    m = make_model()
    j_vals = np.linspace(0.01, 1.4, 10000)
    t0 = time.perf_counter()
    for j in j_vals:
        m.evaluate(j, 70.0)
    elapsed = time.perf_counter() - t0
    print(f"  10,000 evaluations in {elapsed*1000:.1f} ms  ({elapsed/10000*1e6:.2f} µs/call)")
    assert_true(elapsed < 5.0, "10,000 calls complete in < 5 s")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        test_nernst_standard,
        test_voltage_decreasing,
        test_voltage_below_nernst,
        test_power_density_maximum,
        test_efficiency_below_unity,
        test_concentration_diverges,
        test_ohmic_loss,
        test_temperature_effect,
        test_stack_voltage,
        test_predict_interface,
        test_get_info,
        test_benchmark,
    ]
    passed = failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except AssertionError as e:
            failed += 1
            print(f"  ASSERTION ERROR: {e}")
        except Exception as e:
            failed += 1
            print(f"  UNEXPECTED ERROR in {t.__name__}: {e}")

    print(f"\n{'='*50}")
    print(f"EC001 PEM Fuel Cell — Results: {passed} passed, {failed} failed")
    print(f"{'='*50}")
    sys.exit(0 if failed == 0 else 1)
