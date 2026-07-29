"""
EC008 — PEM Electrolyser (PEMEL) — F1a V-I Characteristic
Test suite: physics sanity checks, edge cases, benchmark.
"""

import sys
import os
import math
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from model import PEMElectrolyserModel
from predict import ComponentModel

# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

DEFAULT_PARAMS = {
    "T": 353.15,           # K  (80 °C)
    "N_cells": 20,
    "electrode_area": 100.0,  # cm²
    "j0": 1e-4,            # A/cm²
    "alpha": 0.5,
    "R_membrane": 0.2,     # Ω·cm²
}


def make_model():
    return PEMElectrolyserModel(DEFAULT_PARAMS)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PASS = "\u2713"
FAIL = "\u2717"


def assert_true(cond, msg):
    if cond:
        print(f"  {PASS}  {msg}")
    else:
        print(f"  {FAIL}  FAILED: {msg}")
        raise AssertionError(msg)


# ---------------------------------------------------------------------------
# Test 1 — Reversible voltage physics
# ---------------------------------------------------------------------------

def test_reversible_voltage():
    print("\n[Test 1] Reversible voltage")
    m = make_model()

    E_rev_ref = m.reversible_voltage(298.0)
    assert_true(abs(E_rev_ref - 1.229) < 1e-6,
                "E_rev = 1.229 V at 298 K")

    E_rev_80 = m.reversible_voltage(353.15)
    assert_true(E_rev_80 < 1.229,
                "E_rev decreases with temperature (T > 298 K)")

    E_rev_cold = m.reversible_voltage(313.15)
    E_rev_hot = m.reversible_voltage(363.15)
    assert_true(E_rev_cold > E_rev_hot,
                "E_rev is monotonically decreasing with temperature")


# ---------------------------------------------------------------------------
# Test 2 — Cell voltage > E_rev at all positive current densities
# ---------------------------------------------------------------------------

def test_cell_voltage_above_erev():
    print("\n[Test 2] V_cell > E_rev for j > 0")
    m = make_model()
    for j in [0.01, 0.1, 0.5, 1.0, 2.0]:
        V = m.cell_voltage(j)
        E = m.reversible_voltage()
        assert_true(V > E, f"V_cell ({V:.4f} V) > E_rev ({E:.4f} V) at j={j} A/cm²")


# ---------------------------------------------------------------------------
# Test 3 — Voltage increases monotonically with current density
# ---------------------------------------------------------------------------

def test_voltage_monotone_increasing():
    print("\n[Test 3] V_cell monotonically increases with j")
    m = make_model()
    j_vals = np.linspace(0.01, 2.0, 50)
    V_vals = [m.cell_voltage(j) for j in j_vals]
    for i in range(1, len(V_vals)):
        assert_true(V_vals[i] >= V_vals[i - 1],
                    f"V({j_vals[i]:.3f}) >= V({j_vals[i-1]:.3f})")
    print("  All 49 consecutive pairs checked.")


# ---------------------------------------------------------------------------
# Test 4 — Hydrogen rate proportional to current (Faraday's law)
# ---------------------------------------------------------------------------

def test_h2_rate_faraday():
    print("\n[Test 4] H2 rate proportional to current density")
    m = make_model()
    j1, j2 = 0.5, 1.0
    r1 = m.hydrogen_production_rate(j1)
    r2 = m.hydrogen_production_rate(j2)
    ratio = r2 / r1
    assert_true(abs(ratio - 2.0) < 1e-9,
                f"Doubling j doubles H2 rate (ratio={ratio:.10f})")


# ---------------------------------------------------------------------------
# Test 5 — Efficiency < 1 for all operating points
# ---------------------------------------------------------------------------

def test_efficiency_below_unity():
    print("\n[Test 5] Efficiency < 1.0 for all j > 0")
    m = make_model()
    for j in [0.05, 0.1, 0.5, 1.0, 1.5, 2.0]:
        eta = m.efficiency(j)
        assert_true(eta < 1.0,
                    f"eta={eta:.4f} < 1.0 at j={j} A/cm²")
        assert_true(eta > 0.0,
                    f"eta={eta:.4f} > 0.0 at j={j} A/cm²")


# ---------------------------------------------------------------------------
# Test 6 — Ohmic scaling
# ---------------------------------------------------------------------------

def test_ohmic_scaling():
    print("\n[Test 6] Ohmic overpotential linear in j")
    m = make_model()
    j_vals = [0.1, 0.5, 1.0, 2.0]
    for j in j_vals:
        expected = j * m.R_membrane
        got = m.ohmic_overpotential(j)
        assert_true(abs(got - expected) < 1e-12,
                    f"V_ohm = R_mem * j at j={j}")


# ---------------------------------------------------------------------------
# Test 7 — Stack voltage = N_cells * cell voltage
# ---------------------------------------------------------------------------

def test_stack_voltage():
    print("\n[Test 7] Stack voltage = N_cells * V_cell")
    m = make_model()
    for j in [0.2, 1.0]:
        V_cell = m.cell_voltage(j)
        V_stack = m.stack_voltage(j)
        assert_true(abs(V_stack - m.N_cells * V_cell) < 1e-10,
                    f"V_stack = N * V_cell at j={j}")


# ---------------------------------------------------------------------------
# Test 8 — Temperature effect on cell voltage
# ---------------------------------------------------------------------------

def test_temperature_effect():
    print("\n[Test 8] Higher temperature lowers cell voltage (activation)")
    m = make_model()
    j = 1.0
    V_cold = m.cell_voltage(j, T=313.15)
    V_hot  = m.cell_voltage(j, T=363.15)
    assert_true(V_cold > V_hot,
                f"V_cold ({V_cold:.4f}) > V_hot ({V_hot:.4f}) at j=1 A/cm²")


# ---------------------------------------------------------------------------
# Test 9 — Edge case: j = 0
# ---------------------------------------------------------------------------

def test_zero_current():
    print("\n[Test 9] Edge case: j = 0")
    m = make_model()
    V = m.cell_voltage(0.0)
    E = m.reversible_voltage()
    assert_true(abs(V - E) < 1e-9,
                f"V_cell = E_rev when j=0 (V={V:.6f}, E_rev={E:.6f})")
    eta = m.efficiency(0.0)
    assert_true(eta == 0.0, "Efficiency = 0 when j = 0")


# ---------------------------------------------------------------------------
# Test 10 — ComponentModel predict() interface
# ---------------------------------------------------------------------------

def test_predict_interface():
    print("\n[Test 10] ComponentModel predict() interface")
    cm = ComponentModel()
    out = cm.predict({"current_density": 1.0, "temperature": 80.0})
    required_keys = [
        "cell_voltage_V", "stack_voltage_V", "hydrogen_rate_mol_s",
        "power_W", "efficiency", "E_rev_V", "V_act_V", "V_ohm_V"
    ]
    for k in required_keys:
        assert_true(k in out, f"Output key '{k}' present")

    assert_true(out["cell_voltage_V"] > 0, "cell_voltage > 0")
    assert_true(out["stack_voltage_V"] > out["cell_voltage_V"],
                "stack_voltage > cell_voltage")
    assert_true(out["hydrogen_rate_mol_s"] > 0, "hydrogen_rate > 0")
    assert_true(0 < out["efficiency"] < 1.0, "efficiency in (0, 1)")


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
# Test 12 — Benchmark (performance)
# ---------------------------------------------------------------------------

def test_benchmark():
    print("\n[Test 12] Benchmark: 10,000 evaluate() calls")
    m = make_model()
    j_vals = np.linspace(0.01, 2.0, 10000)
    t0 = time.perf_counter()
    for j in j_vals:
        m.evaluate(j, 80.0)
    elapsed = time.perf_counter() - t0
    print(f"  10,000 evaluations in {elapsed*1000:.1f} ms  "
          f"({elapsed/10000*1e6:.2f} µs/call)")
    assert_true(elapsed < 5.0, "10,000 calls complete in < 5 s")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        test_reversible_voltage,
        test_cell_voltage_above_erev,
        test_voltage_monotone_increasing,
        test_h2_rate_faraday,
        test_efficiency_below_unity,
        test_ohmic_scaling,
        test_stack_voltage,
        test_temperature_effect,
        test_zero_current,
        test_predict_interface,
        test_get_info,
        test_benchmark,
    ]
    passed = 0
    failed = 0
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
    print(f"EC008 PEM Electrolyser — Results: {passed} passed, {failed} failed")
    print(f"{'='*50}")
    sys.exit(0 if failed == 0 else 1)
