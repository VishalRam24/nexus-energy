"""
EC206 -- CO2 Mineralization F2a -- Test suite.
Physics sanity: conversion bounds/monotonicity, Arrhenius T-dependence,
particle-size effect, carbon/mass conservation, exothermic energy balance,
slow kinetics, permanence stoichiometry, predict() interface, benchmark.
Run:  python3 scripts/test_model.py   (system python3, NOT pytest)
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from model import CO2Mineralization_F2a
from predict import ComponentModel

PASS = "✓"
FAIL = "✗"


def assert_true(cond, msg):
    if cond:
        print(f"  {PASS}  {msg}")
    else:
        print(f"  {FAIL}  FAILED: {msg}")
        raise AssertionError(msg)


def make_model():
    cm = ComponentModel()
    return cm._model, cm


# ---------------------------------------------------------------------------
def test_conversion_bounds():
    print("\n[Test 1] Conversion stays in [0, 1]")
    m, _ = make_model()
    r = m.simulate(458.15, 115.0, 3.7e-5, 30.0, 7200.0)
    assert_true(np.all(r["conversion"] >= -1e-9), "X >= 0 everywhere")
    assert_true(np.all(r["conversion"] <= 1.0 + 1e-9), "X <= 1 everywhere")
    assert_true(r["conversion"][0] < 1e-6, f"X(0)={r['conversion'][0]:.2e} ~ 0")


def test_conversion_monotone():
    print("\n[Test 2] Conversion is non-decreasing in time")
    m, _ = make_model()
    r = m.simulate(458.15, 115.0, 3.7e-5, 30.0, 3600.0)
    dX = np.diff(r["conversion"])
    assert_true(np.all(dX >= -1e-9), "X monotonically non-decreasing")


def test_reference_yield():
    print("\n[Test 3] Reference conditions give ~80% in 1 h (O'Connor 2005)")
    m, _ = make_model()
    X = m.conversion_analytic(458.15, 115.0, 3.7e-5, 3600.0)
    assert_true(0.70 < X < 0.90, f"X(1h) = {X:.3f} in literature band [0.70,0.90]")


def test_arrhenius_temperature():
    print("\n[Test 4] Arrhenius: higher T -> faster reaction")
    m, _ = make_model()
    X_hot = m.conversion_analytic(458.15, 115.0, 3.7e-5, 3600.0)
    X_cold = m.conversion_analytic(298.15, 115.0, 3.7e-5, 3600.0)
    assert_true(X_hot > X_cold, f"X(185C)={X_hot:.3f} > X(25C)={X_cold:.4f}")
    assert_true(m.rate_constant(458.15) > m.rate_constant(298.15),
                "k(T) increases with T")


def test_particle_size_effect():
    print("\n[Test 5] Smaller particles carbonate faster")
    m, _ = make_model()
    X_fine = m.conversion_analytic(458.15, 115.0, 1.85e-5, 3600.0)
    X_coarse = m.conversion_analytic(458.15, 115.0, 7.4e-5, 3600.0)
    assert_true(X_fine > X_coarse,
                f"X(18um)={X_fine:.3f} > X(74um)={X_coarse:.3f}")


def test_pressure_effect():
    print("\n[Test 6] Higher CO2 pressure -> faster reaction")
    m, _ = make_model()
    X_hi = m.conversion_analytic(458.15, 150.0, 3.7e-5, 3600.0)
    X_lo = m.conversion_analytic(458.15, 50.0, 3.7e-5, 3600.0)
    assert_true(X_hi > X_lo, f"X(150atm)={X_hi:.3f} > X(50atm)={X_lo:.3f}")


def test_carbon_conservation():
    print("\n[Test 7] Carbon/mass conservation: CO2 bound = stoich * mol reacted")
    m, _ = make_model()
    r = m.simulate(458.15, 115.0, 3.7e-5, 60.0, 3600.0)
    Xf = r["conversion"][-1]
    expected_mol = m.stoich * m.n_mineral0 * Xf
    expected_kg = expected_mol * m.M_CO2
    assert_true(abs(r["co2_bound_kg"][-1] - expected_kg) < 1e-6,
                f"CO2 bound {r['co2_bound_kg'][-1]:.3f} kg == stoich*n*X*M_CO2")
    # CO2 bound cannot exceed full-conversion theoretical maximum
    max_kg = m.stoich * m.n_mineral0 * m.M_CO2
    assert_true(np.all(r["co2_bound_kg"] <= max_kg + 1e-9),
                f"CO2 bound <= theoretical max {max_kg:.1f} kg")


def test_permanence_stoichiometry():
    print("\n[Test 8] Permanence: 2 mol CO2 bound per mol forsterite")
    m, _ = make_model()
    assert_true(abs(m.stoich - 2.0) < 1e-9, "stoich = 2 mol CO2 / mol Mg2SiO4")
    # carbonate (MgCO3) mass must exceed bound CO2 mass (carbonate = CO2 + MgO)
    r = m.simulate(458.15, 115.0, 3.7e-5, 60.0, 3600.0)
    assert_true(r["carbonate_kg"][-1] > r["co2_bound_kg"][-1] > 0,
                "MgCO3 mass > bound-CO2 mass (CO2 locked into mineral)")


def test_exothermic_energy_balance():
    print("\n[Test 9] Exothermic: dH<0, heat released >=0, T rises above coolant")
    m, _ = make_model()
    assert_true(m.dH < 0, f"dH_rxn = {m.dH:.0f} J/mol < 0 (exothermic)")
    r = m.simulate(458.15, 115.0, 3.7e-5, 30.0, 3600.0)
    assert_true(np.all(r["heat_released_J"] >= -1e-6), "cumulative heat >= 0")
    assert_true(np.all(np.diff(r["heat_released_J"]) >= -1e-6),
                "cumulative heat non-decreasing")
    # exotherm pushes slurry above the coolant setpoint during fast early stage
    assert_true(r["temperature"].max() > m.T_coolant,
                f"peak T {r['temperature'].max():.1f} K > coolant {m.T_coolant:.1f} K")
    # energy-balance closure: cumulative heat == (-dH)*mol CO2 bound
    mol_co2 = m.stoich * m.n_mineral0 * r["conversion"][-1]
    assert_true(abs(r["heat_released_J"][-1] - (-m.dH) * mol_co2) < 1e-3,
                "cumulative heat == (-dH)*mol_CO2_bound")


def test_slow_kinetics_ambient():
    print("\n[Test 10] Slow kinetics: negligible conversion at ambient in 1 h")
    m, _ = make_model()
    r = m.simulate(298.15, 1.0, 3.7e-5, 60.0, 3600.0)
    assert_true(r["conversion"][-1] < 0.02,
                f"X={r['conversion'][-1]:.4f} < 2% at 25C/1atm (mineral carb. is slow)")


def test_predict_interface():
    print("\n[Test 11] ComponentModel predict()/get_info() interface")
    _, cm = make_model()
    info = cm.get_info()
    for k in ["component_id", "component_name", "fidelity", "version"]:
        assert_true(k in info, f"get_info has '{k}'")
    assert_true(cm.component_id == "EC206", "component_id == EC206")
    r = cm.predict({"duration_s": 1800.0, "dt": 60.0})
    for key in ["t", "conversion", "temperature", "co2_bound_kg", "carbonate_kg",
                "heat_released_J", "final_conversion", "co2_stored_kg",
                "peak_temperature_K"]:
        assert_true(key in r, f"predict output has '{key}'")
    assert_true(len(r["t"]) == len(r["conversion"]), "arrays same length")


def test_benchmark():
    print("\n[Test 12] Benchmark: 2 h batch sim runtime")
    m, _ = make_model()
    t0 = time.perf_counter()
    m.simulate(458.15, 115.0, 3.7e-5, 30.0, 7200.0)
    elapsed = time.perf_counter() - t0
    print(f"  2 h batch ODE in {elapsed*1000:.1f} ms")
    assert_true(elapsed < 5.0, "completes in < 5 s")


if __name__ == "__main__":
    tests = [
        test_conversion_bounds,
        test_conversion_monotone,
        test_reference_yield,
        test_arrhenius_temperature,
        test_particle_size_effect,
        test_pressure_effect,
        test_carbon_conservation,
        test_permanence_stoichiometry,
        test_exothermic_energy_balance,
        test_slow_kinetics_ambient,
        test_predict_interface,
        test_benchmark,
    ]
    passed = failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except (AssertionError, Exception) as e:
            failed += 1
            print(f"  ERROR: {e}")

    print(f"\n{'='*60}")
    print(f"EC206 CO2 Mineralization F2a -- Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    sys.exit(0 if failed == 0 else 1)
