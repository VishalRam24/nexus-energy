"""
EC089 — Hydrogen Boiler — F1a Constant Efficiency
Test suite: physics sanity checks, edge cases, benchmark.
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from model import HydrogenBoilerModel
from predict import ComponentModel

DEFAULT_PARAMS = {
    "Q_rated":      30.0,
    "eta_nom":      0.92,
    "PLR_min":      0.10,
    "LHV_H2_MJ_kg": 120.0,
    "P_standby_kw": 0.05,
    "co2_factor_g_per_kwh_th": 0.0,
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
    return HydrogenBoilerModel(DEFAULT_PARAMS)


def test_eta_at_full_load():
    print("\n[Test 1] Efficiency = eta_nom at PLR=1")
    m = make_model()
    eta = m.efficiency(1.0)
    assert_true(abs(eta - m.eta_nom) < 1e-9,
                f"eta(1)={eta:.6f} == eta_nom={m.eta_nom}")


def test_eta_below_unity():
    print("\n[Test 2] Efficiency <= 1 for all PLR")
    m = make_model()
    for PLR in np.linspace(0.0, 1.0, 30):
        assert_true(m.efficiency(PLR) <= 1.0, f"eta<=1 at PLR={PLR:.2f}")


def test_fuel_geq_thermal():
    print("\n[Test 3] Fuel input >= thermal output (energy conservation)")
    m = make_model()
    for PLR in np.linspace(0.1, 1.0, 20):
        Q_out = m.thermal_output_kw(PLR)
        Q_fuel = m.fuel_input_kw(PLR)
        assert_true(Q_fuel >= Q_out - 1e-9,
                    f"Q_fuel({Q_fuel:.2f}) >= Q_out({Q_out:.2f}) at PLR={PLR:.2f}")


def test_zero_co2():
    print("\n[Test 4] Point-of-use CO2 = 0")
    cm = ComponentModel()
    out = cm.predict({"part_load_ratio": 0.7})
    assert_true(out["co2_emissions_g_per_kwh_th"] == 0.0,
                "CO2 = 0 g/kWh_th")


def test_water_stoichiometry():
    print("\n[Test 5] Water vapour mass flow = 9.0 * H2 mass flow")
    m = make_model()
    m_h2 = m.h2_mass_flow_kg_h(0.5)
    m_h2o = m.water_vapour_kg_h(0.5)
    ratio = m_h2o / m_h2
    expected = 18.015 / 2.016
    assert_true(abs(ratio - expected) < 1e-9,
                f"H2O/H2 ratio = {ratio:.4f} == {expected:.4f}")


def test_h2_mass_flow_units():
    print("\n[Test 6] H2 mass flow consistent with LHV")
    m = make_model()
    PLR = 1.0
    Q_fuel_kw = m.fuel_input_kw(PLR)         # kW
    m_h2 = m.h2_mass_flow_kg_h(PLR)          # kg/h
    # Recompute Q_fuel from m_h2: kg/h * MJ/kg / 3.6 = kW
    Q_check = m_h2 * m.LHV_H2 / 3.6
    assert_true(abs(Q_check - Q_fuel_kw) < 1e-6,
                f"Q_fuel from m_H2 = {Q_check:.4f} kW == {Q_fuel_kw:.4f} kW")


def test_plr_min_clamping():
    print("\n[Test 7] PLR < PLR_min clamps")
    m = make_model()
    Q_a = m.thermal_output_kw(0.05)
    Q_b = m.thermal_output_kw(m.PLR_min)
    assert_true(abs(Q_a - Q_b) < 1e-9,
                f"Q(0.05)={Q_a:.4f} == Q(PLR_min)={Q_b:.4f}")


def test_predict_interface():
    print("\n[Test 8] ComponentModel predict() interface")
    cm = ComponentModel()
    out = cm.predict({"part_load_ratio": 0.6})
    required = [
        "thermal_output_kw", "fuel_input_kw", "efficiency",
        "h2_mass_flow_kg_h", "water_vapour_kg_h", "PLR_effective",
        "standby_power_kw", "co2_emissions_g_per_kwh_th",
    ]
    for k in required:
        assert_true(k in out, f"Output key '{k}' present")
    assert_true(out["efficiency"] <= 1.0, "efficiency <= 1.0")
    assert_true(out["fuel_input_kw"] >= out["thermal_output_kw"],
                "fuel >= thermal")


def test_get_info():
    print("\n[Test 9] get_info() completeness")
    cm = ComponentModel()
    info = cm.get_info()
    for k in ["component_id", "component_name", "fidelity",
              "inputs", "outputs", "source"]:
        assert_true(k in info, f"'{k}' present")
    assert_true(info["component_id"] == "EC089", "EC ID is EC089")


def test_benchmark():
    print("\n[Test 10] Benchmark: 10,000 evaluate() calls")
    m = make_model()
    plr_vals = np.linspace(0.1, 1.0, 10000)
    t0 = time.perf_counter()
    for plr in plr_vals:
        m.evaluate(float(plr))
    elapsed = time.perf_counter() - t0
    print(f"  10,000 evaluations in {elapsed*1000:.1f} ms")
    assert_true(elapsed < 5.0, "10,000 calls < 5 s")


if __name__ == "__main__":
    tests = [
        test_eta_at_full_load,
        test_eta_below_unity,
        test_fuel_geq_thermal,
        test_zero_co2,
        test_water_stoichiometry,
        test_h2_mass_flow_units,
        test_plr_min_clamping,
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

    print(f"\n{'='*60}")
    print(f"EC089 Hydrogen Boiler — Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    sys.exit(0 if failed == 0 else 1)
