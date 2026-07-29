"""
EC087 — Biomass Boiler — F1a Part-Load Efficiency
Test suite: physics sanity checks, edge cases, benchmark.
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from model import BiomassBoilerModel
from predict import ComponentModel

DEFAULT_PARAMS = {
    "Q_rated":          50.0,
    "eta_nom":          0.88,
    "a0":               0.55,
    "a1":               0.65,
    "a2":              -0.20,
    "PLR_min":          0.30,
    "LHV_fuel_MJ_kg":  17.5,
    "moisture_content": 0.10,
    "co2_factor_g_per_kwh_th": 18.0,
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
    return BiomassBoilerModel(DEFAULT_PARAMS)


def test_polynomial_constraint():
    print("\n[Test 1] Polynomial constraint a0+a1+a2=1")
    m = make_model()
    assert_true(abs(m.a0 + m.a1 + m.a2 - 1.0) < 1e-9, "a0+a1+a2=1")
    bad = dict(DEFAULT_PARAMS)
    bad["a2"] = 0.0
    try:
        BiomassBoilerModel(bad)
        assert_true(False, "Should raise on bad coefficients")
    except ValueError:
        assert_true(True, "ValueError raised for bad coefficients")


def test_eta_at_full_load():
    print("\n[Test 2] Efficiency = eta_nom at PLR=1")
    m = make_model()
    eta = m.efficiency(1.0)
    assert_true(abs(eta - m.eta_nom) < 1e-9,
                f"eta(PLR=1)={eta:.6f} == eta_nom={m.eta_nom}")


def test_eta_below_unity():
    print("\n[Test 3] Efficiency <= 1 for all PLR")
    m = make_model()
    for PLR in np.linspace(0.0, 1.0, 50):
        assert_true(m.efficiency(PLR) <= 1.0, f"eta<=1 at PLR={PLR:.2f}")


def test_eta_drops_at_low_plr():
    print("\n[Test 4] Efficiency at PLR_min < efficiency at PLR=1")
    m = make_model()
    eta_low = m.efficiency(m.PLR_min)
    eta_full = m.efficiency(1.0)
    assert_true(eta_low < eta_full,
                f"eta(PLR_min)={eta_low:.4f} < eta(1)={eta_full:.4f}")


def test_fuel_geq_thermal():
    print("\n[Test 5] Fuel input >= thermal output (energy conservation)")
    m = make_model()
    for PLR in np.linspace(0.3, 1.0, 20):
        Q_out = m.thermal_output_kw(PLR)
        Q_fuel = m.fuel_input_kw(PLR)
        assert_true(Q_fuel >= Q_out - 1e-9,
                    f"Q_fuel({Q_fuel:.2f}) >= Q_out({Q_out:.2f}) PLR={PLR:.2f}")


def test_lhv_moisture_correction():
    print("\n[Test 6] Effective LHV decreases with moisture")
    m = make_model()
    LHV_eff = m.effective_lhv_MJ_kg()
    assert_true(LHV_eff < m.LHV_dry,
                f"LHV_eff={LHV_eff:.2f} < LHV_dry={m.LHV_dry:.2f}")
    # Wetter fuel should give even lower LHV
    wet = dict(DEFAULT_PARAMS)
    wet["moisture_content"] = 0.40
    m2 = BiomassBoilerModel(wet)
    assert_true(m2.effective_lhv_MJ_kg() < LHV_eff,
                "Wetter fuel -> lower effective LHV")


def test_fuel_mass_flow_positive():
    print("\n[Test 7] Fuel mass flow > 0 above PLR_min")
    m = make_model()
    for PLR in np.linspace(0.3, 1.0, 10):
        mf = m.fuel_mass_flow_kg_h(PLR)
        assert_true(mf > 0, f"m_fuel={mf:.3f} > 0 at PLR={PLR:.2f}")


def test_plr_min_clamping():
    print("\n[Test 8] PLR below PLR_min is clamped")
    m = make_model()
    eta_a = m.efficiency(0.10)
    eta_b = m.efficiency(m.PLR_min)
    assert_true(abs(eta_a - eta_b) < 1e-9,
                f"eta(0.10)={eta_a:.6f} == eta(PLR_min)={eta_b:.6f}")


def test_predict_interface():
    print("\n[Test 9] ComponentModel predict() interface")
    cm = ComponentModel()
    out = cm.predict({"part_load_ratio": 0.6})
    required = [
        "thermal_output_kw", "fuel_input_kw", "efficiency",
        "fuel_mass_flow_kg_h", "LHV_effective_MJ_kg",
        "PLR_effective", "co2_emissions_g_per_kwh_th",
    ]
    for k in required:
        assert_true(k in out, f"Output key '{k}' present")
    assert_true(out["efficiency"] <= 1.0, "efficiency <= 1.0")
    assert_true(out["fuel_input_kw"] >= out["thermal_output_kw"],
                "fuel >= thermal")


def test_get_info():
    print("\n[Test 10] get_info() completeness")
    cm = ComponentModel()
    info = cm.get_info()
    for k in ["component_id", "component_name", "fidelity",
              "inputs", "outputs", "source"]:
        assert_true(k in info, f"'{k}' present")
    assert_true(info["component_id"] == "EC087", "EC ID is EC087")


def test_benchmark():
    print("\n[Test 11] Benchmark: 10,000 evaluate() calls")
    m = make_model()
    plr_vals = np.linspace(0.3, 1.0, 10000)
    t0 = time.perf_counter()
    for plr in plr_vals:
        m.evaluate(float(plr))
    elapsed = time.perf_counter() - t0
    print(f"  10,000 evaluations in {elapsed*1000:.1f} ms")
    assert_true(elapsed < 5.0, "10,000 calls < 5 s")


if __name__ == "__main__":
    tests = [
        test_polynomial_constraint,
        test_eta_at_full_load,
        test_eta_below_unity,
        test_eta_drops_at_low_plr,
        test_fuel_geq_thermal,
        test_lhv_moisture_correction,
        test_fuel_mass_flow_positive,
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
    print(f"EC087 Biomass Boiler — Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    sys.exit(0 if failed == 0 else 1)
