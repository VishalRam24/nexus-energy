"""
EC202 -- DAC Liquid Solvent -- F2a Contactor + Calciner
Test suite: physics sanity, conservation, dilute-air capture, regeneration
energy, edge cases, predict() interface, benchmark timing.
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from model import DAC_F2a
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
def test_single_pass_capture_fraction():
    print("\n[Test 1] Single-pass capture is a partial fraction in (0,1)")
    m, _ = make_model()
    eta = m.single_pass_capture()
    assert_true(0.0 < eta < 1.0, f"eta_single={eta:.4f} in (0,1)")
    # DAC contactors are mass-transfer limited -> modest single-pass capture
    assert_true(eta < 0.6, f"eta_single={eta:.4f} < 0.6 (mass-transfer limited)")


def test_capture_from_dilute_air():
    print("\n[Test 2] CO2 captured from genuinely dilute air (~420 ppm)")
    m, _ = make_model()
    c_in = m.inlet_co2_conc()
    # ideal gas at 420 ppm, 20C, 1 atm -> ~0.0175 mol/m3, tiny
    assert_true(0.010 < c_in < 0.030, f"inlet CO2 conc={c_in:.4f} mol/m3 (dilute)")
    # higher ppm -> proportionally higher absorption
    r_lo = m.absorption_rate(ppm=420.0)
    r_hi = m.absorption_rate(ppm=840.0)
    assert_true(abs(r_hi / r_lo - 2.0) < 1e-6, "Absorption rate linear in CO2 ppm")


def test_faster_air_lowers_single_pass_but_raises_throughput():
    print("\n[Test 3] Faster air: lower single-pass eta, higher total absorption")
    m, _ = make_model()
    eta_slow = m.single_pass_capture(u_air=1.0)
    eta_fast = m.single_pass_capture(u_air=3.0)
    assert_true(eta_fast < eta_slow, f"eta(3 m/s)={eta_fast:.3f} < eta(1 m/s)={eta_slow:.3f}")
    R_slow = m.absorption_rate(u_air=1.0)
    R_fast = m.absorption_rate(u_air=3.0)
    assert_true(R_fast > R_slow, f"More air throughput captures more total CO2")


def test_calciner_holds_setpoint():
    print("\n[Test 4] Thermostatic calciner holds ~900 C")
    m, _ = make_model()
    r = m.simulate(dt=3600.0, duration_s=3600.0 * 24 * 20)
    T_final_C = r["T_calciner_K"][-1] - 273.15
    assert_true(850.0 < T_final_C < 950.0, f"T_calciner={T_final_C:.1f} C near 900 C")


def test_mass_balance_steady_state():
    print("\n[Test 5] Mass balance: calcination rate -> carbonate feed rate at SS")
    m, _ = make_model()
    r = m.simulate(dt=3600.0, duration_s=3600.0 * 24 * 60)
    R_calc_ss = r["R_calcination_mol_s"][-1]
    R_feed = m.loop_eff * m.absorption_rate()
    rel = abs(R_calc_ss - R_feed) / R_feed
    assert_true(rel < 0.02, f"R_calc={R_calc_ss:.2f} ~ R_feed={R_feed:.2f} (rel={rel:.4f})")


def test_carbon_conservation():
    print("\n[Test 6] Carbon conservation: feed = released + inventory")
    m, _ = make_model()
    r = m.simulate(dt=3600.0, duration_s=3600.0 * 24 * 30)
    n_feed = m.loop_eff * r["n_CO2_absorbed_mol"][-1]
    n_released = r["co2_product_kg"][-1] / m.M_CO2
    n_inventory = r["n_CaCO3_mol"][-1]
    rel = abs(n_feed - (n_released + n_inventory)) / n_feed
    assert_true(rel < 1e-2, f"Carbon closes: rel imbalance={rel:.2e}")


def test_regeneration_energy_realistic():
    print("\n[Test 7] Regeneration is thermal-dominant, ~5-9 GJ/tCO2")
    m, _ = make_model()
    r = m.simulate(dt=3600.0, duration_s=3600.0 * 24 * 60)
    sec = r["sec_thermal_GJ_tCO2"][-1]
    assert_true(4.0 < sec < 10.0, f"SEC_thermal={sec:.2f} GJ/tCO2 (Keith 2018 range)")


def test_calcination_endothermic_temperature_drop():
    print("\n[Test 8] Calcination is endothermic: cold start needs net heating")
    m, _ = make_model()
    # start cold; burner must heat the calciner up toward setpoint
    r = m.simulate(dt=600.0, duration_s=3600.0 * 12, T_calc0=1000.0)
    assert_true(r["T_calciner_K"][-1] > 1000.0, "Calciner heats from cold start")
    # reaction enthalpy sign sanity
    assert_true(m.dH_calc > 0, f"dH_calcination={m.dH_calc} J/mol endothermic (>0)")


def test_arrhenius_rate_temperature_dependence():
    print("\n[Test 9] Calcination rate rises with temperature (Arrhenius)")
    m, _ = make_model()
    r_cold = m.calcination_rate(1e6, 1073.15)   # 800 C
    r_hot = m.calcination_rate(1e6, 1173.15)    # 900 C
    assert_true(r_hot > r_cold, f"k(900C) > k(800C): {r_hot:.2e} > {r_cold:.2e}")
    assert_true(m.calcination_rate(0.0, 1173.15) == 0.0, "No CaCO3 -> no calcination")


def test_predict_interface():
    print("\n[Test 10] ComponentModel predict() interface")
    _, cm = make_model()
    r = cm.predict({"dt": 3600.0, "duration_s": 3600.0 * 24})
    for key in ["t", "n_CO2_absorbed_mol", "n_CaCO3_mol", "T_calciner_K",
                "R_absorption_mol_s", "R_calcination_mol_s", "co2_captured_kg",
                "co2_product_kg", "sec_thermal_GJ_tCO2", "single_pass_capture"]:
        assert_true(key in r, f"Key '{key}' in output")
    assert_true(len(r["t"]) == len(r["T_calciner_K"]), "Arrays same length")
    info = cm.get_info()
    assert_true(info["component_id"] == "EC202", "get_info component_id == EC202")


def test_co2_monotone_accumulation():
    print("\n[Test 11] Cumulative absorbed CO2 is monotonically non-decreasing")
    m, _ = make_model()
    r = m.simulate(dt=3600.0, duration_s=3600.0 * 24 * 5)
    n = r["n_CO2_absorbed_mol"]
    assert_true(np.all(np.diff(n) >= -1e-6), "n_CO2_absorbed monotone non-decreasing")
    assert_true(n[-1] > 0.0, f"Captured {n[-1]:.0f} mol CO2 > 0")


def test_benchmark():
    print("\n[Test 12] Benchmark: 30-day sim")
    m, _ = make_model()
    t0 = time.perf_counter()
    m.simulate(dt=3600.0, duration_s=3600.0 * 24 * 30)
    elapsed = time.perf_counter() - t0
    print(f"  30-day simulation in {elapsed*1000:.1f} ms")
    assert_true(elapsed < 5.0, "Completes in < 5 s")


if __name__ == "__main__":
    tests = [
        test_single_pass_capture_fraction,
        test_capture_from_dilute_air,
        test_faster_air_lowers_single_pass_but_raises_throughput,
        test_calciner_holds_setpoint,
        test_mass_balance_steady_state,
        test_carbon_conservation,
        test_regeneration_energy_realistic,
        test_calcination_endothermic_temperature_drop,
        test_arrhenius_rate_temperature_dependence,
        test_predict_interface,
        test_co2_monotone_accumulation,
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
    print(f"EC202 DAC Liquid Solvent F2a -- Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    sys.exit(0 if failed == 0 else 1)
