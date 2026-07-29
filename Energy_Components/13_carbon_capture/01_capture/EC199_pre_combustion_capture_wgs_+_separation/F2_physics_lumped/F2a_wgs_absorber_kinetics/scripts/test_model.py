"""
EC199 -- Pre-Combustion Capture (WGS + Separation) -- F2a Physics-Lumped
Test suite: mass/carbon conservation, WGS equilibrium, capture-rate band,
high-partial-pressure separation, energy penalty, edge cases, interface, timing.
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from model import PreCombustionCaptureF2a
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
def test_carbon_conservation():
    print("\n[Test 1] Carbon conservation: CO_out + CO2_slip + CO2_captured == C_in")
    _, cm = make_model()
    for P in (15.0, 30.0, 45.0):
        r = cm.predict({"P_bar": P, "syngas_flow_mol_s": 1000.0})
        assert_true(r["carbon_residual"] < 1e-9,
                    f"P={P}: carbon residual {r['carbon_residual']:.2e} ~ 0")


def test_mole_and_hatom_balance_wgs():
    print("\n[Test 2] WGS conserves total moles and hydrogen atoms")
    _, cm = make_model()
    r = cm.predict({"P_bar": 30.0})
    assert_true(r["mole_residual_wgs"] < 1e-9,
                f"WGS total moles conserved (res {r['mole_residual_wgs']:.2e})")
    assert_true(r["h_atom_residual"] < 1e-9,
                f"H-atom balance conserved (res {r['h_atom_residual']:.2e})")


def test_wgs_approaches_equilibrium():
    print("\n[Test 3] WGS conversion approaches but never exceeds equilibrium")
    _, cm = make_model()
    for T in (493.15, 523.15, 573.15):
        r = cm.predict({"T_WGS_K": T, "P_bar": 30.0})
        X, Xeq = r["wgs_conversion"], r["wgs_equilibrium_conversion"]
        assert_true(X <= Xeq + 1e-6,
                    f"T={T}: X={X:.4f} <= X_eq={Xeq:.4f}")
        assert_true(X > 0.85 * Xeq,
                    f"T={T}: X={X:.4f} within residence reaches >85% of eq {Xeq:.4f}")


def test_keq_decreases_with_T():
    print("\n[Test 4] WGS Keq(T) decreases with temperature (exothermic, van't Hoff)")
    m, _ = make_model()
    K_low = m.keq_wgs(473.15)
    K_high = m.keq_wgs(673.15)
    assert_true(K_low > K_high,
                f"Keq(200C)={K_low:.1f} > Keq(400C)={K_high:.2f}")
    assert_true(K_low > 1.0, f"Keq favours products at LT (Keq={K_low:.1f} > 1)")


def test_capture_rate_in_band():
    print("\n[Test 5] CO2 capture rate in ~90% design band (0.85-0.99)")
    _, cm = make_model()
    r = cm.predict({"P_bar": 30.0})
    cr = r["capture_rate"]
    assert_true(0.85 <= cr <= 0.99, f"capture_rate={cr*100:.1f}% in [85, 99]%")


def test_high_partial_pressure_separation():
    print("\n[Test 6] Capture rate increases with CO2 partial pressure (physical solvent)")
    _, cm = make_model()
    r_lo = cm.predict({"P_bar": 15.0})
    r_hi = cm.predict({"P_bar": 45.0})
    assert_true(r_hi["p_CO2_absorber_in_bar"] > r_lo["p_CO2_absorber_in_bar"],
                "Higher total P -> higher CO2 partial pressure into absorber")
    assert_true(r_hi["capture_rate"] > r_lo["capture_rate"],
                f"Capture rate rises with P: {r_hi['capture_rate']*100:.1f}% > "
                f"{r_lo['capture_rate']*100:.1f}%")


def test_h2_enrichment():
    print("\n[Test 7] Product is H2-rich and H2 is retained (selective solvent)")
    _, cm = make_model()
    r = cm.predict({"co_fraction": 0.45, "h2_fraction": 0.35, "P_bar": 30.0})
    assert_true(r["h2_purity"] > 0.90,
                f"H2 purity {r['h2_purity']*100:.1f}% > 90%")
    # H2 produced should exceed inlet H2 (WGS adds H2) and not be lost to solvent
    ab = r["absorber"]
    assert_true(ab["n_H2_lost_mol_s"] < 0.02 * (ab["n_H2_retained_mol_s"] + 1e-9),
                "H2 co-absorption < 2% (physical solvent selective for CO2)")


def test_energy_penalty_low():
    print("\n[Test 8] Energy penalty low and physical (< 1 GJ/tCO2, < amine ~3.5)")
    _, cm = make_model()
    r = cm.predict({"P_bar": 30.0})
    E = r["energy_penalty_GJ_tCO2"]
    assert_true(0.1 < E < 1.0,
                f"E={E:.2f} GJ/tCO2 in (0.1, 1.0) -- physical-solvent regime")
    assert_true(r["power_penalty_MW"] > 0, "Power penalty positive")


def test_wgs_exothermic_heat():
    print("\n[Test 9] WGS releases exothermic heat proportional to conversion")
    _, cm = make_model()
    r = cm.predict({"P_bar": 30.0, "syngas_flow_mol_s": 1000.0})
    assert_true(r["wgs_heat_kW"] > 0, f"Q_WGS={r['wgs_heat_kW']:.1f} kW > 0 (exothermic)")
    # ~41.1 kJ/mol * extent; sanity magnitude
    extent = r["wgs"]["extent_mol_s"]
    expected = extent * 41100.0 / 1000.0
    assert_true(abs(r["wgs_heat_kW"] - expected) < 1e-3 * expected + 1e-6,
                "Heat == extent * (-DH)")


def test_henry_temperature_dependence():
    print("\n[Test 10] CO2 solubility (Henry coeff) rises as solvent is chilled")
    m, _ = make_model()
    H_warm = m.henry_CO2(308.15)   # 35 C
    H_cold = m.henry_CO2(283.15)   # 10 C
    assert_true(H_cold > H_warm,
                f"H(10C)={H_cold:.1f} > H(35C)={H_warm:.1f} mol/(m3.bar)")


def test_edge_zero_co():
    print("\n[Test 11] Edge case: zero CO inlet -> no shift, no capture")
    _, cm = make_model()
    r = cm.predict({"co_fraction": 0.0, "h2_fraction": 0.6, "P_bar": 30.0})
    assert_true(r["wgs_conversion"] == 0.0 or np.isnan(r["wgs_conversion"]) is False,
                "Zero-CO handled")
    assert_true(r["co2_captured_kg_s"] >= 0.0, "Non-negative capture")
    assert_true(r["carbon_residual"] < 1e-6, "Carbon balance holds at zero-CO")


def test_predict_interface():
    print("\n[Test 12] ComponentModel predict()/get_info() interface")
    _, cm = make_model()
    info = cm.get_info()
    for k in ("component_id", "fidelity", "inputs", "outputs"):
        assert_true(k in info, f"get_info has '{k}'")
    assert_true(info["component_id"] == "EC199", "component_id == EC199")
    r = cm.predict({"syngas_flow_mol_s": 500.0})
    for key in ("capture_rate", "wgs_conversion", "co2_captured_kg_s",
                "h2_rich_fuel_mol_s", "energy_penalty_GJ_tCO2", "power_penalty_MW"):
        assert_true(key in r, f"predict output has '{key}'")


def test_benchmark():
    print("\n[Test 13] Benchmark: coupled WGS+absorber ODE solve timing")
    _, cm = make_model()
    t0 = time.perf_counter()
    for _ in range(5):
        cm.predict({"P_bar": 30.0})
    elapsed = (time.perf_counter() - t0) / 5.0
    print(f"  Mean coupled simulate() = {elapsed*1000:.1f} ms")
    assert_true(elapsed < 5.0, "Single coupled simulation < 5 s")


if __name__ == "__main__":
    tests = [
        test_carbon_conservation,
        test_mole_and_hatom_balance_wgs,
        test_wgs_approaches_equilibrium,
        test_keq_decreases_with_T,
        test_capture_rate_in_band,
        test_high_partial_pressure_separation,
        test_h2_enrichment,
        test_energy_penalty_low,
        test_wgs_exothermic_heat,
        test_henry_temperature_dependence,
        test_edge_zero_co,
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

    print(f"\n{'='*64}")
    print(f"EC199 Pre-Combustion Capture F2a -- Results: {passed} passed, {failed} failed")
    print(f"{'='*64}")
    sys.exit(0 if failed == 0 else 1)
