"""
EC148 -- Bioethanol Fermentation -- F2a Monod + Luong Inhibition
Test suite: kinetics sanity, mass conservation, inhibition, thermal, interface.
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from model import BioethanolFermentationF2a
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
def test_monod_saturation():
    print("\n[Test 1] Monod growth: saturates with S, zero at S=0")
    m, _ = make_model()
    mu0 = m.specific_growth_rate(0.0, 0.0, m.T_opt)
    mu_lo = m.specific_growth_rate(1.0, 0.0, m.T_opt)
    mu_hi = m.specific_growth_rate(200.0, 0.0, m.T_opt)
    assert_true(abs(mu0) < 1e-12, f"mu(S=0)={mu0:.3e} == 0")
    assert_true(mu_hi > mu_lo, f"mu saturates: mu(200)={mu_hi:.4f} > mu(1)={mu_lo:.4f}")
    assert_true(mu_hi <= m.mu_max + 1e-9, f"mu={mu_hi:.4f} <= mu_max={m.mu_max}")


def test_product_inhibition():
    print("\n[Test 2] Luong inhibition: mu decreases with ethanol, ->0 at P*")
    m, _ = make_model()
    mu_a = m.specific_growth_rate(100.0, 0.0, m.T_opt)
    mu_b = m.specific_growth_rate(100.0, 50.0, m.T_opt)
    mu_c = m.specific_growth_rate(100.0, m.P_star, m.T_opt)
    mu_d = m.specific_growth_rate(100.0, m.P_star + 20.0, m.T_opt)
    assert_true(mu_b < mu_a, f"mu(P=50)={mu_b:.4f} < mu(P=0)={mu_a:.4f}")
    assert_true(abs(mu_c) < 1e-12, f"mu(P=P*)={mu_c:.3e} == 0 (growth ceases)")
    assert_true(abs(mu_d) < 1e-12, f"mu(P>P*)={mu_d:.3e} == 0 (no negative growth)")


def test_temperature_factor():
    print("\n[Test 3] Temperature factor peaks at T_opt, in [0,1]")
    m, _ = make_model()
    f_opt = m.temperature_factor(m.T_opt)
    f_cold = m.temperature_factor(m.T_opt - 15.0)
    f_hot = m.temperature_factor(m.T_opt + 15.0)
    assert_true(abs(f_opt - 1.0) < 1e-9, f"f(T_opt)={f_opt:.4f} == 1")
    assert_true(0.0 <= f_cold < f_opt, f"f(cold)={f_cold:.4f} < 1")
    assert_true(0.0 <= f_hot < f_opt, f"f(hot)={f_hot:.4f} < 1")


def test_mass_conservation():
    print("\n[Test 4] Mass balance: consumed C >= produced (ethanol+biomass)")
    m, _ = make_model()
    r = m.simulate(duration_h=48.0, dt=0.5)
    consumed = r["glucose_consumed_g_L"]
    produced = r["ethanol_final_g_L"] + (r["biomass_final_g_L"] - m.X0)
    assert_true(consumed > 0, f"glucose consumed={consumed:.2f} g/L > 0")
    assert_true(produced <= consumed + 1e-6,
                f"products({produced:.2f}) <= consumed({consumed:.2f}) g/L")


def test_yield_below_theoretical():
    print("\n[Test 5] Ethanol yield <= theoretical 0.511 g/g glucose")
    m, _ = make_model()
    r = m.simulate(duration_h=60.0, dt=0.5)
    Y = r["ethanol_yield_g_g"]
    assert_true(0 < Y <= m.Yps_th + 1e-6, f"yield={Y:.4f} g/g <= 0.511")
    assert_true(r["ferment_efficiency"] <= 1.0 + 1e-6,
                f"ferment efficiency={r['ferment_efficiency']:.3f} <= 1")


def test_substrate_nonnegative():
    print("\n[Test 6] Glucose never negative, monotone non-increasing (batch)")
    m, _ = make_model()
    r = m.simulate(duration_h=72.0, dt=0.5)
    assert_true(np.all(r["glucose"] >= -1e-6), "glucose >= 0 throughout")
    dS = np.diff(r["glucose"])
    assert_true(np.all(dS <= 1e-6), "glucose monotone non-increasing (no feed)")


def test_ethanol_monotone_increasing():
    print("\n[Test 7] Ethanol monotone non-decreasing in batch")
    m, _ = make_model()
    r = m.simulate(duration_h=48.0, dt=0.5)
    dP = np.diff(r["ethanol"])
    assert_true(np.all(dP >= -1e-6), "ethanol monotone non-decreasing")
    assert_true(r["ethanol_final_g_L"] > 50.0,
                f"meaningful ethanol made: {r['ethanol_final_g_L']:.1f} g/L")


def test_thermal_exotherm_and_cooling():
    print("\n[Test 8] Exothermic: T rises during active fermentation, cooling bounds it")
    m, _ = make_model()
    r = m.simulate(T0=m.T_opt, duration_h=48.0, dt=0.5)
    T = r["temperature"]
    assert_true(np.max(T) > m.T_opt, f"T peaks above start: {np.max(T):.2f} > {m.T_opt:.2f} K")
    assert_true(np.max(T) < 318.15, f"cooling caps T: max={np.max(T):.2f} K < 45 C")


def test_inhibition_limits_conversion():
    print("\n[Test 9] High initial ethanol suppresses further fermentation")
    m, _ = make_model()
    r_lo = m.simulate(P0=0.0, duration_h=24.0, dt=0.5)
    r_hi = m.simulate(P0=90.0, duration_h=24.0, dt=0.5)
    made_lo = r_lo["ethanol_final_g_L"] - 0.0
    made_hi = r_hi["ethanol_final_g_L"] - 90.0
    assert_true(made_hi < made_lo,
                f"inhibited makes less: {made_hi:.2f} < {made_lo:.2f} g/L")


def test_predict_interface():
    print("\n[Test 10] ComponentModel predict() interface")
    _, cm = make_model()
    r = cm.predict({"duration_h": 10.0, "dt_h": 0.5})
    for key in ["t", "glucose", "biomass", "ethanol", "temperature", "mu",
                "ethanol_yield_g_g", "productivity_g_L_h", "ferment_efficiency"]:
        assert_true(key in r, f"Key '{key}' in output")
    assert_true(len(r["t"]) == len(r["ethanol"]), "Arrays same length")
    info = cm.get_info()
    assert_true(info["component_id"] == "EC148", "get_info component_id == EC148")


def test_benchmark():
    print("\n[Test 11] Benchmark: 48h batch sim at dt=0.1h")
    m, _ = make_model()
    t0 = time.perf_counter()
    m.simulate(duration_h=48.0, dt=0.1)
    elapsed = time.perf_counter() - t0
    print(f"  48h fermentation simulated in {elapsed*1000:.1f} ms")
    assert_true(elapsed < 5.0, "Completes in < 5 s")


if __name__ == "__main__":
    tests = [
        test_monod_saturation,
        test_product_inhibition,
        test_temperature_factor,
        test_mass_conservation,
        test_yield_below_theoretical,
        test_substrate_nonnegative,
        test_ethanol_monotone_increasing,
        test_thermal_exotherm_and_cooling,
        test_inhibition_limits_conversion,
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
    print(f"EC148 Bioethanol Fermentation F2a -- Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    sys.exit(0 if failed == 0 else 1)
