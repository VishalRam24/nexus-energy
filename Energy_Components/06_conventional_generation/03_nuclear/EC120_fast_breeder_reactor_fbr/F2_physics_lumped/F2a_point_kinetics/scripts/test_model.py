"""
EC120 -- Fast Breeder Reactor (FBR) -- F2a Fast-Spectrum Point Kinetics
Test suite: fast-spectrum sanity, feedback stability, breeding ratio > 1,
energy conservation, edge cases, predict() interface, benchmark.
Custom assert harness (NO pytest). Run: python3 scripts/test_model.py
"""

import sys
import os
import time
import numpy as np
from scipy.integrate import trapezoid

sys.path.insert(0, os.path.dirname(__file__))
from model import FBRPointKineticsF2a
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


# --------------------------------------------------------------------------- #
def test_fast_spectrum_kinetics():
    print("\n[Test 1] Fast spectrum: short Lambda and small beta_eff")
    m, _ = make_model()
    # Fast reactor prompt-neutron lifetime is MUCH shorter than a thermal PWR (2e-5 s)
    assert_true(m.Lambda < 1e-5, f"Lambda={m.Lambda:.2e} s << thermal PWR (2e-5 s)")
    assert_true(m.beta_total < 0.0045,
                f"beta_eff={m.beta_total:.5f} (Pu-239 fast, < U-235 thermal 0.0065)")
    assert_true(len(m.beta) == 6 and len(m.lam) == 6, "Six delayed-neutron groups")


def test_feedback_stability():
    print("\n[Test 2] Net power reactivity coefficient is NEGATIVE (self-stabilising)")
    m, _ = make_model()
    pc = m.power_coefficient()
    assert_true(pc < 0, f"Net power coeff={pc:.3e} dk/k per K < 0 (Doppler+axial+void)")
    assert_true(m.alpha_D < 0, f"Doppler coeff {m.alpha_D:.2e} < 0")
    assert_true(m.alpha_void > 0, f"Sodium-void coeff {m.alpha_void:.2e} > 0 (fast-core characteristic)")


def test_breeding_ratio_gt_one():
    print("\n[Test 3] Breeding ratio > 1 (breeder converts U-238 -> Pu-239)")
    _, cm = make_model()
    r = cm.predict({"rho_ext": 0.0, "dt": 0.5, "duration_s": 20.0})
    assert_true(np.all(r["breeding_ratio"] > 1.0),
                f"BR in [{r['breeding_ratio'].min():.3f}, {r['breeding_ratio'].max():.3f}] > 1")
    # Net fissile bred must be POSITIVE (more bred than burned) for a breeder
    assert_true(r["net_fissile_bred_kg"][-1] > 0,
                f"net fissile bred={r['net_fissile_bred_kg'][-1]*1e3:.3f} g > 0")


def test_initial_equilibrium():
    print("\n[Test 4] Equilibrium IC: precursor and thermal balance")
    m, _ = make_model()
    x0 = m.initial_conditions()
    n0 = x0[0]
    C0 = x0[1:7]
    C_expected = m.beta * n0 / (m.lam * m.Lambda)
    assert_true(np.allclose(C0, C_expected, rtol=1e-10), "dC_i/dt = 0 at IC")
    T_f, T_Na = x0[7], x0[8]
    P0 = m.P_th
    Q_fc = m.hA_fc * (T_f - T_Na)
    assert_true(abs(Q_fc - P0) / P0 < 1e-6, f"Fuel->Na heat balance: {Q_fc:.3e} == {P0:.3e}")
    Q_out = m.W_cp * (T_Na - m.T_in)
    assert_true(abs(Q_out - P0) / P0 < 1e-6, f"Coolant heat removal balance: {Q_out:.3e} == {P0:.3e}")


def test_zero_reactivity_steady():
    print("\n[Test 5] Zero external reactivity -> stays at rated steady state")
    _, cm = make_model()
    r = cm.predict({"rho_ext": 0.0, "dt": 0.1, "duration_s": 30.0})
    assert_true(np.all(np.abs(r["n"] - 1.0) < 1e-3),
                f"n stable: [{r['n'].min():.6f}, {r['n'].max():.6f}]")
    assert_true((r["T_f"].max() - r["T_f"].min()) < 0.5, "Fuel T stable")
    assert_true(abs(r["P_thermal_W"][-1] - m_P_rated()) / m_P_rated() < 1e-3,
                "Thermal power == rated at steady state")


def m_P_rated():
    m, _ = make_model()
    return m.P_th


def test_negative_feedback_limits_excursion():
    print("\n[Test 6] Delayed-supercritical step bounded by negative feedback")
    _, cm = make_model()
    # rho_step = 200 pcm < beta_eff (~330 pcm): delayed supercritical, must stay bounded
    r = cm.predict_step({"rho_step": 0.002, "dt": 0.02, "duration_s": 60.0})
    assert_true(np.isfinite(r["n"]).all() and r["n"].max() < 50.0,
                f"n_max={r['n'].max():.2f} bounded (negative feedback, no runaway)")
    # Fuel temperature rises -> negative reactivity brings net rho back toward 0
    assert_true(r["T_f"][-1] > r["T_f"][0], "Fuel T rises with power")
    assert_true(r["rho"][-1] < r["rho_ext"][-1],
                "Feedback makes total rho < external rho (negative feedback acting)")


def test_directional_response():
    print("\n[Test 7] Power follows reactivity sign")
    _, cm = make_model()
    rp = cm.predict_step({"rho_step": 0.0005, "dt": 0.05, "duration_s": 20.0})
    rn = cm.predict_step({"rho_step": -0.0005, "dt": 0.05, "duration_s": 20.0})
    i = np.searchsorted(rp["t"], 3.0)
    assert_true(rp["n"][i] > 1.0, f"+step: n={rp['n'][i]:.4f} > 1")
    assert_true(rn["n"][i] < 1.0, f"-step: n={rn['n'][i]:.4f} < 1")


def test_energy_conservation():
    print("\n[Test 8] Energy conservation: stored + removed == generated")
    m, cm = make_model()
    r = cm.predict_step({"rho_step": 0.001, "dt": 0.05, "duration_s": 40.0})
    t = r["t"]
    # Generated thermal energy
    E_gen = trapezoid(r["P_thermal_W"], t)
    # Heat removed by coolant: W_cp*(T_Na - T_in)
    Q_removed = m.W_cp * (r["T_Na"] - m.T_in)
    E_removed = trapezoid(Q_removed, t)
    # Energy stored in fuel + sodium nodes (relative to start)
    dE_fuel = m.mf_cpf * (r["T_f"][-1] - r["T_f"][0])
    dE_Na = m.mNa_cpNa * (r["T_Na"][-1] - r["T_Na"][0])
    E_stored = dE_fuel + dE_Na
    residual = E_gen - E_removed - E_stored
    rel = abs(residual) / E_gen
    print(f"  E_gen={E_gen:.4e} J, E_removed={E_removed:.4e} J, "
          f"E_stored={E_stored:.4e} J, residual={rel*100:.3f}%")
    assert_true(rel < 1e-3, f"Energy balance closes to {rel*100:.4f}% (< 0.1%)")


def test_sodium_void_effect():
    print("\n[Test 9] Sodium-void reactivity adds positive feedback when coolant heats")
    m, _ = make_model()
    rho_cold = m.reactivity(m.T_f0, m.T_Na0, 0.0)
    rho_hot_Na = m.reactivity(m.T_f0, m.T_Na0 + 50.0, 0.0)
    assert_true(rho_hot_Na > rho_cold,
                f"Hotter sodium -> more positive void reactivity ({rho_hot_Na:.2e} > {rho_cold:.2e})")
    # But fuel heating (Doppler+axial) is strongly negative
    rho_hot_fuel = m.reactivity(m.T_f0 + 50.0, m.T_Na0, 0.0)
    assert_true(rho_hot_fuel < rho_cold, "Hotter fuel -> negative Doppler+axial reactivity")


def test_predict_interface():
    print("\n[Test 10] ComponentModel predict()/get_info() interface")
    _, cm = make_model()
    r = cm.predict({"rho_ext": 0.0, "dt": 0.5, "duration_s": 5.0})
    for key in ["t", "n", "C", "T_f", "T_Na", "P_thermal_W", "P_elec_W",
                "rho", "breeding_ratio", "net_fissile_bred_kg"]:
        assert_true(key in r, f"Output key '{key}' present")
    assert_true(r["C"].shape[0] == 6, "Precursor array has 6 groups")
    assert_true(len(r["t"]) == len(r["n"]), "Time-series arrays aligned")
    info = cm.get_info()
    assert_true(info["ec_id"] == "EC120" and info["fidelity"] == "F2a",
                "get_info() reports EC120 / F2a")


def test_bdf_radau_agree():
    print("\n[Test 11] BDF and Radau stiff solvers agree")
    _, cm = make_model()
    r_radau = cm.predict({"rho_ext": 0.0005, "dt": 0.1, "duration_s": 20.0, "method": "Radau"})
    r_bdf = cm.predict({"rho_ext": 0.0005, "dt": 0.1, "duration_s": 20.0, "method": "BDF"})
    rel = abs(r_radau["n"][-1] - r_bdf["n"][-1]) / r_radau["n"][-1]
    assert_true(rel < 5e-3, f"n_final agrees: Radau={r_radau['n'][-1]:.5f}, BDF={r_bdf['n'][-1]:.5f}")


def test_benchmark():
    print("\n[Test 12] Benchmark: 60s stiff transient < 5 s wall time")
    _, cm = make_model()
    t0 = time.perf_counter()
    cm.predict_step({"rho_step": 0.001, "dt": 0.05, "duration_s": 60.0})
    elapsed = time.perf_counter() - t0
    print(f"  60s transient in {elapsed*1000:.1f} ms")
    assert_true(elapsed < 5.0, "Completes in < 5 s")


if __name__ == "__main__":
    tests = [
        test_fast_spectrum_kinetics,
        test_feedback_stability,
        test_breeding_ratio_gt_one,
        test_initial_equilibrium,
        test_zero_reactivity_steady,
        test_negative_feedback_limits_excursion,
        test_directional_response,
        test_energy_conservation,
        test_sodium_void_effect,
        test_predict_interface,
        test_bdf_radau_agree,
        test_benchmark,
    ]
    passed = failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"  ERROR: {e}")

    print(f"\n{'='*60}")
    print(f"EC120 FBR F2a -- Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    sys.exit(0 if failed == 0 else 1)
