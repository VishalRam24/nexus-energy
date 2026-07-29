"""
EC222 — Betavoltaic Cell — F2a Single-Diode I-V with Beta-Flux Photocurrent and Decay ODE
Test suite: physics sanity (energy consistency, decay, diode limits), edge cases,
predict() interface, benchmark timing. Custom harness — NO pytest.
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from model import BetavoltaicF2a, MeV_to_J, q_e, ln2
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
def test_decay_law():
    print("\n[Test 1] Decay ODE reproduces exp(-lambda t) and half-life")
    m, _ = make_model()
    # After one half-life, activity halves
    A_half = m.activity(m.t_half)
    assert_true(abs(A_half / m.A0 - 0.5) < 1e-9, f"A(t_half)/A0 = {A_half/m.A0:.6f} ~ 0.5")
    # ODE integration matches closed-form decay
    r = m.simulate((0.0, m.t_half), n_eval=20, with_iv=False)
    assert_true(abs(r["fraction_remaining"][-1] - 0.5) < 1e-4,
                f"ODE fraction at t_half = {r['fraction_remaining'][-1]:.5f} ~ 0.5")
    assert_true(np.all(np.diff(r["activity_Bq"]) <= 1e-3),
                "Activity monotonically non-increasing")


def test_energy_consistency():
    print("\n[Test 2] Energy consistency: P_out <= absorbed beta power")
    m, _ = make_model()
    for t in [0.0, 10.0, 50.0]:
        iv = m.iv_curve(t)
        A = m.activity(t)
        P_abs = m.beta_power_absorbed(A)
        assert_true(iv["P_mpp_W"] <= P_abs + 1e-30,
                    f"t={t}: P_mpp={iv['P_mpp_W']:.3e} W <= P_abs={P_abs:.3e} W")


def test_ehp_pair_energy():
    print("\n[Test 3] Pair-creation energy (Klein) and EHP count are physical")
    m, _ = make_model()
    E_pair = m.pair_creation_energy_eV()
    assert_true(E_pair > m.E_gap, f"E_pair={E_pair:.2f} eV > E_gap={m.E_gap:.2f} eV")
    ehp = m.ehp_per_beta()
    E_beta_eV = m.E_beta * 1e6
    assert_true(abs(ehp - E_beta_eV / E_pair) < 1e-9, f"EHP/beta = {ehp:.3f} = E_beta/E_pair")
    # Generated current must be consistent with q * EHP rate
    I_L = m.beta_current(0.0)
    expected = q_e * m.A0 * m.eta_cap * m.eta_coll * ehp
    assert_true(abs(I_L - expected) / expected < 1e-12, "I_L = q*A*eta_cap*eta_coll*EHP")


def test_diode_isc_voc():
    print("\n[Test 4] Single-diode limits: 0 < Isc <= I_L and Voc > 0")
    m, _ = make_model()
    iv = m.iv_curve(0.0)
    I_L = iv["I_L_A"]
    assert_true(0 < iv["Isc_A"] <= I_L * 1.0001, f"Isc={iv['Isc_A']:.3e} A in (0, I_L]")
    assert_true(iv["Voc_V"] > 0, f"Voc={iv['Voc_V']:.4f} V > 0")
    # Isc should be very close to I_L for high Rsh
    assert_true(abs(iv["Isc_A"] - I_L) / I_L < 0.05, "Isc ~ I_L (good shunt)")


def test_fill_factor_range():
    print("\n[Test 5] Fill factor in physical (0,1) and MPP bracketed")
    m, _ = make_model()
    iv = m.iv_curve(0.0)
    assert_true(0.0 < iv["FF"] < 1.0, f"FF={iv['FF']:.3f} in (0,1)")
    assert_true(0.0 < iv["V_mpp_V"] < iv["Voc_V"], "0 < V_mpp < Voc")
    assert_true(0.0 < iv["I_mpp_A"] < iv["Isc_A"] * 1.0001, "0 < I_mpp < Isc")
    assert_true(iv["P_mpp_W"] <= iv["Isc_A"] * iv["Voc_V"] + 1e-30, "P_mpp <= Isc*Voc")


def test_iv_curve_shape():
    print("\n[Test 6] I-V curve: I decreases with V, P has interior maximum")
    m, _ = make_model()
    iv = m.iv_curve(0.0, n_points=120)
    # Current non-increasing along V
    assert_true(np.all(np.diff(iv["I"]) <= 1e-15 + 1e-3 * iv["Isc_A"]),
                "I(V) non-increasing")
    idx = int(np.argmax(iv["P"]))
    assert_true(0 < idx < len(iv["P"]) - 1, f"MPP interior (idx={idx})")


def test_efficiency_range():
    print("\n[Test 7] Conversion efficiency in (0,1) and low (few %)")
    m, _ = make_model()
    r = m.simulate((0.0, 20.0), n_eval=10)
    for e in r["eta"]:
        assert_true(0.0 < e < 1.0, f"eta={e:.4f} in (0,1)")
    assert_true(r["eta"][0] < 0.30, f"eta(0)={r['eta'][0]*100:.2f}% is low (betavoltaic)")


def test_low_power_density():
    print("\n[Test 8] Very low absolute output power (microwatt scale)")
    m, _ = make_model()
    iv = m.iv_curve(0.0)
    assert_true(iv["P_mpp_W"] < 1e-3, f"P_mpp={iv['P_mpp_W']*1e6:.4f} uW < 1 mW")
    assert_true(iv["P_mpp_W"] > 0.0, "P_mpp > 0")


def test_power_decays_with_time():
    print("\n[Test 9] Output power decays over isotope life (P ~ exp(-t/tau))")
    m, _ = make_model()
    r = m.simulate((0.0, m.t_half), n_eval=15)
    assert_true(r["P_out_uW"][-1] < r["P_out_uW"][0],
                f"P_out 0->t_half: {r['P_out_uW'][0]:.4f} -> {r['P_out_uW'][-1]:.4f} uW")
    # Power tracks current ~ activity; at half-life roughly halves (within FF/Voc drift)
    ratio = r["P_out_uW"][-1] / r["P_out_uW"][0]
    assert_true(0.3 < ratio < 0.65, f"P ratio at t_half = {ratio:.3f} (~0.5)")


def test_thermal_near_ambient():
    print("\n[Test 10] Thermal ODE: self-heating negligible (T ~ ambient)")
    m, _ = make_model()
    r = m.simulate((0.0, 1.0), n_eval=10, with_iv=False)
    dT = abs(r["temperature_K"][-1] - m.T_amb)
    assert_true(dT < 1.0, f"|T - T_amb| = {dT:.2e} K << 1 K (tiny deposited power)")


def test_temperature_voc_effect():
    print("\n[Test 11] Higher T lowers Voc (Shockley I0(T) rises)")
    m, _ = make_model()
    iv_cold = m.iv_curve(0.0, T_K=270.0)
    iv_hot = m.iv_curve(0.0, T_K=350.0)
    assert_true(iv_hot["Voc_V"] < iv_cold["Voc_V"],
                f"Voc(350K)={iv_hot['Voc_V']:.3f} < Voc(270K)={iv_cold['Voc_V']:.3f}")
    assert_true(iv_hot["I0_A"] > iv_cold["I0_A"], "I0 rises with T")


def test_predict_interface():
    print("\n[Test 12] ComponentModel predict() interface")
    _, cm = make_model()
    r = cm.predict({"t_years": 30.0, "n_eval": 12})
    for key in ["t_years", "activity_Bq", "fraction_remaining", "P_beta_absorbed_W",
                "temperature_K", "Isc_uA", "Voc_V", "FF", "P_out_uW", "eta",
                "snapshot_t0"]:
        assert_true(key in r, f"Key '{key}' in output")
    assert_true(len(r["t_years"]) == len(r["P_out_uW"]) == 12, "Arrays length n_eval")
    info = cm.get_info()
    assert_true(info["component_id"] == "EC222" and info["version"] == "1.0.0",
                "get_info metadata correct")


def test_benchmark():
    print("\n[Test 13] Benchmark: 50yr life sim with per-sample MPP")
    m, _ = make_model()
    t0 = time.perf_counter()
    m.simulate((0.0, 50.0), n_eval=50, with_iv=True)
    elapsed = time.perf_counter() - t0
    print(f"  50-year sim (50 MPP solves) in {elapsed*1000:.1f} ms")
    assert_true(elapsed < 5.0, "Completes in < 5 s")


if __name__ == "__main__":
    tests = [
        test_decay_law,
        test_energy_consistency,
        test_ehp_pair_energy,
        test_diode_isc_voc,
        test_fill_factor_range,
        test_iv_curve_shape,
        test_efficiency_range,
        test_low_power_density,
        test_power_decays_with_time,
        test_thermal_near_ambient,
        test_temperature_voc_effect,
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
    print(f"EC222 Betavoltaic F2a — Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    sys.exit(0 if failed == 0 else 1)
