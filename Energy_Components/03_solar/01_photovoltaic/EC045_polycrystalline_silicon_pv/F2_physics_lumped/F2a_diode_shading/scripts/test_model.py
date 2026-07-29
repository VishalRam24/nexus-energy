"""
EC045 -- Poly-Si PV -- F2a Physics-Lumped
Test suite: single-diode physics sanity, Lambert-W I-V, MPP, thermal ODE,
partial shading, predict() interface, benchmark. NO pytest.
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from model import PolySiPVF2a
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
def test_zero_irradiance_zero_power():
    print("\n[Test 1] P = 0 at zero irradiance (night)")
    m, _ = make_model()
    r = m.mpp(0.0, 25.0)
    assert_true(r["p_mp"] == 0.0, f"P_mp={r['p_mp']} at G=0")
    assert_true(r["v_oc"] == 0.0 and r["i_sc"] == 0.0, "V_oc=I_sc=0 at G=0")
    assert_true(m.efficiency(0.0, 25.0) == 0.0, "efficiency=0 at G=0")


def test_efficiency_range():
    print("\n[Test 2] 0 < efficiency < 0.22 over operating range")
    m, _ = make_model()
    for G in [200.0, 600.0, 1000.0, 1100.0]:
        for Tc in [10.0, 25.0, 60.0]:
            eff = m.efficiency(G, Tc)
            assert_true(0.0 < eff < 0.22, f"eff(G={G},T={Tc})={eff*100:.2f}% in (0,22)%")


def test_isc_proportional_to_G():
    print("\n[Test 3] I_sc proportional to irradiance G")
    m, _ = make_model()
    isc1 = m.mpp(500.0, 25.0)["i_sc"]
    isc2 = m.mpp(1000.0, 25.0)["i_sc"]
    ratio = isc2 / isc1
    assert_true(abs(ratio - 2.0) < 0.05, f"I_sc(1000)/I_sc(500)={ratio:.3f} ~ 2.0")
    # near-linearity at a third point
    isc3 = m.mpp(250.0, 25.0)["i_sc"]
    assert_true(abs(isc1 / isc3 - 2.0) < 0.05, f"I_sc(500)/I_sc(250)={isc1/isc3:.3f} ~ 2.0")


def test_pv_curve_monotone_to_mpp():
    print("\n[Test 4] P-V curve rises monotonically to MPP")
    m, _ = make_model()
    V, I, P = m.iv_curve(1000.0, 25.0, n_points=300)
    imax = int(np.argmax(P))
    rising = P[:imax + 1]
    diffs = np.diff(rising)
    assert_true(np.all(diffs >= -1e-6), "P-V monotone non-decreasing up to MPP")
    # and falls after MPP
    falling = P[imax:]
    assert_true(np.all(np.diff(falling) <= 1e-6), "P-V monotone non-increasing after MPP")


def test_mpp_consistent_with_curve():
    print("\n[Test 5] Golden-section MPP matches swept-curve peak")
    m, _ = make_model()
    V, I, P = m.iv_curve(1000.0, 25.0, n_points=2000)
    p_curve = float(np.max(P))
    p_mpp = m.mpp(1000.0, 25.0)["p_mp"]
    assert_true(abs(p_mpp - p_curve) / p_curve < 0.01, f"MPP {p_mpp:.2f} vs curve {p_curve:.2f}")


def test_lambertw_matches_implicit():
    print("\n[Test 6] Lambert-W I(V) satisfies the single-diode equation")
    m, _ = make_model()
    I_L, I_o, R_sh, a = m.calc_params(1000.0, 25.0)
    for V in [5.0, 15.0, 25.0]:
        I = float(m.i_from_v(V, I_L, I_o, R_sh, a))
        resid = (I_L - I_o * (np.exp((V + I * m.R_s) / a) - 1.0)
                 - (V + I * m.R_s) / R_sh - I)
        assert_true(abs(resid) < 1e-6, f"residual at V={V}: {resid:.2e}")


def test_temperature_lowers_power():
    print("\n[Test 7] Higher cell temperature lowers P_mp (V_oc drops)")
    m, _ = make_model()
    p_cold = m.mpp(1000.0, 15.0)["p_mp"]
    p_hot = m.mpp(1000.0, 65.0)["p_mp"]
    voc_cold = m.mpp(1000.0, 15.0)["v_oc"]
    voc_hot = m.mpp(1000.0, 65.0)["v_oc"]
    assert_true(p_hot < p_cold, f"P_mp(65C)={p_hot:.1f} < P_mp(15C)={p_cold:.1f}")
    assert_true(voc_hot < voc_cold, f"V_oc(65C)={voc_hot:.2f} < V_oc(15C)={voc_cold:.2f}")


def test_thermal_ode_heats_and_settles():
    print("\n[Test 8] Thermal ODE: cold module heats above ambient and settles")
    m, _ = make_model()
    r = m.simulate(900.0, T_amb=25.0, wind=1.0, dt=30.0, duration_s=3600.0, T_cell0=25.0)
    assert_true(r["T_cell"][-1] > 25.0, f"T_cell rose to {r['T_cell'][-1]:.1f} C > 25")
    assert_true(r["T_cell"][-1] < 90.0, f"T_cell {r['T_cell'][-1]:.1f} C stays < 90")
    dT = abs(r["T_cell"][-1] - r["T_cell"][-2])
    assert_true(dT < 0.2, f"Near steady state: dT={dT:.4f} C/step")


def test_thermal_noct_consistency():
    print("\n[Test 9] Steady cell temp near NOCT at NOCT conditions")
    m, _ = make_model()
    # Faiman model with default U should give an elevated cell temp at 800 W/m2
    Tc = m._steady_cell_temp(800.0, 20.0, 1.0)
    assert_true(40.0 < Tc < 70.0, f"T_cell at NOCT-like conditions={Tc:.1f} C (warm)")
    assert_true(m._steady_cell_temp(0.0, 20.0, 1.0) == 20.0, "T_cell = T_amb at night")


def test_partial_shading_reduces_power():
    print("\n[Test 10] Partial shading of one substring reduces module power")
    m, _ = make_model()
    p_full = m.mpp(1000.0, 25.0)["p_mp"]
    p_light = m.mpp_partial_shade(1000.0, 25.0, 0.2)["p_mp"]
    p_half = m.mpp_partial_shade(1000.0, 25.0, 0.5)["p_mp"]
    p_dark = m.mpp_partial_shade(1000.0, 25.0, 1.0)["p_mp"]
    assert_true(p_half < p_full, f"50% shade P={p_half:.1f} < full P={p_full:.1f}")
    # heavier shade never produces MORE power than lighter shade (monotone)
    assert_true(p_dark <= p_half + 1e-6, f"100% shade P={p_dark:.1f} <= 50% shade P={p_half:.1f}")
    assert_true(p_light <= p_full + 1e-6 and p_dark <= p_light + 1e-6,
                f"monotone: full {p_full:.1f} >= 20% {p_light:.1f} >= 100% {p_dark:.1f}")
    # one of 3 substrings fully shaded & bypassed -> ~2/3 of power survives
    frac = p_dark / p_full
    assert_true(0.45 < frac < 0.80, f"bypassed-substring fraction={frac:.2f} (~2/3)")


def test_predict_interface():
    print("\n[Test 11] ComponentModel predict() steady + dynamic interface")
    _, cm = make_model()
    r = cm.predict({"irradiance_W_m2": 1000.0, "cell_temp_C": 25.0})
    for key in ["v_mp", "i_mp", "p_mp", "v_oc", "i_sc", "efficiency", "cell_temp_C"]:
        assert_true(key in r, f"steady key '{key}' present")
    assert_true(150.0 < r["p_mp"] < 330.0, f"STC P_mp={r['p_mp']:.1f} W in datasheet band")
    rd = cm.predict({"mode": "dynamic", "irradiance_W_m2": 800.0,
                     "T_ambient_C": 25.0, "dt": 60.0, "duration_s": 600.0})
    assert_true(len(rd["t"]) == len(rd["T_cell"]) == len(rd["p_mp"]), "arrays aligned")


def test_benchmark():
    print("\n[Test 12] Benchmark: 2-hour dynamic sim at dt=60 s")
    m, _ = make_model()
    t0 = time.perf_counter()
    m.simulate(900.0, T_amb=30.0, wind=2.0, dt=60.0, duration_s=7200.0)
    elapsed = time.perf_counter() - t0
    print(f"  2-hour simulation in {elapsed*1000:.1f} ms")
    assert_true(elapsed < 5.0, "Completes in < 5 s")


if __name__ == "__main__":
    tests = [
        test_zero_irradiance_zero_power,
        test_efficiency_range,
        test_isc_proportional_to_G,
        test_pv_curve_monotone_to_mpp,
        test_mpp_consistent_with_curve,
        test_lambertw_matches_implicit,
        test_temperature_lowers_power,
        test_thermal_ode_heats_and_settles,
        test_thermal_noct_consistency,
        test_partial_shading_reduces_power,
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
    print(f"EC045 Poly-Si PV F2a -- Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    sys.exit(0 if failed == 0 else 1)
