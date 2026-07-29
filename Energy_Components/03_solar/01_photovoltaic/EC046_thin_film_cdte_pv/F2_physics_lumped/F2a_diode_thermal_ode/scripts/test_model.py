"""
EC046 -- Thin-Film CdTe PV -- F2a Physics-Lumped
Test suite: I-V physics sanity, Lambert-W vs Newton cross-check, MPP,
irradiance/temperature dependence, lumped thermal ODE, CdTe specifics,
predict() interface, benchmark. Custom harness (NO pytest).
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from model import CdTePV_F2a
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
def test_lambertw_vs_newton():
    print("\n[Test 1] Lambert-W I(V) matches Newton root-find")
    m, _ = make_model()
    I_L, I_o, R_sh, a = m.desoto_params(1000.0, 25.0)
    V_oc = float(m.open_circuit_voltage(I_L, I_o, R_sh, a))
    V = np.linspace(0.0, V_oc, 50)
    I_lw = m.current_from_voltage(V, I_L, I_o, R_sh, a)
    I_nw = m._i_from_v_newton(V, I_L, I_o, R_sh, a)
    max_err = float(np.max(np.abs(I_lw - I_nw)))
    assert_true(max_err < 1e-4, f"max |I_LambertW - I_Newton| = {max_err:.2e} A < 1e-4")


def test_p_zero_at_dark():
    print("\n[Test 2] P = 0 at G = 0 (enforced)")
    m, _ = make_model()
    r = m.mpp(0.0, 25.0)
    assert_true(r["p_mp"] == 0.0, f"P_mp(G=0) = {r['p_mp']} == 0")
    assert_true(m.efficiency(0.0, 25.0) == 0.0, "efficiency(G=0) == 0")


def test_isc_proportional_to_G():
    print("\n[Test 3] Isc roughly proportional to irradiance")
    m, _ = make_model()
    isc_1000 = m._raw_mpp(1000.0, 25.0)["i_sc"]
    isc_500 = m._raw_mpp(500.0, 25.0)["i_sc"]
    ratio = isc_500 / isc_1000
    assert_true(0.48 < ratio < 0.52, f"Isc(500)/Isc(1000) = {ratio:.3f} ~ 0.5")
    isc_200 = m._raw_mpp(200.0, 25.0)["i_sc"]
    assert_true(0.18 < isc_200 / isc_1000 < 0.22, f"Isc(200)/Isc(1000) = {isc_200/isc_1000:.3f} ~ 0.2")


def test_pv_curve_monotone_to_mpp():
    print("\n[Test 4] P-V curve rises monotonically to MPP then falls")
    m, _ = make_model()
    iv = m.iv_curve(1000.0, 25.0, n_points=300)
    P = iv["P"]
    imax = int(np.argmax(P))
    # rising part up to MPP
    rising_ok = np.all(np.diff(P[:imax + 1]) >= -1e-6)
    falling_ok = np.all(np.diff(P[imax:]) <= 1e-6)
    assert_true(rising_ok, f"P monotone non-decreasing up to MPP (idx {imax})")
    assert_true(falling_ok, "P monotone non-increasing past MPP")
    assert_true(0 < imax < len(P) - 1, "MPP is interior to the curve")


def test_efficiency_bounds():
    print("\n[Test 5] Efficiency in (0, 0.19) across operating range")
    m, _ = make_model()
    for G in [200.0, 500.0, 800.0, 1000.0, 1100.0]:
        for Tc in [-10.0, 25.0, 45.0, 70.0]:
            eta = m.efficiency(G, Tc)
            assert_true(0.0 < eta < 0.19, f"eff(G={G},T={Tc}) = {eta*100:.2f}% in (0,19)%")


def test_mpp_consistency():
    print("\n[Test 6] MPP power between 0 and Voc*Isc, FF in (0.5,0.85)")
    m, _ = make_model()
    r = m._raw_mpp(1000.0, 25.0)
    assert_true(0 < r["p_mp"] < r["v_oc"] * r["i_sc"], "0 < Pmp < Voc*Isc")
    assert_true(0.5 < r["fill_factor"] < 0.85, f"FF = {r['fill_factor']:.3f} in (0.5,0.85)")
    assert_true(0 < r["v_mp"] < r["v_oc"], "0 < Vmp < Voc")


def test_cdte_tempco():
    print("\n[Test 7] CdTe Pmp tempco ~ -0.28 %/K (0.578 correction preserved)")
    m, _ = make_model()
    # Use fixed cell temperature so only the device tempco shows.
    p25 = m.mpp(1000.0, 25.0)["p_mp"]
    p65 = m.mpp(1000.0, 65.0)["p_mp"]
    gamma = (p65 - p25) / p25 / 40.0  # 1/K
    assert_true(-0.0033 < gamma < -0.0024,
                f"gamma_pmp = {gamma*100:.3f} %/K ~ -0.28 %/K (corrected)")
    # And materially better (less negative) than the bare De Soto tempco.
    assert_true(gamma > m.gamma_desoto,
                f"corrected {gamma*100:.3f}%/K > bare De Soto {m.gamma_desoto*100:.3f}%/K")


def test_low_light_gain():
    print("\n[Test 8] CdTe low-light: efficiency uplift at low irradiance")
    m, _ = make_model()
    eta_1000 = m.efficiency(1000.0, 25.0)
    eta_200 = m.efficiency(200.0, 25.0)
    assert_true(eta_200 > eta_1000, f"eff(200)={eta_200*100:.2f}% > eff(1000)={eta_1000*100:.2f}%")


def test_thermal_ode_heats_up():
    print("\n[Test 9] Thermal ODE: module heats above ambient under sun")
    m, _ = make_model()
    r = m.simulate(900.0, 25.0, T_cell0_c=25.0, wind=1.0,
                   duration_s=1800.0, dt=30.0)
    T_final = r["temperature"][-1]
    assert_true(T_final > 25.0, f"T_final = {T_final:.1f} C > ambient 25 C")
    assert_true(T_final < 90.0, f"T_final = {T_final:.1f} C < 90 C (reasonable)")
    # steady state reached
    dT = abs(r["temperature"][-1] - r["temperature"][-2])
    assert_true(dT < 0.2, f"near steady state: dT = {dT:.3f} C")


def test_thermal_dark_cools():
    print("\n[Test 10] Thermal ODE: hot module cools to ambient in the dark")
    m, _ = make_model()
    r = m.simulate(0.0, 20.0, T_cell0_c=60.0, wind=2.0,
                   duration_s=1800.0, dt=30.0)
    T_final = r["temperature"][-1]
    assert_true(T_final < 25.0, f"T_final = {T_final:.1f} C cools toward 20 C")
    assert_true(np.all(r["p_mp"] == 0.0), "P_mp = 0 throughout dark run")


def test_predict_interface():
    print("\n[Test 11] ComponentModel predict() interface (mpp + dynamic)")
    _, cm = make_model()
    r = cm.predict({"irradiance": 1000.0, "cell_temp_c": 25.0})
    for key in ["v_mp", "i_mp", "p_mp", "v_oc", "i_sc", "fill_factor",
                "efficiency", "cell_temp_c", "iv_curve"]:
        assert_true(key in r, f"mpp key '{key}' present")
    # STC nameplate ~445 W within 10%
    assert_true(abs(r["p_mp"] - 445.0) / 445.0 < 0.10,
                f"STC Pmp = {r['p_mp']:.1f} W within 10% of 445 W")
    d = cm.predict({"mode": "dynamic", "irradiance": 800.0, "T_ambient_c": 25.0,
                    "duration_s": 300.0, "dt": 30.0})
    for key in ["t", "temperature", "p_mp", "efficiency"]:
        assert_true(key in d, f"dynamic key '{key}' present")
    assert_true(len(d["t"]) == len(d["temperature"]), "time-series arrays aligned")


def test_benchmark():
    print("\n[Test 12] Benchmark: 1800 s thermal sim")
    m, _ = make_model()
    t0 = time.perf_counter()
    m.simulate(900.0, 25.0, T_cell0_c=25.0, duration_s=1800.0, dt=30.0)
    elapsed = time.perf_counter() - t0
    print(f"  1800 s thermal ODE in {elapsed*1000:.1f} ms")
    assert_true(elapsed < 5.0, "Completes in < 5 s")


if __name__ == "__main__":
    tests = [
        test_lambertw_vs_newton,
        test_p_zero_at_dark,
        test_isc_proportional_to_G,
        test_pv_curve_monotone_to_mpp,
        test_efficiency_bounds,
        test_mpp_consistency,
        test_cdte_tempco,
        test_low_light_gain,
        test_thermal_ode_heats_up,
        test_thermal_dark_cools,
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
    print(f"EC046 CdTe PV F2a -- Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    sys.exit(0 if failed == 0 else 1)
