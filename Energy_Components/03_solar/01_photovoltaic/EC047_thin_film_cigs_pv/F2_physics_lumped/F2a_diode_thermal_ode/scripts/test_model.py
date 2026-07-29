"""
EC047 -- Thin-Film CIGS PV -- F2a Physics-Lumped
Test suite: single-diode physics sanity, Lambert-W vs root-find, MPP,
irradiance/temperature dependence, thermal ODE, edge cases, predict interface.
Run:  python3 scripts/test_model.py   (system python3, NOT pytest)
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from model import CIGSPvF2a
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
def test_zero_power_at_dark():
    print("\n[Test 1] P = 0 at G = 0 (dark)")
    m, _ = make_model()
    r = m.mpp(0.0, 298.15)
    assert_true(r["p_mp"] == 0.0, f"P_mp={r['p_mp']:.4f} == 0 at G=0")
    assert_true(r["i_sc"] == 0.0, f"I_sc={r['i_sc']:.4f} == 0 at G=0")
    assert_true(m.efficiency(0.0, 298.15) == 0.0, "efficiency == 0 at G=0")


def test_efficiency_bounds():
    print("\n[Test 2] 0 < efficiency < 0.21 across conditions")
    m, _ = make_model()
    for G in [100, 400, 700, 1000, 1100]:
        for Tc in [0, 25, 45, 65]:
            eff = m.efficiency(G, Tc + 273.15)
            assert_true(0.0 < eff < 0.21, f"eff(G={G},T={Tc})={eff*100:.2f}% in (0,21%)")


def test_isc_proportional_to_G():
    print("\n[Test 3] Isc proportional to irradiance")
    m, _ = make_model()
    base = m.mpp(1000.0, 298.15)["i_sc"]
    half = m.mpp(500.0, 298.15)["i_sc"]
    quarter = m.mpp(250.0, 298.15)["i_sc"]
    assert_true(abs(half / base - 0.5) < 0.02, f"Isc(500)/Isc(1000)={half/base:.3f} ~ 0.5")
    assert_true(abs(quarter / base - 0.25) < 0.02, f"Isc(250)/Isc(1000)={quarter/base:.3f} ~ 0.25")


def test_lambertw_matches_residual():
    print("\n[Test 4] Lambert-W I(V) satisfies the implicit diode equation")
    m, _ = make_model()
    I_L, I_o, R_sh, a = m._calc_params(800.0, 308.15)
    for V in [0.0, 20.0, 50.0, 90.0]:
        I = float(m._i_from_v(V, I_L, I_o, R_sh, a))
        resid = (I_L - I_o * (np.exp((V + I * m.R_s) / a) - 1.0)
                 - (V + I * m.R_s) / R_sh - I)
        assert_true(abs(resid) < 1e-6, f"residual(V={V})={resid:.2e} ~ 0")


def test_pv_monotone_to_mpp():
    print("\n[Test 5] P-V curve monotone increasing up to MPP")
    m, _ = make_model()
    V, I, P = m.iv_curve(1000.0, 298.15, n=400)
    imax = int(np.argmax(P))
    diffs = np.diff(P[:imax + 1])
    assert_true(np.all(diffs >= -1e-6), "P rises monotonically from 0 to MPP")
    assert_true(np.all(np.diff(P[imax:]) <= 1e-6), "P falls monotonically after MPP")
    # MPP from golden-section agrees with curve max
    p_mpp = m.mpp(1000.0, 298.15)["p_mp"]
    assert_true(abs(p_mpp - P[imax]) / P[imax] < 0.01, "golden-section MPP ~ curve max")


def test_stc_power_near_rating():
    print("\n[Test 6] STC MPP power near 170 W rating")
    m, _ = make_model()
    r = m.mpp(1000.0, 298.15)
    assert_true(140.0 < r["p_mp"] < 190.0, f"P_mp(STC)={r['p_mp']:.1f} W near 170 W")
    assert_true(0.5 < r["fill_factor"] < 0.85, f"FF={r['fill_factor']:.3f} physical")
    assert_true(r["v_mp"] < r["v_oc"], f"Vmp={r['v_mp']:.1f} < Voc={r['v_oc']:.1f}")
    assert_true(r["i_mp"] < r["i_sc"] + 1e-6, f"Imp={r['i_mp']:.2f} < Isc={r['i_sc']:.2f}")


def test_temperature_coefficient():
    print("\n[Test 7] Power falls with temperature; CIGS low tempco")
    m, _ = make_model()
    p25 = m.mpp(1000.0, 298.15)["p_mp"]
    p65 = m.mpp(1000.0, 338.15)["p_mp"]
    assert_true(p65 < p25, f"P(65C)={p65:.1f} < P(25C)={p25:.1f}")
    gamma = (p65 - p25) / p25 / 40.0  # per K
    assert_true(-0.006 < gamma < -0.0015,
                f"gamma_pmp={gamma*100:.3f} %/K (CIGS low tempco band)")


def test_voc_rises_with_irradiance():
    print("\n[Test 8] Voc increases with irradiance, ~log")
    m, _ = make_model()
    voc_low = m.mpp(200.0, 298.15)["v_oc"]
    voc_high = m.mpp(1000.0, 298.15)["v_oc"]
    assert_true(voc_high > voc_low, f"Voc(1000)={voc_high:.1f} > Voc(200)={voc_low:.1f}")


def test_thermal_ode_dynamics():
    print("\n[Test 9] Thermal ODE: heats above ambient, bounded, lagged")
    m, _ = make_model()
    r = m.simulate(irradiance=900.0, T_amb_c=30.0, wind=1.0, T_cell0_c=30.0,
                   dt=60.0, duration_s=7200.0)
    T_end = r["T_cell_c"][-1]
    assert_true(T_end > 30.0, f"T_cell heats above ambient: {T_end:.1f} > 30 C")
    assert_true(T_end < 95.0, f"T_cell bounded: {T_end:.1f} < 95 C")
    assert_true(r["T_cell_c"][0] < T_end, "thermal lag: starts cold, warms up")
    # near steady state at the end
    dT = abs(r["T_cell_c"][-1] - r["T_cell_c"][-2])
    assert_true(dT < 0.5, f"approaches steady state: dT={dT:.3f} C/step")
    # cross-check against Faiman steady-state algebraic temperature
    T_ss = m.steady_cell_temperature(900.0, 30.0, 1.0)
    assert_true(abs(T_end - T_ss) < 2.0,
                f"ODE end {T_end:.1f} ~ Faiman steady {T_ss:.1f} C")


def test_bandgap_tunable():
    print("\n[Test 10] CIGS tunable bandgap shifts Voc")
    m, _ = make_model()
    voc_low_gap = m.mpp(1000.0, 298.15)["v_oc"]
    m.set_bandgap(1.30)  # higher Ga content -> wider gap -> higher Voc
    voc_high_gap = m.mpp(1000.0, 298.15)["v_oc"]
    assert_true(voc_high_gap > voc_low_gap,
                f"Voc(Eg=1.30)={voc_high_gap:.1f} > Voc(Eg=1.15)={voc_low_gap:.1f}")
    raised = False
    try:
        m.set_bandgap(2.5)
    except ValueError:
        raised = True
    assert_true(raised, "rejects unphysical bandgap")


def test_predict_interface():
    print("\n[Test 11] ComponentModel predict() interface (3 modes)")
    _, cm = make_model()
    r = cm.predict({"mode": "mpp", "irradiance": 1000.0, "cell_temperature_c": 25.0})
    for key in ["v_mp", "i_mp", "p_mp", "v_oc", "i_sc", "fill_factor", "efficiency"]:
        assert_true(key in r, f"mpp key '{key}' present")
    iv = cm.predict({"mode": "iv", "irradiance": 800.0, "cell_temperature_c": 40.0, "n": 50})
    assert_true(len(iv["V"]) == len(iv["I"]) == len(iv["P"]) == 50, "iv arrays aligned")
    tr = cm.predict({"mode": "transient", "irradiance": 700.0, "T_ambient_c": 20.0,
                     "dt": 120.0, "duration_s": 1200.0})
    assert_true(len(tr["t"]) == len(tr["power_W"]), "transient arrays aligned")
    info = cm.get_info()
    assert_true(info["component_id"] == "EC047", "get_info component_id")


def test_benchmark():
    print("\n[Test 12] Benchmark: 24h transient at dt=300s")
    m, _ = make_model()
    t0 = time.perf_counter()
    m.simulate(irradiance=800.0, T_amb_c=25.0, wind=1.5, dt=300.0, duration_s=86400.0)
    elapsed = time.perf_counter() - t0
    print(f"  24h simulation in {elapsed*1000:.1f} ms")
    assert_true(elapsed < 5.0, "Completes in < 5 s")


if __name__ == "__main__":
    tests = [
        test_zero_power_at_dark,
        test_efficiency_bounds,
        test_isc_proportional_to_G,
        test_lambertw_matches_residual,
        test_pv_monotone_to_mpp,
        test_stc_power_near_rating,
        test_temperature_coefficient,
        test_voc_rises_with_irradiance,
        test_thermal_ode_dynamics,
        test_bandgap_tunable,
        test_predict_interface,
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
    print(f"EC047 CIGS PV F2a -- Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    sys.exit(0 if failed == 0 else 1)
