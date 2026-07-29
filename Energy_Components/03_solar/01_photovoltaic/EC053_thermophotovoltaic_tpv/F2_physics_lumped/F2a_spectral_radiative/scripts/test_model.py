"""
EC053 -- Thermophotovoltaic (TPV) -- F2a Spectral Radiative
Test suite: spectral/radiative physics sanity, single-diode I-V, MPP,
thermal ODE, edge cases, predict() interface, benchmark timing.
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from model import TPV_F2a, SIGMA
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
def test_planck_stefan_boltzmann():
    print("\n[Test 1] Planck integral recovers Stefan-Boltzmann (sigma*T^4)")
    m, _ = make_model()
    T = 1500.0
    lam = m._lambda_grid(T, n=4000)
    M = m.planck_exitance(lam, T)
    M_total = np.trapz(M, lam)
    sb = SIGMA * T ** 4
    rel = abs(M_total - sb) / sb
    assert_true(rel < 0.02, f"integral={M_total:.3e} vs sigma*T^4={sb:.3e} (rel err {rel*100:.2f}%)")


def test_power_zero_below_useful_T():
    print("\n[Test 2] Electrical power -> 0 at low emitter T (P=0 below useful T)")
    m, _ = make_model()
    P_low = m.mpp(300.0, 300.0)["P_W"]
    P_mid = m.mpp(1500.0, 300.0)["P_W"]
    assert_true(P_low < 1e-6, f"P(300K)={P_low:.3e} W ~ 0 (negligible above-gap photons)")
    assert_true(P_mid > P_low, f"P(1500K)={P_mid:.3e} W > P(300K)")


def test_power_rises_steeply_with_T():
    print("\n[Test 3] In-band power rises >= T^4 with emitter temperature")
    m, _ = make_model()
    T1, T2 = 1200.0, 1500.0
    P1 = m.spectral_power(T1)["M_inband"]
    P2 = m.spectral_power(T2)["M_inband"]
    ratio = P2 / P1
    t4 = (T2 / T1) ** 4
    assert_true(P2 > P1, f"M_inband rises: {P2:.3e} > {P1:.3e}")
    # near/above band edge the in-band band rises STEEPER than T^4
    assert_true(ratio >= t4 * 0.95, f"ratio={ratio:.2f} >= T^4 ratio={t4:.2f} (steep, band-edge)")


def test_power_monotone_in_T():
    print("\n[Test 4] MPP power monotonically increases with emitter T")
    m, _ = make_model()
    Ts = np.linspace(1000.0, 2000.0, 12)
    P_prev = -1.0
    for T in Ts:
        P = m.mpp(T, 300.0)["P_W"]
        assert_true(P >= P_prev - 1e-12, f"P({T:.0f}K)={P*1e3:.3f} mW >= prev")
        P_prev = P
    print("  All 12 points monotone.")


def test_efficiency_bounds():
    print("\n[Test 5] 0 < eta_spectral < 1 and 0 < eta_system < 1")
    m, _ = make_model()
    for T in [1200.0, 1500.0, 1800.0]:
        eff = m.efficiencies(T, 300.0)
        assert_true(0.0 < eff["eta_spectral"] < 1.0,
                    f"eta_spec({T:.0f})={eff['eta_spectral']:.3f} in (0,1)")
        assert_true(0.0 < eff["eta_system"] < 1.0,
                    f"eta_sys({T:.0f})={eff['eta_system']:.3f} in (0,1)")


def test_subbandgap_loss_accounted():
    print("\n[Test 6] Sub-bandgap photons exist and recycling reduces net loss")
    m, _ = make_model()
    rb = m.radiative_balance(1500.0, 300.0)
    assert_true(rb["P_subband_W"] > 0.0, f"sub-bandgap radiated > 0: {rb['P_subband_W']:.3e} W")
    # recycling: lost < total sub-band (reflector returns some)
    assert_true(rb["P_subband_lost_W"] < rb["P_subband_W"],
                f"recycled: lost {rb['P_subband_lost_W']:.3e} < total {rb['P_subband_W']:.3e}")
    # net radiated less than naive total (inband+subband) due to recycling
    naive = rb["P_inband_W"] + rb["P_subband_W"]
    assert_true(rb["P_rad_net_W"] < naive,
                f"net {rb['P_rad_net_W']:.3e} < naive {naive:.3e} (recycling)")


def test_single_diode_iv():
    print("\n[Test 7] Single-diode I-V: J(0)=Jsc~Jph, J(Voc)~0, V<Voc gives V*J>0")
    m, _ = make_model()
    Jph = m.photocurrent_density(1500.0)
    Voc = m.open_circuit_voltage(1500.0, 300.0, Jph=Jph)
    Jsc = m.current_density(0.0, 1500.0, 300.0, Jph=Jph)
    Joc = m.current_density(Voc, 1500.0, 300.0, Jph=Jph)
    assert_true(Jph > 0, f"Jph={Jph:.2f} A/m^2 > 0")
    assert_true(abs(Jsc - Jph) / Jph < 0.05, f"Jsc={Jsc:.2f} ~ Jph={Jph:.2f}")
    assert_true(abs(Joc) < 0.05 * Jph, f"J(Voc)={Joc:.3e} ~ 0")
    assert_true(0.0 < Voc < m.E_g_eV, f"0 < Voc={Voc:.3f} < Eg={m.E_g_eV:.2f} V")


def test_mpp_and_ff():
    print("\n[Test 8] MPP power <= Voc*Jsc*A and fill factor in (0,1)")
    m, _ = make_model()
    r = m.mpp(1500.0, 300.0)
    ceiling = r["Voc"] * r["Jsc"] * m.A_cell
    assert_true(0.0 < r["P_W"] <= ceiling + 1e-12,
                f"Pmp={r['P_W']*1e3:.3f} mW <= Voc*Jsc*A={ceiling*1e3:.3f} mW")
    assert_true(0.0 < r["FF"] < 1.0, f"FF={r['FF']:.3f} in (0,1)")


def test_thermal_ode_heats_and_cools():
    print("\n[Test 9] Cell thermal ODE: heats above coolant, bounded, near steady")
    m, _ = make_model()
    r = m.simulate(1500.0, 300.0, dt=0.5, duration_s=120.0)
    Tf = r["T_cell"][-1]
    assert_true(Tf > 300.0, f"cell warms: T_final={Tf:.2f} K > 300 K (coolant)")
    assert_true(Tf < 500.0, f"bounded: T_final={Tf:.2f} K < 500 K")
    dT = abs(r["T_cell"][-1] - r["T_cell"][-2])
    assert_true(dT < 0.5, f"near steady-state: dT={dT:.4f} K between last steps")


def test_predict_interface():
    print("\n[Test 10] ComponentModel predict() interface + arrays consistent")
    _, cm = make_model()
    r = cm.predict({"T_emitter_K": 1500.0, "dt": 1.0, "duration_s": 10.0})
    for key in ["t", "T_cell", "T_emitter", "P_elec_W", "Vmp",
                "eta_system", "eta_spectral", "mpp_summary"]:
        assert_true(key in r, f"Key '{key}' in output")
    assert_true(len(r["t"]) == len(r["P_elec_W"]) == len(r["T_cell"]),
                "Time-series arrays same length")
    assert_true(np.all(r["P_elec_W"] >= 0.0), "Power non-negative over trajectory")


def test_time_varying_emitter():
    print("\n[Test 11] Time-varying emitter ramp -> power tracks emitter up")
    m, _ = make_model()
    def ramp(t):
        return 1100.0 + (1800.0 - 1100.0) * min(t / 60.0, 1.0)
    r = m.simulate(ramp, 300.0, dt=2.0, duration_s=60.0)
    assert_true(r["P_elec_W"][-1] > r["P_elec_W"][0],
                f"P rises with ramp: {r['P_elec_W'][-1]*1e3:.2f} > {r['P_elec_W'][0]*1e3:.2f} mW")


def test_benchmark():
    print("\n[Test 12] Benchmark: 60 s thermal sim")
    m, _ = make_model()
    t0 = time.perf_counter()
    m.simulate(1500.0, 300.0, dt=0.5, duration_s=60.0)
    elapsed = time.perf_counter() - t0
    print(f"  60 s simulation in {elapsed*1000:.1f} ms")
    assert_true(elapsed < 10.0, "Completes in < 10 s")


if __name__ == "__main__":
    tests = [
        test_planck_stefan_boltzmann,
        test_power_zero_below_useful_T,
        test_power_rises_steeply_with_T,
        test_power_monotone_in_T,
        test_efficiency_bounds,
        test_subbandgap_loss_accounted,
        test_single_diode_iv,
        test_mpp_and_ff,
        test_thermal_ode_heats_and_cools,
        test_predict_interface,
        test_time_varying_emitter,
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
    print(f"EC053 TPV F2a -- Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    sys.exit(0 if failed == 0 else 1)
