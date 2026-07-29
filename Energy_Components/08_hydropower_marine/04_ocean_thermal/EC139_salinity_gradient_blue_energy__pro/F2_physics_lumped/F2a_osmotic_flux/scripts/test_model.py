"""
EC139 -- Salinity Gradient Blue Energy (PRO) -- F2a Osmotic Flux
Test suite: physics sanity (peak at DeltaP=Dpi/2, mass conservation, CP effects),
edge cases, predict() interface, benchmark timing. NO pytest.
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from model import SalinityGradientPRO_F2a
from predict import ComponentModel

PASS = "✓"
FAIL = "✗"
_BAR = 1.0e5


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
def test_osmotic_pressure_seawater():
    print("\n[Test 1] van't Hoff osmotic pressure of seawater ~27 bar")
    m, _ = make_model()
    pi = m.osmotic_pressure(35.0)  # Pa
    pi_bar = pi / _BAR
    assert_true(22.0 < pi_bar < 32.0, f"pi_seawater={pi_bar:.2f} bar in [22,32]")
    pi_f = m.osmotic_pressure(0.5) / _BAR
    assert_true(pi_f < pi_bar, f"feed osmotic pressure {pi_f:.3f} < draw {pi_bar:.2f}")


def test_flux_decreases_with_dP():
    print("\n[Test 2] Water flux Jw decreases monotonically with DeltaP")
    m, _ = make_model()
    dpi = float(m.osmotic_pressure(35.0) - m.osmotic_pressure(0.5))
    dP_vals = np.linspace(0.0, dpi, 40)
    Jw_prev = m.water_flux(dP_vals[0])
    for dP in dP_vals[1:]:
        Jw = m.water_flux(dP)
        assert_true(Jw <= Jw_prev + 1e-12,
                    f"Jw(dP={dP/_BAR:.1f}bar)={Jw*3.6e6:.3f}LMH <= prev {Jw_prev*3.6e6:.3f}")
        Jw_prev = Jw
    print("  All 39 pairs monotone.")


def test_power_peaks_at_half_dpi():
    print("\n[Test 3] Power density W peaks near DeltaP = Delta_pi/2 (Loeb 1976)")
    m, _ = make_model()
    dpi = float(m.osmotic_pressure(35.0) - m.osmotic_pressure(0.5))
    dP_opt, W_max = m.optimal_delta_P()
    ratio = dP_opt / dpi
    # CP + reverse salt flux shift the peak slightly below 0.5; allow a window.
    assert_true(0.30 < ratio < 0.55,
                f"DeltaP_opt/Delta_pi = {ratio:.3f} near 0.5 (W_max={W_max:.2f} W/m2)")
    # confirm it's actually a maximum: lower at the edges
    W_lo = m.power_density(0.1 * dpi)
    W_hi = m.power_density(0.9 * dpi)
    assert_true(W_max > W_lo and W_max > W_hi,
                f"W_max={W_max:.2f} > edges ({W_lo:.2f}, {W_hi:.2f})")


def test_ideal_peak_exact_half():
    print("\n[Test 4] Ideal membrane (no CP, no salt flux): peak EXACTLY at Dpi/2")
    m, _ = make_model()
    # Suppress CP and reverse salt flux -> classic Jw = A*(dpi - dP)
    m.B = 0.0
    m.S = 1e-12      # ~ no ICP
    m.k = 1e6        # ~ no ECP
    dpi = float(m.osmotic_pressure(35.0) - m.osmotic_pressure(0.5))
    dP_opt, W_max = m.optimal_delta_P(n_scan=400)
    ratio = dP_opt / dpi
    assert_true(abs(ratio - 0.5) < 0.02, f"ideal DeltaP_opt/Delta_pi={ratio:.4f} ~ 0.5")
    W_theory = m.A * dpi ** 2 / 4.0
    assert_true(abs(W_max - W_theory) / W_theory < 0.02,
                f"W_max={W_max:.2f} ~ A*Dpi^2/4={W_theory:.2f} W/m2")


def test_flux_zero_above_dpi():
    print("\n[Test 5] Jw <= 0 when DeltaP exceeds effective osmotic pressure")
    m, _ = make_model()
    dpi = float(m.osmotic_pressure(35.0) - m.osmotic_pressure(0.5))
    Jw = m.water_flux(1.5 * dpi)
    assert_true(Jw <= 1e-9, f"Jw(1.5*Dpi)={Jw:.3e} m/s <= 0 (reverse osmosis regime)")


def test_power_density_realistic():
    print("\n[Test 6] Peak power density realistic (0.5-15 W/m2)")
    m, _ = make_model()
    _, W_max = m.optimal_delta_P()
    assert_true(0.5 < W_max < 15.0,
                f"W_max={W_max:.2f} W/m2 in realistic PRO range (Statkraft target ~5)")


def test_reverse_salt_flux_positive():
    print("\n[Test 7] Reverse salt flux Js > 0 (salt leaks draw->feed)")
    m, _ = make_model()
    Jw = m.water_flux(13.0 * _BAR)
    Js = m.reverse_salt_flux(Jw)
    assert_true(Js > 0, f"Js={Js:.3e} mol/m2/s > 0")
    # higher draw salinity -> more salt leakage
    Js_hi = m.reverse_salt_flux(Jw, C_draw_gL=60.0)
    assert_true(Js_hi > Js, f"Js(60g/L)={Js_hi:.3e} > Js(35g/L)={Js:.3e}")


def test_cp_reduces_flux():
    print("\n[Test 8] Concentration polarization reduces flux vs ideal")
    m_real, _ = make_model()
    m_ideal, _ = make_model()
    m_ideal.B = 0.0; m_ideal.S = 1e-12; m_ideal.k = 1e6
    dP = 10.0 * _BAR
    Jw_real = m_real.water_flux(dP)
    Jw_ideal = m_ideal.water_flux(dP)
    assert_true(Jw_real < Jw_ideal,
                f"Jw_real={Jw_real*3.6e6:.2f} < Jw_ideal={Jw_ideal*3.6e6:.2f} LMH (CP penalty)")


def test_mass_conservation():
    print("\n[Test 9] Module ODE conserves salt: draw dilutes, reaches steady state")
    m, _ = make_model()
    dP_opt, _ = m.optimal_delta_P()
    r = m.simulate(dP_opt, dt=5.0, duration_s=600.0)
    # Draw concentration must decrease from inlet (permeate dilution dominates)
    assert_true(r["C_draw_gL"][-1] < m.C_draw0,
                f"C_draw final {r['C_draw_gL'][-1]:.3f} < inlet {m.C_draw0:.1f} g/L")
    assert_true(r["C_draw_gL"][-1] > m.C_feed0,
                f"C_draw final {r['C_draw_gL'][-1]:.3f} > feed {m.C_feed0:.1f} g/L (bounded)")
    # steady state reached
    dC = abs(r["C_draw_gL"][-1] - r["C_draw_gL"][-2])
    assert_true(dC < 1e-2, f"near steady state: dC={dC:.2e} g/L between last steps")
    # all concentrations physical
    assert_true(np.all(r["C_draw_gL"] > 0), "all draw concentrations positive")


def test_net_power_positive():
    print("\n[Test 10] Net power > 0 at optimal DeltaP (after pumping losses)")
    m, _ = make_model()
    dP_opt, _ = m.optimal_delta_P()
    r = m.simulate(dP_opt, dt=5.0, duration_s=300.0)
    P_net = r["P_net_W"][-1]
    P_turb = r["P_turbine_W"][-1]
    assert_true(P_net > 0, f"P_net={P_net:.1f} W > 0")
    assert_true(P_net < P_turb, f"P_net={P_net:.1f} < P_turbine={P_turb:.1f} (pumping cost)")


def test_predict_interface():
    print("\n[Test 11] ComponentModel predict() interface")
    _, cm = make_model()
    r = cm.predict({"delta_P_bar": 13.0, "dt": 10.0, "duration_s": 100.0})
    for key in ["t", "C_draw_gL", "Jw", "Js", "power_density",
                "P_turbine_W", "P_pump_W", "P_net_W"]:
        assert_true(key in r, f"Key '{key}' in output")
    assert_true(len(r["t"]) == len(r["power_density"]), "Arrays same length")
    assert_true("power_density_final_Wm2" in r, "scalar summary present")


def test_benchmark():
    print("\n[Test 12] Benchmark: 600s module ODE simulation")
    m, _ = make_model()
    dP_opt, _ = m.optimal_delta_P()
    t0 = time.perf_counter()
    m.simulate(dP_opt, dt=1.0, duration_s=600.0)
    elapsed = time.perf_counter() - t0
    print(f"  600s simulation in {elapsed*1000:.1f} ms")
    assert_true(elapsed < 5.0, "Completes in < 5 s")


if __name__ == "__main__":
    tests = [
        test_osmotic_pressure_seawater,
        test_flux_decreases_with_dP,
        test_power_peaks_at_half_dpi,
        test_ideal_peak_exact_half,
        test_flux_zero_above_dpi,
        test_power_density_realistic,
        test_reverse_salt_flux_positive,
        test_cp_reduces_flux,
        test_mass_conservation,
        test_net_power_positive,
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
    print(f"EC139 PRO F2a -- Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    sys.exit(0 if failed == 0 else 1)
