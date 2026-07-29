"""
EC135 -- Point Absorber WEC -- F2a Heaving-Buoy Linear Hydrodynamic Model
Test suite: physics sanity (resonance, optimal damping, energy balance,
capture-width bound), edge cases, predict() interface, benchmark timing.

Run with system python3:  python3 scripts/test_model.py   (NOT pytest)
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from model import PointAbsorberF2a
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
def test_natural_period():
    print("\n[Test 1] Natural period T_n = 2 pi sqrt(M/C_hyd) is sensible")
    m, _ = make_model()
    Tn = m.natural_period()
    assert_true(4.0 < Tn < 20.0, f"T_n={Tn:.2f} s in ocean-wave band")
    # consistency: C_hyd must equal rho g A_wp
    assert_true(abs(m.C_hyd - m.rho * 9.81 * m.A_wp) < 1e-6, "C_hyd = rho g A_wp")


def test_ode_matches_frequency_domain():
    print("\n[Test 2] solve_ivp steady amplitude matches closed-form |X|")
    m, _ = make_model()
    Tn = m.natural_period()
    for T in [0.7 * Tn, Tn, 1.3 * Tn]:
        r = m.simulate(H=1.5, T=T, B_pto=1.5e5, duration_s=80 * T, dt=0.05)
        X_analytic = m.steady_amplitude(1.5, T, B_pto=1.5e5)
        rel = abs(r["amplitude"] - X_analytic) / X_analytic
        assert_true(rel < 0.03,
                    f"T={T:.1f}s: ODE amp {r['amplitude']:.3f} vs analytic "
                    f"{X_analytic:.3f} (rel {rel*100:.2f}%)")


def test_power_peaks_at_resonance():
    print("\n[Test 3] Absorbed power peaks at the natural period")
    m, _ = make_model()
    Tn = m.natural_period()
    periods = np.linspace(0.6 * Tn, 1.6 * Tn, 11)
    powers = [m.mean_power_analytic(1.5, T, B_pto=9.0e4) for T in periods]
    i_peak = int(np.argmax(powers))
    T_peak = periods[i_peak]
    assert_true(abs(T_peak - Tn) / Tn < 0.12,
                f"Power peaks near T_n: T_peak={T_peak:.2f}s vs T_n={Tn:.2f}s")


def test_optimal_pto_damping():
    print("\n[Test 4] Optimal B_pto = |Z_i| maximises absorbed power")
    m, _ = make_model()
    Tn = m.natural_period()
    T = 1.2 * Tn   # off-resonance so |Z_i| > B_rad (non-trivial optimum)
    B_opt = m.optimal_B_pto(T)
    P_opt = m.mean_power_analytic(1.5, T, B_pto=B_opt)
    # Perturb around the optimum: power must not increase
    for f in [0.5, 0.7, 1.4, 2.0]:
        P = m.mean_power_analytic(1.5, T, B_pto=f * B_opt)
        assert_true(P <= P_opt + 1e-6,
                    f"P(B={f:.1f}*B_opt)={P/1e3:.2f}kW <= P_opt={P_opt/1e3:.2f}kW")
    assert_true(B_opt > m.B_rad, f"|Z_i|={B_opt:.2e} > B_rad off-resonance")


def test_optimal_equals_radiation_at_resonance():
    print("\n[Test 5] At resonance reactance=0 so B_opt = B_rad")
    m, _ = make_model()
    Tn = m.natural_period()
    B_opt = m.optimal_B_pto(Tn)
    assert_true(abs(B_opt - m.B_rad) / m.B_rad < 0.02,
                f"B_opt({Tn:.2f}s)={B_opt:.3e} ~= B_rad={m.B_rad:.3e}")


def test_energy_balance():
    print("\n[Test 6] Energy balance: wave input = PTO + radiated (steady state)")
    m, _ = make_model()
    Tn = m.natural_period()
    r = m.simulate(H=2.0, T=Tn, B_pto=1.2e5, duration_s=80 * Tn, dt=0.02)
    # Mean power into body by excitation = mean PTO + mean radiated
    # (mean inertial & spring power vanish over a period). No kinetic-energy drift.
    lhs = r["P_exc_mean"]
    rhs = r["P_pto_mean"] + r["P_rad_mean"]
    rel = abs(lhs - rhs) / abs(lhs)
    assert_true(rel < 0.02,
                f"P_exc={lhs/1e3:.2f}kW = P_pto+P_rad={rhs/1e3:.2f}kW "
                f"(rel {rel*100:.2f}%)")
    assert_true(r["P_pto_mean"] > 0, "Net positive power absorbed")


def test_capture_width_below_theoretical_max():
    print("\n[Test 7] Capture width < point-absorber theoretical max")
    m, _ = make_model()
    Tn = m.natural_period()
    for T in [0.8 * Tn, Tn, 1.2 * Tn]:
        r = m.simulate(H=1.5, T=T, B_pto="optimal" and m.optimal_B_pto(T),
                       duration_s=60 * T, dt=0.05)
        cw = r["capture_width"]
        cw_max = r["capture_width_max"]
        assert_true(0 < cw < cw_max,
                    f"T={T:.1f}s: CW={cw:.2f}m < CW_max={cw_max:.2f}m")


def test_power_scales_with_wave_height_squared():
    print("\n[Test 8] Linear model: mean power scales as H^2")
    m, _ = make_model()
    Tn = m.natural_period()
    P1 = m.mean_power_analytic(1.0, Tn, B_pto=9.0e4)
    P2 = m.mean_power_analytic(2.0, Tn, B_pto=9.0e4)
    ratio = P2 / P1
    assert_true(abs(ratio - 4.0) < 0.05, f"P(2H)/P(H)={ratio:.3f} ~= 4 (H^2 law)")


def test_zero_pto_no_extraction():
    print("\n[Test 9] B_pto=0 -> zero absorbed power")
    m, _ = make_model()
    Tn = m.natural_period()
    r = m.simulate(H=1.5, T=Tn, B_pto=0.0, duration_s=60 * Tn, dt=0.05)
    assert_true(abs(r["P_pto_mean"]) < 1.0, f"P_pto={r['P_pto_mean']:.4f} W ~ 0")
    assert_true(r["capture_width"] < 1e-3, "Capture width ~ 0 with no PTO")


def test_predict_interface():
    print("\n[Test 10] ComponentModel predict() interface + efficiency bound")
    _, cm = make_model()
    r = cm.predict({"H_m": 1.5, "T_s": 10.0, "B_pto": "optimal", "duration_s": 200.0})
    for key in ["t", "x", "x_dot", "P_pto_mean", "P_elec_mean",
                "capture_width", "capture_width_max", "T_natural"]:
        assert_true(key in r, f"Key '{key}' in output")
    assert_true(len(r["t"]) == len(r["x"]), "Time and state arrays same length")
    # Electrical <= mechanical (eta in (0,1))
    assert_true(0 < r["P_elec_mean"] < r["P_pto_mean"],
                f"0 < P_elec ({r['P_elec_mean']/1e3:.1f}kW) < "
                f"P_pto ({r['P_pto_mean']/1e3:.1f}kW)")


def test_benchmark():
    print("\n[Test 11] Benchmark: resonant simulation runtime")
    m, _ = make_model()
    Tn = m.natural_period()
    t0 = time.perf_counter()
    m.simulate(H=1.5, T=Tn, B_pto=9.0e4, duration_s=60 * Tn, dt=0.05)
    elapsed = time.perf_counter() - t0
    print(f"  60-period simulation in {elapsed*1000:.1f} ms")
    assert_true(elapsed < 5.0, "Representative simulate() finishes < 5 s")


if __name__ == "__main__":
    tests = [
        test_natural_period,
        test_ode_matches_frequency_domain,
        test_power_peaks_at_resonance,
        test_optimal_pto_damping,
        test_optimal_equals_radiation_at_resonance,
        test_energy_balance,
        test_capture_width_below_theoretical_max,
        test_power_scales_with_wave_height_squared,
        test_zero_pto_no_extraction,
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
    print(f"EC135 Point Absorber WEC F2a -- Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    sys.exit(0 if failed == 0 else 1)
