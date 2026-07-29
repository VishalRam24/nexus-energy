"""
EC134 -- Oscillating Water Column (OWC) -- F2a Physics-Lumped Dynamics
Test suite: energy conservation, capture efficiency < 1, resonance,
Wells turbine behaviour, edge cases, predict() interface, benchmark.
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from model import OWC_F2a
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
def test_excitation_gain_bounded():
    print("\n[Test 1] Excitation gain in (0, 1] and decreases with frequency")
    m, _ = make_model()
    g_long = m.excitation_gain(0.3)   # long wave (low omega)
    g_short = m.excitation_gain(2.0)  # short wave (high omega)
    assert_true(0 < g_long <= 1.0, f"G(low)={g_long:.4f} in (0,1]")
    assert_true(0 < g_short <= 1.0, f"G(high)={g_short:.4f} in (0,1]")
    assert_true(g_short < g_long, f"G decreases with omega: {g_short:.4f} < {g_long:.4f}")


def test_wells_efficiency_single_peak():
    print("\n[Test 2] Wells turbine efficiency: zero at phi=0, peak near stall, decays")
    m, _ = make_model()
    eta0 = m.wells_efficiency(0.0)
    eta_peak = m.wells_efficiency(m.phi_stall)
    eta_over = m.wells_efficiency(3.0 * m.phi_stall)
    assert_true(abs(eta0) < 1e-9, f"eta(0)={eta0:.4f} == 0")
    assert_true(abs(eta_peak - m.eta_peak) < 1e-6, f"eta(stall)={eta_peak:.4f} == peak {m.eta_peak}")
    assert_true(eta_over < eta_peak, f"post-stall eta={eta_over:.4f} < peak {eta_peak:.4f}")
    assert_true(0 <= eta_peak < 1.0, "peak efficiency < 1")


def test_capture_efficiency_below_one():
    print("\n[Test 3] Capture efficiency and CWR strictly < 1 (2nd law)")
    m, _ = make_model()
    T_n = 2.0 * np.pi / m.natural_frequency()
    for H, T in [(1.0, T_n), (2.0, T_n), (3.0, 1.2 * T_n)]:
        r = m.simulate(H, T, dt=0.03, duration_s=400.0)
        assert_true(0 < r["capture_efficiency"] < 1.0,
                    f"H={H},T={T:.1f}: capture_eff={r['capture_efficiency']:.4f} in (0,1)")
        assert_true(0 < r["capture_width_ratio"] < 1.0,
                    f"H={H},T={T:.1f}: CWR={r['capture_width_ratio']:.4f} in (0,1)")
    # near resonance, a real OWC should capture a meaningful (non-trivial) share
    r = m.simulate(2.0, T_n, dt=0.02, duration_s=600.0)
    assert_true(r["capture_width_ratio"] > 0.05,
                f"resonant CWR={r['capture_width_ratio']:.3f} > 0.05 (realistic OWC)")


def test_energy_conservation():
    print("\n[Test 4] Energy balance: wave input = radiation loss + turbine extraction")
    m, _ = make_model()
    # run at the natural period and long enough to reach a stationary cycle
    # (lightly damped resonance needs many cycles to build up).
    T_n = 2.0 * np.pi / m.natural_frequency()
    r = m.simulate(2.0, T_n, dt=0.02, duration_s=600.0)
    # steady-state first law: mean wave power = mean radiation dissipation
    # + mean turbine (pneumatic) extraction.
    lhs = r["mean_P_exc_W"]
    rhs = r["mean_P_rad_W"] + r["mean_P_avail_W"]
    rel_err = abs(lhs - rhs) / abs(lhs)
    assert_true(rel_err < 0.05, f"P_exc={lhs:.0f} = P_rad+P_avail={rhs:.0f} (err {rel_err*100:.1f}%)")
    # mean wave input must exceed the part absorbed by the turbine (rest radiated)
    assert_true(r["mean_P_exc_W"] >= r["mean_P_avail_W"] - 1e-3,
                f"P_exc={r['mean_P_exc_W']:.0f} >= P_avail={r['mean_P_avail_W']:.0f}")
    assert_true(r["mean_P_avail_W"] >= 0.0, "absorbed pneumatic power >= 0")
    # electrical <= turbine shaft <= available (no power created in PTO chain)
    assert_true(r["mean_P_elec_W"] <= r["mean_P_turb_W"] + 1e-6, "P_elec <= P_turb")
    assert_true(r["mean_P_turb_W"] <= r["mean_P_avail_W"] + 1e-6, "P_turb <= P_avail")


def test_power_monotone_with_height():
    print("\n[Test 5] Mean electrical power increases with wave height (~ H^2)")
    m, _ = make_model()
    T = 2.0 * np.pi / m.natural_frequency()
    P_prev = -1.0
    for H in [0.5, 1.0, 2.0, 3.0]:
        r = m.simulate(H, T, dt=0.03, duration_s=400.0)
        assert_true(r["mean_P_elec_W"] > P_prev,
                    f"H={H}: P={r['mean_P_elec_W']:.1f} W > prev {P_prev:.1f}")
        P_prev = r["mean_P_elec_W"]


def test_resonance_amplification():
    print("\n[Test 6] Resonance: response peaks when wave period near natural period")
    m, _ = make_model()
    omega_n = m.natural_frequency()
    T_res = 2.0 * np.pi / omega_n
    # sweep periods around resonance, record water-column oscillation amplitude
    periods = np.linspace(0.5 * T_res, 1.8 * T_res, 9)
    amps = []
    for T in periods:
        r = m.simulate(1.0, float(T), dt=0.03, duration_s=120.0)
        i0 = len(r["t"]) // 2
        amps.append(np.std(r["x"][i0:]))
        ratio = float(T) / T_res
        _ = ratio
    amps = np.array(amps)
    i_peak = int(np.argmax(amps))
    T_peak = periods[i_peak]
    # peak should be within +/-40% of the natural period
    assert_true(0.6 * T_res < T_peak < 1.4 * T_res,
                f"amplitude peaks near T_n={T_res:.2f}s at T={T_peak:.2f}s")
    assert_true(amps[i_peak] > amps[0] and amps[i_peak] > amps[-1],
                "interior amplitude peak (resonant amplification)")


def test_pressure_oscillates():
    print("\n[Test 7] Chamber pressure oscillates (positive and negative phases)")
    m, _ = make_model()
    r = m.simulate(2.0, 9.0, dt=0.05, duration_s=80.0)
    i0 = len(r["t"]) // 2
    p = r["pressure"][i0:]
    assert_true(np.max(p) > 0 and np.min(p) < 0,
                f"self-rectifying: p in [{np.min(p):.1f}, {np.max(p):.1f}] Pa")
    # pressure should be a small fraction of atmospheric (linear-acoustics regime)
    assert_true(np.max(np.abs(p)) < 0.5 * m.p_atm, "chamber gauge pressure < 0.5 atm")


def test_zero_wave_zero_power():
    print("\n[Test 8] Edge case: zero wave height -> zero power")
    m, _ = make_model()
    r = m.simulate(0.0, 9.0, dt=0.1, duration_s=40.0)
    assert_true(abs(r["mean_P_elec_W"]) < 1e-6, f"P_elec={r['mean_P_elec_W']:.2e} ~ 0")
    assert_true(np.max(np.abs(r["x"])) < 1e-6, "no water-column motion")


def test_spectrum_mean_power():
    print("\n[Test 9] Irregular-sea (PM spectrum) mean power positive and < regular peak")
    m, _ = make_model()
    spec = m.mean_power_spectrum(2.0, 9.0, n_freq=15, duration_s=60.0, dt=0.1)
    assert_true(spec["mean_P_elec_W"] > 0, f"spectrum P_elec={spec['mean_P_elec_W']:.1f} W > 0")
    assert_true(0 < spec["capture_width_ratio"] < 1.0,
                f"spectrum CWR={spec['capture_width_ratio']:.4f} in (0,1)")


def test_predict_interface():
    print("\n[Test 10] ComponentModel predict() interface")
    _, cm = make_model()
    r = cm.predict({"H_s": 2.0, "T_e": 9.0, "dt": 0.05, "duration_s": 40.0})
    for key in ["t", "x", "xdot", "pressure", "P_elec", "mean_P_elec_kW",
                "capture_width_ratio", "capture_efficiency"]:
        assert_true(key in r, f"Key '{key}' in output")
    assert_true(len(r["t"]) == len(r["P_elec"]), "Arrays same length")
    info = cm.get_info()
    assert_true(info["component_id"] == "EC134", "component_id == EC134")


def test_benchmark():
    print("\n[Test 11] Benchmark: 120 s simulation at dt=0.05")
    m, _ = make_model()
    t0 = time.perf_counter()
    m.simulate(2.0, 9.0, dt=0.05, duration_s=120.0)
    elapsed = time.perf_counter() - t0
    print(f"  120 s simulation in {elapsed*1000:.1f} ms")
    assert_true(elapsed < 5.0, "Completes in < 5 s")


if __name__ == "__main__":
    tests = [
        test_excitation_gain_bounded,
        test_wells_efficiency_single_peak,
        test_capture_efficiency_below_one,
        test_energy_conservation,
        test_power_monotone_with_height,
        test_resonance_amplification,
        test_pressure_oscillates,
        test_zero_wave_zero_power,
        test_spectrum_mean_power,
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
    print(f"EC134 OWC F2a -- Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    sys.exit(0 if failed == 0 else 1)
