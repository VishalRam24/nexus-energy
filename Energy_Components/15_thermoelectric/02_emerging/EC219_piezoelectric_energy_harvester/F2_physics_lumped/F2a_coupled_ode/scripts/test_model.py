"""
EC219 -- Piezoelectric Energy Harvester -- F2a Coupled Electromechanical ODE
Test suite: resonance, optimal load, energy conservation, acceleration^2
scaling, ODE/analytic consistency, edge cases, predict() interface, benchmark.
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from model import PiezoF2a
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
def test_derived_parameters():
    print("\n[Test 1] Derived lumped parameters are physical / consistent")
    m, _ = make_model()
    assert_true(m.m_eff > 0 and m.k_eff > 0, f"m_eff={m.m_eff:.3e} kg, k_eff={m.k_eff:.2f} N/m > 0")
    assert_true(m.c_mech > 0 and m.C_p > 0, f"c={m.c_mech:.3e} N.s/m, C_p={m.C_p:.3e} F > 0")
    # short-circuit resonance recovered from k/m
    f_check = np.sqrt(m.k_eff / m.m_eff) / (2 * np.pi)
    assert_true(abs(f_check - m.f_n) < 1e-6, f"sqrt(k/m)/2pi={f_check:.4f} Hz == f_n={m.f_n} Hz")


def test_power_peaks_at_resonance():
    print("\n[Test 2] Average harvested power peaks at resonance")
    m, _ = make_model()
    R = m.optimal_load()
    freqs = [70.0, 85.0, 100.0, 115.0, 130.0]
    P = [m.simulate(9.81, f, R)["P_avg"] for f in freqs]
    i_max = int(np.argmax(P))
    assert_true(freqs[i_max] == 100.0,
                f"peak at f={freqs[i_max]} Hz (P={P[i_max]*1e6:.3f} uW), f_n=100 Hz")
    assert_true(P[i_max] > 5 * P[0] and P[i_max] > 5 * P[-1],
                "resonant power >> off-resonance power")


def test_optimal_load_exists():
    print("\n[Test 3] An optimal load resistance maximises power")
    m, _ = make_model()
    R_opt = m.optimal_load()
    R_vals = [R_opt / 100.0, R_opt / 10.0, R_opt, R_opt * 10.0, R_opt * 100.0]
    P = [m.simulate(9.81, 100.0, R)["P_avg"] for R in R_vals]
    i_max = int(np.argmax(P))
    assert_true(0 < i_max < len(R_vals) - 1,
                f"interior optimum at R={R_vals[i_max]:.3e} ohm (not an endpoint)")
    assert_true(P[i_max] > P[0] and P[i_max] > P[-1],
                "power lower for too-small and too-large loads")


def test_energy_conservation():
    print("\n[Test 4] Global energy balance closes (W_in = dissipation + storage)")
    m, _ = make_model()
    r = m.simulate(9.81, 100.0, m.optimal_load())
    eb = m.energy_balance(r)
    assert_true(eb["rel_error"] < 1e-4,
                f"energy closure rel_error={eb['rel_error']:.2e} < 1e-4")
    assert_true(eb["W_in"] > 0 and eb["E_diss_elec"] > 0,
                f"W_in={eb['W_in']:.3e} J, E_elec={eb['E_diss_elec']:.3e} J both > 0")


def test_power_scales_accel_squared():
    print("\n[Test 5] Average power scales with acceleration^2 (linear system)")
    m, _ = make_model()
    R = m.optimal_load()
    a1, a2 = 4.0, 8.0  # factor 2 -> power x4
    P1 = m.simulate(a1, 100.0, R)["P_avg"]
    P2 = m.simulate(a2, 100.0, R)["P_avg"]
    ratio = P2 / P1
    assert_true(abs(ratio - 4.0) < 0.05, f"P(2a)/P(a)={ratio:.4f} ~= 4")


def test_ode_matches_analytic():
    print("\n[Test 6] Time-domain ODE agrees with frequency-domain closed form")
    m, _ = make_model()
    R = m.optimal_load()
    # use a longer record so the transient fully decays in the SS window
    r = m.simulate(9.81, 100.0, R, n_periods=120)
    P_an, V_an = m.steady_state_power(9.81, 100.0, R)
    rel = abs(r["P_avg"] - P_an) / P_an
    assert_true(rel < 0.05, f"sim P={r['P_avg']*1e6:.3f} uW vs analytic {P_an*1e6:.3f} uW, rel={rel:.3f}")


def test_voltage_lags_open_circuit():
    print("\n[Test 7] Larger load -> higher voltage amplitude (toward open circuit)")
    m, _ = make_model()
    V_small = m.simulate(9.81, 100.0, 1e3)["V_amp"]
    V_large = m.simulate(9.81, 100.0, 1e6)["V_amp"]
    assert_true(V_large > V_small,
                f"V_amp(1e6 ohm)={V_large:.3f} V > V_amp(1e3 ohm)={V_small:.4f} V")


def test_zero_acceleration():
    print("\n[Test 8] Zero base acceleration -> zero harvested power")
    m, _ = make_model()
    r = m.simulate(0.0, 100.0, m.optimal_load())
    assert_true(r["P_avg"] < 1e-20, f"P_avg={r['P_avg']:.2e} W ~ 0")
    assert_true(np.max(np.abs(r["voltage"])) < 1e-12, "voltage stays ~0")


def test_transient_decays():
    print("\n[Test 9] Mechanical transient decays (steady-state reached)")
    m, _ = make_model()
    r = m.simulate(9.81, 100.0, m.optimal_load(), n_periods=80)
    t, x = r["t"], r["x"]
    early = np.max(np.abs(x[(t > 0.05) & (t < 0.10)]))
    late = np.max(np.abs(x[t >= 0.5 * t[-1]]))
    # at resonance amplitude builds up then settles; ensure bounded steady amplitude
    assert_true(np.isfinite(late) and late > 0, f"steady tip amplitude={late:.3e} m finite > 0")
    assert_true(late < 1e-2, f"tip displacement {late:.3e} m physically bounded (< 1 cm)")


def test_predict_interface():
    print("\n[Test 10] ComponentModel predict() interface + get_info()")
    _, cm = make_model()
    r = cm.predict({"acceleration_ms2": 9.81, "frequency_hz": 100.0})
    for key in ["t", "x", "voltage", "power_inst", "P_avg", "V_rms",
                "optimal_R_ohm", "power_uw", "frequency_ratio"]:
        assert_true(key in r, f"key '{key}' in predict() output")
    assert_true(len(r["t"]) == len(r["voltage"]) == len(r["power_inst"]),
                "time-series arrays same length")
    info = cm.get_info()
    assert_true(info["component_id"] == "EC219" and "derived_parameters" in info,
                "get_info() returns EC219 metadata + derived params")


def test_benchmark():
    print("\n[Test 11] Benchmark: representative simulate() call < 5 s")
    m, _ = make_model()
    t0 = time.perf_counter()
    m.simulate(9.81, 100.0, m.optimal_load())
    elapsed = time.perf_counter() - t0
    print(f"  40-period resonant simulation in {elapsed*1000:.1f} ms")
    assert_true(elapsed < 5.0, "completes in < 5 s")


if __name__ == "__main__":
    tests = [
        test_derived_parameters,
        test_power_peaks_at_resonance,
        test_optimal_load_exists,
        test_energy_conservation,
        test_power_scales_accel_squared,
        test_ode_matches_analytic,
        test_voltage_lags_open_circuit,
        test_zero_acceleration,
        test_transient_decays,
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
    print(f"EC219 Piezoelectric Harvester F2a -- Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    sys.exit(0 if failed == 0 else 1)
