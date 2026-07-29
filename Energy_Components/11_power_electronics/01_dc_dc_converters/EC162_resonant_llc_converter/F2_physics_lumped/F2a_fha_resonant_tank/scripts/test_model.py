"""
EC162 -- Resonant LLC Converter -- F2a Physics-Lumped (FHA)
Test suite: gain-curve behavior, ZVS, energy conservation, efficiency bounds,
ODE settling, predict() interface, benchmark timing. Custom harness (no pytest).
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from model import LLCConverterF2a
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
def test_resonant_frequency():
    print("\n[Test 1] Derived resonant frequency matches 1/(2pi sqrt(LrCr))")
    m, _ = make_model()
    f_expect = 1.0 / (2.0 * np.pi * np.sqrt(m.L_r * m.C_r))
    assert_true(abs(m.f_r - f_expect) < 1.0, f"f_r={m.f_r/1e3:.2f} kHz")
    assert_true(abs(m.f_r - 1.0e5) / 1.0e5 < 0.02, "f_r ~ 100 kHz as designed")


def test_unity_gain_at_resonance():
    print("\n[Test 2] Gain M = 1 at resonance, load-independent")
    m, _ = make_model()
    for r in [0.05, 0.072, 0.2, 1.0, 5.0]:
        M = m.gain_from_load(1.0, r)
        assert_true(abs(M - 1.0) < 1e-9, f"M(fn=1, R={r})={M:.6f}")


def test_gain_curve_peak_near_resonance():
    print("\n[Test 3] Gain curve peaks at/below resonance, bucks above")
    m, _ = make_model()
    r = 0.072
    fn = np.linspace(0.4, 2.0, 400)
    M = np.array([m.gain_from_load(f, r) for f in fn])
    fn_peak = fn[np.argmax(M)]
    assert_true(fn_peak <= 1.0 + 1e-6, f"peak at fn={fn_peak:.3f} <= 1 (boost region)")
    assert_true(M[np.argmin(np.abs(fn - 1.5))] < 1.0, "M(fn=1.5) < 1 (buck region)")
    assert_true(M.max() > 1.0, f"peak gain {M.max():.3f} > 1 (boost capable)")


def test_gain_monotone_above_resonance():
    print("\n[Test 4] Gain decreases monotonically for fn > 1")
    m, _ = make_model()
    r = 0.072
    fn = np.linspace(1.0, 2.5, 60)
    M = np.array([m.gain_from_load(f, r) for f in fn])
    diffs = np.diff(M)
    assert_true(np.all(diffs <= 1e-9), "M strictly non-increasing for fn>=1")


def test_zvs_above_resonance():
    print("\n[Test 5] ZVS holds above resonance (inductive tank)")
    m, _ = make_model()
    r = 0.072
    assert_true(m.is_zvs(1.05, r), "ZVS at fn=1.05")
    assert_true(m.is_zvs(1.3, r), "ZVS at fn=1.3")
    # Below the gain peak the tank turns capacitive (no ZVS)
    assert_true(not m.is_zvs(0.40, r), "No ZVS at fn=0.40 (capacitive region, below gain peak)")


def test_output_voltage_scaling():
    print("\n[Test 6] V_out = M*(V_in/2)/n; doubles with V_in at fixed fn")
    m, _ = make_model()
    v1 = m.output_voltage(1.0, 0.072, 400.0)
    v2 = m.output_voltage(1.0, 0.072, 800.0)
    assert_true(abs(v1 - 400.0 / 2.0 / m.n) < 1e-9, f"V_out={v1:.3f} V at resonance")
    assert_true(abs(v2 - 2.0 * v1) < 1e-9, "V_out scales linearly with V_in")


def test_efficiency_bounds():
    print("\n[Test 7] Efficiency strictly in (0,1) and high near rated")
    m, _ = make_model()
    for r in [0.05, 0.072, 0.2, 1.0]:
        eta = m.efficiency(1.0, r)
        assert_true(0.0 < eta < 1.0, f"eta(R={r})={eta:.4f} in (0,1)")
    eta_rated = m.efficiency(1.0, 0.072)
    assert_true(eta_rated > 0.90, f"eta at rated = {eta_rated*100:.2f}% > 90% (soft-switching)")


def test_energy_conservation():
    print("\n[Test 8] Energy conservation: P_in = P_out + P_loss")
    m, _ = make_model()
    r = 0.072
    op = m.operating_point(1.0, r)
    p_loss = m.loss_breakdown(1.0, r)["p_total_w"]
    p_in = op["p_out"] + p_loss
    eta = m.efficiency(1.0, r)
    assert_true(abs(eta - op["p_out"] / p_in) < 1e-9, "eta == P_out/(P_out+P_loss)")
    assert_true(p_loss > 0, f"losses positive ({p_loss:.2f} W)")


def test_ode_settles_to_steady_state():
    print("\n[Test 9] Output-filter ODE settles to FHA steady state")
    m, _ = make_model()
    r = 0.072
    sim = m.simulate(1.0, r, v_out0=0.0, dt=2e-6, duration_s=3e-3)
    v_final = sim["v_out"][-1]
    v_ss = sim["v_out_ss"]
    assert_true(sim["v_out"][0] < v_final, "Output ramps up from cold start")
    assert_true(abs(v_final - v_ss) / v_ss < 0.02, f"settles to V_ss: {v_final:.3f} vs {v_ss:.3f}")
    assert_true(np.all(sim["v_out"] >= -1e-9), "V_out never negative")


def test_ode_monotone_charge():
    print("\n[Test 10] Output cap charges monotonically toward target (no overshoot)")
    m, _ = make_model()
    sim = m.simulate(1.0, 0.072, v_out0=0.0, dt=2e-6, duration_s=2e-3)
    v = sim["v_out"]
    assert_true(np.all(np.diff(v) >= -1e-6), "Monotone non-decreasing charge")
    assert_true(v[-1] <= sim["v_out_ss"] + 1e-6, "No overshoot past steady state")


def test_predict_interface():
    print("\n[Test 11] ComponentModel predict() interface")
    _, cm = make_model()
    r = cm.predict({"f_sw_Hz": 100000.0, "duration_s": 2e-3, "dt": 4e-6})
    for key in ["t", "v_out", "i_load", "p_out", "v_out_ss", "gain",
                "efficiency", "zvs", "operating_point", "losses"]:
        assert_true(key in r, f"Key '{key}' in output")
    assert_true(len(r["t"]) == len(r["v_out"]), "Arrays same length")
    info = cm.get_info()
    assert_true(info["component_id"] == "EC162", "component_id EC162")


def test_benchmark():
    print("\n[Test 12] Benchmark: 3 ms transient at dt=1 us")
    m, _ = make_model()
    t0 = time.perf_counter()
    m.simulate(1.0, 0.072, dt=1e-6, duration_s=3e-3)
    elapsed = time.perf_counter() - t0
    print(f"  3 ms transient in {elapsed*1000:.1f} ms")
    assert_true(elapsed < 5.0, "Completes in < 5 s")


if __name__ == "__main__":
    tests = [
        test_resonant_frequency,
        test_unity_gain_at_resonance,
        test_gain_curve_peak_near_resonance,
        test_gain_monotone_above_resonance,
        test_zvs_above_resonance,
        test_output_voltage_scaling,
        test_efficiency_bounds,
        test_energy_conservation,
        test_ode_settles_to_steady_state,
        test_ode_monotone_charge,
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
    print(f"EC162 LLC F2a (FHA) -- Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    sys.exit(0 if failed == 0 else 1)
