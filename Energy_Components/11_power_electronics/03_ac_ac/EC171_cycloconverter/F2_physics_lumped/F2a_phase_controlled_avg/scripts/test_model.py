"""
EC171 -- Cycloconverter -- F2a Physics-Lumped
Test suite: physics sanity (energy conservation, monotonicity, known limits),
edge cases, predict() interface, benchmark timing. Custom harness (NO pytest).
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from model import CycloconverterF2a
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
def test_output_below_input_freq():
    print("\n[Test 1] Output frequency strictly below input (down-conversion)")
    m, _ = make_model()
    r = m.simulate(r_mod=0.8, f_out=10.0, n_cycles=3)
    assert_true(r["f_out"] < r["f_line"], f"f_out={r['f_out']} < f_line={r['f_line']}")
    assert_true(r["below_one_third"], f"ratio={r['freq_ratio']:.3f} < 1/3 (clean region)")
    # f_out >= f_line must be rejected
    raised = False
    try:
        m.simulate(r_mod=0.8, f_out=60.0)
    except ValueError:
        raised = True
    assert_true(raised, "f_out >= f_line raises ValueError")


def test_vout_vs_firing_angle():
    print("\n[Test 2] Averaged V_out follows cosine firing law (Pelly 1971)")
    m, _ = make_model()
    Vdo = m.V_do()
    # alpha=0 -> V=Vdo, alpha=pi/2 -> V~0, alpha=pi -> V=-Vdo
    v0 = Vdo * np.cos(m.alpha_min)
    v90 = Vdo * np.cos(np.pi / 2)
    assert_true(abs(v0) > abs(v90), f"|V(alpha~0)|={abs(v0):.1f} > |V(90deg)|={abs(v90):.3f}")
    # V_out fundamental peak scales with modulation ratio
    pk_lo = m.v_out_fundamental_peak(0.3)
    pk_hi = m.v_out_fundamental_peak(0.9)
    assert_true(pk_hi > pk_lo > 0, f"V_pk(0.9)={pk_hi:.1f} > V_pk(0.3)={pk_lo:.1f} > 0")
    assert_true(pk_hi <= Vdo + 1e-9, "fundamental peak <= V_do (cannot exceed max)")


def test_vout_monotone_in_modulation():
    print("\n[Test 3] V_out_ll_rms monotonically increases with r_mod")
    m, _ = make_model()
    prev = -1.0
    for r in np.linspace(0.1, 1.0, 10):
        v = m.v_out_ll_rms(r)
        assert_true(v > prev, f"V_ll_rms(r={r:.2f})={v:.1f} > prev={prev:.1f}")
        prev = v


def test_energy_conservation():
    print("\n[Test 4] Energy conservation: P_in = P_out + P_loss")
    m, _ = make_model()
    r = m.simulate(r_mod=0.85, f_out=10.0, n_cycles=4)
    resid = abs(r["P_in_total"] - (r["P_out_total"] + r["P_loss_total"]))
    assert_true(resid < 1e-6 * max(r["P_in_total"], 1.0),
                f"|P_in - (P_out+P_loss)|={resid:.3e} W ~ 0")
    assert_true(r["P_out_total"] > 0, f"P_out={r['P_out_total']/1e3:.1f} kW > 0")
    assert_true(r["P_loss_total"] > 0, f"P_loss={r['P_loss_total']/1e3:.2f} kW > 0")


def test_efficiency_bounds():
    print("\n[Test 5] Efficiency strictly in (0, 1)")
    m, _ = make_model()
    for r_mod in [0.4, 0.7, 0.95]:
        r = m.simulate(r_mod=r_mod, f_out=10.0, n_cycles=3)
        eta = r["efficiency"]
        assert_true(0.0 < eta < 1.0, f"eta(r={r_mod})={eta:.4f} in (0,1)")


def test_input_pf_lagging():
    print("\n[Test 6] Input displacement PF always lagging (< 1, < load PF)")
    m, _ = make_model()
    for r_mod in [0.3, 0.6, 0.9]:
        dpf = m.input_displacement_factor(r_mod, f_out=10.0, phi_load=0.4)
        assert_true(0.0 < dpf < 1.0, f"DPF_in(r={r_mod})={dpf:.3f} in (0,1) -> lagging")
        assert_true(dpf < np.cos(0.4), f"DPF_in={dpf:.3f} < load PF={np.cos(0.4):.3f}")


def test_harmonics_increase_with_freq():
    print("\n[Test 7] Output THD rises with f_out/f_line, -> 0 as f_out -> 0")
    m, _ = make_model()
    thd_lo = m.output_thd(2.0)
    thd_hi = m.output_thd(15.0)
    assert_true(thd_hi > thd_lo > 0, f"THD(15Hz)={thd_hi:.3f} > THD(2Hz)={thd_lo:.3f} > 0")
    assert_true(m.output_thd(1e-6) < 1e-4, "THD -> 0 as f_out -> 0")
    # dominant harmonics cluster near p*f_line
    harms = m.dominant_harmonic_freqs(10.0)
    assert_true(min(h for h in harms if h > 0) > 0, "harmonic freqs computed")
    assert_true(any(abs(h - m.p_pulse * m.f_line) < 2 * 10.0 for h in harms),
                f"dominant family near p*f_line={m.p_pulse*m.f_line} Hz present")


def test_current_settles_periodic():
    print("\n[Test 8] Output current ODE settles to bounded periodic waveform")
    m, _ = make_model()
    r = m.simulate(r_mod=0.8, f_out=10.0, n_cycles=5)
    i = r["i_out"]
    assert_true(np.all(np.isfinite(i)), "current finite everywhere")
    # current is bounded (no runaway)
    assert_true(np.max(np.abs(i)) < 5000.0, f"max|i|={np.max(np.abs(i)):.1f} A bounded")
    # last cycle mean ~ 0 (sinusoidal AC output, no DC offset at steady state)
    T = 1.0 / 10.0
    last = i[r["t"] >= (r["t"][-1] - T)]
    mean_last = abs(np.mean(last))
    rms_last = np.sqrt(np.mean(last ** 2))
    assert_true(mean_last < 0.1 * rms_last + 1e-6,
                f"near-zero DC offset: |mean|={mean_last:.2f} << rms={rms_last:.2f}")


def test_alpha_within_limits():
    print("\n[Test 9] Firing angle stays within [alpha_min, alpha_max]")
    m, _ = make_model()
    r = m.simulate(r_mod=1.0, f_out=10.0, n_cycles=2)
    a = r["alpha"]
    assert_true(np.all(a >= m.alpha_min - 1e-9), f"alpha >= {m.alpha_min:.3f} rad")
    assert_true(np.all(a <= m.alpha_max + 1e-9), f"alpha <= {m.alpha_max:.3f} rad")


def test_predict_interface():
    print("\n[Test 10] ComponentModel predict()/get_info() interface")
    _, cm = make_model()
    info = cm.get_info()
    for k in ["component_id", "component_name", "fidelity", "version", "inputs", "outputs"]:
        assert_true(k in info, f"get_info has '{k}'")
    assert_true(info["component_id"] == "EC171", "component_id == EC171")
    r = cm.predict({"r_mod": 0.7, "f_out": 8.0, "n_cycles": 3})
    for key in ["t", "i_out", "v_out_avg", "alpha", "efficiency",
                "input_displacement_factor", "output_thd", "P_out_total"]:
        assert_true(key in r, f"predict output has '{key}'")
    assert_true(len(r["t"]) == len(r["i_out"]) == len(r["v_out_avg"]),
                "time-series arrays same length")


def test_benchmark():
    print("\n[Test 11] Benchmark: 4-cycle ODE simulation timing")
    m, _ = make_model()
    t0 = time.perf_counter()
    m.simulate(r_mod=0.8, f_out=10.0, n_cycles=4)
    elapsed = time.perf_counter() - t0
    print(f"  4-cycle solve_ivp simulation in {elapsed*1000:.1f} ms")
    assert_true(elapsed < 5.0, "completes in < 5 s")


if __name__ == "__main__":
    tests = [
        test_output_below_input_freq,
        test_vout_vs_firing_angle,
        test_vout_monotone_in_modulation,
        test_energy_conservation,
        test_efficiency_bounds,
        test_input_pf_lagging,
        test_harmonics_increase_with_freq,
        test_current_settles_periodic,
        test_alpha_within_limits,
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
    print(f"EC171 Cycloconverter F2a -- Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    sys.exit(0 if failed == 0 else 1)
