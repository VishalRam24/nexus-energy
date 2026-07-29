"""
EC163 -- Single-Phase DC-AC Inverter -- F2a Averaged SPWM + LC Filter
Test suite: physics sanity, ODE filter dynamics, energy balance, edge cases.
Run with system python3 (NOT pytest):  python3 scripts/test_model.py
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from model import SinglePhaseInverterF2a
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
def test_vac_scales_with_ma_vdc():
    print("\n[Test 1] Fundamental V_ac scales with m_a * V_dc")
    m, _ = make_model()
    # Linearity in m_a
    v1 = m.fundamental_rms(400.0, 0.4)
    v2 = m.fundamental_rms(400.0, 0.8)
    assert_true(abs(v2 - 2.0 * v1) < 1e-9, f"V_rms(0.8)=2*V_rms(0.4): {v2:.2f}=={2*v1:.2f}")
    # Linearity in V_dc
    vA = m.fundamental_rms(200.0, 0.85)
    vB = m.fundamental_rms(800.0, 0.85)
    assert_true(abs(vB - 4.0 * vA) < 1e-9, f"V_rms(800V)=4*V_rms(200V): {vB:.2f}=={4*vA:.2f}")
    # Exact formula m_a*Vdc/sqrt(2)
    assert_true(abs(m.fundamental_rms(400.0, 1.0) - 400.0/np.sqrt(2)) < 1e-9,
                "V_fund_rms(m=1) = Vdc/sqrt(2)")


def test_output_tracks_reference():
    print("\n[Test 2] Filtered output RMS tracks ideal fundamental")
    m, _ = make_model()
    s = m.simulate(m_a=0.85, duration_s=0.14)
    ratio = s["v_out_rms"] / s["v_fund_rms_ideal"]
    assert_true(0.8 < ratio < 1.2,
                f"V_out_rms/V_fund = {ratio:.3f} within +-20% of reference")


def test_filter_attenuates_harmonics():
    print("\n[Test 3] LC filter attenuates switching harmonics, passes fundamental")
    m, _ = make_model()
    a_fund = float(m.filter_attenuation(m.f_grid))     # at 50 Hz
    a_sw = float(m.filter_attenuation(m.f_sw))          # at 10 kHz
    assert_true(a_fund > 0.9, f"Fundamental passed: |H(50Hz)|={a_fund:.3f} ~ 1")
    assert_true(a_sw < 0.05, f"Switching attenuated: |H(10kHz)|={a_sw:.4f} << 1")
    assert_true(a_sw < a_fund, "High-freq attenuation < fundamental gain")


def test_lc_corner_between_grid_and_sw():
    print("\n[Test 4] LC corner frequency sits between f_grid and f_sw")
    m, _ = make_model()
    f_lc = m.lc_corner_frequency()
    assert_true(m.f_grid < f_lc < m.f_sw,
                f"f_grid({m.f_grid}) < f_LC({f_lc:.0f}) < f_sw({m.f_sw})")


def test_thd_postfilter_lower():
    print("\n[Test 5] Post-filter THD << pre-filter THD")
    m, _ = make_model()
    thd_pre = m.thd_prefilter(0.85)
    thd_post = m.thd_postfilter(0.85)
    assert_true(thd_post < thd_pre, f"THD_post({thd_post:.4f}) < THD_pre({thd_pre:.3f})")
    assert_true(thd_post < 0.05, f"Output THD={thd_post*100:.2f}% < 5% (good filtered SPWM)")


def test_efficiency_range():
    print("\n[Test 6] Efficiency strictly in (0, 1)")
    m, _ = make_model()
    for m_a in [0.4, 0.7, 0.95]:
        s = m.simulate(m_a=m_a, duration_s=0.12)
        eta = m.efficiency(400.0, s["p_out_w"], s["i_out_rms"], m_a)
        assert_true(0.0 < eta < 1.0, f"eta(m_a={m_a})={eta:.4f} in (0,1)")


def test_energy_conservation():
    print("\n[Test 7] Energy conservation: P_in = P_out + P_loss")
    m, _ = make_model()
    s = m.simulate(m_a=0.85, duration_s=0.12)
    loss = m.losses(400.0, s["i_out_rms"], 0.85)["p_loss_total_w"]
    p_in = s["p_out_w"] + loss
    eta = m.efficiency(400.0, s["p_out_w"], s["i_out_rms"], 0.85)
    assert_true(abs(eta - s["p_out_w"] / p_in) < 1e-9, "eta = P_out/(P_out+P_loss)")
    assert_true(loss >= 0, f"Losses non-negative: {loss:.1f} W")
    assert_true(p_in > s["p_out_w"], f"P_in({p_in:.0f}) > P_out({s['p_out_w']:.0f})")


def test_losses_split():
    print("\n[Test 8] Loss breakdown: conduction + switching = total, both > 0")
    m, _ = make_model()
    s = m.simulate(m_a=0.85, duration_s=0.12)
    L = m.losses(400.0, s["i_out_rms"], 0.85)
    assert_true(L["p_conduction_w"] > 0, f"P_cond={L['p_conduction_w']:.1f} W > 0")
    assert_true(L["p_switching_w"] > 0, f"P_sw={L['p_switching_w']:.1f} W > 0")
    assert_true(abs(L["p_conduction_w"] + L["p_switching_w"] - L["p_loss_total_w"]) < 1e-6,
                "Conduction + switching = total")


def test_higher_load_more_power_more_loss():
    print("\n[Test 9] Heavier load (lower R) -> more output power and more loss")
    m, _ = make_model()
    s_light = m.operating_point(m_a=0.85, R_load=40.0, duration_s=0.12)
    s_heavy = m.operating_point(m_a=0.85, R_load=8.0, duration_s=0.12)
    assert_true(s_heavy["p_out_w"] > s_light["p_out_w"],
                f"P_out heavy({s_heavy['p_out_w']:.0f}) > light({s_light['p_out_w']:.0f})")
    assert_true(s_heavy["p_loss_total_w"] > s_light["p_loss_total_w"],
                "Heavier load -> larger conduction loss")


def test_zero_modulation_zero_output():
    print("\n[Test 10] m_a = 0 -> zero output voltage and power")
    m, _ = make_model()
    s = m.simulate(m_a=0.0, duration_s=0.10)
    assert_true(s["v_out_rms"] < 1e-6, f"V_out_rms={s['v_out_rms']:.2e} ~ 0")
    assert_true(abs(s["p_out_w"]) < 1e-6, f"P_out={s['p_out_w']:.2e} ~ 0")


def test_predict_interface():
    print("\n[Test 11] ComponentModel predict() interface keys present")
    _, cm = make_model()
    r = cm.predict({"m_a": 0.8, "duration_s": 0.10})
    for key in ["t", "v_inv", "i_L", "v_out", "v_out_rms", "i_out_rms",
                "p_out_w", "p_loss_total_w", "efficiency",
                "thd_postfilter", "f_lc_hz"]:
        assert_true(key in r, f"Key '{key}' in output")
    assert_true(len(r["t"]) == len(r["v_out"]), "Time-series arrays same length")


def test_benchmark():
    print("\n[Test 12] Benchmark: 6-cycle LC-filter ODE solve")
    m, _ = make_model()
    t0 = time.perf_counter()
    m.simulate(m_a=0.85, duration_s=0.12)
    elapsed = time.perf_counter() - t0
    print(f"  120 ms simulation solved in {elapsed*1000:.1f} ms wall")
    assert_true(elapsed < 5.0, "Completes in < 5 s")


if __name__ == "__main__":
    tests = [
        test_vac_scales_with_ma_vdc,
        test_output_tracks_reference,
        test_filter_attenuates_harmonics,
        test_lc_corner_between_grid_and_sw,
        test_thd_postfilter_lower,
        test_efficiency_range,
        test_energy_conservation,
        test_losses_split,
        test_higher_load_more_power_more_loss,
        test_zero_modulation_zero_output,
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
    print(f"EC163 Single-Phase Inverter F2a -- Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    sys.exit(0 if failed == 0 else 1)
