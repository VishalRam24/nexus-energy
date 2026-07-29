"""
EC166 -- AC-DC Rectifier (Diode Bridge) -- F2a Averaged Cap-Filter
Test suite: physics sanity, conservation, known limits, edge cases, benchmark.
Run:  python3 scripts/test_model.py   (system python3, NOT pytest)
"""

import sys
import os
import time
import json
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from model import DiodeBridgeRectifierF2a
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


def make_1phase():
    with open(os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")) as f:
        p = json.load(f)
    p["unit"]["n_phases"]["value"] = 1
    return DiodeBridgeRectifierF2a(p)


# ---------------------------------------------------------------------------
def test_vdc_relation_3phase():
    print("\n[Test 1] 3-phase ideal Vdc = 1.3505 * V_LL (Mohan 5-68)")
    m, _ = make_model()
    vdc = m.ideal_dc_voltage(400.0)
    assert_true(abs(vdc - 1.3505 * 400.0) < 1.0, f"Vdc_ideal={vdc:.2f} ~ 540.2 V")


def test_vdc_relation_1phase():
    print("\n[Test 2] 1-phase ideal Vdc = 0.9003 * V_rms (Mohan 5-9)")
    m1 = make_1phase()
    vdc = m1.ideal_dc_voltage(230.0)
    assert_true(abs(vdc - 0.9003 * 230.0) < 1.0, f"Vdc_ideal={vdc:.2f} ~ 207.1 V")


def test_vdc_mean_near_peak():
    print("\n[Test 3] With cap filter, V_dc_mean sits between Vdc_ideal and V_peak")
    m, _ = make_model()
    r = m.simulate(400.0, R_load=8.0, dt=2e-5, duration_s=0.12)
    vpk = float(m.peak_voltage(400.0))
    assert_true(r["v_dc_mean"] < vpk + 1.0, f"V_dc_mean={r['v_dc_mean']:.1f} <= V_peak={vpk:.1f}")
    assert_true(r["v_dc_mean"] > r["v_dc_ideal"] * 0.9,
                f"V_dc_mean={r['v_dc_mean']:.1f} > 0.9*Vdc_ideal={0.9*r['v_dc_ideal']:.1f}")


def test_efficiency_bounds():
    print("\n[Test 4] Efficiency strictly in (0, 1)")
    m, _ = make_model()
    for R in [4.0, 8.0, 20.0]:
        r = m.simulate(400.0, R_load=R, dt=2e-5, duration_s=0.1)
        assert_true(0.0 < r["efficiency"] < 1.0,
                    f"R={R}: eff={r['efficiency']:.4f} in (0,1)")


def test_energy_conservation():
    print("\n[Test 5] Energy balance: P_in = P_out + P_cond (within 2%)")
    m, _ = make_model()
    r = m.simulate(400.0, R_load=8.0, dt=2e-5, duration_s=0.12)
    p_in = r["p_out_w"] + r["p_cond_w"]
    recon_eff = r["p_out_w"] / p_in
    assert_true(abs(recon_eff - r["efficiency"]) < 0.02,
                f"eff reconstructs: {recon_eff:.4f} ~ {r['efficiency']:.4f}")
    assert_true(r["p_cond_w"] > 0, f"Conduction loss positive: {r['p_cond_w']:.1f} W")


def test_ripple_decreases_with_C():
    print("\n[Test 6] DC ripple decreases as output capacitance increases")
    with open(os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")) as f:
        base = json.load(f)
    ripples = []
    for C in [1e-3, 4.7e-3, 2e-2]:
        p = json.loads(json.dumps(base))
        p["unit"]["C_out"]["value"] = C
        mm = DiodeBridgeRectifierF2a(p)
        r = mm.simulate(400.0, R_load=8.0, dt=2e-5, duration_s=0.14)
        ripples.append(r["v_ripple_pp"])
        print(f"    C={C*1e6:.0f}uF -> ripple_pp={r['v_ripple_pp']:.2f} V")
    assert_true(ripples[0] > ripples[1] > ripples[2],
                f"ripple monotone decreasing: {ripples}")


def test_ripple_approx_formula():
    print("\n[Test 7] Ripple ~ I_dc/(f_pulse*C) order-of-magnitude (Rashid)")
    m, _ = make_model()
    r = m.simulate(400.0, R_load=8.0, dt=2e-5, duration_s=0.14)
    f_pulse = m._pulses * m.f_line              # 6*50 = 300 Hz ripple
    approx = r["i_dc_mean"] / (f_pulse * m.C_out)
    # within factor of 3 of the textbook estimate
    assert_true(0.3 * approx < r["v_ripple_pp"] < 3.0 * approx,
                f"ripple={r['v_ripple_pp']:.2f} V vs Idc/(f*C)={approx:.2f} V")


def test_low_power_factor():
    print("\n[Test 8] Capacitor-input bridge has poor input PF (< 0.85)")
    m, _ = make_model()
    r = m.simulate(400.0, R_load=8.0, dt=2e-5, duration_s=0.14)
    assert_true(0.0 < r["power_factor"] < 0.85,
                f"PF={r['power_factor']:.3f} (poor, < 0.85)")


def test_overlap_drop_monotone():
    print("\n[Test 9] Commutation overlap drop grows with load current (Mohan 5-75)")
    m, _ = make_model()
    d_lo = m.overlap_drop(10.0)
    d_hi = m.overlap_drop(60.0)
    assert_true(d_hi > d_lo > 0, f"overlap: {d_lo:.3f} V (10A) < {d_hi:.3f} V (60A)")


def test_higher_load_lowers_vdc():
    print("\n[Test 10] Heavier load (smaller R) lowers mean DC voltage")
    m, _ = make_model()
    r_light = m.simulate(400.0, R_load=40.0, dt=2e-5, duration_s=0.12)
    r_heavy = m.simulate(400.0, R_load=4.0, dt=2e-5, duration_s=0.12)
    assert_true(r_heavy["v_dc_mean"] < r_light["v_dc_mean"],
                f"V_dc heavy={r_heavy['v_dc_mean']:.1f} < light={r_light['v_dc_mean']:.1f}")


def test_predict_interface():
    print("\n[Test 11] ComponentModel predict() interface")
    _, cm = make_model()
    r = cm.predict({"v_ac_rms": 400.0, "R_load": 8.0, "duration_s": 0.06})
    for key in ["t", "v_dc", "i_load", "i_diode", "v_dc_mean",
                "v_ripple_pp", "efficiency", "power_factor", "v_dc_ideal"]:
        assert_true(key in r, f"Key '{key}' in output")
    assert_true(len(r["t"]) == len(r["v_dc"]), "Time/voltage arrays same length")
    info = cm.get_info()
    assert_true(info["component_id"] == "EC166", "get_info component_id == EC166")


def test_benchmark():
    print("\n[Test 12] Benchmark: 0.1 s (5 cycles) sim")
    m, _ = make_model()
    t0 = time.perf_counter()
    m.simulate(400.0, R_load=8.0, dt=2e-5, duration_s=0.1)
    elapsed = time.perf_counter() - t0
    print(f"  0.1 s simulation in {elapsed*1000:.1f} ms")
    assert_true(elapsed < 5.0, "Completes in < 5 s")


if __name__ == "__main__":
    tests = [
        test_vdc_relation_3phase,
        test_vdc_relation_1phase,
        test_vdc_mean_near_peak,
        test_efficiency_bounds,
        test_energy_conservation,
        test_ripple_decreases_with_C,
        test_ripple_approx_formula,
        test_low_power_factor,
        test_overlap_drop_monotone,
        test_higher_load_lowers_vdc,
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
    print(f"EC166 Diode Bridge F2a -- Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    sys.exit(0 if failed == 0 else 1)
