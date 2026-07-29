"""
EC167 -- Active Front End / PFC -- F2a Averaged Boost-PFC Dual-Loop
Test suite: PFC physics sanity, dual-loop regulation, conservation, edge cases.
Run with system python3 (NOT pytest):  python3 scripts/test_model.py
"""

import sys
import os
import time
import numpy as np

try:
    from scipy.integrate import trapezoid as _trapz
except ImportError:
    _trapz = np.trapz

sys.path.insert(0, os.path.dirname(__file__))
from model import BoostPFC_F2a
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
def test_near_unity_pf():
    print("\n[Test 1] Near-unity power factor (PFC works)")
    m, _ = make_model()
    res = m.simulate(duration_s=0.14)
    pf = m.power_factor(res)
    assert_true(pf > 0.99, f"PF={pf:.4f} > 0.99 (near unity)")
    assert_true(pf <= 1.0 + 1e-9, f"PF={pf:.4f} <= 1 (physical bound)")


def test_vdc_above_peak():
    print("\n[Test 2] DC-link regulated above line peak (boost)")
    m, _ = make_model()
    res = m.simulate(duration_s=0.14)
    s = m.summary(res)
    assert_true(s["v_dc_mean"] > s["V_peak"],
                f"V_dc={s['v_dc_mean']:.1f} > V_peak={s['V_peak']:.1f}")
    assert_true(abs(s["v_dc_mean"] - m.V_dc_ref) < 8.0,
                f"V_dc tracks setpoint {m.V_dc_ref}: mean={s['v_dc_mean']:.1f}")


def test_efficiency_bounds():
    print("\n[Test 3] Efficiency strictly in (0, 1)")
    m, _ = make_model()
    for P in [800.0, 1800.0, 3000.0]:
        res = m.simulate(P_load=P, duration_s=0.14)
        eta = m.efficiency(res)
        assert_true(0.0 < eta < 1.0, f"P={P:.0f}W: eta={eta:.4f} in (0,1)")


def test_low_thd():
    print("\n[Test 4] Low input current THD")
    m, _ = make_model()
    res = m.simulate(duration_s=0.14)
    thd = m.thd_current(res)
    assert_true(0.0 <= thd < 0.15, f"THD={thd*100:.2f}% < 15% (low-harmonic rectifier)")


def test_energy_conservation():
    print("\n[Test 5] Energy conservation: P_in ~= P_out + P_loss")
    m, _ = make_model()
    res = m.simulate(P_load=3000.0, duration_s=0.16)
    t, p_in = m._steady_window(res["t"], res["p_in_inst"], periods=1)
    win = t[-1] - t[0]
    P_in = _trapz(p_in, t) / win
    s = m.summary(res)
    P_out = 3000.0
    # mismatch (cap energy ripple + model averaging) should be small at steady state
    bal = P_in - (P_out + s["p_loss_w"])
    rel = abs(bal) / P_in
    assert_true(rel < 0.05, f"|P_in-(P_out+P_loss)|/P_in={rel*100:.2f}% < 5% "
                            f"(P_in={P_in:.0f}, P_out={P_out:.0f}, P_loss={s['p_loss_w']:.1f})")


def test_line_current_in_phase():
    print("\n[Test 6] Line current in phase with line voltage (cos phi ~ 1)")
    m, _ = make_model()
    res = m.simulate(duration_s=0.14)
    t, v, i = m._steady_window(res["t"], res["v_line"], res["i_line"], periods=2)
    # displacement: sign of v*i averaged should be strongly positive
    p = _trapz(v * i, t)
    neg = _trapz(np.minimum(v * i, 0.0), t)
    assert_true(p > 0, f"net real power positive (rectifier draws power): {p:.1f}")
    assert_true(abs(neg) / abs(p) < 0.05, "negligible reverse power (in-phase)")


def test_voltage_loop_recovers():
    print("\n[Test 7] Outer voltage loop pulls V_dc back to setpoint from offset")
    m, _ = make_model()
    res = m.simulate(P_load=2000.0, duration_s=0.30, v_dc0=360.0)
    s = m.summary(res)
    assert_true(s["v_dc_mean"] > 360.0,
                f"V_dc recovered upward from 360 V start: {s['v_dc_mean']:.1f}")
    assert_true(abs(s["v_dc_mean"] - m.V_dc_ref) < 10.0,
                f"V_dc near setpoint after recovery: {s['v_dc_mean']:.1f}")


def test_higher_load_more_current():
    print("\n[Test 8] Higher load -> larger line current amplitude")
    m, _ = make_model()
    r1 = m.simulate(P_load=1000.0, duration_s=0.14)
    r2 = m.simulate(P_load=3000.0, duration_s=0.14)
    a1 = np.max(np.abs(r1["i_line"][-1000:]))
    a2 = np.max(np.abs(r2["i_line"][-1000:]))
    assert_true(a2 > a1, f"I_line peak grows with load: {a2:.2f} > {a1:.2f} A")


def test_duty_bounded():
    print("\n[Test 9] Inductor current finite & DC-link bounded (stable CCM)")
    m, _ = make_model()
    res = m.simulate(duration_s=0.16)
    assert_true(np.all(np.isfinite(res["i_L"])), "i_L finite")
    assert_true(np.all(res["v_dc"] > 0), "v_dc stays positive")
    assert_true(np.max(res["v_dc"]) < 600.0, f"v_dc bounded < 600 V: {np.max(res['v_dc']):.1f}")


def test_universal_input():
    print("\n[Test 10] Universal input: regulates across 120-265 Vac")
    m, _ = make_model()
    for V in [120.0, 230.0, 265.0]:
        res = m.simulate(V_line_rms=V, P_load=1500.0, duration_s=0.16)
        s = m.summary(res)
        assert_true(s["v_dc_mean"] > s["V_peak"],
                    f"Vrms={V}: V_dc={s['v_dc_mean']:.1f} > V_peak={s['V_peak']:.1f}")
        assert_true(s["power_factor"] > 0.98, f"Vrms={V}: PF={s['power_factor']:.4f} > 0.98")


def test_predict_interface():
    print("\n[Test 11] ComponentModel predict() interface")
    _, cm = make_model()
    r = cm.predict({"P_load": 2500.0, "duration_s": 0.12})
    for key in ["power_factor", "thd_current", "efficiency",
                "v_dc_mean", "V_peak", "p_loss_w", "waveforms"]:
        assert_true(key in r, f"Key '{key}' in output")
    wf = r["waveforms"]
    assert_true(len(wf["t"]) == len(wf["v_dc"]), "Waveform arrays same length")
    info = cm.get_info()
    assert_true(info["component_id"] == "EC167", "get_info reports EC167")


def test_benchmark():
    print("\n[Test 12] Benchmark: averaged sim timing")
    m, _ = make_model()
    t0 = time.perf_counter()
    res = m.simulate(duration_s=0.12, n_points=4000)
    m.summary(res)
    elapsed = time.perf_counter() - t0
    print(f"  0.12 s sim + KPIs in {elapsed*1000:.1f} ms")
    assert_true(elapsed < 5.0, "Completes in < 5 s")


if __name__ == "__main__":
    tests = [
        test_near_unity_pf,
        test_vdc_above_peak,
        test_efficiency_bounds,
        test_low_thd,
        test_energy_conservation,
        test_line_current_in_phase,
        test_voltage_loop_recovers,
        test_higher_load_more_current,
        test_duty_bounded,
        test_universal_input,
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
    print(f"EC167 Boost-PFC F2a -- Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    sys.exit(0 if failed == 0 else 1)
