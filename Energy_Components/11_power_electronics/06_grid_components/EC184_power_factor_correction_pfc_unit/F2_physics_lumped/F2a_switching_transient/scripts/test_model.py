"""
EC184 -- Power Factor Correction Unit -- F2a Physics-Lumped
Test suite: compensation algebra, reactive/energy balance, resonance,
RLC energization ODE, edge cases, predict() interface, benchmark timing.
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from model import PFCUnit_F2a
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
def test_compensation_formula():
    print("\n[Test 1] Qc = P*(tan phi1 - tan phi2) exact")
    m, _ = make_model()
    P, pf1, pf2 = 800.0, 0.80, 0.95
    r = m.compensate(P, pf1, pf2)
    phi1 = np.arccos(pf1); phi2 = np.arccos(pf2)
    Qc_expected = P * (np.tan(phi1) - np.tan(phi2))
    assert_true(abs(r["Q_required_kVAR"] - Qc_expected) < 1e-6,
                f"Qc_req={r['Q_required_kVAR']:.3f} == {Qc_expected:.3f}")


def test_pf_improves_to_target():
    print("\n[Test 2] PF improves toward target (Qc unclamped)")
    m, _ = make_model()
    for pf1 in [0.70, 0.80, 0.85]:
        r = m.compensate(500.0, pf1, 0.95)
        assert_true(r["pf_achieved"] > pf1 - 1e-9,
                    f"PF {pf1:.2f} -> {r['pf_achieved']:.4f} (improved)")
        assert_true(abs(r["pf_achieved"] - 0.95) < 1e-3,
                    f"reaches target 0.95 (got {r['pf_achieved']:.4f})")


def test_reactive_balance():
    print("\n[Test 3] Reactive balance: Q_residual = Q_load - Qc")
    m, _ = make_model()
    r = m.compensate(800.0, 0.80, 0.95)
    bal = r["Q_load_kVAR"] - r["Q_compensated_kVAR"]
    assert_true(abs(bal - r["Q_residual_kVAR"]) < 1e-9,
                f"Q_load-Qc={bal:.4f} == Q_residual={r['Q_residual_kVAR']:.4f}")


def test_released_capacity_positive():
    print("\n[Test 4] Released capacity S1 - S2 >= 0 and matches kVA")
    m, _ = make_model()
    r = m.compensate(800.0, 0.75, 0.98)
    assert_true(r["released_capacity_kVA"] > 0,
                f"released={r['released_capacity_kVA']:.1f} kVA > 0")
    expect = r["S_before_kVA"] - r["S_after_kVA"]
    assert_true(abs(r["released_capacity_kVA"] - expect) < 1e-6, "matches S1-S2")
    assert_true(r["S_after_kVA"] <= r["S_before_kVA"] + 1e-9,
                "apparent power reduced after compensation")


def test_capacitance_sizing():
    print("\n[Test 5] C = Q/(2 pi f V^2) sizing consistent")
    m, _ = make_model()
    C = m.capacitance_for_Q(1000.0)
    V = m.V_rated_kV * 1e3
    Q_back = m.w * C * V ** 2 / 1e3   # kVAR
    assert_true(abs(Q_back - 1000.0) < 1e-3, f"Q recovered={Q_back:.3f} == 1000")
    assert_true(C > 0, f"C={C*1e6:.2f} uF > 0")


def test_resonance_frequency():
    print("\n[Test 6] Parallel resonance h = sqrt(Xc/Xsys); detuned h=1/sqrt(p)")
    m, _ = make_model()
    res = m.resonance(1000.0, detuning_pct=7.0)
    # cross-check h_parallel against sqrt(Ssc/Qc)
    h_alt = np.sqrt(res["Ssc_MVA"] * 1e3 / 1000.0)
    assert_true(abs(res["h_parallel"] - h_alt) < 1e-6,
                f"h_par={res['h_parallel']:.3f} == sqrt(Ssc/Qc)={h_alt:.3f}")
    h_tune_expected = 1.0 / np.sqrt(0.07)
    assert_true(abs(res["h_tune"] - h_tune_expected) < 1e-6,
                f"h_tune={res['h_tune']:.3f} == {h_tune_expected:.3f} (~3.78)")
    assert_true(res["f_tune_Hz"] < 250.0 + 1e-6,
                f"detuned below 5th harmonic: {res['f_tune_Hz']:.1f} Hz < 250")


def test_voltage_rise():
    print("\n[Test 7] Voltage rise dV/V ~ Qc/Ssc, larger Qc -> larger rise")
    m, _ = make_model()
    dv1 = m.voltage_rise(500.0)
    dv2 = m.voltage_rise(1000.0)
    assert_true(0 < dv1 < dv2, f"dV(500)={dv1*100:.2f}% < dV(1000)={dv2*100:.2f}%")
    assert_true(dv2 < 0.2, f"rise reasonable < 20% (got {dv2*100:.2f}%)")


def test_energize_inrush_ode():
    print("\n[Test 8] Energization RLC ODE: inrush > steady, cap charges")
    m, _ = make_model()
    tr = m.energize(Qc_kVAR=1000.0, duration_s=0.06)
    assert_true(tr["inrush_factor"] > 1.0,
                f"inrush_factor={tr['inrush_factor']:.2f} > 1 (overshoot)")
    Vm = np.sqrt(2.0 / 3.0) * (m.V_rated_kV * 1e3)
    assert_true(tr["v_cap_max_V"] > 0.5 * Vm,
                f"cap charges up: v_cap_max={tr['v_cap_max_V']:.0f} V")
    assert_true(np.all(np.isfinite(tr["i"])) and np.all(np.isfinite(tr["v_cap"])),
                "ODE solution finite")


def test_energize_damping_and_freq():
    print("\n[Test 9] Natural freq f0=1/(2pi sqrt(LC)); ESR damps ring")
    m, _ = make_model()
    tr = m.energize(Qc_kVAR=1000.0, duration_s=0.08)
    f0_expected = 1.0 / (2.0 * np.pi * np.sqrt(tr["L_eff_H"] * tr["C_F"]))
    assert_true(abs(tr["f_natural_Hz"] - f0_expected) < 1e-3,
                f"f0={tr['f_natural_Hz']:.1f} Hz == {f0_expected:.1f} Hz")
    assert_true(0 < tr["damping_ratio"], f"zeta={tr['damping_ratio']:.4f} > 0 (damped)")
    # higher loop R damps -> lower inrush peak
    tr_hi = m.energize(Qc_kVAR=1000.0, R_eff=2.0, duration_s=0.08)
    assert_true(tr_hi["inrush_factor"] <= tr["inrush_factor"] + 1e-6,
                f"more R -> less inrush ({tr_hi['inrush_factor']:.2f} <= {tr['inrush_factor']:.2f})")


def test_clamp_and_edge():
    print("\n[Test 10] Edge: Qc clamped to rating; zero load -> no Qc")
    m, _ = make_model()
    r_big = m.compensate(50000.0, 0.5, 1.0)   # huge demand
    assert_true(r_big["Q_compensated_kVAR"] <= m.Q_rated_kVAR + 1e-9,
                f"Qc clamped to rating {r_big['Q_compensated_kVAR']:.1f}")
    r_zero = m.compensate(0.0, 0.80, 0.95)
    assert_true(r_zero["Q_compensated_kVAR"] == 0.0, "zero load -> Qc=0")
    assert_true(r_zero["stages_on"] == 0, "zero load -> 0 stages on")


def test_predict_interface():
    print("\n[Test 11] ComponentModel predict() interface")
    _, cm = make_model()
    r = cm.predict({"P_kW": 800.0, "pf_initial": 0.80, "pf_target": 0.95})
    for key in ["Q_compensated_kVAR", "pf_achieved", "released_capacity_kVA",
                "voltage_rise_pu", "resonance", "transient", "C_F"]:
        assert_true(key in r, f"Key '{key}' in output")
    assert_true(len(r["transient"]["t"]) == len(r["transient"]["i"]),
                "transient arrays same length")
    info = cm.get_info()
    assert_true(info["component_id"] == "EC184", "get_info id EC184")


def test_benchmark():
    print("\n[Test 12] Benchmark: full predict() with inrush ODE")
    _, cm = make_model()
    t0 = time.perf_counter()
    cm.predict({"P_kW": 800.0, "pf_initial": 0.80, "pf_target": 0.95})
    elapsed = time.perf_counter() - t0
    print(f"  full predict() in {elapsed*1000:.1f} ms")
    assert_true(elapsed < 5.0, "Completes in < 5 s")


if __name__ == "__main__":
    tests = [
        test_compensation_formula,
        test_pf_improves_to_target,
        test_reactive_balance,
        test_released_capacity_positive,
        test_capacitance_sizing,
        test_resonance_frequency,
        test_voltage_rise,
        test_energize_inrush_ode,
        test_energize_damping_and_freq,
        test_clamp_and_edge,
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
    print(f"EC184 PFC F2a -- Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    sys.exit(0 if failed == 0 else 1)
