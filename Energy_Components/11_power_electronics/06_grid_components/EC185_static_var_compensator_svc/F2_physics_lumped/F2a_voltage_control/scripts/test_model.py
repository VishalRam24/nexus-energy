"""
EC185 -- Static VAR Compensator (SVC) -- F2a Physics-Lumped Voltage Control
Test suite: B(alpha) physics, Q range, droop, voltage regulation ODE, edge cases.
Run: python3 scripts/test_model.py   (system python3, NOT pytest)
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from model import SVC_F2a
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
def test_tcr_susceptance_limits():
    print("\n[Test 1] B_L(alpha): full conduction at 90deg, zero at 180deg")
    m, _ = make_model()
    B_full = m.tcr_susceptance(np.radians(90.0))
    B_zero = m.tcr_susceptance(np.radians(180.0))
    assert_true(abs(B_full - 1.0 / m.X_L) < 1e-6,
                f"B_L(90)={B_full:.5f} == 1/X_L={1.0/m.X_L:.5f} (max inductive)")
    assert_true(abs(B_zero) < 1e-9, f"B_L(180)={B_zero:.2e} == 0 (no conduction)")


def test_tcr_monotone():
    print("\n[Test 2] B_L(alpha) monotonically decreasing on [90,180]deg")
    m, _ = make_model()
    alphas = np.radians(np.linspace(90.0, 180.0, 100))
    B = m.tcr_susceptance(alphas)
    diffs = np.diff(B)
    assert_true(np.all(diffs <= 1e-12), "B_L strictly non-increasing in alpha")
    print(f"  B_L range: {B[0]:.4f} -> {B[-1]:.4f} pu over 99 steps")


def test_alpha_inversion():
    print("\n[Test 3] alpha_from_B inverts B_svc(alpha) round-trip")
    m, _ = make_model()
    for B_target in np.linspace(m.B_svc_min, m.B_svc_max, 9):
        a = m.alpha_from_B(B_target)
        B_back = m.net_susceptance(a)
        assert_true(abs(B_back - B_target) < 1e-3,
                    f"B={B_target:+.3f} -> alpha={np.degrees(a):6.2f}deg -> "
                    f"B={B_back:+.3f}")


def test_q_range_inductive_to_capacitive():
    print("\n[Test 4] Reactive power spans inductive (-) to capacitive (+)")
    m, _ = make_model()
    a_ind = m.alpha_min   # full TCR -> inductive
    a_cap = m.alpha_max   # TCR off  -> capacitive
    Q_ind = m.reactive_power_MVAR(m.net_susceptance(a_ind), 1.0)
    Q_cap = m.reactive_power_MVAR(m.net_susceptance(a_cap), 1.0)
    assert_true(Q_ind < 0 < Q_cap, f"Q_ind={Q_ind:.1f} < 0 < Q_cap={Q_cap:.1f} MVAR")
    assert_true(abs(Q_cap - 100.0) < 1e-3, f"Q_cap={Q_cap:.2f} == +100 MVAR rated")
    assert_true(abs(Q_ind + 50.0) < 1e-3, f"Q_ind={Q_ind:.2f} == -50 MVAR rated")


def test_q_proportional_v_squared():
    print("\n[Test 5] Q = B*V^2 scales with V^2 (shunt-susceptance law)")
    m, _ = make_model()
    B = m.B_svc_max
    Q1 = m.reactive_power_pu(B, 1.0)
    Q2 = m.reactive_power_pu(B, 1.1)
    assert_true(abs(Q2 / Q1 - 1.1 ** 2) < 1e-9,
                f"Q(1.1)/Q(1.0)={Q2/Q1:.4f} == 1.21")


def test_droop_characteristic():
    print("\n[Test 6] V-Q droop: capacitive raises V, inductive lowers V")
    m, _ = make_model()
    V_cap = m.droop_voltage(+1.0)   # +1 pu Q (cap)
    V_ind = m.droop_voltage(-0.5)   # -0.5 pu Q (ind)
    assert_true(V_cap > m.V_ref > V_ind,
                f"V(cap)={V_cap:.4f} > Vref={m.V_ref} > V(ind)={V_ind:.4f}")
    # slope magnitude check
    slope = (m.droop_voltage(1.0) - m.droop_voltage(0.0)) / 1.0
    assert_true(abs(slope - m.X_SL) < 1e-9, f"droop slope={slope:.4f} == X_SL")


def test_regulation_overvoltage():
    print("\n[Test 7] ODE regulates an over-voltage down toward V_ref")
    m, _ = make_model()
    # Source over-voltage E=1.05 ; with SVC off the bus would sit ~1.05 pu.
    r = m.simulate(E_thev=1.05, X_thev=0.10, dt=0.002, duration_s=0.6)
    V_uncomp = 1.05
    V_final = r["V_bus"][-1]
    assert_true(V_final < V_uncomp,
                f"V_bus {V_uncomp:.3f} -> {V_final:.4f} pu (pulled down)")
    assert_true(abs(V_final - m.V_ref) < 0.04,
                f"V_final={V_final:.4f} within droop band of Vref={m.V_ref}")
    assert_true(r["Q_MVAR"][-1] < 0.0,
                f"Inductive absorption to cut over-voltage: Q={r['Q_MVAR'][-1]:.1f} MVAR")


def test_regulation_undervoltage():
    print("\n[Test 8] ODE regulates an under-voltage up toward V_ref")
    m, _ = make_model()
    r = m.simulate(E_thev=0.95, X_thev=0.10, dt=0.002, duration_s=0.6)
    V_final = r["V_bus"][-1]
    assert_true(V_final > 0.95, f"V_bus 0.950 -> {V_final:.4f} pu (boosted)")
    assert_true(r["Q_MVAR"][-1] > 0.0,
                f"Capacitive injection to lift under-voltage: Q={r['Q_MVAR'][-1]:.1f} MVAR")


def test_steady_state_consistency():
    print("\n[Test 9] ODE settles to the algebraic droop steady state")
    m, _ = make_model()
    E, X = 1.04, 0.08
    B_ss, V_ss, sat = m.steady_state_susceptance(E, X)
    r = m.simulate(E_thev=E, X_thev=X, dt=0.001, duration_s=1.0)
    assert_true(abs(r["V_bus"][-1] - V_ss) < 5e-3,
                f"ODE V_final={r['V_bus'][-1]:.4f} ~ algebraic V_ss={V_ss:.4f}")
    assert_true(abs(r["B_act"][-1] - B_ss) < 1e-2,
                f"ODE B_final={r['B_act'][-1]:.4f} ~ algebraic B_ss={B_ss:.4f}")


def test_susceptance_bounds():
    print("\n[Test 10] Realised susceptance stays within [B_min, B_max]")
    m, _ = make_model()
    r = m.simulate(E_thev=1.15, X_thev=0.30, dt=0.002, duration_s=0.6)
    assert_true(np.all(r["B_act"] >= m.B_svc_min - 1e-9), "B_act >= B_svc_min")
    assert_true(np.all(r["B_act"] <= m.B_svc_max + 1e-9), "B_act <= B_svc_max")
    assert_true(np.all((r["alpha_deg"] >= 89.9) & (r["alpha_deg"] <= 180.1)),
                "alpha within [90,180] deg")


def test_loss_energy_consistency():
    print("\n[Test 11] Losses >= 0 and proportional to |Q|")
    m, _ = make_model()
    r = m.simulate(E_thev=1.05, X_thev=0.10, dt=0.002, duration_s=0.4)
    assert_true(np.all(r["P_loss_MW"] >= -1e-12), "P_loss >= 0 everywhere")
    i = int(np.argmax(np.abs(r["Q_MVAR"])))
    expect = m.loss_factor * abs(r["Q_MVAR"][i])
    assert_true(abs(r["P_loss_MW"][i] - expect) < 1e-9,
                f"P_loss={r['P_loss_MW'][i]:.4f} == 1%*|Q|={expect:.4f} MW")


def test_predict_interface():
    print("\n[Test 12] ComponentModel predict() interface + metadata")
    _, cm = make_model()
    r = cm.predict({"E_thev": 1.05, "X_thev": 0.1, "dt": 0.005, "duration_s": 0.2})
    for key in ["t", "V_bus", "B_act", "alpha_deg", "Q_MVAR", "Q_pu",
                "P_loss_MW", "mode"]:
        assert_true(key in r, f"Key '{key}' in output")
    assert_true(len(r["t"]) == len(r["V_bus"]) == len(r["Q_MVAR"]),
                "Output arrays equal length")
    info = cm.get_info()
    assert_true(info["component_id"] == "EC185" and info["version"] == "1.0.0",
                "get_info() id/version correct")


def test_benchmark():
    print("\n[Test 13] Benchmark: 0.5 s transient at dt=0.001")
    m, _ = make_model()
    t0 = time.perf_counter()
    m.simulate(E_thev=1.05, X_thev=0.10, dt=0.001, duration_s=0.5)
    elapsed = time.perf_counter() - t0
    print(f"  0.5 s simulation in {elapsed*1000:.1f} ms")
    assert_true(elapsed < 5.0, "Completes in < 5 s")


if __name__ == "__main__":
    tests = [
        test_tcr_susceptance_limits,
        test_tcr_monotone,
        test_alpha_inversion,
        test_q_range_inductive_to_capacitive,
        test_q_proportional_v_squared,
        test_droop_characteristic,
        test_regulation_overvoltage,
        test_regulation_undervoltage,
        test_steady_state_consistency,
        test_susceptance_bounds,
        test_loss_energy_consistency,
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
    print(f"EC185 SVC F2a -- Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    sys.exit(0 if failed == 0 else 1)
