"""
EC186 -- STATCOM -- F2a VSC Reactive-Current Control
Test suite: physics sanity, Q-injection relation, constant-current at low V,
fast response, energy consistency, edge cases, predict() interface, benchmark.
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from model import STATCOM_F2a
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
def test_q_tracking_capacitive():
    print("\n[Test 1] Capacitive Q command tracked at steady state")
    _, cm = make_model()
    r = cm.predict({"Q_ref_MVAR": 80.0, "V_bus_pu": 1.0, "duration_s": 0.05})
    Qf = r["Q_out_MVAR"][-1]
    assert_true(abs(Qf - 80.0) < 1.0, f"Q_out={Qf:.2f} MVAR tracks +80 MVAR")
    assert_true(Qf > 0, "Capacitive output is positive")


def test_q_tracking_inductive():
    print("\n[Test 2] Inductive Q command tracked (symmetric VSC range)")
    _, cm = make_model()
    r = cm.predict({"Q_ref_MVAR": -80.0, "V_bus_pu": 1.0, "duration_s": 0.05})
    Qf = r["Q_out_MVAR"][-1]
    assert_true(abs(Qf + 80.0) < 1.0, f"Q_out={Qf:.2f} MVAR tracks -80 MVAR")
    assert_true(Qf < 0, "Inductive output is negative")


def test_q_injection_relation():
    print("\n[Test 3] Q-injection relation Q = V_bus*(V_bus - V_conv)/X holds")
    m, cm = make_model()
    r = cm.predict({"Q_ref_MVAR": 60.0, "V_bus_pu": 1.0, "duration_s": 0.05})
    Q = r["Q_out_VAR"][-1]
    V_conv = r["V_conv_V"][-1]           # line-line RMS
    V_bus = 1.0 * m.V_LL
    # per-unit reconstruction of Q from V_conv
    Q_recon = V_bus * (V_bus - V_conv) / (m.X_pu * m.Z_base)
    rel = abs(Q_recon - Q) / max(abs(Q), 1.0)
    assert_true(rel < 1e-3, f"Q from V_conv matches: {Q_recon/1e6:.2f} vs {Q/1e6:.2f} MVAR")


def test_constant_current_at_low_voltage():
    print("\n[Test 4] Constant-current capability at low bus voltage (vs SVC)")
    m, cm = make_model()
    # Full capacitive command at nominal and at 0.3 pu
    r_hi = cm.predict({"Q_ref_MVAR": 100.0, "V_bus_pu": 1.0, "duration_s": 0.05})
    r_lo = cm.predict({"Q_ref_MVAR": 100.0, "V_bus_pu": 0.3, "duration_s": 0.05})
    I_hi = r_hi["I_mag"][-1]
    I_lo = r_lo["I_mag"][-1]
    # Current stays at (near) its limit even though V dropped to 0.3 pu
    assert_true(abs(I_lo - I_hi) / I_hi < 0.1,
                f"Current held ~constant: I_lo={I_lo:.0f} A vs I_hi={I_hi:.0f} A")
    # An SVC (Q ~ B*V^2) would drop Q by 0.3^2 = 0.09x; STATCOM only ~0.3x
    Q_lo = r_lo["Q_out_MVAR"][-1]
    svc_like = 100.0 * 0.3 ** 2
    assert_true(Q_lo > svc_like * 2.5,
                f"STATCOM Q={Q_lo:.1f} MVAR >> SVC-equivalent {svc_like:.1f} MVAR at 0.3pu")


def test_current_limit_enforced():
    print("\n[Test 5] Reactive current clamps at I_max (over-command)")
    m, cm = make_model()
    r = cm.predict({"Q_ref_MVAR": 200.0, "V_bus_pu": 1.0, "duration_s": 0.05})
    I = r["I_mag"][-1]
    assert_true(I <= m.I_max * 1.001, f"I_mag={I:.0f} A <= I_max={m.I_max:.0f} A")
    Qf = r["Q_out_MVAR"][-1]
    assert_true(Qf <= 100.5, f"Q clamped to rating: {Qf:.1f} <= 100 MVAR")


def test_fast_response():
    print("\n[Test 6] Fast sub-cycle response (rise time ~ few tau_i)")
    m, cm = make_model()
    r = cm.predict({"Q_ref_MVAR": 100.0, "V_bus_pu": 1.0, "dt": 5e-5, "duration_s": 0.02})
    t = r["t"]
    Q = r["Q_out_MVAR"]
    Qss = Q[-1]
    idx90 = np.argmax(Q >= 0.9 * Qss)
    t90 = t[idx90]
    # closed-loop tau_i = 1.5 ms; 90% in ~2.3*tau ~ 3.5 ms, well under one 50Hz cycle (20 ms)
    assert_true(t90 < 0.010, f"90% rise in {t90*1e3:.2f} ms (< 10 ms, sub-cycle)")
    assert_true(t90 > 0.0, "Non-instantaneous (has dynamics)")


def test_dc_link_recovers():
    print("\n[Test 7] DC-link voltage stays bounded / regulated")
    m, cm = make_model()
    r = cm.predict({"Q_ref_MVAR": 100.0, "V_bus_pu": 1.0, "duration_s": 0.1})
    Vdc = r["Vdc"]
    assert_true(np.all(Vdc > 0.5 * m.Vdc_rated), "Vdc stays > 0.5 rated")
    assert_true(np.all(Vdc < 1.5 * m.Vdc_rated), "Vdc stays < 1.5 rated")
    # near steady DC-link at end
    dV = abs(Vdc[-1] - Vdc[-2])
    assert_true(dV < 0.01 * m.Vdc_rated, f"Vdc near steady: dV={dV:.1f} V")


def test_energy_consistency():
    print("\n[Test 8] Energy consistency: ideal-VSC Q exchange has near-zero net P")
    m, cm = make_model()
    r = cm.predict({"Q_ref_MVAR": 80.0, "V_bus_pu": 1.0, "duration_s": 0.1})
    P_in = r["P_in_W"][-1]      # real power drawn from bus
    P_loss = r["P_loss_W"][-1]  # converter losses
    Q_out = abs(r["Q_out_VAR"][-1])
    # At steady state real power in == losses (capacitor neither charges nor
    # discharges); real power is a tiny fraction of reactive throughput.
    assert_true(abs(P_in - P_loss) < 0.05 * P_loss + 5e4,
                f"Steady DC-link: P_in={P_in/1e6:.3f} ~ P_loss={P_loss/1e6:.3f} MW")
    assert_true(P_in < 0.06 * Q_out,
                f"Real power {P_in/1e6:.3f} MW << reactive {Q_out/1e6:.1f} MVAR")


def test_capacitive_vs_inductive_vconv():
    print("\n[Test 9] V_conv ordering follows Q = V_bus*(V_bus - V_conv)/X")
    m, cm = make_model()
    r_cap = cm.predict({"Q_ref_MVAR": 80.0, "V_bus_pu": 1.0, "duration_s": 0.05})
    r_ind = cm.predict({"Q_ref_MVAR": -80.0, "V_bus_pu": 1.0, "duration_s": 0.05})
    Vc_cap = r_cap["V_conv_V"][-1]
    Vc_ind = r_ind["V_conv_V"][-1]
    # Per the stated relation, capacitive (Q>0) => V_conv < V_bus < inductive V_conv
    assert_true(Vc_cap < Vc_ind, f"V_conv(cap)={Vc_cap/1e3:.1f} < V_conv(ind)={Vc_ind/1e3:.1f} kV")


def test_zero_command_standby():
    print("\n[Test 10] Zero Q command -> ~zero reactive output, only standby loss")
    m, cm = make_model()
    r = cm.predict({"Q_ref_MVAR": 0.0, "V_bus_pu": 1.0, "duration_s": 0.05})
    Qf = abs(r["Q_out_MVAR"][-1])
    assert_true(Qf < 1.0, f"|Q_out|={Qf:.3f} MVAR ~ 0 at zero command")
    assert_true(r["P_loss_W"][-1] >= m.P_standby - 1.0, "Loss >= standby")


def test_predict_interface():
    print("\n[Test 11] ComponentModel predict() interface")
    _, cm = make_model()
    r = cm.predict({"Q_ref_MVAR": 50.0, "V_bus_pu": 1.0, "dt": 1e-4, "duration_s": 0.02})
    for key in ["t", "i_d", "i_q", "I_mag", "Vdc", "Q_out_MVAR",
                "P_loss_W", "V_conv_V", "V_bus_pu"]:
        assert_true(key in r, f"Key '{key}' in output")
    assert_true(len(r["t"]) == len(r["Q_out_MVAR"]), "Arrays same length")
    info = cm.get_info()
    assert_true(info["component_id"] == "EC186", "component_id == EC186")


def test_benchmark():
    print("\n[Test 12] Benchmark: 0.1 s sim at dt=1e-4")
    _, cm = make_model()
    t0 = time.perf_counter()
    cm.predict({"Q_ref_MVAR": 80.0, "V_bus_pu": 1.0, "dt": 1e-4, "duration_s": 0.1})
    elapsed = time.perf_counter() - t0
    print(f"  0.1 s simulation in {elapsed*1000:.1f} ms")
    assert_true(elapsed < 5.0, "Completes in < 5 s")


if __name__ == "__main__":
    tests = [
        test_q_tracking_capacitive,
        test_q_tracking_inductive,
        test_q_injection_relation,
        test_constant_current_at_low_voltage,
        test_current_limit_enforced,
        test_fast_response,
        test_dc_link_recovers,
        test_energy_consistency,
        test_capacitive_vs_inductive_vconv,
        test_zero_command_standby,
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
    print(f"EC186 STATCOM F2a -- Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    sys.exit(0 if failed == 0 else 1)
