"""
EC077 -- Microchannel Heat Exchanger -- F2a Lumped Transient
Test suite: energy conservation, e-NTU limit, microchannel physics, edges.
NO pytest -- custom assert harness, run as __main__.
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from model import MicrochannelHX_F2a
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
def test_steady_energy_conservation():
    print("\n[Test 1] Energy conservation: Q_hot == Q_cold at steady state")
    m, _ = make_model()
    r = m.simulate(T_h_in=80.0, T_c_in=20.0, mdot_h=0.10, mdot_c=0.08,
                   dt=5.0, duration_s=300.0)
    cp_h, cp_c = m.cp_h, m.cp_c
    Q_h = 0.10 * cp_h * (80.0 - r["T_h_out"][-1])
    Q_c = 0.08 * cp_c * (r["T_c_out"][-1] - 20.0)
    rel = abs(Q_h - Q_c) / max(Q_h, 1.0)
    assert_true(rel < 0.01, f"Q_h={Q_h:.1f} W ~= Q_c={Q_c:.1f} W (rel err {rel*100:.3f}%)")


def test_matches_epsilon_ntu():
    print("\n[Test 2] Steady-state effectiveness matches e-NTU (counterflow)")
    m, _ = make_model()
    r = m.simulate(T_h_in=80.0, T_c_in=20.0, mdot_h=0.10, mdot_c=0.08,
                   dt=5.0, duration_s=400.0)
    eps_ode = r["effectiveness"][-1]
    eps_ntu, NTU, _, _ = m.epsilon_ntu_counterflow(0.10, 0.08)
    err = abs(eps_ode - eps_ntu)
    print(f"  eps_ODE={eps_ode:.4f}, eps_NTU={eps_ntu:.4f}, NTU={NTU:.3f}")
    assert_true(err < 0.02, f"|eps_ODE - eps_NTU| = {err:.4f} < 0.02")


def test_effectiveness_bounds():
    print("\n[Test 3] Effectiveness in (0, 1)")
    m, _ = make_model()
    r = m.simulate(dt=10.0, duration_s=300.0)
    eps = r["effectiveness"][-1]
    assert_true(0.0 < eps < 1.0, f"eps={eps:.4f} in (0,1)")


def test_high_effectiveness():
    print("\n[Test 4] Microchannel is compact: high effectiveness (eps > 0.7)")
    m, _ = make_model()
    r = m.simulate(dt=10.0, duration_s=400.0)
    eps = r["effectiveness"][-1]
    assert_true(eps > 0.7, f"eps={eps:.4f} > 0.7 (high UA/compactness)")


def test_outlet_bounds():
    print("\n[Test 5] Outlet temps within inlet bounds (2nd law)")
    m, _ = make_model()
    r = m.simulate(T_h_in=80.0, T_c_in=20.0, dt=10.0, duration_s=400.0)
    Th_out = r["T_h_out"][-1]
    Tc_out = r["T_c_out"][-1]
    assert_true(20.0 <= Th_out <= 80.0, f"T_h_out={Th_out:.2f} in [20,80]")
    assert_true(20.0 <= Tc_out <= 80.0, f"T_c_out={Tc_out:.2f} in [20,80]")
    # 2nd law (counterflow): hot can only cool, cold can only warm, and a
    # counterflow cold-outlet MAY exceed the hot-outlet -- so the valid
    # constraints are T_h_out >= T_c_in and T_c_out <= T_h_in.
    assert_true(Th_out >= 20.0 - 1e-6, f"T_h_out={Th_out:.2f} >= T_c_in=20")
    assert_true(Tc_out <= 80.0 + 1e-6, f"T_c_out={Tc_out:.2f} <= T_h_in=80")


def test_microchannel_high_h():
    print("\n[Test 6] Microchannel laminar h is very large (Dh<1mm)")
    m, _ = make_model()
    h = m.htc("hot")
    # h = Nu*k/Dh = 4.36*0.607/5e-4 ~ 5300 W/m2K
    assert_true(h > 2000.0, f"h_h={h:.0f} W/m2.K > 2000 (compact)")
    assert_true(m.Dh < 1e-3, f"Dh={m.Dh*1e3:.2f} mm < 1 mm (microchannel)")


def test_laminar_regime():
    print("\n[Test 7] Liquid microchannel flow is laminar (Re < 2300)")
    m, _ = make_model()
    Re = m.reynolds(0.10, "hot")
    print(f"  Re_h={Re:.1f}")
    assert_true(Re < 2300.0, f"Re={Re:.1f} < 2300 -> laminar Nu=4.36 valid")


def test_pressure_drop_notable():
    print("\n[Test 8] Notable pressure drop from small Dh")
    m, _ = make_model()
    dP = m.pressure_drop(0.10, "hot")
    print(f"  dP_h={dP:.1f} Pa")
    assert_true(dP > 100.0, f"dP={dP:.1f} Pa > 100 Pa (notable)")
    # f = 64/Re scaling: doubling flow at fixed laminar f-Re raises dP
    dP2 = m.pressure_drop(0.20, "hot")
    assert_true(dP2 > dP, f"dP({0.20})={dP2:.1f} > dP({0.10})={dP:.1f}")


def test_transient_monotone_warmup():
    print("\n[Test 9] Cold-start transient: cold-outlet warms monotonically")
    m, _ = make_model()
    r = m.simulate(T_h_in=80.0, T_c_in=20.0, dt=2.0, duration_s=120.0,
                   T_init=20.0)
    Tc = r["T_c_out"]
    diffs = np.diff(Tc)
    assert_true(np.all(diffs >= -0.05), "T_c_out non-decreasing during warm-up")
    assert_true(Tc[-1] > Tc[0] + 1.0, f"warmed: {Tc[0]:.2f}->{Tc[-1]:.2f} C")


def test_flow_sensitivity():
    print("\n[Test 10] Lower hot flow -> lower NTU? higher eps (counterflow)")
    m, _ = make_model()
    r_hi = m.simulate(mdot_h=0.20, mdot_c=0.08, dt=10.0, duration_s=400.0)
    r_lo = m.simulate(mdot_h=0.05, mdot_c=0.08, dt=10.0, duration_s=400.0)
    # Lower C_min raises NTU=UA/C_min -> higher effectiveness
    assert_true(r_lo["effectiveness"][-1] > r_hi["effectiveness"][-1],
                f"eps(low mdot_h)={r_lo['effectiveness'][-1]:.3f} > "
                f"eps(high)={r_hi['effectiveness'][-1]:.3f}")


def test_predict_interface():
    print("\n[Test 11] ComponentModel predict() interface")
    _, cm = make_model()
    r = cm.predict({"T_h_in": 80.0, "T_c_in": 20.0, "dt": 10.0, "duration_s": 100.0})
    for key in ["t", "T_h_out", "T_c_out", "Q_kW", "effectiveness",
                "T_wall_profile", "dP_h_Pa", "UA"]:
        assert_true(key in r, f"Key '{key}' in output")
    assert_true(len(r["t"]) == len(r["T_h_out"]), "Arrays same length")
    info = cm.get_info()
    assert_true(info["component_id"] == "EC077", "get_info id == EC077")


def test_benchmark():
    print("\n[Test 12] Benchmark: 120s transient sim")
    m, _ = make_model()
    t0 = time.perf_counter()
    m.simulate(dt=1.0, duration_s=120.0)
    elapsed = time.perf_counter() - t0
    print(f"  120s simulation in {elapsed*1000:.1f} ms")
    assert_true(elapsed < 5.0, "Completes in < 5 s")


if __name__ == "__main__":
    tests = [
        test_steady_energy_conservation,
        test_matches_epsilon_ntu,
        test_effectiveness_bounds,
        test_high_effectiveness,
        test_outlet_bounds,
        test_microchannel_high_h,
        test_laminar_regime,
        test_pressure_drop_notable,
        test_transient_monotone_warmup,
        test_flow_sensitivity,
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
    print(f"EC077 MCHX F2a -- Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    sys.exit(0 if failed == 0 else 1)
