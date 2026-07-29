"""
EC073 -- Shell-and-Tube Heat Exchanger -- F2a Lumped-Capacitance Transient
Test suite: physics sanity (conservation, monotonicity, steady-state limit),
edge cases, predict() interface, and a benchmark timing test.
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from model import ShellAndTubeHEX_F2a
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
def test_steady_state_matches_epsilon_ntu():
    print("\n[Test 1] Transient steady state matches epsilon-NTU duty")
    m, _ = make_model()
    Th, Tc, mh, mc = 90.0, 20.0, 2.0, 2.0
    r = m.simulate(Th, Tc, mh, mc, duration_s=2500.0, dt=10.0)
    ss = m.steady_state_duty(Th, Tc, mh, mc)
    err_Q = abs(r["Q_kw"][-1] - ss["Q_kw"]) / abs(ss["Q_kw"])
    err_Th = abs(r["T_h_out"][-1] - ss["T_h_out"])
    err_Tc = abs(r["T_c_out"][-1] - ss["T_c_out"])
    assert_true(err_Q < 0.02, f"Q rel-err {err_Q*100:.2f}% < 2%")
    assert_true(err_Th < 1.0, f"T_h_out err {err_Th:.3f} degC < 1.0")
    assert_true(err_Tc < 1.0, f"T_c_out err {err_Tc:.3f} degC < 1.0")


def test_energy_conservation():
    print("\n[Test 2] Energy conservation: hot drop == cold rise (W)")
    m, _ = make_model()
    Th, Tc, mh, mc = 95.0, 15.0, 3.0, 1.5
    r = m.simulate(Th, Tc, mh, mc, duration_s=1500.0, dt=10.0)
    Q_hot = mh * m.cp_h * (Th - r["T_h_out"][-1])
    Q_cold = mc * m.cp_c * (r["T_c_out"][-1] - Tc)
    rel = abs(Q_hot - Q_cold) / abs(Q_hot)
    assert_true(rel < 0.01, f"|Q_hot-Q_cold|/Q_hot = {rel*100:.3f}% < 1%")


def test_second_law_outlet_bounds():
    print("\n[Test 3] Outlet temps respect thermodynamic bounds")
    m, _ = make_model()
    Th, Tc, mh, mc = 90.0, 20.0, 2.0, 2.0
    r = m.simulate(Th, Tc, mh, mc, duration_s=1200.0, dt=10.0)
    Tho, Tco = r["T_h_out"][-1], r["T_c_out"][-1]
    assert_true(Tc <= Tho <= Th, f"T_h_out={Tho:.2f} in [{Tc},{Th}]")
    assert_true(Tc <= Tco <= Th, f"T_c_out={Tco:.2f} in [{Tc},{Th}]")
    # No temperature cross beyond physical: cold cannot exceed hot inlet
    assert_true(Tco <= Th + 1e-6, "Cold outlet does not exceed hot inlet")


def test_heat_flows_hot_to_cold():
    print("\n[Test 4] Heat flows hot->cold (hot cools, cold heats)")
    m, _ = make_model()
    Th, Tc, mh, mc = 80.0, 25.0, 2.0, 2.0
    r = m.simulate(Th, Tc, mh, mc, duration_s=1000.0, dt=10.0)
    assert_true(r["T_h_out"][-1] < Th, f"hot cooled: {r['T_h_out'][-1]:.2f} < {Th}")
    assert_true(r["T_c_out"][-1] > Tc, f"cold heated: {r['T_c_out'][-1]:.2f} > {Tc}")
    assert_true(r["Q_kw"][-1] > 0, f"Q={r['Q_kw'][-1]:.2f} kW > 0")


def test_transient_warmup_monotone():
    print("\n[Test 5] Cold-start: cold outlet rises monotonically to SS")
    m, _ = make_model()
    # Start whole HX at cold inlet temp -> cold outlet should climb
    r = m.simulate(90.0, 20.0, 2.0, 2.0, duration_s=600.0, dt=5.0,
                   T_init=20.0)
    Tco = r["T_c_out"]
    diffs = np.diff(Tco)
    assert_true(np.all(diffs >= -1e-6), "Cold outlet non-decreasing during warmup")
    assert_true(Tco[-1] > Tco[0], f"Final {Tco[-1]:.2f} > initial {Tco[0]:.2f}")


def test_duty_increases_with_flow():
    print("\n[Test 6] Steady duty increases with hot mass flow")
    m, _ = make_model()
    ss_lo = m.steady_state_duty(90.0, 20.0, 1.0, 2.0)
    ss_hi = m.steady_state_duty(90.0, 20.0, 4.0, 2.0)
    assert_true(ss_hi["Q_kw"] > ss_lo["Q_kw"],
                f"Q(4 kg/s)={ss_hi['Q_kw']:.1f} > Q(1 kg/s)={ss_lo['Q_kw']:.1f} kW")


def test_effectiveness_range():
    print("\n[Test 7] Effectiveness and NTU physical")
    m, _ = make_model()
    eps, NTU, C_r, _ = m.epsilon_ntu_counterflow(2.0, 2.0)
    assert_true(0.0 < eps < 1.0, f"0 < eps={eps:.3f} < 1")
    assert_true(NTU > 0.0, f"NTU={NTU:.3f} > 0")
    assert_true(0.0 <= C_r <= 1.0, f"0 <= C_r={C_r:.3f} <= 1")


def test_balanced_flow_eps_formula():
    print("\n[Test 8] Balanced flow uses C_r=1 limit eps=NTU/(1+NTU)")
    m, _ = make_model()
    eps, NTU, C_r, _ = m.epsilon_ntu_counterflow(2.0, 2.0)
    assert_true(abs(C_r - 1.0) < 1e-9, f"C_r={C_r:.6f} == 1 (cp equal, flows equal)")
    expected = NTU / (1.0 + NTU)
    assert_true(abs(eps - expected) < 1e-9, f"eps={eps:.4f} == NTU/(1+NTU)={expected:.4f}")


def test_higher_inlet_dt_more_duty():
    print("\n[Test 9] Larger inlet dT gives proportionally larger duty")
    m, _ = make_model()
    ss1 = m.steady_state_duty(60.0, 20.0, 2.0, 2.0)   # dT=40
    ss2 = m.steady_state_duty(100.0, 20.0, 2.0, 2.0)  # dT=80
    ratio = ss2["Q_kw"] / ss1["Q_kw"]
    assert_true(abs(ratio - 2.0) < 0.01, f"Q ratio={ratio:.3f} ~ 2.0 (linear in dT)")


def test_solver_success_and_finite():
    print("\n[Test 10] Solver succeeds and outputs finite")
    m, _ = make_model()
    r = m.simulate(120.0, 10.0, 5.0, 5.0, duration_s=300.0, dt=5.0)
    assert_true(r["success"], "solve_ivp reported success")
    assert_true(np.all(np.isfinite(r["T_h_out"])), "T_h_out finite")
    assert_true(np.all(np.isfinite(r["T_c_out"])), "T_c_out finite")
    assert_true(np.all(np.isfinite(r["Q_kw"])), "Q_kw finite")


def test_predict_interface():
    print("\n[Test 11] ComponentModel predict() interface")
    _, cm = make_model()
    r = cm.predict({"T_h_in": 90.0, "T_c_in": 20.0,
                    "m_dot_hot": 2.0, "m_dot_cold": 2.0,
                    "duration_s": 200.0, "dt": 5.0})
    for key in ["t", "T_h_out", "T_c_out", "Q_kw", "steady_state_reference"]:
        assert_true(key in r, f"Key '{key}' in output")
    assert_true(len(r["t"]) == len(r["T_h_out"]), "Arrays same length")
    info = cm.get_info()
    assert_true(info["component_id"] == "EC073", "component_id == EC073")
    assert_true(info["version"] == "1.0.0", "version == 1.0.0")


def test_benchmark():
    print("\n[Test 12] Benchmark: 300 s transient sim")
    m, _ = make_model()
    t0 = time.perf_counter()
    m.simulate(90.0, 20.0, 2.0, 2.0, duration_s=300.0, dt=1.0)
    elapsed = time.perf_counter() - t0
    print(f"  300 s simulation in {elapsed*1000:.1f} ms")
    assert_true(elapsed < 5.0, "Completes in < 5 s")


if __name__ == "__main__":
    tests = [
        test_steady_state_matches_epsilon_ntu,
        test_energy_conservation,
        test_second_law_outlet_bounds,
        test_heat_flows_hot_to_cold,
        test_transient_warmup_monotone,
        test_duty_increases_with_flow,
        test_effectiveness_range,
        test_balanced_flow_eps_formula,
        test_higher_inlet_dt_more_duty,
        test_solver_success_and_finite,
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
    print(f"EC073 Shell-and-Tube HX F2a -- Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    sys.exit(0 if failed == 0 else 1)
