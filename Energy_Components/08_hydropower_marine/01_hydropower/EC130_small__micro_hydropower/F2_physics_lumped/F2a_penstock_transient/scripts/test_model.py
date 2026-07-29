"""
EC130 -- Small/Micro Hydropower -- F2a Penstock Transient
Test suite: physics sanity (energy conservation, P~Q*H, head-loss law),
ODE convergence, surge transient, edge cases, predict() interface, benchmark.
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from model import MicroHydroF2a
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
def test_efficiency_range():
    print("\n[Test 1] Turbine efficiency strictly in (0, 1)")
    m, _ = make_model()
    for q in [0.3, 0.5, 0.8, 1.0, 1.05]:
        eta = float(m.turbine_efficiency(q * m.Q_design))
        assert_true(0.0 < eta < 1.0, f"eta(q={q})={eta:.4f} in (0,1)")
    # peak at design flow
    eta_design = float(m.turbine_efficiency(m.Q_design))
    assert_true(abs(eta_design - m.eta_t_peak) < 1e-9,
                f"eta peaks at design flow: {eta_design:.4f}")


def test_power_scales_with_QH():
    print("\n[Test 2] Power scales with Q*H (hydraulic power law)")
    m, _ = make_model()
    # double flow at same head -> hydraulic power doubles
    P1 = float(m.hydraulic_power_kw(0.5, 40.0))
    P2 = float(m.hydraulic_power_kw(1.0, 40.0))
    assert_true(abs(P2 / P1 - 2.0) < 1e-9, f"P(2Q)/P(Q)={P2/P1:.4f} == 2")
    # double head at same flow -> hydraulic power doubles
    P3 = float(m.hydraulic_power_kw(0.5, 80.0))
    assert_true(abs(P3 / P1 - 2.0) < 1e-9, f"P(2H)/P(H)={P3/P1:.4f} == 2")


def test_energy_conservation():
    print("\n[Test 3] Energy conservation: P_el = eta_t*eta_gen*rho*g*Q*H")
    m, _ = make_model()
    Q, H = 1.2, 43.0
    eta_t = float(m.turbine_efficiency(Q))
    P_hyd = float(m.hydraulic_power_kw(Q, H))
    P_el = float(m.electrical_power_kw(Q, H))
    expected = eta_t * m.eta_gen * P_hyd
    assert_true(abs(P_el - expected) < 1e-6,
                f"P_el={P_el:.3f} == eta_t*eta_gen*P_hyd={expected:.3f}")
    # electrical output never exceeds hydraulic input (no free energy)
    assert_true(P_el < P_hyd, f"P_el={P_el:.1f} < P_hyd={P_hyd:.1f} (losses)")


def test_head_loss_law():
    print("\n[Test 4] Darcy-Weisbach head loss scales with v^2")
    m, _ = make_model()
    hl1 = float(m.head_loss(1.0))
    hl2 = float(m.head_loss(2.0))
    assert_true(abs(hl2 / hl1 - 4.0) < 1e-9, f"H_loss(2v)/H_loss(v)={hl2/hl1:.4f} == 4")
    assert_true(hl1 > 0.0, "Head loss positive for v>0")
    assert_true(float(m.head_loss(0.0)) == 0.0, "Zero loss at zero velocity")


def test_net_head_reduced_by_loss():
    print("\n[Test 5] Net head < gross head (friction reduces head)")
    m, _ = make_model()
    H_net = float(m.net_head(m.v_design, 0.0))
    assert_true(H_net < m.H_gross, f"H_net={H_net:.2f} < H_gross={m.H_gross:.2f}")
    assert_true(H_net > 0.9 * m.H_gross, f"loss reasonable: H_net={H_net:.2f} m")


def test_steady_convergence():
    print("\n[Test 6] ODE converges to steady state from cold start")
    m, _ = make_model()
    r = m.simulate(1.0, v0=0.1, dt=0.2, duration_s=300.0)
    dv = abs(r["velocity"][-1] - r["velocity"][-2])
    assert_true(dv < 1e-3, f"Near steady: dv={dv:.2e} m/s between last steps")
    assert_true(abs(r["velocity"][-1] - m.v_design) < 0.05,
                f"Full gate -> v->v_design: {r['velocity'][-1]:.3f} vs {m.v_design:.3f}")


def test_gate_step_response():
    print("\n[Test 7] Gate step up -> flow and power increase")
    m, _ = make_model()

    def gate(t):
        return 0.5 if t < 20.0 else 1.0

    r = m.simulate(gate, dt=0.1, duration_s=120.0)
    i_before = np.argmin(np.abs(r["t"] - 19.0))
    i_after = np.argmin(np.abs(r["t"] - 119.0))
    assert_true(r["flow"][i_after] > r["flow"][i_before],
                f"Flow rises: {r['flow'][i_before]:.3f} -> {r['flow'][i_after]:.3f}")
    assert_true(r["power_el"][i_after] > r["power_el"][i_before],
                f"Power rises: {r['power_el'][i_before]:.1f} -> {r['power_el'][i_after]:.1f} kW")


def test_surge_oscillation():
    print("\n[Test 8] Surge/forebay level oscillates after disturbance")
    m, _ = make_model()
    # supply held at design flow while gate cycles -> surge tank buffers
    def gate(t):
        return 1.0 if t < 30.0 else 0.6
    r = m.simulate(gate, Q_in=m.Q_design, dt=0.1, duration_s=200.0)
    z = r["surge_level"]
    assert_true(np.max(np.abs(z)) > 1e-3, f"Surge excites: max|z|={np.max(np.abs(z)):.4f} m")
    assert_true(np.max(np.abs(z)) < 20.0, "Surge bounded (mass-oscillation stable)")


def test_part_load_efficiency_drops():
    print("\n[Test 9] Part-load efficiency below peak (monotone toward design)")
    m, _ = make_model()
    eta_full = float(m.turbine_efficiency(m.Q_design))
    eta_part = float(m.turbine_efficiency(0.5 * m.Q_design))
    assert_true(eta_part < eta_full, f"eta(50%)={eta_part:.4f} < eta(100%)={eta_full:.4f}")
    # below q_min the turbine shuts off
    eta_off = float(m.turbine_efficiency(0.1 * m.Q_design))
    assert_true(eta_off == 0.0, f"eta below q_min == 0 (got {eta_off})")


def test_gate_closed_zero_power():
    print("\n[Test 10] Closed gate -> flow and power decay to ~0")
    m, _ = make_model()
    r = m.simulate(0.0, v0=m.v_design, dt=0.1, duration_s=120.0)
    assert_true(r["velocity"][-1] < 0.1, f"Flow stops: v_final={r['velocity'][-1]:.4f} m/s")
    assert_true(r["power_el"][-1] < 1.0, f"Power -> 0: {r['power_el'][-1]:.4f} kW")


def test_predict_interface():
    print("\n[Test 11] ComponentModel predict() interface + get_info()")
    _, cm = make_model()
    r = cm.predict({"gate_command": 0.8, "dt": 0.5, "duration_s": 30.0})
    for key in ["t", "velocity", "flow", "head_net", "head_loss",
                "surge_level", "gate", "power_el", "power_hyd", "efficiency"]:
        assert_true(key in r, f"Key '{key}' in output")
    assert_true(len(r["t"]) == len(r["power_el"]), "Output arrays same length")
    info = cm.get_info()
    assert_true(info["component_id"] == "EC130", "get_info id == EC130")
    assert_true("Harvey" in info["source"], "Harvey 1993 cited in source")


def test_benchmark():
    print("\n[Test 12] Benchmark: 120 s transient at dt=0.1")
    m, _ = make_model()
    t0 = time.perf_counter()
    m.simulate(1.0, dt=0.1, duration_s=120.0)
    elapsed = time.perf_counter() - t0
    print(f"  120 s simulation in {elapsed*1000:.1f} ms")
    assert_true(elapsed < 5.0, "Completes in < 5 s")


if __name__ == "__main__":
    tests = [
        test_efficiency_range,
        test_power_scales_with_QH,
        test_energy_conservation,
        test_head_loss_law,
        test_net_head_reduced_by_loss,
        test_steady_convergence,
        test_gate_step_response,
        test_surge_oscillation,
        test_part_load_efficiency_drops,
        test_gate_closed_zero_power,
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
    print(f"EC130 Micro-Hydro F2a -- Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    sys.exit(0 if failed == 0 else 1)
