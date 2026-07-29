"""
EC098 -- Organic Rankine Cycle (ORC) -- F2a Thermo Cycle Steady-State
Test suite: physics sanity, energy conservation, edge cases, benchmarks.
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from model import ORC_F2a, R245faProperties
from predict import ComponentModel

PASS = "\u2713"
FAIL = "\u2717"


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
def test_r245fa_saturation():
    print("\n[Test 1] R245fa saturation properties consistency")
    fluid = R245faProperties
    for T in [300.0, 340.0, 380.0, 410.0]:
        P = fluid.P_sat(T)
        T_back = fluid.T_sat(P)
        assert_true(abs(T_back - T) < 0.5, f"T_sat(P_sat({T:.0f}))={T_back:.2f} ~ {T:.0f} K")
        h_fg = fluid.h_fg(T)
        assert_true(h_fg > 0, f"h_fg({T:.0f})={h_fg:.0f} > 0 J/kg")


def test_cycle_efficiency_below_carnot():
    print("\n[Test 2] Thermal efficiency below Carnot limit")
    m, _ = make_model()
    r = m.compute_cycle()
    eta = r["eta_thermal"]
    eta_c = r["eta_carnot"]
    assert_true(eta < eta_c, f"eta={eta:.4f} < eta_carnot={eta_c:.4f}")
    assert_true(eta > 0.01, f"eta={eta:.4f} > 0.01 (reasonable)")
    assert_true(eta < 0.30, f"eta={eta:.4f} < 0.30 (ORC range)")


def test_energy_conservation():
    print("\n[Test 3] Energy conservation: W_net = Q_in - Q_out (approx)")
    m, _ = make_model()
    r = m.compute_cycle()
    W = r["W_net"]
    Q_in = r["Q_in"]
    Q_out = r["Q_out"]
    balance = abs(W - (Q_in - Q_out)) / max(abs(W), 1.0)
    assert_true(balance < 0.01, f"Energy balance error: {balance*100:.2f}% < 1%")


def test_state_points_monotonicity():
    print("\n[Test 4] State point temperatures physically consistent")
    m, _ = make_model()
    r = m.compute_cycle()
    T = r["state_points"]["T"]
    # T1 < T2 (pump heats liquid slightly)
    assert_true(T[1] >= T[0], f"T2={T[1]:.1f} >= T1={T[0]:.1f}")
    # T3 is highest (evaporator outlet)
    assert_true(T[2] > T[1], f"T3={T[2]:.1f} > T2={T[1]:.1f}")
    # T4 < T3 (expansion cools)
    assert_true(T[3] < T[2], f"T4={T[3]:.1f} < T3={T[2]:.1f}")


def test_pressure_ratio():
    print("\n[Test 5] Pressure consistency across cycle")
    m, _ = make_model()
    r = m.compute_cycle()
    P = r["state_points"]["P"]
    # P1 = P_cond, P2 = P_evap, P3 = P_evap, P4 = P_cond
    assert_true(abs(P[0] - P[3]) < 1.0, "P1 ~ P4 (both at condenser)")
    assert_true(abs(P[1] - P[2]) < 1.0, "P2 ~ P3 (both at evaporator)")
    assert_true(P[1] > P[0], f"P_evap={P[1]:.0f} > P_cond={P[0]:.0f}")


def test_part_load_efficiency_drops():
    print("\n[Test 6] Part-load efficiency decreases")
    m, _ = make_model()
    r_full = m.compute_cycle(load_fraction=1.0)
    r_half = m.compute_cycle(load_fraction=0.5)
    r_low = m.compute_cycle(load_fraction=0.2)
    assert_true(r_full["eta_thermal"] > r_half["eta_thermal"],
                f"eta(1.0)={r_full['eta_thermal']:.4f} > eta(0.5)={r_half['eta_thermal']:.4f}")
    assert_true(r_half["eta_thermal"] > r_low["eta_thermal"],
                f"eta(0.5)={r_half['eta_thermal']:.4f} > eta(0.2)={r_low['eta_thermal']:.4f}")


def test_part_load_power_scales():
    print("\n[Test 7] Part-load power output scales approximately with load")
    m, _ = make_model()
    r_full = m.compute_cycle(load_fraction=1.0)
    r_half = m.compute_cycle(load_fraction=0.5)
    ratio = r_half["W_net"] / r_full["W_net"]
    assert_true(0.3 < ratio < 0.7, f"W_half/W_full={ratio:.3f} in [0.3, 0.7]")


def test_higher_pressure_improves_efficiency():
    print("\n[Test 8] Higher evaporator pressure improves efficiency (to a point)")
    m, _ = make_model()
    r_low = m.compute_cycle(P_evap=500000.0)
    r_high = m.compute_cycle(P_evap=1800000.0)
    assert_true(r_high["eta_thermal"] > r_low["eta_thermal"],
                f"eta(2.5MPa)={r_high['eta_thermal']:.4f} > eta(1.0MPa)={r_low['eta_thermal']:.4f}")


def test_mass_flow_positive():
    print("\n[Test 9] Mass flow rate positive and reasonable")
    m, _ = make_model()
    r = m.compute_cycle()
    assert_true(r["m_dot"] > 0, f"m_dot={r['m_dot']:.4f} > 0")
    assert_true(r["m_dot"] < 50.0, f"m_dot={r['m_dot']:.4f} < 50 kg/s (reasonable for 100kW ORC)")


def test_predict_interface():
    print("\n[Test 10] ComponentModel predict() interface")
    _, cm = make_model()
    r = cm.predict({"load_fraction": 0.8})
    for key in ["W_net", "eta_thermal", "Q_in", "Q_out", "m_dot", "state_points"]:
        assert_true(key in r, f"Key '{key}' in output")


def test_dynamic_simulation():
    print("\n[Test 11] Dynamic simulation runs and returns arrays")
    _, cm = make_model()
    r = cm.predict({
        "mode": "dynamic",
        "load_fraction": 0.8,
        "T_ambient_K": 293.15,
        "dt": 10.0,
        "duration_s": 100.0,
    })
    assert_true("t" in r, "Has time array")
    assert_true(len(r["t"]) > 5, f"Has {len(r['t'])} time steps")
    assert_true(np.all(r["W_net"] > 0), "All W_net > 0")


def test_benchmark():
    print("\n[Test 12] Benchmark: 100 cycle calculations")
    m, _ = make_model()
    t0 = time.perf_counter()
    for _ in range(100):
        m.compute_cycle(load_fraction=0.8)
    elapsed = time.perf_counter() - t0
    print(f"  100 cycle calcs in {elapsed*1000:.1f} ms")
    assert_true(elapsed < 5.0, "Completes in < 5 s")


if __name__ == "__main__":
    tests = [
        test_r245fa_saturation,
        test_cycle_efficiency_below_carnot,
        test_energy_conservation,
        test_state_points_monotonicity,
        test_pressure_ratio,
        test_part_load_efficiency_drops,
        test_part_load_power_scales,
        test_higher_pressure_improves_efficiency,
        test_mass_flow_positive,
        test_predict_interface,
        test_dynamic_simulation,
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
    print(f"EC098 ORC F2a -- Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    sys.exit(0 if failed == 0 else 1)
