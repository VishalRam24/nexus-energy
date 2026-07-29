"""
EC192 -- Gas Pressure Regulator -- F2a Physics-Lumped Diaphragm/Valve Dynamics
Test suite: regulation, droop, lockup, choked-flow limit, JT cooling,
mass conservation, ODE convergence, edge cases, predict() interface.
Run with system python3 (NOT pytest):  python3 scripts/test_model.py
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from model import GasPressureRegulatorF2a
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


def load_kg_s(model, Qm3h):
    return Qm3h * model.rho_std / 3600.0


# ---------------------------------------------------------------------------
def test_regulation_to_setpoint():
    print("\n[Test 1] Downstream pressure regulated near setpoint")
    m, _ = make_model()
    s = m.steady_state(50.0, load_kg_s(m, 15000.0), 288.15, duration_s=150.0)
    P_set = m.P_set / 1e5
    err = abs(s["P_down_bar"] - P_set)
    assert_true(err < 0.5,
                f"P_down={s['P_down_bar']:.3f} bar within 0.5 bar of setpoint {P_set:.2f}")


def test_valve_self_adjusts_to_load():
    print("\n[Test 2] Valve travel passes exactly the load flow (mass balance SS)")
    m, _ = make_model()
    s = m.steady_state(50.0, load_kg_s(m, 12000.0), 288.15, duration_s=150.0)
    # At steady state inflow must equal the load draw
    rel = abs(s["flow_std_m3_per_h"] - 12000.0) / 12000.0
    assert_true(rel < 0.02,
                f"SS inflow {s['flow_std_m3_per_h']:.0f} = load 12000 m3/h (rel err {rel*100:.2f}%)")


def test_droop_monotonic():
    print("\n[Test 3] Droop: regulated P_down decreases with increasing load")
    m, _ = make_model()
    loads = [0.0, 10000.0, 20000.0, 30000.0, 40000.0]
    P = [m.steady_state(50.0, load_kg_s(m, q), 288.15, duration_s=150.0)["P_down_bar"]
         for q in loads]
    for i in range(1, len(P)):
        assert_true(P[i] <= P[i - 1] + 1e-3,
                    f"P_down({loads[i]:.0f})={P[i]:.3f} <= prev {P[i-1]:.3f}")
    print(f"  Droop band = {P[0]-P[-1]:.3f} bar over 0..40000 m3/h")


def test_lockup_highest_pressure():
    print("\n[Test 4] Lockup: zero-flow pressure is the highest and valve seats")
    m, _ = make_model()
    s0 = m.steady_state(50.0, 0.0, 288.15, duration_s=150.0)
    s1 = m.steady_state(50.0, load_kg_s(m, 20000.0), 288.15, duration_s=150.0)
    assert_true(s0["P_down_bar"] >= s1["P_down_bar"],
                f"lockup P {s0['P_down_bar']:.3f} >= loaded P {s1['P_down_bar']:.3f}")
    assert_true(s0["valve_travel_frac"] < 0.05,
                f"valve seated at lockup: travel {s0['valve_travel_frac']*100:.1f}%")


def test_choked_flow_limit():
    print("\n[Test 5] Choked-flow: flow independent of P_down once choked")
    m, _ = make_model()
    # full open valve, high upstream -> choked across a range of P_down
    P_up = 50.0
    Q_a = float(m.flow_std_m3_per_h(P_up, 4.0, 288.15, m.x_max))
    Q_b = float(m.flow_std_m3_per_h(P_up, 8.0, 288.15, m.x_max))
    assert_true(m.is_choked(P_up, 4.0) and m.is_choked(P_up, 8.0), "both states choked")
    rel = abs(Q_a - Q_b) / Q_a
    assert_true(rel < 1e-9,
                f"Q(P_d=4)={Q_a:.0f} == Q(P_d=8)={Q_b:.0f} (choked, rel {rel:.1e})")


def test_subsonic_depends_on_pdown():
    print("\n[Test 6] Subsonic: flow rises as P_down falls (not choked)")
    m, _ = make_model()
    P_up = 10.0
    Q_hi = float(m.flow_std_m3_per_h(P_up, 9.0, 288.15, m.x_max))   # small dP
    Q_lo = float(m.flow_std_m3_per_h(P_up, 8.0, 288.15, m.x_max))   # larger dP
    assert_true(not m.is_choked(P_up, 9.0), "10->9 bar is subsonic")
    assert_true(Q_lo > Q_hi, f"more dP -> more flow: {Q_lo:.0f} > {Q_hi:.0f}")


def test_jt_cooling():
    print("\n[Test 7] Joule-Thomson cooling: T_down < T_up on expansion")
    m, _ = make_model()
    T_out = float(m.temperature_out(288.15, 50.0, 4.0))
    assert_true(T_out < 288.15, f"T_down={T_out:.2f} K < T_up=288.15 K (cooling)")
    # larger pressure drop -> more cooling
    T_small = float(m.temperature_out(288.15, 50.0, 45.0))
    assert_true((288.15 - T_out) > (288.15 - T_small),
                "bigger dP gives more JT cooling")


def test_mass_conservation():
    print("\n[Test 8] Mass conservation: dP_d/dt sign follows net mass flux")
    m, _ = make_model()
    # start below setpoint with no load -> valve opens, fills -> P rises
    r = m.simulate(50.0, 0.0, 288.15, P_d0=2.0e5, duration_s=60.0, dt=0.05)
    assert_true(r["P_down_bar"][-1] > r["P_down_bar"][0],
                f"empty volume fills: {r['P_down_bar'][0]:.2f} -> {r['P_down_bar'][-1]:.2f} bar")
    # overpressured with load -> pressure drains toward setpoint
    r2 = m.simulate(50.0, load_kg_s(m, 20000.0), 288.15, P_d0=8.0e5,
                    duration_s=60.0, dt=0.05)
    assert_true(r2["P_down_bar"][-1] < r2["P_down_bar"][0],
                f"overpressure relieves: {r2['P_down_bar'][0]:.2f} -> {r2['P_down_bar'][-1]:.2f} bar")


def test_load_step_response():
    print("\n[Test 9] Load step: P dips then recovers, valve opens further")
    m, _ = make_model()
    def load(t):
        return load_kg_s(m, 8000.0) if t < 30.0 else load_kg_s(m, 30000.0)
    r = m.simulate(50.0, load, 288.15, duration_s=80.0, dt=0.02)
    i_before = np.argmin(np.abs(r["t"] - 29.0))
    i_after = np.argmin(np.abs(r["t"] - 79.0))
    assert_true(r["valve_travel_frac"][i_after] > r["valve_travel_frac"][i_before],
                "valve opens more after load step up")
    # post-step regulated pressure stays in a sane band (droop, not collapse)
    assert_true(2.0 < r["P_down_bar"][i_after] < 6.0,
                f"post-step P_down={r['P_down_bar'][i_after]:.2f} bar held in band")


def test_pressure_positive_bounded():
    print("\n[Test 10] State bounds: P_down>0, travel in [0,1] throughout")
    m, _ = make_model()
    r = m.simulate(80.0, load_kg_s(m, 25000.0), 300.0, duration_s=60.0, dt=0.02)
    assert_true(np.all(r["P_down_bar"] > 0.0), "downstream pressure stays positive")
    assert_true(np.all((r["valve_travel_frac"] >= -1e-9) &
                       (r["valve_travel_frac"] <= 1.0 + 1e-9)),
                "valve travel within [0,1]")


def test_predict_interface():
    print("\n[Test 11] ComponentModel predict() interface")
    _, cm = make_model()
    r = cm.predict({"P_up_bar": 50.0, "load_flow_m3_h": 10000.0,
                    "duration_s": 20.0, "dt": 0.1})
    for key in ["t", "P_down_bar", "P_set_bar", "valve_travel_frac",
                "flow_std_m3_per_h", "T_downstream_K", "JT_cooling_K", "is_choked"]:
        assert_true(key in r, f"Key '{key}' in output")
    assert_true(len(r["t"]) == len(r["P_down_bar"]), "arrays same length")
    info = cm.get_info()
    assert_true(info["component_id"] == "EC192" and info["version"] == "1.0.0",
                "get_info id/version correct")


def test_benchmark():
    print("\n[Test 12] Benchmark: 60 s simulation")
    m, _ = make_model()
    t0 = time.perf_counter()
    m.simulate(50.0, load_kg_s(m, 10000.0), 288.15, duration_s=60.0, dt=0.05)
    elapsed = time.perf_counter() - t0
    print(f"  60 s simulation in {elapsed*1000:.1f} ms")
    assert_true(elapsed < 5.0, "completes in < 5 s")


if __name__ == "__main__":
    tests = [
        test_regulation_to_setpoint,
        test_valve_self_adjusts_to_load,
        test_droop_monotonic,
        test_lockup_highest_pressure,
        test_choked_flow_limit,
        test_subsonic_depends_on_pdown,
        test_jt_cooling,
        test_mass_conservation,
        test_load_step_response,
        test_pressure_positive_bounded,
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

    print(f"\n{'='*62}")
    print(f"EC192 Gas Pressure Regulator F2a -- Results: {passed} passed, {failed} failed")
    print(f"{'='*62}")
    sys.exit(0 if failed == 0 else 1)
