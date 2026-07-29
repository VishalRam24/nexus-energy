"""
EC070 -- Water-Source Heat Pump -- F2a Vapor-Compression Cycle
Test suite: thermodynamic sanity, COP bounds, energy balance, transient ODE,
edge cases, predict() interface, benchmark. NO pytest -- run with:
    python3 scripts/test_model.py
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from model import WaterSourceHP_F2a
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
def test_cop_above_one():
    print("\n[Test 1] COP > 1 across operating envelope")
    m, _ = make_model()
    for Ts in [5, 12, 20, 30]:
        for Tk in [30, 45, 55]:
            st = m.cycle(Ts, Tk)
            assert_true(st["cop_heat"] > 1.0,
                        f"COP_h(src={Ts},sink={Tk})={st['cop_heat']:.3f} > 1")


def test_cop_below_carnot():
    print("\n[Test 2] COP < Carnot COP (second law)")
    m, _ = make_model()
    for Ts in [5, 12, 20, 30]:
        for Tk in [30, 45, 55]:
            st = m.cycle(Ts, Tk)
            assert_true(st["cop_heat"] < st["cop_carnot"],
                        f"COP={st['cop_heat']:.3f} < Carnot={st['cop_carnot']:.3f} "
                        f"(src={Ts},sink={Tk})")


def test_energy_balance():
    print("\n[Test 3] First-law energy balance Q_cond = Q_evap + W_comp")
    m, _ = make_model()
    for Ts, Tk in [(12, 45), (5, 55), (20, 35)]:
        st = m.cycle(Ts, Tk)
        lhs = st["Q_cond"]
        rhs = st["Q_evap"] + st["W_comp"]
        rel = abs(lhs - rhs) / lhs
        assert_true(rel < 1e-6,
                    f"Q_cond={lhs/1000:.3f}kW == Q_evap+W={rhs/1000:.3f}kW "
                    f"(rel err {rel:.2e})")


def test_cop_decreases_with_lift():
    print("\n[Test 4] COP decreases as temperature lift increases")
    m, _ = make_model()
    cop_prev = None
    for Tk in [30, 40, 50, 60]:
        st = m.cycle(12.0, Tk)
        if cop_prev is not None:
            assert_true(st["cop_heat"] < cop_prev + 1e-9,
                        f"COP(sink={Tk})={st['cop_heat']:.3f} <= prev {cop_prev:.3f}")
        cop_prev = st["cop_heat"]


def test_cop_increases_with_source():
    print("\n[Test 5] COP increases as source temperature rises (less lift)")
    m, _ = make_model()
    cop_prev = None
    for Ts in [5, 12, 20, 30]:
        st = m.cycle(Ts, 45.0)
        if cop_prev is not None:
            assert_true(st["cop_heat"] > cop_prev - 1e-9,
                        f"COP(src={Ts})={st['cop_heat']:.3f} >= prev {cop_prev:.3f}")
        cop_prev = st["cop_heat"]


def test_pressure_and_states_ordering():
    print("\n[Test 6] Thermodynamic state ordering is physical")
    m, _ = make_model()
    st = m.cycle(12.0, 45.0)
    assert_true(st["P_cond"] > st["P_evap"], "P_cond > P_evap")
    assert_true(st["pressure_ratio"] > 1.0, f"PR={st['pressure_ratio']:.2f} > 1")
    assert_true(st["h2"] > st["h1"], "compression raises enthalpy h2 > h1")
    assert_true(st["h2"] > st["h2s"] - 1e-9, "actual work >= isentropic (h2 >= h2s)")
    assert_true(st["h1"] > st["h3"], "evaporator vapor enthalpy > liquid h1 > h3")
    assert_true(0.2 < st["eta_vol"] <= 1.0, f"eta_vol={st['eta_vol']:.3f} in (0,1]")
    assert_true(0.3 < st["eta_is"] < 0.95, f"eta_is={st['eta_is']:.3f} in range")


def test_capacity_magnitude():
    print("\n[Test 7] Heating capacity is in the expected 50 kW class")
    m, _ = make_model()
    st = m.cycle(12.0, 45.0)
    Q_kw = st["Q_cond"] / 1000.0
    assert_true(10.0 < Q_kw < 120.0, f"Q_cond={Q_kw:.1f} kW in 10-120 kW window")
    assert_true(st["m_dot"] > 0, f"m_dot={st['m_dot']:.4f} kg/s > 0")


def test_transient_warms_up():
    print("\n[Test 8] Transient ODE: load loop warms toward demand balance")
    m, _ = make_model()
    r = m.simulate(T_source_c=12.0, T_load0_c=25.0, Q_demand_W=15000.0,
                   duration_s=2400.0, dt=60.0)
    assert_true(r["T_load"][-1] > r["T_load"][0],
                f"T_load {r['T_load'][0]:.2f} -> {r['T_load'][-1]:.2f} degC (rises)")
    assert_true(r["T_load"][-1] < 70.0, f"T_load_final={r['T_load'][-1]:.2f} < 70 degC")
    assert_true(np.all(r["cop"] > 1.0), "COP > 1 throughout transient")


def test_transient_setpoint_cutoff():
    print("\n[Test 9] Transient with setpoint stops heating near setpoint")
    m, _ = make_model()
    r = m.simulate(T_source_c=12.0, T_load0_c=30.0, Q_demand_W=2000.0,
                   duration_s=3600.0, dt=60.0, T_setpoint_c=45.0)
    assert_true(r["T_load"][-1] <= 46.0,
                f"T_load capped near setpoint: {r['T_load'][-1]:.2f} <= 46 degC")


def test_steady_state_balance():
    print("\n[Test 10] Transient reaches quasi steady-state (dT/dt -> 0)")
    m, _ = make_model()
    r = m.simulate(T_source_c=12.0, T_load0_c=20.0, Q_demand_W=40000.0,
                   duration_s=6000.0, dt=120.0)
    dT = abs(r["T_load"][-1] - r["T_load"][-2])
    assert_true(dT < 0.5, f"Near SS: dT={dT:.4f} K between last two steps")


def test_predict_interface():
    print("\n[Test 11] ComponentModel predict() interface (both modes)")
    _, cm = make_model()
    c = cm.predict({"mode": "cycle", "T_source_c": 12.0, "T_sink_c": 45.0})
    for k in ["cop_heat", "cop_carnot", "Q_cond", "Q_evap", "W_comp", "m_dot"]:
        assert_true(k in c, f"cycle key '{k}' present")
    t = cm.predict({"mode": "transient", "T_source_c": 12.0, "T_load0_c": 30.0,
                    "Q_demand_W": 20000.0, "duration_s": 600.0, "dt": 60.0})
    for k in ["t", "T_load", "cop", "Q_cond", "W_comp"]:
        assert_true(k in t, f"transient key '{k}' present")
    assert_true(len(t["t"]) == len(t["T_load"]), "transient arrays same length")


def test_benchmark():
    print("\n[Test 12] Benchmark: 1 h transient at dt=30 s")
    m, _ = make_model()
    t0 = time.perf_counter()
    m.simulate(T_source_c=12.0, T_load0_c=30.0, Q_demand_W=20000.0,
               duration_s=3600.0, dt=30.0)
    elapsed = time.perf_counter() - t0
    print(f"  1 h transient in {elapsed*1000:.1f} ms")
    assert_true(elapsed < 5.0, "Completes in < 5 s")


if __name__ == "__main__":
    tests = [
        test_cop_above_one,
        test_cop_below_carnot,
        test_energy_balance,
        test_cop_decreases_with_lift,
        test_cop_increases_with_source,
        test_pressure_and_states_ordering,
        test_capacity_magnitude,
        test_transient_warms_up,
        test_transient_setpoint_cutoff,
        test_steady_state_balance,
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

    print(f"\n{'='*64}")
    print(f"EC070 WSHP F2a vapor-compression -- Results: {passed} passed, {failed} failed")
    print(f"{'='*64}")
    sys.exit(0 if failed == 0 else 1)
