"""
EC153 -- Binary Cycle Geothermal Plant -- F2a ORC-Brine Coupling
Test suite: physics sanity, ODE convergence, edge cases.
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from model import BinaryCycleGeothermal_F2a, IsobutaneProperties
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
def test_orc_cycle_energy_balance():
    print("\n[Test 1] ORC cycle energy balance: q_in = w_turbine + q_out + w_pump")
    m, _ = make_model()
    cycle = m.orc_cycle(393.15)
    # Energy balance per unit mass: q_in = w_net_cycle + q_out
    q_in = cycle["q_in"]
    q_out = cycle["q_out"]
    w_turb = cycle["w_turbine"]
    w_pump = cycle["w_pump"]
    # w_net = w_turb - w_pump, so q_in ~ w_turb - w_pump + q_out
    balance = abs(q_in - (w_turb - w_pump + q_out))
    rel_err = balance / q_in if q_in > 0 else 0
    assert_true(rel_err < 0.05, f"Energy balance relative error = {rel_err:.4f} < 5%")


def test_thermal_efficiency_range():
    print("\n[Test 2] Thermal efficiency in realistic range (5-20%)")
    m, _ = make_model()
    cycle = m.orc_cycle(393.15)
    eta = cycle["eta_thermal"]
    assert_true(0.05 < eta < 0.25, f"eta_thermal = {eta:.4f} in [0.05, 0.25]")


def test_net_power_positive():
    print("\n[Test 3] Net power output is positive")
    m, _ = make_model()
    cycle = m.orc_cycle(393.15)
    assert_true(cycle["W_net"] > 0, f"W_net = {cycle['W_net']/1e6:.3f} MW > 0")


def test_parasitic_less_than_gross():
    print("\n[Test 4] Parasitic power < gross turbine power")
    m, _ = make_model()
    cycle = m.orc_cycle(393.15)
    assert_true(cycle["W_parasitic"] < cycle["W_turbine"],
                f"W_parasitic={cycle['W_parasitic']/1e3:.1f} kW < "
                f"W_turbine={cycle['W_turbine']/1e3:.1f} kW")


def test_higher_brine_more_power():
    print("\n[Test 5] Higher T_evap -> more net power")
    m, _ = make_model()
    # Directly test ORC cycle at two different evaporator temperatures
    c_lo = m.orc_cycle(363.15)   # 90°C evaporator
    c_hi = m.orc_cycle(403.15)   # 130°C evaporator
    assert_true(c_hi["W_net"] > c_lo["W_net"],
                f"W_net(T_evap=130C)={c_hi['W_net']/1e3:.0f} kW > "
                f"W_net(T_evap=90C)={c_lo['W_net']/1e3:.0f} kW")


def test_brine_outlet_above_minimum():
    print("\n[Test 6] Brine outlet temperature >= minimum reinjection T")
    m, _ = make_model()
    T_out = m.brine_outlet_temperature(443.15, 393.15)
    assert_true(T_out >= m.T_brine_min,
                f"T_brine_out={T_out-273.15:.1f} C >= "
                f"T_min={m.T_brine_min-273.15:.1f} C")


def test_pinch_point_constraint():
    print("\n[Test 7] Pinch-point constraint respected")
    m, _ = make_model()
    # With very low brine T, evap T should adapt
    T_evap = m.effective_evap_temperature(383.15)  # 110 C brine
    assert_true(T_evap < 383.15 - m.dT_pinch,
                f"T_evap={T_evap-273.15:.1f} C < "
                f"T_brine-dT_pinch={383.15-m.dT_pinch-273.15:.1f} C")


def test_thermal_transient_startup():
    print("\n[Test 8] Startup transient: evaporator approaches steady state")
    m, _ = make_model()
    # Start from condenser-side temperature — should heat up toward equilibrium
    T_init = m.T_cond + 5.0  # just above condenser
    r = m.simulate(443.15, T_init, 5.0, 5000.0)
    assert_true(r["T_evap"][-1] > T_init,
                f"T_evap_final={r['T_evap'][-1]-273.15:.1f} C > "
                f"{T_init-273.15:.1f} C (initial)")
    assert_true(r["T_evap"][-1] < 443.15,
                f"T_evap_final < T_brine_in (thermodynamic limit)")


def test_step_response():
    print("\n[Test 9] Step response: brine T drop -> power decrease")
    m, _ = make_model()
    def brine_step(t):
        return 443.15 if t < 500 else 413.15
    r = m.simulate(brine_step, 393.15, 5.0, 1500.0)
    idx_before = np.argmin(np.abs(r["t"] - 490))
    idx_after = np.argmin(np.abs(r["t"] - 1400))
    assert_true(r["W_net"][idx_after] < r["W_net"][idx_before],
                "Power decreases after brine temperature drop")


def test_predict_interface():
    print("\n[Test 10] ComponentModel predict() interface")
    _, cm = make_model()
    r = cm.predict({"dt": 5.0, "duration_s": 30.0})
    for key in ["t", "T_evap", "T_brine_in", "T_brine_out", "W_net",
                "W_turbine", "W_parasitic", "Q_in", "eta_thermal"]:
        assert_true(key in r, f"Key '{key}' in output")
    assert_true(len(r["t"]) == len(r["W_net"]), "Arrays same length")


def test_isobutane_properties_monotone():
    print("\n[Test 11] Isobutane properties: P_sat increases with T")
    wf = IsobutaneProperties()
    T_arr = np.linspace(280, 400, 20)
    P_prev = wf.P_sat(T_arr[0])
    for T in T_arr[1:]:
        P = wf.P_sat(T)
        assert_true(P > P_prev, f"P_sat({T:.0f} K)={P:.0f} > P_prev={P_prev:.0f}")
        P_prev = P


def test_benchmark():
    print("\n[Test 12] Benchmark: 3000s sim at dt=5")
    m, _ = make_model()
    t0 = time.perf_counter()
    m.simulate(443.15, 393.15, 5.0, 3000.0)
    elapsed = time.perf_counter() - t0
    print(f"  3000s simulation in {elapsed*1000:.1f} ms")
    assert_true(elapsed < 10.0, "Completes in < 10 s")


if __name__ == "__main__":
    tests = [
        test_orc_cycle_energy_balance,
        test_thermal_efficiency_range,
        test_net_power_positive,
        test_parasitic_less_than_gross,
        test_higher_brine_more_power,
        test_brine_outlet_above_minimum,
        test_pinch_point_constraint,
        test_thermal_transient_startup,
        test_step_response,
        test_predict_interface,
        test_isobutane_properties_monotone,
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
    print(f"EC153 Binary Cycle Geothermal F2a -- Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    sys.exit(0 if failed == 0 else 1)
