"""
EC042 -- Pseudocapacitor -- F2a RC-Faradaic (physics-lumped)
Test suite: physics sanity, charge conservation, energy bounds, redox rate
fade, thermal balance, efficiency bounds, predict() interface, benchmark.
NO pytest -- run as `python3 scripts/test_model.py`.
"""

import sys
import os
import time
import numpy as np
from scipy.integrate import trapezoid

sys.path.insert(0, os.path.dirname(__file__))
from model import PseudocapacitorF2a
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
def test_pseudo_exceeds_edlc():
    print("\n[Test 1] Total capacitance > bare double-layer (faradaic boost)")
    m, _ = make_model()
    for v in [0.2, 0.5, 0.8]:
        C = m.differential_capacitance(v, m.T_ref)
        assert_true(C > m.C_dl, f"C_diff({v})={C:.1f} F > C_dl={m.C_dl:.1f} F")
    # peak of faradaic hump is at v_redox
    C_peak = m.differential_capacitance(m.v_redox, m.T_ref)
    C_edge = m.differential_capacitance(0.0, m.T_ref)
    assert_true(C_peak > C_edge, f"C(redox center)={C_peak:.1f} > C(0V)={C_edge:.1f} (voltage dependence)")


def test_charge_conservation():
    print("\n[Test 2] Charge conservation: dQ matches integral of -current")
    m, _ = make_model()
    I = 30.0  # discharge
    r = m.simulate(I, 1.0, m.T_ref, 0.01, 2.0)
    Q0 = m.charge(r["v_cap"][0], r["temperature"][0])
    Q1 = m.charge(r["v_cap"][-1], r["temperature"][-1])
    dQ = Q1 - Q0  # should be negative (discharging)
    # expected charge removed = integral (I + I_leak) dt ~ I*t for small leak
    t = r["t"]
    i_total = I + r["v_cap"] / m.R_leak
    dQ_expected = -trapezoid(i_total, t)
    rel = abs(dQ - dQ_expected) / abs(dQ_expected)
    assert_true(rel < 0.02, f"dQ={dQ:.2f} C vs expected {dQ_expected:.2f} C (rel err {rel*100:.2f}%)")


def test_energy_bounds():
    print("\n[Test 3] Stored energy bounded: 0 <= E <= E(V_max)")
    m, _ = make_model()
    E_max = m.energy_max(m.T_ref)
    for v in [0.0, 0.25, 0.5, 0.75, 1.0]:
        E = m.stored_energy(v, m.T_ref)
        assert_true(0.0 <= E <= E_max + 1e-9, f"E({v})={E:.2f} J in [0, {E_max:.2f}]")
    # monotonic in V
    Es = [m.stored_energy(v, m.T_ref) for v in np.linspace(0, 1, 20)]
    assert_true(all(np.diff(Es) >= -1e-9), "Stored energy monotonically increases with V")


def test_terminal_below_internal_on_discharge():
    print("\n[Test 4] V_term < V_cap on discharge (ESR + R_ct drop)")
    m, _ = make_model()
    for I in [10.0, 50.0]:
        v_term = float(m.terminal_voltage(0.8, I, m.T_ref))
        assert_true(v_term < 0.8, f"I={I}: V_term={v_term:.4f} < V_cap=0.8")
    # on charge (I<0) terminal exceeds internal
    v_term_ch = float(m.terminal_voltage(0.5, -50.0, m.T_ref))
    assert_true(v_term_ch > 0.5, f"charge: V_term={v_term_ch:.4f} > V_cap=0.5")


def test_redox_rate_fade():
    print("\n[Test 5] High-rate redox kinetics reduce accessible capacitance")
    m, _ = make_model()
    f_low = float(m.access_factor(0.5, 5.0, m.T_ref))
    f_high = float(m.access_factor(0.5, 80.0, m.T_ref))
    assert_true(0 < f_high < f_low <= 1.0,
                f"access_factor falls with rate: {f_low:.3f} (5A) -> {f_high:.3f} (80A)")
    # slow branch carries less of the current at high rate
    Is_low = float(m.slow_branch_current(0.5, 5.0, m.T_ref))
    Is_high = float(m.slow_branch_current(0.5, 80.0, m.T_ref))
    assert_true(Is_low / 5.0 > Is_high / 80.0, "Faradaic current share drops at high rate")


def test_voltage_decreases_on_discharge():
    print("\n[Test 6] V_cap monotonically decreases on constant discharge")
    m, _ = make_model()
    r = m.simulate(40.0, 1.0, m.T_ref, 0.02, 3.0)
    v = r["v_cap"]
    assert_true(np.all(np.diff(v) <= 1e-9), "V_cap non-increasing during discharge")
    assert_true(v[-1] < v[0], f"V dropped: {v[0]:.3f} -> {v[-1]:.3f} V")


def test_thermal_heats_and_balances():
    print("\n[Test 7] Thermal ODE: cell heats under load, dT/dt -> 0 at balance")
    m, _ = make_model()
    # hold at fixed v_cap by zero net current? use a moderate continuous discharge
    r = m.simulate(60.0, 0.8, m.T_ref, 0.05, 10.0)
    assert_true(r["temperature"][-1] > m.T_ref, f"T rose: {r['temperature'][-1]:.4f} > {m.T_ref}")
    assert_true(r["temperature"][-1] < m.T_ref + 50.0, "T stays physically reasonable (<+50K)")
    # at thermal balance Q_gen == Q_cool => dT/dt == 0
    # find T where dTdt=0 for a held operating point
    dT_lo = m.dTdt(0.8, 60.0, m.T_ref)
    assert_true(dT_lo > 0, "dT/dt > 0 when Q_gen > Q_cool (cold)")
    T_hot = m.T_ref + 200.0
    dT_hi = m.dTdt(0.8, 0.0, T_hot)  # no load, hot -> cooling
    assert_true(dT_hi < 0, "dT/dt < 0 when only cooling acts (hot, no load)")


def test_leakage_self_discharge():
    print("\n[Test 8] Open-circuit leakage causes self-discharge")
    m, _ = make_model()
    r = m.simulate(0.0, 1.0, m.T_ref, 1.0, 200.0)
    assert_true(r["v_cap"][-1] < r["v_cap"][0], f"self-discharge: {r['v_cap'][0]:.4f} -> {r['v_cap'][-1]:.4f} V")
    assert_true(r["v_cap"][-1] > 0.0, "voltage stays >= 0")


def test_efficiency_strictly_between_0_and_1():
    print("\n[Test 9] Round-trip efficiency strictly in (0,1), lower at high rate")
    m, _ = make_model()
    eta_lo = m.round_trip_efficiency(10.0, 0.2, m.T_ref)
    eta_hi = m.round_trip_efficiency(60.0, 0.2, m.T_ref)
    assert_true(0.0 < eta_lo < 1.0, f"eta@10A={eta_lo:.4f} in (0,1)")
    assert_true(0.0 < eta_hi < 1.0, f"eta@60A={eta_hi:.4f} in (0,1)")
    assert_true(eta_hi < eta_lo, f"higher rate -> lower efficiency: {eta_hi:.4f} < {eta_lo:.4f}")


def test_heat_nonneg_irreversible():
    print("\n[Test 10] Heat generation finite; ohmic+redox terms non-negative")
    m, _ = make_model()
    for I in [-50.0, 0.0, 50.0]:
        # isolate irreversible part: entropic term can be signed, but total at
        # |I|>0 discharge must be positive; ohmic+redox always >=0
        Islow = float(m.slow_branch_current(0.6, I, m.T_ref))
        q_irr = I ** 2 * float(m.esr(m.T_ref)) + Islow ** 2 * float(m.r_ct(m.T_ref))
        assert_true(q_irr >= 0.0, f"I={I}: irreversible heat {q_irr:.3f} W >= 0")
    q_dis = float(m.heat_generation(0.6, 50.0, m.T_ref))
    assert_true(q_dis > 0.0, f"discharge heat {q_dis:.2f} W > 0")


def test_predict_interface():
    print("\n[Test 11] ComponentModel predict() / get_info() interface")
    _, cm = make_model()
    info = cm.get_info()
    for k in ["component_id", "component_name", "fidelity", "version", "inputs", "outputs"]:
        assert_true(k in info, f"get_info has '{k}'")
    assert_true(info["component_id"] == "EC042", "component_id == EC042")
    r = cm.predict({"current_A": 40.0, "v_cap0_V": 1.0, "dt": 0.1, "duration_s": 2.0})
    for key in ["t", "v_cap", "terminal_voltage", "power", "soc", "temperature",
                "stored_energy", "capacitance", "heat"]:
        assert_true(key in r, f"output has '{key}'")
    assert_true(len(r["t"]) == len(r["v_cap"]) == len(r["temperature"]), "arrays same length")


def test_benchmark():
    print("\n[Test 12] Benchmark: 20 s coupled sim at dt=0.01")
    m, _ = make_model()
    t0 = time.perf_counter()
    m.simulate(40.0, 1.0, m.T_ref, 0.01, 20.0)
    elapsed = time.perf_counter() - t0
    print(f"  20 s coupled (V_cap+T) simulation in {elapsed*1000:.1f} ms")
    assert_true(elapsed < 5.0, "completes in < 5 s")


if __name__ == "__main__":
    tests = [
        test_pseudo_exceeds_edlc,
        test_charge_conservation,
        test_energy_bounds,
        test_terminal_below_internal_on_discharge,
        test_redox_rate_fade,
        test_voltage_decreases_on_discharge,
        test_thermal_heats_and_balances,
        test_leakage_self_discharge,
        test_efficiency_strictly_between_0_and_1,
        test_heat_nonneg_irreversible,
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
    print(f"EC042 Pseudocapacitor F2a RC-Faradaic -- Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    sys.exit(0 if failed == 0 else 1)
