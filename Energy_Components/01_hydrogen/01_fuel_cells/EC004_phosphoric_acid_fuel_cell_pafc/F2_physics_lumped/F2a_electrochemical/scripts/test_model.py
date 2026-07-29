"""
EC004 -- Phosphoric Acid Fuel Cell (PAFC) -- F2a Electrochemical
Test suite: physics sanity, ODE convergence, PAFC-specific behaviour, edge cases.
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from model import PAFC_F2a
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
def test_nernst_range():
    print("\n[Test 1] Nernst voltage in physical PAFC range")
    m, _ = make_model()
    E = m.nernst_voltage(453.15, 1.0, 0.21)
    # At 180C E0 drops below 1.1 V; Nernst air term lowers it further
    assert_true(0.95 < E < 1.15, f"E_nernst={E:.4f} V in [0.95, 1.15]")
    E2 = m.nernst_voltage(453.15, 3.0, 0.5)
    assert_true(E2 > E, f"Higher reactant pressures raise E: {E2:.4f} > {E:.4f}")


def test_voltage_below_nernst():
    print("\n[Test 2] V_cell < E_nernst for j > 0")
    m, _ = make_model()
    E = m.nernst_voltage(453.15, 1.0, 0.21)
    for j in [0.05, 0.3, 0.6]:
        V = m.cell_voltage(j, 453.15, 1.0, 0.21)
        assert_true(V < E, f"V({j})={V:.4f} < E={E:.4f}")


def test_voltage_monotone():
    print("\n[Test 3] V_cell decreases monotonically with j")
    m, _ = make_model()
    j_vals = np.linspace(0.01, 0.69, 50)
    V_prev = m.cell_voltage(j_vals[0], 453.15, 1.0, 0.21)
    for j in j_vals[1:]:
        V = m.cell_voltage(j, 453.15, 1.0, 0.21)
        assert_true(V <= V_prev + 1e-9, f"V({j:.2f})={V:.4f} <= {V_prev:.4f}")
        V_prev = V
    print("  All 49 pairs checked.")


def test_acid_conductivity_increases_with_T():
    print("\n[Test 4] H3PO4 conductivity rises with temperature (Arrhenius)")
    m, _ = make_model()
    s_low = m.acid_conductivity(423.15)
    s_high = m.acid_conductivity(483.15)
    assert_true(s_high > s_low, f"sigma(483K)={s_high:.4f} > sigma(423K)={s_low:.4f}")
    # Higher T => lower ohmic loss at fixed j
    R_low_T = m.ohmic_overpotential(0.4, 423.15)
    R_high_T = m.ohmic_overpotential(0.4, 483.15)
    assert_true(R_high_T < R_low_T, "Ohmic loss falls as T rises")


def test_co_tolerance():
    print("\n[Test 5] CO penalty positive but small, and decays with T")
    m, _ = make_model()
    eta_co = m.co_overpotential(453.15, 0.01)
    assert_true(0.0 < eta_co < 0.10, f"eta_CO={eta_co:.4f} V (small, PAFC-tolerant)")
    eta_co_hot = m.co_overpotential(483.15, 0.01)
    assert_true(eta_co_hot < eta_co, f"CO penalty falls with T: {eta_co_hot:.4f} < {eta_co:.4f}")
    eta_co_zero = m.co_overpotential(453.15, 0.0)
    assert_true(eta_co_zero == 0.0, "No CO => no penalty")


def test_thermal_ode_heats_up():
    print("\n[Test 6] Thermal ODE: stack heats up from cold start")
    m, _ = make_model()
    r = m.simulate(0.3, 423.15, 1.0, 0.21, 2.0, 600.0)
    assert_true(r["temperature"][-1] > 423.15, f"T_final={r['temperature'][-1]:.2f} > 423 K")
    assert_true(r["temperature"][-1] < 520.0, f"T_final={r['temperature'][-1]:.2f} < 520 K (reasonable)")


def test_thermal_steady_state_energy_balance():
    print("\n[Test 7] Steady state: Q_gen == Q_cool (energy conservation)")
    m, _ = make_model()
    r = m.simulate(0.3, 453.15, 1.0, 0.21, 5.0, 3600.0)
    T_ss = r["temperature"][-1]
    dT = abs(r["temperature"][-1] - r["temperature"][-2])
    assert_true(dT < 0.05, f"Near SS: dT={dT:.5f} K between last two steps")
    # Verify the energy balance directly at steady state
    V = m.cell_voltage(0.3, T_ss, 1.0, 0.21)
    E_th = m.thermoneutral_voltage(T_ss)
    Q_gen = m.N_cells * m.A_cell * 0.3 * (E_th - V)
    Q_cool = m.hA_cool * (T_ss - m.T_coolant)
    rel = abs(Q_gen - Q_cool) / Q_gen
    assert_true(rel < 0.02, f"Q_gen={Q_gen:.0f}W ~ Q_cool={Q_cool:.0f}W (rel {rel:.4f})")


def test_efficiency_range():
    print("\n[Test 8] Efficiency in (0, 1)")
    m, _ = make_model()
    r = m.simulate(0.3, 453.15, 1.0, 0.21, 5.0, 30.0)
    for eta in r["efficiency"]:
        assert_true(0 < eta < 1.0, f"eta={eta:.4f}")


def test_overpotentials_positive():
    print("\n[Test 9] All overpotentials >= 0")
    m, _ = make_model()
    r = m.simulate(0.4, 453.15, 1.0, 0.21, 5.0, 30.0)
    for name, arr in r["overpotentials"].items():
        if name == "E_nernst":
            continue
        assert_true(np.all(arr >= -1e-9), f"{name} all >= 0")


def test_concentration_near_jL():
    print("\n[Test 10] Concentration loss diverges near j_L (0.7 A/cm2)")
    m, _ = make_model()
    v1 = m.concentration_overpotential(0.5)
    v2 = m.concentration_overpotential(0.695)
    assert_true(v2 > v1 * 3, f"eta_conc(0.695)={v2:.4f} >> eta_conc(0.5)={v1:.4f}")


def test_step_response():
    print("\n[Test 11] Step current response -- voltage drops on step-up")
    m, _ = make_model()
    def step_j(t):
        return 0.15 if t < 150 else 0.5
    r = m.simulate(step_j, 453.15, 1.0, 0.21, 2.0, 300.0)
    idx_before = np.argmin(np.abs(r["t"] - 148.0))
    idx_after = np.argmin(np.abs(r["t"] - 154.0))
    assert_true(r["voltage"][idx_after] < r["voltage"][idx_before],
                "Voltage drops after current step up")


def test_predict_interface():
    print("\n[Test 12] ComponentModel predict() interface")
    _, cm = make_model()
    r = cm.predict({"current_density_A_cm2": 0.3, "dt": 5.0, "duration_s": 30.0})
    for key in ["t", "voltage", "power_density", "efficiency", "temperature", "overpotentials"]:
        assert_true(key in r, f"Key '{key}' in output")
    assert_true(len(r["t"]) == len(r["voltage"]), "Arrays same length")
    info = cm.get_info()
    assert_true(info["component_id"] == "EC004", "get_info component_id == EC004")


def test_benchmark():
    print("\n[Test 13] Benchmark: 300s sim at dt=1.0")
    m, _ = make_model()
    t0 = time.perf_counter()
    m.simulate(0.3, 453.15, 1.0, 0.21, 1.0, 300.0)
    elapsed = time.perf_counter() - t0
    print(f"  300s simulation in {elapsed*1000:.1f} ms")
    assert_true(elapsed < 5.0, "Completes in < 5 s")


if __name__ == "__main__":
    tests = [
        test_nernst_range,
        test_voltage_below_nernst,
        test_voltage_monotone,
        test_acid_conductivity_increases_with_T,
        test_co_tolerance,
        test_thermal_ode_heats_up,
        test_thermal_steady_state_energy_balance,
        test_efficiency_range,
        test_overpotentials_positive,
        test_concentration_near_jL,
        test_step_response,
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
    print(f"EC004 PAFC F2a -- Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    sys.exit(0 if failed == 0 else 1)
