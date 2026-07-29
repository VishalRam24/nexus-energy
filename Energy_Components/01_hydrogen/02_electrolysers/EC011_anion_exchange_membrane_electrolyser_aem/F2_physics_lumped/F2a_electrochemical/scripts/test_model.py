"""
EC011 -- Anion Exchange Membrane (AEM) Electrolyser -- F2a Electrochemical
Test suite: physics sanity, ODE behaviour, conservation, edge cases.
Run with system python3:  python3 scripts/test_model.py   (NO pytest)
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from model import AEM_F2a
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
def test_reversible_range():
    print("\n[Test 1] Reversible voltage in physical range, falls with T, rises with P")
    m, _ = make_model()
    E = m.reversible_voltage(333.15, 1.0, 1.0)
    assert_true(1.1 < E < 1.25, f"E_rev={E:.4f} V in [1.1, 1.25]")
    E_hot = m.reversible_voltage(353.15, 1.0, 1.0)
    assert_true(E_hot < E, f"E_rev falls with T: {E_hot:.4f} < {E:.4f}")
    E_p = m.reversible_voltage(333.15, 30.0, 30.0)
    assert_true(E_p > E, f"Nernst raises E_rev with pressure: {E_p:.4f} > {E:.4f}")


def test_voltage_above_reversible():
    print("\n[Test 2] ELECTROLYSIS: V_cell > E_rev for j > 0")
    m, _ = make_model()
    for j in [0.1, 0.5, 1.0, 2.0]:
        V = m.cell_voltage(j, 333.15, 1.0, 1.0)
        E = m.reversible_voltage(333.15, 1.0, 1.0)
        assert_true(V > E, f"V({j})={V:.4f} > E_rev={E:.4f}")


def test_voltage_monotone_increasing():
    print("\n[Test 3] V_cell increases with current density j")
    m, _ = make_model()
    j_vals = np.linspace(0.05, 2.8, 50)
    V_prev = m.cell_voltage(j_vals[0], 333.15, 1.0, 1.0)
    for j in j_vals[1:]:
        V = m.cell_voltage(j, 333.15, 1.0, 1.0)
        assert_true(V >= V_prev - 1e-9, f"V({j:.2f})={V:.4f} >= V_prev={V_prev:.4f}")
        V_prev = V
    print("  All 49 pairs checked (monotone increasing).")


def test_faradaic_efficiency_bounds():
    print("\n[Test 4] 0 < faradaic_eff <= 1, increases with j")
    m, _ = make_model()
    e_low = m.faradaic_efficiency(0.1)
    e_high = m.faradaic_efficiency(2.0)
    assert_true(0.0 < e_low <= 1.0, f"eta_F(0.1)={e_low:.4f} in (0,1]")
    assert_true(0.0 < e_high <= 1.0, f"eta_F(2.0)={e_high:.4f} in (0,1]")
    assert_true(e_high > e_low, f"eta_F rises with j: {e_high:.4f} > {e_low:.4f}")
    assert_true(e_high <= m.eta_F_max + 1e-9, f"capped at eta_F_max={m.eta_F_max}")


def test_h2_rate_faraday():
    print("\n[Test 5] H2 rate follows Faraday's law (scales with current, eta_F)")
    m, _ = make_model()
    n1 = m.hydrogen_rate(1.0)
    I = 1.0 * m.A_cell
    expected = m.faradaic_efficiency(1.0) * m.N_cells * I / (2.0 * m.F)
    assert_true(abs(n1 - expected) < 1e-12, f"n_H2={n1:.3e} matches Faraday {expected:.3e}")
    assert_true(m.hydrogen_rate(2.0) > n1, "More current -> more H2")
    assert_true(m.hydrogen_rate(0.0) == 0.0, "Zero current -> zero H2")


def test_thermal_ode_heats_up():
    print("\n[Test 6] Thermal ODE: stack heats up from cold start")
    m, _ = make_model()
    r = m.simulate(1.5, 300.0, 1.0, 1.0, 1.0, 300.0)
    assert_true(r["temperature"][-1] > 300.0, f"T_final={r['temperature'][-1]:.2f} > 300 K")
    assert_true(r["temperature"][-1] < 380.0, f"T_final={r['temperature'][-1]:.2f} < 380 K (reasonable)")


def test_thermal_steady_state():
    print("\n[Test 7] Thermal approaches steady state under constant load")
    m, _ = make_model()
    r = m.simulate(1.0, 313.15, 1.0, 1.0, 2.0, 3600.0)
    dT = abs(r["temperature"][-1] - r["temperature"][-2])
    assert_true(dT < 0.05, f"Near SS: dT={dT:.5f} K between last two steps")


def test_efficiency_range():
    print("\n[Test 8] HHV efficiency in (0, 1)")
    m, _ = make_model()
    r = m.simulate(1.0, 333.15, 1.0, 1.0, 2.0, 20.0)
    for eta in r["efficiency"]:
        assert_true(0.0 < eta < 1.0, f"eff={eta:.4f} in (0,1)")


def test_energy_conservation():
    print("\n[Test 9] Energy conservation: P_el = P_H2(HHV) + Q_gen at steady T")
    m, _ = make_model()
    j, T = 1.0, 333.15
    HHV = 285800.0
    V = m.cell_voltage(j, T, 1.0, 1.0)
    I = j * m.A_cell
    P_el = m.N_cells * V * I                       # W (stack)
    P_h2 = m.hydrogen_rate(j, T) * HHV             # W stored chemically
    # Heat from irreversibilities + Faradaic loss (electrical input not converted to V*I_faradaic H2)
    E_tn = m.thermoneutral_voltage(T)
    Q_irrev = m.N_cells * j * m.A_cell * (V - E_tn)
    # Faradaic loss: part of current does not make H2
    Q_far = m.N_cells * j * m.A_cell * E_tn * (1.0 - m.faradaic_efficiency(j))
    residual = P_el - (P_h2 + Q_irrev + Q_far)
    assert_true(abs(residual) / P_el < 0.02,
                f"Balance residual {residual:.2f} W / {P_el:.1f} W < 2%")


def test_overpotentials_positive():
    print("\n[Test 10] All overpotentials >= 0; ohmic dominates AEM at high j")
    m, _ = make_model()
    r = m.simulate(2.0, 333.15, 1.0, 1.0, 2.0, 10.0)
    for name, arr in r["overpotentials"].items():
        if name == "E_rev":
            continue
        assert_true(np.all(arr >= -1e-9), f"{name} all >= 0")
    # AEM membrane ohmic should be a meaningful share at high current
    assert_true(r["overpotentials"]["ohmic"][-1] > 0.05,
                f"ohmic loss {r['overpotentials']['ohmic'][-1]:.4f} V significant at j=2")


def test_predict_interface():
    print("\n[Test 11] ComponentModel predict() interface")
    _, cm = make_model()
    r = cm.predict({"current_density_A_cm2": 1.0, "dt": 2.0, "duration_s": 10.0})
    for key in ["t", "voltage", "stack_voltage", "power_kW", "h2_rate_mol_s",
                "efficiency", "faradaic_eff", "temperature", "overpotentials"]:
        assert_true(key in r, f"Key '{key}' in output")
    assert_true(len(r["t"]) == len(r["voltage"]), "Arrays same length")
    info = cm.get_info()
    assert_true(info["component_id"] == "EC011", "get_info component_id = EC011")


def test_benchmark():
    print("\n[Test 12] Benchmark: 300s sim at dt=1.0")
    m, _ = make_model()
    t0 = time.perf_counter()
    m.simulate(1.0, 313.15, 1.0, 1.0, 1.0, 300.0)
    elapsed = time.perf_counter() - t0
    print(f"  300s simulation in {elapsed*1000:.1f} ms")
    assert_true(elapsed < 5.0, "Completes in < 5 s")


if __name__ == "__main__":
    tests = [
        test_reversible_range,
        test_voltage_above_reversible,
        test_voltage_monotone_increasing,
        test_faradaic_efficiency_bounds,
        test_h2_rate_faraday,
        test_thermal_ode_heats_up,
        test_thermal_steady_state,
        test_efficiency_range,
        test_energy_conservation,
        test_overpotentials_positive,
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
    print(f"EC011 AEM Electrolyser F2a -- Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    sys.exit(0 if failed == 0 else 1)
