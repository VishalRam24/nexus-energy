"""
EC003 -- Alkaline Fuel Cell (AFC) -- F2a Electrochemical
Test suite: physics sanity, KOH conductivity, ODE convergence, edge cases.
Run with system python3 (NOT pytest):  python3 scripts/test_model.py
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from model import AFC_F2a
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
    print("\n[Test 1] Nernst voltage in physical range + pressure monotonicity")
    m, _ = make_model()
    E = m.nernst_voltage(343.15, 1.0, 0.21)
    assert_true(1.0 < E < 1.3, f"E_nernst={E:.4f} V in [1.0, 1.3]")
    E2 = m.nernst_voltage(343.15, 2.0, 0.5)
    assert_true(E2 > E, f"Higher reactant pressures raise E: {E2:.4f} > {E:.4f}")


def test_koh_conductivity_peak():
    print("\n[Test 2] KOH conductivity (Gilliam 2007): positive + peaks ~6-7 mol/L")
    m, _ = make_model()
    T = 343.15
    kappas = {c: m.koh_conductivity(T, c) for c in [1, 3, 6, 6.9, 9, 12]}
    for c, k in kappas.items():
        assert_true(k > 0, f"kappa({c} mol/L)={k:.4f} S/cm > 0")
    c_peak = max(kappas, key=kappas.get)
    assert_true(5.0 <= c_peak <= 8.0,
                f"Conductivity peaks at c={c_peak} mol/L (literature ~6-7)")
    assert_true(1.0 < kappas[6.9] < 1.5,
                f"Peak kappa={kappas[6.9]:.3f} S/cm in physical [1.0,1.5] range")


def test_voltage_below_nernst():
    print("\n[Test 3] V_cell < E_nernst for j > 0")
    m, _ = make_model()
    for j in [0.05, 0.4, 0.9]:
        V = m.cell_voltage(j, 343.15, 1.0, 0.21)
        E = m.nernst_voltage(343.15, 1.0, 0.21)
        assert_true(V < E, f"V({j})={V:.4f} < E={E:.4f}")


def test_voltage_monotone():
    print("\n[Test 4] V_cell decreases monotonically with j")
    m, _ = make_model()
    j_vals = np.linspace(0.005, 0.95, 50)
    V_prev = m.cell_voltage(j_vals[0], 343.15, 1.0, 0.21)
    for j in j_vals[1:]:
        V = m.cell_voltage(j, 343.15, 1.0, 0.21)
        assert_true(V <= V_prev + 1e-9, f"V({j:.2f})={V:.4f} <= V_prev={V_prev:.4f}")
        V_prev = V
    print("  All 49 pairs checked.")


def test_thermal_ode_heats_up():
    print("\n[Test 5] Thermal ODE: stack heats up from cold start, stays bounded")
    m, _ = make_model()
    r = m.simulate(0.5, 313.15, 1.0, 0.21, 0.5, 120.0)
    assert_true(r["temperature"][-1] > 313.15,
                f"T_final={r['temperature'][-1]:.2f} > 313.15 K")
    assert_true(r["temperature"][-1] < 380.0,
                f"T_final={r['temperature'][-1]:.2f} < 380 K (physical)")


def test_thermal_steady_state():
    print("\n[Test 6] Thermal reaches approximate steady state")
    m, _ = make_model()
    r = m.simulate(0.4, 343.15, 1.0, 0.21, 1.0, 1200.0)
    dT = abs(r["temperature"][-1] - r["temperature"][-2])
    assert_true(dT < 0.1, f"Near SS: dT={dT:.5f} K between last two steps")


def test_energy_conservation():
    print("\n[Test 7] Energy balance: electrical + heat-out + storage = enthalpy input")
    m, _ = make_model()
    j = 0.5
    r = m.simulate(j, 343.15, 1.0, 0.21, 1.0, 300.0)
    t = r["t"]
    # Enthalpy (HHV) power released by reaction = N*A*j*E_tn
    E_tn = m.thermoneutral_voltage(r["temperature"])
    P_chem = m.N_cells * m.A_cell * j * E_tn               # W
    P_elec = m.N_cells * m.A_cell * r["power_density"]      # W
    P_cool = m.hA_cool * (r["temperature"] - m.T_coolant)   # W
    # Stored thermal power = m*cp*dT/dt
    dTdt = np.gradient(r["temperature"], t)
    P_store = m.m_stack * m.cp_stack * dTdt                 # W
    residual = P_chem - (P_elec + P_cool + P_store)
    rel = np.max(np.abs(residual)) / np.max(P_chem)
    assert_true(rel < 1e-3, f"Energy balance closes: max rel residual={rel:.2e}")


def test_efficiency_range():
    print("\n[Test 8] Efficiency strictly in (0, 1)")
    m, _ = make_model()
    r = m.simulate(0.4, 343.15, 1.0, 0.21, 1.0, 10.0)
    for eta in r["efficiency"]:
        assert_true(0 < eta < 1.0, f"eta={eta:.4f} in (0,1)")


def test_overpotentials_positive():
    print("\n[Test 9] All overpotentials >= 0")
    m, _ = make_model()
    r = m.simulate(0.5, 343.15, 1.0, 0.21, 1.0, 10.0)
    for name, arr in r["overpotentials"].items():
        if name == "E_nernst":
            continue
        assert_true(np.all(arr >= -1e-9), f"{name} all >= 0")


def test_step_response():
    print("\n[Test 10] Step current up -> voltage drops")
    m, _ = make_model()

    def step_j(t):
        return 0.2 if t < 30 else 0.7

    r = m.simulate(step_j, 343.15, 1.0, 0.21, 0.5, 60.0)
    idx_before = np.argmin(np.abs(r["t"] - 29.5))
    idx_after = np.argmin(np.abs(r["t"] - 31.0))
    assert_true(r["voltage"][idx_after] < r["voltage"][idx_before],
                "Voltage drops after current step up")


def test_concentration_near_jL():
    print("\n[Test 11] Concentration loss diverges near j_L")
    m, _ = make_model()
    v1 = m.concentration_overpotential(0.5)
    v2 = m.concentration_overpotential(0.97)
    assert_true(v2 > v1 * 3,
                f"eta_conc(0.97)={v2:.4f} >> eta_conc(0.5)={v1:.4f}")


def test_predict_interface_and_benchmark():
    print("\n[Test 12] ComponentModel predict() interface + benchmark timing")
    _, cm = make_model()
    r = cm.predict({"current_density_A_cm2": 0.4, "dt": 1.0, "duration_s": 5.0})
    for key in ["t", "voltage", "power_density", "efficiency",
                "temperature", "koh_conductivity", "overpotentials"]:
        assert_true(key in r, f"Key '{key}' in output")
    assert_true(len(r["t"]) == len(r["voltage"]), "Arrays same length")

    m, _ = make_model()
    t0 = time.perf_counter()
    m.simulate(0.4, 343.15, 1.0, 0.21, 0.1, 60.0)
    elapsed = time.perf_counter() - t0
    print(f"  60s simulation (dt=0.1) in {elapsed*1000:.1f} ms")
    assert_true(elapsed < 5.0, "Representative sim completes in < 5 s")


if __name__ == "__main__":
    tests = [
        test_nernst_range,
        test_koh_conductivity_peak,
        test_voltage_below_nernst,
        test_voltage_monotone,
        test_thermal_ode_heats_up,
        test_thermal_steady_state,
        test_energy_conservation,
        test_efficiency_range,
        test_overpotentials_positive,
        test_step_response,
        test_concentration_near_jL,
        test_predict_interface_and_benchmark,
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
    print(f"EC003 AFC F2a -- Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    sys.exit(0 if failed == 0 else 1)
