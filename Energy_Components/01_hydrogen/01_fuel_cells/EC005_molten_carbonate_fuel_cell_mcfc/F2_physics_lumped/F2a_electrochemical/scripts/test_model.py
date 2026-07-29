"""
EC005 -- Molten Carbonate Fuel Cell (MCFC) -- F2a Electrochemical
Test suite: physics sanity, ODE convergence, MCFC-specific (CO2/carbonate), edge cases.
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from model import MCFC_F2a
from predict import ComponentModel

PASS = "✓"
FAIL = "✗"

# Default operating gas composition (anode: H2/H2O/CO2 ; cathode: O2/CO2)
P = dict(pH2=0.7, pO2=0.15, pH2O=0.20, pCO2_cat=0.15, pCO2_an=0.10)


def assert_true(cond, msg):
    if cond:
        print(f"  {PASS}  {msg}")
    else:
        print(f"  {FAIL}  FAILED: {msg}")
        raise AssertionError(msg)


def make_model():
    cm = ComponentModel()
    return cm._model, cm


def V(m, j, T):
    return m.cell_voltage(j, T, P["pH2"], P["pO2"], P["pH2O"], P["pCO2_cat"], P["pCO2_an"])


def E(m, T):
    return m.nernst_voltage(T, P["pH2"], P["pO2"], P["pH2O"], P["pCO2_cat"], P["pCO2_an"])


# ---------------------------------------------------------------------------
def test_nernst_range():
    print("\n[Test 1] Nernst voltage in physical MCFC range")
    m, _ = make_model()
    e = E(m, 923.15)
    assert_true(0.9 < e < 1.2, f"E_nernst={e:.4f} V in [0.9, 1.2]")


def test_nernst_co2_dependence():
    print("\n[Test 2] Nernst rises with cathode CO2, falls with anode CO2 (MCFC signature)")
    m, _ = make_model()
    base = m.nernst_voltage(923.15, 0.7, 0.15, 0.20, 0.15, 0.10)
    more_cat = m.nernst_voltage(923.15, 0.7, 0.15, 0.20, 0.30, 0.10)
    more_an = m.nernst_voltage(923.15, 0.7, 0.15, 0.20, 0.15, 0.20)
    assert_true(more_cat > base, f"more cathode CO2 raises E: {more_cat:.4f} > {base:.4f}")
    assert_true(more_an < base, f"more anode CO2 lowers E: {more_an:.4f} < {base:.4f}")


def test_voltage_below_nernst():
    print("\n[Test 3] V_cell < E_nernst for j > 0")
    m, _ = make_model()
    e = E(m, 923.15)
    for j in [0.05, 0.2, 0.4]:
        v = V(m, j, 923.15)
        assert_true(v < e, f"V({j})={v:.4f} < E={e:.4f}")


def test_voltage_monotone():
    print("\n[Test 4] V_cell decreases with j")
    m, _ = make_model()
    j_vals = np.linspace(0.01, 0.55, 50)
    V_prev = V(m, j_vals[0], 923.15)
    for j in j_vals[1:]:
        v = V(m, j, 923.15)
        assert_true(v <= V_prev + 1e-9, f"V({j:.2f})={v:.4f} <= V_prev={V_prev:.4f}")
        V_prev = v
    print("  All 49 pairs checked.")


def test_carbonate_conductivity_arrhenius():
    print("\n[Test 5] Carbonate conductivity rises with T (Arrhenius)")
    m, _ = make_model()
    s_low = m.carbonate_conductivity(873.15)
    s_high = m.carbonate_conductivity(973.15)
    assert_true(s_high > s_low, f"sigma(973)={s_high:.3f} > sigma(873)={s_low:.3f} S/cm")
    assert_true(1.0 < s_low < 10.0, f"sigma in physical range ~1-10 S/cm: {s_low:.3f}")


def test_thermal_ode_heats_up():
    print("\n[Test 6] Thermal ODE: stack heats up from cold start")
    m, _ = make_model()
    r = m.simulate(0.3, 873.15, P["pH2"], P["pO2"], P["pH2O"], P["pCO2_cat"], P["pCO2_an"], 5.0, 1200.0)
    assert_true(r["temperature"][-1] > 873.15, f"T_final={r['temperature'][-1]:.2f} > 873 K")
    assert_true(r["temperature"][-1] < 1100.0, f"T_final={r['temperature'][-1]:.2f} < 1100 K (reasonable)")


def test_thermal_steady_state():
    print("\n[Test 7] Thermal reaches approximate steady state")
    m, _ = make_model()
    r = m.simulate(0.3, 923.15, P["pH2"], P["pO2"], P["pH2O"], P["pCO2_cat"], P["pCO2_an"], 10.0, 7200.0)
    dT = abs(r["temperature"][-1] - r["temperature"][-2])
    assert_true(dT < 0.1, f"Near SS: dT={dT:.5f} K between last two steps")


def test_efficiency_range():
    print("\n[Test 8] Efficiency in (0, 1) and energy conservation (V_cell < E_tn)")
    m, _ = make_model()
    r = m.simulate(0.3, 923.15, P["pH2"], P["pO2"], P["pH2O"], P["pCO2_cat"], P["pCO2_an"], 5.0, 60.0)
    for eta in r["efficiency"]:
        assert_true(0 < eta < 1.0, f"eta={eta:.4f}")


def test_overpotentials_positive():
    print("\n[Test 9] All overpotentials >= 0")
    m, _ = make_model()
    r = m.simulate(0.4, 923.15, P["pH2"], P["pO2"], P["pH2O"], P["pCO2_cat"], P["pCO2_an"], 5.0, 60.0)
    for name, arr in r["overpotentials"].items():
        if name == "E_nernst":
            continue
        assert_true(np.all(arr >= -1e-9), f"{name} all >= 0")


def test_step_response():
    print("\n[Test 10] Step current response -- voltage drops on step-up")
    m, _ = make_model()
    def step_j(t):
        return 0.15 if t < 300 else 0.45
    r = m.simulate(step_j, 923.15, P["pH2"], P["pO2"], P["pH2O"], P["pCO2_cat"], P["pCO2_an"], 5.0, 600.0)
    idx_before = np.argmin(np.abs(r["t"] - 295.0))
    idx_after = np.argmin(np.abs(r["t"] - 310.0))
    assert_true(r["voltage"][idx_after] < r["voltage"][idx_before],
                "Voltage drops after current step up")


def test_concentration_near_jL():
    print("\n[Test 11] Concentration loss diverges near j_L")
    m, _ = make_model()
    v1 = m.concentration_overpotential(0.4)
    v2 = m.concentration_overpotential(0.595)
    assert_true(v2 > v1 * 3, f"eta_conc(0.595)={v2:.4f} >> eta_conc(0.40)={v1:.4f}")


def test_predict_interface():
    print("\n[Test 12] ComponentModel predict() interface")
    _, cm = make_model()
    r = cm.predict({"current_density_A_cm2": 0.3, "dt": 5.0, "duration_s": 30.0})
    for key in ["t", "voltage", "power_density", "efficiency", "temperature", "overpotentials"]:
        assert_true(key in r, f"Key '{key}' in output")
    assert_true(len(r["t"]) == len(r["voltage"]), "Arrays same length")


def test_benchmark():
    print("\n[Test 13] Benchmark: 1200s sim at dt=5")
    m, _ = make_model()
    t0 = time.perf_counter()
    m.simulate(0.3, 923.15, P["pH2"], P["pO2"], P["pH2O"], P["pCO2_cat"], P["pCO2_an"], 5.0, 1200.0)
    elapsed = time.perf_counter() - t0
    print(f"  1200s simulation in {elapsed*1000:.1f} ms")
    assert_true(elapsed < 5.0, "Completes in < 5 s")


if __name__ == "__main__":
    tests = [
        test_nernst_range,
        test_nernst_co2_dependence,
        test_voltage_below_nernst,
        test_voltage_monotone,
        test_carbonate_conductivity_arrhenius,
        test_thermal_ode_heats_up,
        test_thermal_steady_state,
        test_efficiency_range,
        test_overpotentials_positive,
        test_step_response,
        test_concentration_near_jL,
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
    print(f"EC005 MCFC F2a -- Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    sys.exit(0 if failed == 0 else 1)
