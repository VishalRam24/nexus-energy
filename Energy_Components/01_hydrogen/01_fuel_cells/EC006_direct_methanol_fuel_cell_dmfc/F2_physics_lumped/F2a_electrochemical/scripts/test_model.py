"""
EC006 -- Direct Methanol Fuel Cell (DMFC) -- F2a Electrochemical
Test suite: physics sanity, methanol-crossover behaviour, ODE convergence, edges.
Run with system python3:  python3 scripts/test_model.py   (NOT pytest)
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from model import DMFC_F2a
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
    print("\n[Test 1] Reversible (thermodynamic) voltage in physical range")
    m, _ = make_model()
    E = m.reversible_voltage(353.15)
    assert_true(1.1 < E < 1.25, f"E_rev={E:.4f} V in [1.1, 1.25]")
    assert_true(m.reversible_voltage(383.15) < E,
                "E_rev decreases with T (dE/dT < 0)")


def test_ocv_depressed_by_crossover():
    print("\n[Test 2] Crossover depresses OCV well below thermodynamic E_rev")
    m, _ = make_model()
    E = m.reversible_voltage(353.15)
    OCV = m.cell_voltage(1e-6, 353.15)  # near zero useful current
    assert_true(OCV < E - 0.2,
                f"OCV={OCV:.3f} V << E_rev={E:.3f} V (mixed-potential)")
    assert_true(0.3 < OCV < 0.9,
                f"OCV={OCV:.3f} V in realistic DMFC band [0.3,0.9]")


def test_voltage_below_reversible():
    print("\n[Test 3] V_cell < E_rev for all j > 0")
    m, _ = make_model()
    E = m.reversible_voltage(353.15)
    for j in [0.05, 0.1, 0.2, 0.35]:
        V = m.cell_voltage(j, 353.15)
        assert_true(V < E, f"V({j})={V:.4f} < E_rev={E:.4f}")


def test_voltage_monotone():
    print("\n[Test 4] V_cell decreases monotonically with j")
    m, _ = make_model()
    j_vals = np.linspace(0.005, 0.38, 50)
    V_prev = m.cell_voltage(j_vals[0], 353.15)
    for j in j_vals[1:]:
        V = m.cell_voltage(j, 353.15)
        assert_true(V <= V_prev + 1e-9, f"V({j:.3f})={V:.4f} <= {V_prev:.4f}")
        V_prev = V
    print("  All 49 pairs checked.")


def test_crossover_arrhenius_and_concentration():
    print("\n[Test 5] Crossover current rises with T and with [MeOH]")
    m, _ = make_model()
    jx_lo = m.crossover_current(323.15, 1.0)
    jx_hi = m.crossover_current(363.15, 1.0)
    assert_true(jx_hi > jx_lo, f"j_cross(363K)={jx_hi:.4f} > j_cross(323K)={jx_lo:.4f}")
    jx_1m = m.crossover_current(353.15, 1.0)
    jx_2m = m.crossover_current(353.15, 2.0)
    assert_true(jx_2m > jx_1m, f"j_cross(2M)={jx_2m:.4f} > j_cross(1M)={jx_1m:.4f}")
    assert_true(abs(jx_2m - 2 * jx_1m) < 1e-9, "crossover linear in concentration")


def test_fuel_efficiency_bounds():
    print("\n[Test 6] Fuel (Faradaic) efficiency in (0,1), rises with load")
    m, _ = make_model()
    fe_lo = m.fuel_efficiency(0.05, 353.15, 1.0)
    fe_hi = m.fuel_efficiency(0.30, 353.15, 1.0)
    assert_true(0 < fe_lo < 1, f"fuel_eff(0.05)={fe_lo:.3f} in (0,1)")
    assert_true(0 < fe_hi < 1, f"fuel_eff(0.30)={fe_hi:.3f} in (0,1)")
    assert_true(fe_hi > fe_lo,
                f"fuel_eff rises with j: {fe_hi:.3f} > {fe_lo:.3f}")


def test_higher_concentration_lowers_voltage():
    print("\n[Test 7] Higher [MeOH] -> more crossover -> lower OCV")
    m, _ = make_model()
    V_1m = m.cell_voltage(1e-6, 353.15, 1.0)
    V_2m = m.cell_voltage(1e-6, 353.15, 2.0)
    assert_true(V_2m < V_1m,
                f"OCV(2M)={V_2m:.3f} < OCV(1M)={V_1m:.3f} (crossover penalty)")


def test_overpotentials_positive():
    print("\n[Test 8] All overpotentials >= 0")
    m, _ = make_model()
    r = m.simulate(0.2, 353.15, 1.0, 1.0, 10.0)
    for name, arr in r["overpotentials"].items():
        if name == "E_rev":
            continue
        assert_true(np.all(arr >= -1e-9), f"{name} all >= 0")


def test_thermal_ode_heats_up():
    print("\n[Test 9] Thermal ODE: stack warms from cold start, stays bounded")
    m, _ = make_model()
    r = m.simulate(0.25, 318.15, 1.0, 1.0, 200.0)
    assert_true(r["temperature"][-1] > 318.15,
                f"T_final={r['temperature'][-1]:.2f} > 318.15 K")
    assert_true(r["temperature"][-1] < 420.0,
                f"T_final={r['temperature'][-1]:.2f} < 420 K (bounded)")


def test_energy_conservation():
    print("\n[Test 10] Energy balance: electrical + heat + crossover-heat = enthalpy")
    m, _ = make_model()
    T, j, c = 353.15, 0.2, 1.0
    V = m.cell_voltage(j, T, c)
    jx = m.crossover_current(T, c)
    E_th = m.thermoneutral_voltage(T)
    A = m.A_cell
    # Per cm2: useful electrical + useful-current heat = j*E_th
    P_elec = j * V
    Q_useful = j * (E_th - V)
    Q_cross = jx * E_th
    chem_in = (j + jx) * E_th          # total enthalpy released
    out = P_elec + Q_useful + Q_cross
    assert_true(abs(out - chem_in) < 1e-9,
                f"balance closes: out={out:.5f} == chem_in={chem_in:.5f} (W/cm2)")


def test_efficiency_range():
    print("\n[Test 11] Overall efficiency in (0,1)")
    m, _ = make_model()
    r = m.simulate(0.2, 353.15, 1.0, 1.0, 10.0)
    for eta in r["efficiency"]:
        assert_true(0 < eta < 1.0, f"eta={eta:.4f} in (0,1)")


def test_predict_interface():
    print("\n[Test 12] ComponentModel predict() interface")
    _, cm = make_model()
    r = cm.predict({"current_density_A_cm2": 0.2, "dt": 1.0, "duration_s": 5.0})
    for key in ["t", "voltage", "power_density", "efficiency",
                "fuel_efficiency", "temperature", "crossover_current",
                "overpotentials"]:
        assert_true(key in r, f"Key '{key}' in output")
    assert_true(len(r["t"]) == len(r["voltage"]), "Arrays same length")


def test_benchmark():
    print("\n[Test 13] Benchmark: 60s sim at dt=0.1")
    m, _ = make_model()
    t0 = time.perf_counter()
    m.simulate(0.2, 343.15, 1.0, 0.1, 60.0)
    elapsed = time.perf_counter() - t0
    print(f"  60s simulation in {elapsed*1000:.1f} ms")
    assert_true(elapsed < 5.0, "Completes in < 5 s")


if __name__ == "__main__":
    tests = [
        test_reversible_range,
        test_ocv_depressed_by_crossover,
        test_voltage_below_reversible,
        test_voltage_monotone,
        test_crossover_arrhenius_and_concentration,
        test_fuel_efficiency_bounds,
        test_higher_concentration_lowers_voltage,
        test_overpotentials_positive,
        test_thermal_ode_heats_up,
        test_energy_conservation,
        test_efficiency_range,
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
    print(f"EC006 DMFC F2a -- Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    sys.exit(0 if failed == 0 else 1)
