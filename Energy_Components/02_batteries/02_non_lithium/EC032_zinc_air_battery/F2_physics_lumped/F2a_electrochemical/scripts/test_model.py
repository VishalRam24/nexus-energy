"""
EC032 -- Zinc-Air Battery -- F2a Air-Cathode Electrochemical
Test suite: physics sanity (V<E_eq, Coulomb conservation, plateau, limiting
current), thermal balance, efficiency bounds, edge cases, predict() interface,
and benchmark timing. Custom harness (NO pytest).
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from model import ZincAirF2a
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
def test_equilibrium_range():
    print("\n[Test 1] Equilibrium voltage in physical Zn-air range")
    m, _ = make_model()
    E = m.equilibrium_voltage(298.15, 0.21)
    assert_true(1.3 < E < 1.7, f"E_eq={E:.4f} V in [1.3, 1.7] (Zn-air ~1.65 V)")
    E_o2 = m.equilibrium_voltage(298.15, 1.0)
    assert_true(E_o2 > E, f"Pure O2 raises E: {E_o2:.4f} > air {E:.4f}")


def test_voltage_below_eq():
    print("\n[Test 2] V_cell < E_eq for j > 0 (irreversibility)")
    m, _ = make_model()
    E = m.equilibrium_voltage(298.15, 0.21)
    for j in [0.01, 0.05, 0.10, 0.20]:
        V = m.cell_voltage(j, 298.15, 0.21)
        assert_true(V < E, f"V({j})={V:.4f} < E_eq={E:.4f}")


def test_discharge_plateau():
    print("\n[Test 3] Flat discharge plateau near 1.2-1.4 V at moderate load")
    m, _ = make_model()
    V = m.cell_voltage(0.05, 298.15, 0.21)
    assert_true(1.1 < V < 1.45, f"Plateau V(0.05 A/cm2)={V:.4f} in [1.1, 1.45] V")


def test_voltage_monotone():
    print("\n[Test 4] V_cell decreases monotonically with j")
    m, _ = make_model()
    j_vals = np.linspace(0.002, 0.29, 60)
    V_prev = m.cell_voltage(j_vals[0], 298.15, 0.21)
    for j in j_vals[1:]:
        V = m.cell_voltage(j, 298.15, 0.21)
        assert_true(V <= V_prev + 1e-9, f"V({j:.3f})={V:.4f} <= V_prev={V_prev:.4f}")
        V_prev = V
    print("  All 59 pairs checked.")


def test_air_limiting_current():
    print("\n[Test 5] Air-electrode limiting current: voltage collapses near j_L")
    m, _ = make_model()
    jL = m.limiting_current(298.15, 0.21)
    eta_lo = m.concentration_overpotential(0.5 * jL, 298.15, 0.21)
    eta_hi = m.concentration_overpotential(0.98 * jL, 298.15, 0.21)
    assert_true(eta_hi > eta_lo * 3, f"eta_conc diverges: {eta_hi:.4f} >> {eta_lo:.4f}")
    V_at_jL = m.cell_voltage(jL, 298.15, 0.21)
    assert_true(V_at_jL < 0.5, f"V collapses at j_L: V={V_at_jL:.4f} < 0.5 V")


def test_limiting_scales_with_po2():
    print("\n[Test 6] Limiting current scales with O2 partial pressure")
    m, _ = make_model()
    jL_air = m.limiting_current(298.15, 0.21)
    jL_o2 = m.limiting_current(298.15, 1.0)
    assert_true(jL_o2 > jL_air, f"j_L(O2)={jL_o2:.4f} > j_L(air)={jL_air:.4f}")
    ratio = jL_o2 / jL_air
    assert_true(abs(ratio - 1.0 / 0.21) < 0.2, f"j_L ratio ~ 1/0.21: got {ratio:.2f}")


def test_coulomb_conservation():
    print("\n[Test 7] Coulomb conservation: SOC drop matches charge passed")
    m, _ = make_model()
    j = 0.05
    dur = 3600.0
    r = m.simulate(j, 298.15, 0.21, 10.0, dur, soc0=1.0)
    I = m.A * j                              # A
    charge_Ah = I * dur / 3600.0            # Ah
    expected_dsoc = charge_Ah / m.Q_cap
    actual_dsoc = r["soc"][0] - r["soc"][-1]
    err = abs(actual_dsoc - expected_dsoc)
    assert_true(err < 1e-3,
                f"dSOC actual={actual_dsoc:.4f} vs expected={expected_dsoc:.4f} (err={err:.2e})")


def test_thermal_balance():
    print("\n[Test 8] Thermal ODE: cell warms then approaches steady state")
    m, _ = make_model()
    r = m.simulate(0.15, 298.15, 0.21, 10.0, 4000.0, soc0=1.0)
    assert_true(r["temperature"][-1] > 298.15, f"Warms up: T_final={r['temperature'][-1]:.2f} K")
    assert_true(r["temperature"][-1] < 360.0, f"Bounded: T_final={r['temperature'][-1]:.2f} < 360 K")
    dT = abs(r["temperature"][-1] - r["temperature"][-2])
    assert_true(dT < 0.05, f"Near steady state: dT={dT:.4f} K between last steps")


def test_efficiency_range():
    print("\n[Test 9] Voltaic efficiency strictly in (0, 1)")
    m, _ = make_model()
    r = m.simulate(0.05, 298.15, 0.21, 10.0, 200.0)
    for eta in r["efficiency"]:
        assert_true(0.0 < eta < 1.0, f"eta={eta:.4f} in (0,1)")


def test_overpotentials_positive():
    print("\n[Test 10] All overpotentials >= 0")
    m, _ = make_model()
    r = m.simulate(0.08, 298.15, 0.21, 10.0, 200.0)
    for name, arr in r["overpotentials"].items():
        if name == "E_eq":
            continue
        assert_true(np.all(arr >= -1e-9), f"{name} all >= 0")


def test_predict_interface():
    print("\n[Test 11] ComponentModel predict() interface")
    _, cm = make_model()
    r = cm.predict({"current_density_A_cm2": 0.05, "dt": 10.0, "duration_s": 100.0})
    for key in ["t", "voltage", "power_density", "efficiency", "temperature", "soc", "overpotentials"]:
        assert_true(key in r, f"Key '{key}' in output")
    assert_true(len(r["t"]) == len(r["voltage"]), "Arrays same length")
    info = cm.get_info()
    assert_true(info["component_id"] == "EC032", "component_id == EC032")


def test_benchmark():
    print("\n[Test 12] Benchmark: 1-hour discharge sim at dt=1 s")
    m, _ = make_model()
    t0 = time.perf_counter()
    m.simulate(0.05, 298.15, 0.21, 1.0, 3600.0)
    elapsed = time.perf_counter() - t0
    print(f"  3600 s simulation in {elapsed*1000:.1f} ms")
    assert_true(elapsed < 5.0, "Completes in < 5 s")


if __name__ == "__main__":
    tests = [
        test_equilibrium_range,
        test_voltage_below_eq,
        test_discharge_plateau,
        test_voltage_monotone,
        test_air_limiting_current,
        test_limiting_scales_with_po2,
        test_coulomb_conservation,
        test_thermal_balance,
        test_efficiency_range,
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
    print(f"EC032 Zn-Air F2a -- Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    sys.exit(0 if failed == 0 else 1)
