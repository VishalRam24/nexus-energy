"""
EC106 -- Fuel Cell CHP (SOFC-Based) -- F2a SOFC Cogeneration
Test suite: physics sanity, CHP-efficiency bounds, energy conservation,
ODE behaviour, edge cases, predict() interface, benchmark timing.
Run with system python3 (NOT pytest):  python3 scripts/test_model.py
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from model import SOFC_CHP_F2a
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
    print("\n[Test 1] Nernst voltage in SOFC physical range (~0.9-1.1 V)")
    m, _ = make_model()
    E = m.nernst_voltage(1073.15)
    assert_true(0.8 < E < 1.2, f"E_nernst={E:.4f} V in [0.8, 1.2] at 800C")


def test_voltage_below_nernst():
    print("\n[Test 2] V_cell < E_nernst for all j > 0 (irreversibility)")
    m, _ = make_model()
    E = m.nernst_voltage(1073.15)
    for j in [0.1, 0.5, 1.0, 1.5]:
        V = m.cell_voltage(j, 1073.15)
        assert_true(V < E, f"V({j})={V:.4f} < E={E:.4f}")


def test_voltage_monotone():
    print("\n[Test 3] V_cell decreases monotonically with j")
    m, _ = make_model()
    j_vals = np.linspace(0.05, 2.0, 40)
    V_prev = m.cell_voltage(j_vals[0], 1073.15)
    for j in j_vals[1:]:
        V = m.cell_voltage(j, 1073.15)
        assert_true(V <= V_prev + 1e-9, f"V({j:.2f})={V:.4f} <= {V_prev:.4f}")
        V_prev = V
    print("  All 39 pairs checked.")


def test_efficiency_bounds():
    print("\n[Test 4] 0 < eta_e, eta_th, eta_total < 1 across load sweep")
    m, _ = make_model()
    for j in [0.1, 0.3, 0.5, 0.8, 1.2]:
        ph = m.power_and_heat(j, 1073.15)
        assert_true(0 < ph["eta_electrical"] < 1.0,
                    f"j={j}: eta_e={ph['eta_electrical']:.4f} in (0,1)")
        assert_true(0 < ph["eta_thermal"] < 1.0,
                    f"j={j}: eta_th={ph['eta_thermal']:.4f} in (0,1)")
        assert_true(0 < ph["eta_total"] < 1.0,
                    f"j={j}: eta_total={ph['eta_total']:.4f} in (0,1)")


def test_total_eff_gt_electrical():
    print("\n[Test 5] eta_total > eta_electrical (CHP recovers extra heat)")
    m, _ = make_model()
    for j in [0.2, 0.5, 1.0]:
        ph = m.power_and_heat(j, 1073.15)
        assert_true(ph["eta_total"] > ph["eta_electrical"] + 1e-9,
                    f"j={j}: eta_total={ph['eta_total']:.4f} > "
                    f"eta_e={ph['eta_electrical']:.4f}")
        assert_true(abs(ph["eta_total"]
                        - (ph["eta_electrical"] + ph["eta_thermal"])) < 1e-9,
                    "eta_total == eta_e + eta_th identity")


def test_energy_conservation():
    print("\n[Test 6] Energy conservation: P_e + Q_useful + Q_loss == Q_fuel")
    m, _ = make_model()
    for j in [0.2, 0.5, 1.0, 1.5]:
        ph = m.power_and_heat(j, 1073.15)
        lhs = ph["P_e_W"] + ph["Q_useful_thermal_W"] + ph["Q_loss_W"]
        rhs = ph["Q_fuel_W"]
        rel = abs(lhs - rhs) / max(rhs, 1e-9)
        assert_true(rel < 1e-6, f"j={j}: balance closes (rel err={rel:.2e})")
        # useful thermal cannot exceed liberated heat
        assert_true(ph["Q_useful_thermal_W"] <= ph["Q_heat_total_W"] + 1e-6,
                    f"j={j}: Q_useful <= Q_heat_total")


def test_power_to_heat_positive():
    print("\n[Test 7] Power-to-heat ratio finite & positive under load")
    m, _ = make_model()
    ph = m.power_and_heat(0.5, 1073.15)
    assert_true(np.isfinite(ph["power_to_heat"]) and ph["power_to_heat"] > 0,
                f"P/H={ph['power_to_heat']:.3f} > 0")
    # SOFC: electrical typically exceeds recovered heat -> P/H > 1
    assert_true(ph["power_to_heat"] > 1.0,
                f"SOFC P/H={ph['power_to_heat']:.3f} > 1 (electric-led)")


def test_thermal_ode_dynamics():
    print("\n[Test 8] Thermal ODE: stack relaxes toward bounded steady state")
    m, _ = make_model()
    r = m.simulate(0.5, 1023.15, dt=10.0, duration_s=3000.0)
    Tf = r["temperature"][-1]
    assert_true(873.15 < Tf < 1273.15, f"T_final={Tf:.1f} K in valid window")
    dT = abs(r["temperature"][-1] - r["temperature"][-2])
    assert_true(dT < 0.5, f"Near steady state: dT={dT:.4f} K last step")


def test_higher_load_more_power():
    print("\n[Test 9] Higher current density -> more electrical power")
    m, _ = make_model()
    p_lo = m.power_and_heat(0.3, 1073.15)["P_e_W"]
    p_hi = m.power_and_heat(0.9, 1073.15)["P_e_W"]
    assert_true(p_hi > p_lo, f"P_e(0.9)={p_hi:.1f} > P_e(0.3)={p_lo:.1f} W")


def test_zero_current_edge():
    print("\n[Test 10] Zero current edge case: no power, no fuel, no NaN")
    m, _ = make_model()
    ph = m.power_and_heat(0.0, 1073.15)
    assert_true(ph["P_e_W"] == 0.0, "P_e=0 at j=0")
    assert_true(ph["eta_electrical"] == 0.0 and ph["eta_total"] == 0.0,
                "efficiencies 0 at j=0 (no fuel flow)")


def test_predict_interface():
    print("\n[Test 11] ComponentModel predict() interface + array integrity")
    _, cm = make_model()
    r = cm.predict({"current_density_A_cm2": 0.5, "dt": 20.0, "duration_s": 200.0})
    for key in ["t", "temperature", "voltage", "P_e_W", "Q_useful_thermal_W",
                "eta_electrical", "eta_thermal", "eta_total", "power_to_heat",
                "steady_state"]:
        assert_true(key in r, f"Key '{key}' in output")
    assert_true(len(r["t"]) == len(r["P_e_W"]) == len(r["eta_total"]),
                "All time-series arrays equal length")
    ss = r["steady_state"]
    assert_true(0 < ss["eta_total"] < 1 and ss["eta_total"] > ss["eta_electrical"],
                "steady_state CHP metrics consistent")


def test_benchmark():
    print("\n[Test 12] Benchmark: 1200 s transient sim at dt=5 s")
    m, _ = make_model()
    t0 = time.perf_counter()
    m.simulate(0.5, 1023.15, dt=5.0, duration_s=1200.0)
    elapsed = time.perf_counter() - t0
    print(f"  1200 s simulation in {elapsed*1000:.1f} ms")
    assert_true(elapsed < 5.0, "Completes in < 5 s")


if __name__ == "__main__":
    tests = [
        test_nernst_range,
        test_voltage_below_nernst,
        test_voltage_monotone,
        test_efficiency_bounds,
        test_total_eff_gt_electrical,
        test_energy_conservation,
        test_power_to_heat_positive,
        test_thermal_ode_dynamics,
        test_higher_load_more_power,
        test_zero_current_edge,
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
    print(f"EC106 SOFC-CHP F2a -- Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    sys.exit(0 if failed == 0 else 1)
