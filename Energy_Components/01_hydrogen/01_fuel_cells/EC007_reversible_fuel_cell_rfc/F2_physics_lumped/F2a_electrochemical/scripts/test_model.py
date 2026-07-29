"""
EC007 -- Reversible Fuel Cell (RFC) -- F2a Bidirectional Electrochemical
Test suite: bidirectional physics sanity, ODE convergence, round-trip, edge cases.
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from model import RFC_F2a
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
    print("\n[Test 1] Nernst voltage in physical range, rises with pressure")
    m, _ = make_model()
    E = m.nernst_voltage(353.15, 1.0, 0.21)
    assert_true(1.0 < E < 1.3, f"E_nernst={E:.4f} V in [1.0, 1.3]")
    E2 = m.nernst_voltage(353.15, 2.0, 0.5)
    assert_true(E2 > E, f"Higher pressures raise E: {E2:.4f} > {E:.4f}")


def test_mode_switching_by_sign():
    print("\n[Test 2] Mode selected by sign of current: FC<E<EL")
    m, _ = make_model()
    E = m.nernst_voltage(353.15, 1.0, 0.21)
    V_fc = m.cell_voltage(0.5, 353.15, 1.0, 0.21)   # discharge
    V_el = m.cell_voltage(-0.5, 353.15, 1.0, 0.21)  # charge
    V_ocv = m.cell_voltage(0.0, 353.15, 1.0, 0.21)
    assert_true(V_fc < E, f"FC discharge V={V_fc:.4f} < E={E:.4f}")
    assert_true(V_el > E, f"EL charge V={V_el:.4f} > E={E:.4f}")
    assert_true(abs(V_ocv - E) < 1e-9, f"OCV V={V_ocv:.4f} == E={E:.4f}")


def test_fc_voltage_monotone_down():
    print("\n[Test 3] FC mode: V decreases with j")
    m, _ = make_model()
    j_vals = np.linspace(0.01, 1.4, 50)
    V_prev = m.cell_voltage(j_vals[0], 353.15, 1.0, 0.21)
    for j in j_vals[1:]:
        V = m.cell_voltage(j, 353.15, 1.0, 0.21)
        assert_true(V <= V_prev + 1e-9, f"V({j:.2f})={V:.4f} <= prev")
        V_prev = V
    print("  All 49 FC pairs checked.")


def test_el_voltage_monotone_up():
    print("\n[Test 4] EL mode: V increases with |j|")
    m, _ = make_model()
    j_vals = -np.linspace(0.01, 2.8, 50)
    V_prev = m.cell_voltage(j_vals[0], 353.15, 1.0, 0.21)
    for j in j_vals[1:]:
        V = m.cell_voltage(j, 353.15, 1.0, 0.21)
        assert_true(V >= V_prev - 1e-9, f"V({j:.2f})={V:.4f} >= prev")
        V_prev = V
    print("  All 49 EL pairs checked.")


def test_round_trip_efficiency():
    print("\n[Test 5] Round-trip efficiency in (0,1) and decreases with |j|")
    m, _ = make_model()
    rt_low = m.round_trip_efficiency(0.2, 353.15, 1.0, 0.21)
    rt_high = m.round_trip_efficiency(1.0, 353.15, 1.0, 0.21)
    assert_true(0.0 < rt_high < rt_low < 1.0,
                f"eta_rt: 0 < {rt_high:.3f} < {rt_low:.3f} < 1")


def test_thermal_heats_up_both_modes():
    print("\n[Test 6] Both modes dissipate heat -> stack warms from cold")
    m, _ = make_model()
    rfc = m.simulate(0.8, 300.0, 1.0, 0.21, 0.5, 120.0)
    rel = m.simulate(-1.0, 300.0, 1.0, 0.21, 0.5, 120.0)
    assert_true(300.0 < rfc["temperature"][-1] < 400.0,
                f"FC T_final={rfc['temperature'][-1]:.2f} K warmed")
    assert_true(300.0 < rel["temperature"][-1] < 400.0,
                f"EL T_final={rel['temperature'][-1]:.2f} K warmed")


def test_thermal_steady_state():
    print("\n[Test 7] Thermal ODE reaches approximate steady state")
    m, _ = make_model()
    r = m.simulate(0.5, 343.15, 1.0, 0.21, 1.0, 800.0)
    dT = abs(r["temperature"][-1] - r["temperature"][-2])
    assert_true(dT < 0.1, f"Near SS: dT={dT:.5f} K between last two steps")


def test_efficiency_range():
    print("\n[Test 8] Efficiency in (0,1) for both FC and EL operation")
    m, _ = make_model()
    rfc = m.simulate(0.5, 353.15, 1.0, 0.21, 1.0, 10.0)
    rel = m.simulate(-0.5, 353.15, 1.0, 0.21, 1.0, 10.0)
    for eta in rfc["efficiency"]:
        assert_true(0 < eta < 1.0, f"FC eta={eta:.4f}")
    for eta in rel["efficiency"]:
        assert_true(0 < eta < 1.0, f"EL eta={eta:.4f}")


def test_power_sign_convention():
    print("\n[Test 9] Power sign: + delivered (FC), - consumed (EL)")
    m, _ = make_model()
    rfc = m.simulate(0.6, 353.15, 1.0, 0.21, 1.0, 5.0)
    rel = m.simulate(-0.6, 353.15, 1.0, 0.21, 1.0, 5.0)
    assert_true(np.all(rfc["power_density"] > 0), "FC power density > 0 (delivered)")
    assert_true(np.all(rel["power_density"] < 0), "EL power density < 0 (consumed)")


def test_overpotentials_positive():
    print("\n[Test 10] Overpotential magnitudes >= 0 in both modes")
    m, _ = make_model()
    for j in (0.6, -0.6):
        r = m.simulate(j, 353.15, 1.0, 0.21, 1.0, 5.0)
        for name, arr in r["overpotentials"].items():
            if name == "E_nernst":
                continue
            assert_true(np.all(arr >= -1e-9), f"j={j}: {name} all >= 0")


def test_concentration_diverges():
    print("\n[Test 11] Concentration loss diverges near limiting current (both modes)")
    m, _ = make_model()
    fc1 = m.concentration_overpotential(1.0, 353.15, "FC")
    fc2 = m.concentration_overpotential(1.45, 353.15, "FC")
    el1 = m.concentration_overpotential(2.0, 353.15, "EL")
    el2 = m.concentration_overpotential(2.9, 353.15, "EL")
    assert_true(fc2 > fc1 * 3, f"FC conc(1.45)={fc2:.4f} >> conc(1.0)={fc1:.4f}")
    assert_true(el2 > el1 * 3, f"EL conc(2.9)={el2:.4f} >> conc(2.0)={el1:.4f}")


def test_predict_interface():
    print("\n[Test 12] ComponentModel predict() interface")
    _, cm = make_model()
    r = cm.predict({"current_density_A_cm2": -0.5, "dt": 1.0, "duration_s": 5.0})
    for key in ["t", "voltage", "power_density", "efficiency", "temperature",
                "mode", "overpotentials"]:
        assert_true(key in r, f"Key '{key}' in output")
    assert_true(len(r["t"]) == len(r["voltage"]) == len(r["mode"]),
                "Arrays same length")
    assert_true(all(mo == "EL" for mo in r["mode"]), "Negative j -> EL mode")


def test_benchmark():
    print("\n[Test 13] Benchmark: 60s sim at dt=0.1")
    m, _ = make_model()
    t0 = time.perf_counter()
    m.simulate(0.5, 343.15, 1.0, 0.21, 0.1, 60.0)
    elapsed = time.perf_counter() - t0
    print(f"  60s simulation in {elapsed*1000:.1f} ms")
    assert_true(elapsed < 10.0, "Completes in < 10 s")


if __name__ == "__main__":
    tests = [
        test_nernst_range,
        test_mode_switching_by_sign,
        test_fc_voltage_monotone_down,
        test_el_voltage_monotone_up,
        test_round_trip_efficiency,
        test_thermal_heats_up_both_modes,
        test_thermal_steady_state,
        test_efficiency_range,
        test_power_sign_convention,
        test_overpotentials_positive,
        test_concentration_diverges,
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
    print(f"EC007 RFC F2a -- Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    sys.exit(0 if failed == 0 else 1)
