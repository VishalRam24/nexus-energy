"""
EC040 -- Hydrogen-Bromine Flow Battery (HBrFB) -- F2a Physics-Lumped Stack Model
Test suite: physics sanity (Nernst, conservation, V_charge>V_disch, efficiency
bounds, thermal balance), edge cases, predict() interface, benchmark timing.
Run with system python3 (NOT pytest):  python3 scripts/test_model.py
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from model import HydrogenBromineFlowF2a
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
    print("\n[Test 1] Nernst cell voltage near 1.1 V and SOC-monotone")
    m, _ = make_model()
    E = m.nernst_voltage(0.5, 298.15)
    assert_true(1.0 < E < 1.2, f"E_nernst(SOC=0.5)={E:.4f} V near 1.09 V")
    E_hi = m.nernst_voltage(0.9, 298.15)
    E_lo = m.nernst_voltage(0.1, 298.15)
    assert_true(E_hi > E > E_lo, f"Nernst rises with SOC: {E_lo:.3f} < {E:.3f} < {E_hi:.3f}")


def test_discharge_below_nernst():
    print("\n[Test 2] V_discharge < E_nernst < V_charge (overpotentials)")
    m, _ = make_model()
    for I in [20.0, 80.0, 150.0]:
        E = m.nernst_voltage(0.5, 298.15)
        V_dis = m.cell_voltage(+I, 0.5, 298.15)
        V_chg = m.cell_voltage(-I, 0.5, 298.15)
        assert_true(V_dis < E < V_chg,
                    f"I={I}: V_dis={V_dis:.4f} < E={E:.4f} < V_chg={V_chg:.4f}")


def test_v_charge_gt_v_discharge():
    print("\n[Test 3] V_charge > V_discharge at equal |I| (hysteresis)")
    m, _ = make_model()
    for I in [10.0, 50.0, 120.0, 200.0]:
        V_dis = m.cell_voltage(+I, 0.5, 298.15)
        V_chg = m.cell_voltage(-I, 0.5, 298.15)
        assert_true(V_chg > V_dis, f"I={I}: V_chg={V_chg:.4f} > V_dis={V_dis:.4f}")


def test_h2_kinetics_fast():
    print("\n[Test 4] H2/H+ activation << Br2/Br- activation (fast Pt/C)")
    m, _ = make_model()
    jabs = 100.0 / m.A_cell
    eta_H2 = m.activation_overpotential_H2(jabs, 298.15)
    eta_Br = m.activation_overpotential_Br(jabs, 298.15)
    assert_true(eta_H2 < eta_Br, f"eta_H2={eta_H2*1000:.2f} mV < eta_Br={eta_Br*1000:.2f} mV")
    assert_true(eta_H2 < 0.05, f"H2 activation small ({eta_H2*1000:.2f} mV)")


def test_coulomb_conservation():
    print("\n[Test 5] Coulomb conservation: SOC drop matches charge passed (k_cross=0)")
    m, _ = make_model()
    m.k_cross = 0.0  # isolate Faradaic term
    I = 40.0
    dur = 1800.0
    r = m.simulate(I, soc0=0.8, T0=298.15, dt=60.0, duration_s=dur)
    dSOC_actual = r["soc"][0] - r["soc"][-1]
    dSOC_expected = I * dur / m.Q_nom_C
    rel_err = abs(dSOC_actual - dSOC_expected) / dSOC_expected
    assert_true(rel_err < 1e-3,
                f"dSOC actual={dSOC_actual:.5f} vs Faraday={dSOC_expected:.5f} (err {rel_err*100:.3f}%)")


def test_crossover_self_discharge():
    print("\n[Test 6] Br crossover self-discharges at I=0 (Coulombic loss)")
    m, _ = make_model()
    r = m.simulate(0.0, soc0=0.9, T0=298.15, dt=600.0, duration_s=7200.0)
    assert_true(r["soc"][-1] < r["soc"][0],
                f"SOC self-discharges {r['soc'][0]:.4f} -> {r['soc'][-1]:.4f} via Br2 crossover")
    assert_true(r["soc"][-1] > 0.0, "SOC stays physical (>0)")


def test_charge_raises_soc():
    print("\n[Test 7] Charging (I<0) raises SOC")
    m, _ = make_model()
    r = m.simulate(-40.0, soc0=0.3, T0=298.15, dt=60.0, duration_s=1800.0)
    assert_true(r["soc"][-1] > r["soc"][0],
                f"Charge raises SOC {r['soc'][0]:.3f} -> {r['soc'][-1]:.3f}")


def test_thermal_balance():
    print("\n[Test 8] Thermal ODE: heats under load, reaches bounded steady state")
    m, _ = make_model()
    m.k_cross = 0.0
    r = m.simulate(120.0, soc0=0.5, T0=298.15, dt=30.0, duration_s=7200.0)
    assert_true(r["temperature"][-1] > 298.15, f"Heats up: T_final={r['temperature'][-1]:.2f} K")
    assert_true(r["temperature"][-1] < 333.15, f"Bounded: T_final={r['temperature'][-1]:.2f} K < 60 C")
    dT = abs(r["temperature"][-1] - r["temperature"][-2])
    assert_true(dT < 0.05, f"Near steady state: dT={dT:.4f} K/step")


def test_round_trip_efficiency():
    print("\n[Test 9] Round-trip efficiency high but strictly < 1")
    m, _ = make_model()
    rt = m.round_trip_efficiency(50.0, soc=0.5, T=298.15)
    assert_true(0.5 < rt < 1.0, f"RTE={rt*100:.1f}% in (50%, 100%)")
    rt_lowI = m.round_trip_efficiency(150.0, soc=0.5, T=298.15)
    assert_true(rt_lowI < rt, f"Higher current lowers voltaic eff: {rt_lowI*100:.1f}% < {rt*100:.1f}%")


def test_concentration_diverges():
    print("\n[Test 10] Br2 concentration overpotential diverges near j_L")
    m, _ = make_model()
    I_mid = 0.5 * m.j_L * m.A_cell
    I_hi = 0.95 * m.j_L * m.A_cell
    v1 = m.concentration_overpotential(abs(I_mid) / m.A_cell)
    v2 = m.concentration_overpotential(abs(I_hi) / m.A_cell)
    assert_true(v2 > v1 * 3, f"eta_conc(0.95 jL)={v2:.4f} >> eta_conc(0.5 jL)={v1:.4f}")


def test_predict_interface():
    print("\n[Test 11] ComponentModel predict() interface + array lengths")
    _, cm = make_model()
    r = cm.predict({"current_A": 50.0, "soc0": 0.6, "dt": 60.0, "duration_s": 600.0})
    for key in ["t", "soc", "temperature", "cell_voltage", "stack_voltage",
                "power_W", "E_nernst", "overpotentials"]:
        assert_true(key in r, f"Key '{key}' in output")
    assert_true(len(r["t"]) == len(r["stack_voltage"]) == len(r["soc"]),
                "Arrays same length")
    info = cm.get_info()
    assert_true(info["component_id"] == "EC040", "get_info reports EC040")


def test_benchmark():
    print("\n[Test 12] Benchmark: 1 h sim at dt=10 s")
    m, _ = make_model()
    t0 = time.perf_counter()
    m.simulate(60.0, 0.5, 298.15, 10.0, 3600.0)
    elapsed = time.perf_counter() - t0
    print(f"  3600 s simulation in {elapsed*1000:.1f} ms")
    assert_true(elapsed < 5.0, "Completes in < 5 s")


if __name__ == "__main__":
    tests = [
        test_nernst_range,
        test_discharge_below_nernst,
        test_v_charge_gt_v_discharge,
        test_h2_kinetics_fast,
        test_coulomb_conservation,
        test_crossover_self_discharge,
        test_charge_raises_soc,
        test_thermal_balance,
        test_round_trip_efficiency,
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
    print(f"EC040 HBrFB F2a -- Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    sys.exit(0 if failed == 0 else 1)
