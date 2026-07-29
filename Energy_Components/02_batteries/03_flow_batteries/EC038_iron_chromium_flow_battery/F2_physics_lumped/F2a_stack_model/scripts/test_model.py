"""
EC038 -- Iron-Chromium Flow Battery (ICFB) -- F2a Physics-Lumped Stack Model
Test suite: physics sanity (conservation, monotonicity, limits), edge cases,
predict() interface, benchmark timing. Custom harness (NO pytest).
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from model import FeCrFlowBatteryF2a
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
def test_ocv_range_and_soc_monotone():
    print("\n[Test 1] OCV ~ E0_cell at SOC=0.5, rises with SOC")
    m, _ = make_model()
    E_half = m.nernst_voltage(0.5, 298.15)
    assert_true(abs(E_half - m.E0_cell) < 1e-6,
                f"OCV(0.5)={E_half:.4f} V ~ E0_cell={m.E0_cell:.2f} V")
    socs = np.linspace(0.1, 0.9, 9)
    prev = m.nernst_voltage(socs[0], 298.15)
    for s in socs[1:]:
        E = m.nernst_voltage(s, 298.15)
        assert_true(E > prev, f"OCV rises with SOC: {E:.4f} > {prev:.4f}")
        prev = E
    assert_true(1.0 < E_half < 1.4, f"OCV in physical band: {E_half:.3f} V")


def test_charge_above_discharge():
    print("\n[Test 2] V_charge > V_ocv > V_discharge (round-trip eff < 1)")
    m, _ = make_model()
    soc, T = 0.5, 308.15
    V_ocv = m.nernst_voltage(soc, T) * m.N_cells
    V_chg = m.terminal_voltage(-50.0, soc, T)
    V_dis = m.terminal_voltage(+50.0, soc, T)
    assert_true(V_chg > V_ocv, f"V_charge={V_chg:.2f} > V_ocv={V_ocv:.2f}")
    assert_true(V_dis < V_ocv, f"V_discharge={V_dis:.2f} < V_ocv={V_ocv:.2f}")
    assert_true(V_chg > V_dis, f"V_charge={V_chg:.2f} > V_discharge={V_dis:.2f}")


def test_cr_dominates_activation():
    print("\n[Test 3] Sluggish Cr kinetics dominate activation loss")
    m, _ = make_model()
    T = 308.15
    j = 50.0 / m.A_cell
    j0_Fe = m._arrhenius(m.j0_Fe, m.E_act_Fe, T)
    j0_Cr = m._arrhenius(m.j0_Cr, m.E_act_Cr, T)
    eta_Fe = m._bv_overpotential(j, j0_Fe, T)
    eta_Cr = m._bv_overpotential(j, j0_Cr, T)
    assert_true(eta_Cr > eta_Fe, f"eta_Cr={eta_Cr*1000:.1f} mV > eta_Fe={eta_Fe*1000:.1f} mV")
    assert_true(j0_Cr < j0_Fe, f"j0_Cr={j0_Cr:.2e} < j0_Fe={j0_Fe:.2e} (Cr sluggish)")


def test_butler_volmer_consistency():
    print("\n[Test 4] Butler-Volmer solution reproduces current density")
    m, _ = make_model()
    T = 308.15
    j0 = m._arrhenius(m.j0_Cr, m.E_act_Cr, T)
    j_target = 0.05
    eta = m._bv_overpotential(j_target, j0, T)
    f = m.F / (m.R * T)
    a = m.alpha
    j_recon = j0 * (np.exp(a * f * eta) - np.exp(-(1 - a) * f * eta))
    assert_true(abs(j_recon - j_target) < 1e-6,
                f"BV inverse: j_recon={j_recon:.5f} ~ j_target={j_target:.5f}")


def test_coulombic_efficiency_bounds():
    print("\n[Test 5] Coulombic eff in (0,1) on charge; H2 parasitic on Cr side")
    m, _ = make_model()
    I_H2 = m.h2_parasitic_current(-50.0, 0.7, 313.15)
    assert_true(0.0 < I_H2 < 50.0, f"0 < I_H2={I_H2:.3f} A < |I|=50 A")
    # no HER on discharge
    assert_true(m.h2_parasitic_current(+50.0, 0.7, 313.15) == 0.0,
                "No H2 evolution during discharge")
    r = m.simulate(-50.0, 0.4, 308.15, 60.0, 1800.0)
    ce = r["coulombic_eff"]
    assert_true(np.all(ce > 0) and np.all(ce < 1.0),
                f"coulombic_eff in (0,1): min={ce.min():.4f} max={ce.max():.4f}")


def test_charge_discharge_soc_direction():
    print("\n[Test 6] Charge raises SOC, discharge lowers SOC (Coulomb balance)")
    m, _ = make_model()
    r_chg = m.simulate(-50.0, 0.4, 308.15, 60.0, 1800.0)
    r_dis = m.simulate(+50.0, 0.6, 308.15, 60.0, 1800.0)
    assert_true(r_chg["soc"][-1] > r_chg["soc"][0],
                f"charge: SOC {r_chg['soc'][0]:.3f} -> {r_chg['soc'][-1]:.3f}")
    assert_true(r_dis["soc"][-1] < r_dis["soc"][0],
                f"discharge: SOC {r_dis['soc'][0]:.3f} -> {r_dis['soc'][-1]:.3f}")


def test_coulomb_conservation_with_parasitic():
    print("\n[Test 7] dSOC matches (I - I_H2 - crossover) integrated charge")
    m, _ = make_model()
    # discharge: no HER, no crossover at... crossover small; check charge balance
    I = 50.0
    dur = 600.0
    r = m.simulate(I, 0.7, 308.15, 30.0, dur)
    dSOC = r["soc"][0] - r["soc"][-1]            # discharge -> positive drop
    Q_drawn = I * dur                            # Coulomb out
    # SOC drop * Q_cap should approx equal Q_drawn + crossover loss (>= Q_drawn)
    dSOC_charge = dSOC * m.Q_cap
    assert_true(dSOC_charge >= Q_drawn * 0.99,
                f"SOC-charge {dSOC_charge:.0f} C >= drawn {Q_drawn:.0f} C (+ self-disch)")
    assert_true(dSOC_charge < Q_drawn * 1.5,
                f"SOC-charge {dSOC_charge:.0f} C reasonable vs drawn {Q_drawn:.0f} C")


def test_self_discharge():
    print("\n[Test 8] Idle (I=0) self-discharges via crossover")
    m, _ = make_model()
    r = m.simulate(0.0, 0.8, 308.15, 600.0, 36000.0)   # 10 h idle
    assert_true(r["soc"][-1] < r["soc"][0],
                f"idle self-discharge: SOC {r['soc'][0]:.4f} -> {r['soc'][-1]:.4f}")


def test_thermal_balance():
    print("\n[Test 9] Thermal ODE heats under load, bounded, reaches steady-ish state")
    m, _ = make_model()
    r = m.simulate(120.0, 0.6, 298.15, 60.0, 7200.0)   # 2 h heavy discharge
    T = r["temperature"]
    assert_true(T[-1] > T[0], f"warms under load: {T[0]:.2f} -> {T[-1]:.2f} K")
    assert_true(T[-1] < 360.0, f"bounded temperature: {T[-1]:.2f} K < 360 K")
    dT = abs(T[-1] - T[-2])
    assert_true(dT < 0.5, f"approaching steady state: last dT={dT:.4f} K")


def test_overpotentials_and_temperature_effect():
    print("\n[Test 10] Overpotentials >= 0; warmer electrolyte lowers activation loss")
    m, _ = make_model()
    r = m.simulate(50.0, 0.5, 308.15, 60.0, 600.0)
    for name, arr in r["overpotentials"].items():
        assert_true(np.all(arr >= -1e-9), f"{name} overpotential >= 0")
    j = 50.0 / m.A_cell
    eta_cold = m.activation_overpotential(j, 288.15)
    eta_hot = m.activation_overpotential(j, 328.15)
    assert_true(eta_hot < eta_cold,
                f"warm helps Cr kinetics: eta(55C)={eta_hot*1000:.1f} < eta(15C)={eta_cold*1000:.1f} mV")


def test_predict_interface():
    print("\n[Test 11] ComponentModel predict() interface + get_info()")
    _, cm = make_model()
    r = cm.predict({"current_A": -50.0, "soc0": 0.5, "dt": 60.0, "duration_s": 600.0})
    for key in ["t", "soc", "temperature", "voltage", "power", "ocv",
                "efficiency", "coulombic_eff", "I_H2", "overpotentials"]:
        assert_true(key in r, f"Key '{key}' in output")
    assert_true(len(r["t"]) == len(r["voltage"]), "Arrays same length")
    info = cm.get_info()
    assert_true(info["component_id"] == "EC038" and info["version"] == "1.0.0",
                "get_info metadata correct")


def test_benchmark():
    print("\n[Test 12] Benchmark: 1 h sim at dt=10 s")
    m, _ = make_model()
    t0 = time.perf_counter()
    m.simulate(-50.0, 0.4, 308.15, 10.0, 3600.0)
    elapsed = time.perf_counter() - t0
    print(f"  1h simulation in {elapsed*1000:.1f} ms")
    assert_true(elapsed < 5.0, "Representative sim completes in < 5 s")


if __name__ == "__main__":
    tests = [
        test_ocv_range_and_soc_monotone,
        test_charge_above_discharge,
        test_cr_dominates_activation,
        test_butler_volmer_consistency,
        test_coulombic_efficiency_bounds,
        test_charge_discharge_soc_direction,
        test_coulomb_conservation_with_parasitic,
        test_self_discharge,
        test_thermal_balance,
        test_overpotentials_and_temperature_effect,
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
    print(f"EC038 Fe-Cr Flow Battery F2a -- Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    sys.exit(0 if failed == 0 else 1)
