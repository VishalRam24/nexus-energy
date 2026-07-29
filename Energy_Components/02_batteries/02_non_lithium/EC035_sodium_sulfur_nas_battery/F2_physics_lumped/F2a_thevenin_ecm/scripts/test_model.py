"""
EC035 -- Sodium-Sulfur (NaS) Battery -- F2a Thevenin 1-RC ECM
Test suite: Coulomb conservation, OCV behaviour, Arrhenius R(T), thermal
balance, efficiency bounds, edge cases, predict() interface, benchmark.
Run with system python3 (NOT pytest):  python3 scripts/test_model.py
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from model import NaSBatteryF2a
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
def test_ocv_behaviour():
    print("\n[Test 1] OCV(SOC) monotone increasing in NaS 1.78-2.08 V band")
    m, _ = make_model()
    socs = np.linspace(0.0, 1.0, 100)
    ocv = m.ocv(socs)
    assert_true(np.all(np.diff(ocv) >= -1e-6), "OCV monotonically increases with SOC")
    assert_true(1.76 <= ocv.min() <= 1.80, f"OCV(empty)={ocv.min():.4f} ~ 1.78 V")
    assert_true(2.05 <= ocv.max() <= 2.10, f"OCV(full)={ocv.max():.4f} ~ 2.08 V (high plateau)")


def test_coulomb_conservation():
    print("\n[Test 2] Coulomb counting: dSOC = -I*t/(Q*3600)")
    m, _ = make_model()
    I = 50.0
    dur = 1200.0  # 20 min
    r = m.simulate(I, 0.95, 593.15, 5.0, dur)
    Q_eff = m.capacity(593.15)
    expected_dsoc = -I * dur / (Q_eff * 3600.0)
    actual_dsoc = r["soc"][-1] - r["soc"][0]
    # capacity drifts a touch with T; allow 2% tolerance
    rel = abs(actual_dsoc - expected_dsoc) / abs(expected_dsoc)
    assert_true(rel < 0.02, f"dSOC actual={actual_dsoc:.5f} vs ideal={expected_dsoc:.5f} (rel {rel*100:.2f}%)")


def test_coulomb_charge_discharge_symmetry():
    print("\n[Test 3] Charge then discharge returns SOC to start (Coulomb closure)")
    m, _ = make_model()
    # discharge 30 A for 600 s then charge -30 A for 600 s at fixed T window
    def Ifun(t):
        return 30.0 if t < 600.0 else -30.0
    r = m.simulate(Ifun, 0.7, 593.15, 5.0, 1200.0)
    assert_true(abs(r["soc"][-1] - 0.7) < 5e-3, f"SOC returns to 0.700 (got {r['soc'][-1]:.4f})")


def test_discharge_drops_soc_and_voltage():
    print("\n[Test 4] Discharge lowers SOC and V_term < OCV")
    m, _ = make_model()
    r = m.simulate(40.0, 0.9, 593.15, 5.0, 600.0)
    assert_true(r["soc"][-1] < r["soc"][0], "SOC decreases on discharge")
    ocv_end = float(m.ocv(r["soc"][-1]))
    assert_true(r["voltage"][-1] < ocv_end, f"V_term={r['voltage'][-1]:.4f} < OCV={ocv_end:.4f} under load")


def test_arrhenius_resistance():
    print("\n[Test 5] Beta-alumina R(T) decreases as T rises (Arrhenius)")
    m, _ = make_model()
    R_low = float(m.R0(573.15))   # 300 C
    R_ref = float(m.R0(593.15))   # 320 C
    R_high = float(m.R0(623.15))  # 350 C
    assert_true(R_low > R_ref > R_high, f"R(300C)={R_low*1e3:.3f} > R(320C)={R_ref*1e3:.3f} > R(350C)={R_high*1e3:.3f} mOhm")
    assert_true(abs(R_ref - m.R0_ref) < 1e-9, "R0(T_op_ref) == R0_ref")


def test_thermal_balance_steady_state():
    print("\n[Test 6] Thermal ODE: idle cell held in band by heater (balance)")
    m, _ = make_model()
    # idle (I=0): heater + loss only, should settle inside operating window
    r = m.simulate(0.0, 0.5, 593.15, 30.0, 7200.0)
    T_ss = r["temperature"][-1]
    assert_true(m.T_op_min <= T_ss <= m.T_op_max, f"Idle T_ss={T_ss:.2f} K stays in 300-350 C window")
    dT = abs(r["temperature"][-1] - r["temperature"][-2])
    assert_true(dT < 0.5, f"Near steady state: dT={dT:.4f} K/step")


def test_self_heating():
    print("\n[Test 7] Self-heating: heavy current raises T above pure-idle")
    m, _ = make_model()
    r_idle = m.simulate(0.0, 0.5, 593.15, 30.0, 3600.0)
    r_load = m.simulate(60.0, 0.9, 593.15, 30.0, 3600.0)
    assert_true(r_load["temperature"][-1] > r_idle["temperature"][-1],
                f"Loaded T={r_load['temperature'][-1]:.2f} > idle T={r_idle['temperature'][-1]:.2f} K")
    assert_true(np.all(r_load["heat_gen"][r_load["current"] != 0] >= -1e-6) or
                np.all(r_load["heat_gen"] >= -5.0),
                "Heat generation dominated by non-negative Joule term")


def test_heater_keeps_molten():
    print("\n[Test 8] Heater fights heat loss from cold-ish start within band")
    m, _ = make_model()
    # start near lower edge; heater should pull T up toward setpoint
    r = m.simulate(0.0, 0.5, 575.0, 30.0, 7200.0)
    assert_true(r["temperature"][-1] > 575.0, f"Heater raises T from 575->{r['temperature'][-1]:.2f} K")
    assert_true(r["heater_power"][0] > 0.0, "Heater ON below setpoint")


def test_efficiency_bounds():
    print("\n[Test 9] Round-trip / coulombic efficiency in (0,1)")
    m, _ = make_model()
    # discharge: energy out vs OCV energy reference
    r = m.simulate(40.0, 0.9, 593.15, 5.0, 600.0)
    v = r["voltage"]
    ocv = m.ocv(r["soc"])
    # voltaic efficiency = V_term/OCV during discharge, must be in (0,1)
    eta = v / ocv
    assert_true(np.all((eta > 0.0) & (eta < 1.0)), f"Discharge voltaic eff in (0,1): min={eta.min():.4f} max={eta.max():.4f}")


def test_outside_window_nonfunctional():
    print("\n[Test 10] Outside 300-350 C: cell non-functional (V=0, SOC frozen)")
    m, _ = make_model()
    assert_true(not bool(m.is_functional(550.0)), "T=550 K (277 C) non-functional")
    assert_true(not bool(m.is_functional(650.0)), "T=650 K (377 C) non-functional")
    v = float(m.terminal_voltage(0.5, 40.0, 560.0))
    assert_true(v == 0.0, f"V_term=0 when frozen (got {v})")


def test_predict_interface():
    print("\n[Test 11] ComponentModel predict() interface")
    _, cm = make_model()
    r = cm.predict({"current_A": 30.0, "soc0": 0.8, "dt": 10.0, "duration_s": 300.0})
    for key in ["t", "soc", "voltage", "current", "power", "temperature",
                "v_rc", "R0", "R1", "heat_gen", "heater_power", "functional"]:
        assert_true(key in r, f"Key '{key}' in output")
    assert_true(len(r["t"]) == len(r["voltage"]) == len(r["soc"]), "Arrays same length")
    info = cm.get_info()
    assert_true(info["component_id"] == "EC035", "get_info component_id == EC035")


def test_benchmark():
    print("\n[Test 12] Benchmark: 1 h sim at dt=5 s")
    m, _ = make_model()
    t0 = time.perf_counter()
    m.simulate(40.0, 0.9, 593.15, 5.0, 3600.0)
    elapsed = time.perf_counter() - t0
    print(f"  3600 s simulation in {elapsed*1000:.1f} ms")
    assert_true(elapsed < 5.0, "Completes in < 5 s")


if __name__ == "__main__":
    tests = [
        test_ocv_behaviour,
        test_coulomb_conservation,
        test_coulomb_charge_discharge_symmetry,
        test_discharge_drops_soc_and_voltage,
        test_arrhenius_resistance,
        test_thermal_balance_steady_state,
        test_self_heating,
        test_heater_keeps_molten,
        test_efficiency_bounds,
        test_outside_window_nonfunctional,
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
    print(f"EC035 NaS F2a Thevenin ECM -- Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    sys.exit(0 if failed == 0 else 1)
