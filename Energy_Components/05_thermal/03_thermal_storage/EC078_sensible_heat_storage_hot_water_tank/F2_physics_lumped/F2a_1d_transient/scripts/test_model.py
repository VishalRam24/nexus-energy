"""
EC078 -- Hot Water Tank TES -- F2a 1D Transient
Test suite: physics sanity, ODE convergence, edge cases.
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from model import HotWaterTank_F2a
from predict import ComponentModel

PASS = "\u2713"
FAIL = "\u2717"


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
def test_charge_heats_tank():
    print("\n[Test 1] Charging heats the tank")
    m, _ = make_model()
    T_init = np.full(20, 293.15)
    r = m.simulate(0.5, 353.15, 0.0, 288.15, T_init, 10.0, 1800.0)
    assert_true(r["T_mean"][-1] > 293.15, f"T_mean rises: {r['T_mean'][-1]:.2f} > 293.15 K")
    assert_true(r["E_stored_kWh"][-1] > r["E_stored_kWh"][0],
                f"Energy increases: {r['E_stored_kWh'][-1]:.2f} > {r['E_stored_kWh'][0]:.2f} kWh")


def test_discharge_cools_tank():
    print("\n[Test 2] Discharging cools the tank")
    m, _ = make_model()
    T_init = np.full(20, 353.15)
    r = m.simulate(0.0, 353.15, 0.5, 288.15, T_init, 10.0, 1800.0)
    assert_true(r["T_mean"][-1] < 353.15, f"T_mean drops: {r['T_mean'][-1]:.2f} < 353.15 K")
    assert_true(r["E_stored_kWh"][-1] < r["E_stored_kWh"][0],
                f"Energy decreases: {r['E_stored_kWh'][-1]:.2f} < {r['E_stored_kWh'][0]:.2f} kWh")


def test_stratification_during_charge():
    print("\n[Test 3] Stratification: top hotter than bottom during charge")
    m, _ = make_model()
    T_init = np.full(20, 293.15)
    r = m.simulate(0.5, 353.15, 0.0, 288.15, T_init, 10.0, 600.0)
    # After charging, top should be hotter than bottom
    assert_true(r["T_top"][-1] > r["T_bottom"][-1],
                f"T_top={r['T_top'][-1]:.2f} > T_bottom={r['T_bottom'][-1]:.2f}")
    assert_true(r["stratification_K"][-1] > 0,
                f"Stratification positive: {r['stratification_K'][-1]:.2f} K")


def test_no_flow_heat_loss():
    print("\n[Test 4] No flow: tank loses heat to ambient")
    m, _ = make_model()
    T_init = np.full(20, 353.15)
    r = m.simulate(0.0, 353.15, 0.0, 288.15, T_init, 30.0, 7200.0)
    assert_true(r["T_mean"][-1] < 353.15,
                f"Tank cools from heat loss: {r['T_mean'][-1]:.2f} < 353.15 K")
    assert_true(r["T_mean"][-1] > 293.15,
                f"Tank still above ambient: {r['T_mean'][-1]:.2f} > 293.15 K")


def test_temperature_range():
    print("\n[Test 5] All temperatures in physical range")
    m, _ = make_model()
    T_init = np.full(20, 293.15)
    r = m.simulate(0.5, 353.15, 0.0, 288.15, T_init, 10.0, 3600.0)
    assert_true(np.all(r["T_profiles"] >= 273.15), "All T >= 0 C")
    assert_true(np.all(r["T_profiles"] <= 373.15), "All T <= 100 C")


def test_energy_conservation_standby():
    print("\n[Test 6] Energy conservation: standby losses match UA model")
    m, _ = make_model()
    T_init = np.full(20, 353.15)
    r = m.simulate(0.0, 353.15, 0.0, 288.15, T_init, 60.0, 3600.0)
    dE = r["E_stored_J"][0] - r["E_stored_J"][-1]
    # Approximate expected loss: U*A_total*dT*time
    A_total = np.pi * m.D_tank * m.H_tank + 2 * m.A_cross
    avg_dT = np.mean(r["T_mean"]) - m.T_ambient
    Q_loss_est = m.U_loss * A_total * avg_dT * 3600.0
    rel_err = abs(dE - Q_loss_est) / max(Q_loss_est, 1.0)
    assert_true(rel_err < 0.20, f"Energy loss matches UA model: dE={dE/1e6:.2f} MJ, est={Q_loss_est/1e6:.2f} MJ, err={rel_err*100:.1f}%")


def test_charge_discharge_cycle():
    print("\n[Test 7] Charge/discharge cycle recovers energy")
    m, _ = make_model()
    T_init = np.full(20, 293.15)

    def m_ch(t):
        return 0.5 if t < 1800 else 0.0

    def m_dis(t):
        return 0.0 if t < 1800 else 0.5

    r = m.simulate(m_ch, 353.15, m_dis, 288.15, T_init, 10.0, 3600.0)
    idx_mid = np.argmin(np.abs(r["t"] - 1800))
    E_peak = r["E_stored_kWh"][idx_mid]
    E_final = r["E_stored_kWh"][-1]
    assert_true(E_peak > E_final, f"Peak energy > final: {E_peak:.2f} > {E_final:.2f} kWh")


def test_uniform_no_stratification():
    print("\n[Test 8] Uniform initial temp with no flow: minimal stratification change")
    m, _ = make_model()
    T_init = np.full(20, 323.15)
    r = m.simulate(0.0, 353.15, 0.0, 288.15, T_init, 30.0, 600.0)
    # With uniform temp and no flow, stratification should stay near zero
    assert_true(abs(r["stratification_K"][-1]) < 5.0,
                f"Stratification stays small: {r['stratification_K'][-1]:.2f} K")


def test_predict_interface():
    print("\n[Test 9] ComponentModel predict() interface")
    _, cm = make_model()
    r = cm.predict({"duration_s": 60.0, "dt": 10.0})
    for key in ["t", "T_profiles", "T_top", "T_bottom", "T_mean", "E_stored_kWh", "stratification_K"]:
        assert_true(key in r, f"Key '{key}' in output")
    assert_true(len(r["t"]) == len(r["T_top"]), "Arrays same length")


def test_benchmark():
    print("\n[Test 10] Benchmark: 3600s sim at dt=10")
    m, _ = make_model()
    T_init = np.full(20, 293.15)
    t0 = time.perf_counter()
    m.simulate(0.5, 353.15, 0.0, 288.15, T_init, 10.0, 3600.0)
    elapsed = time.perf_counter() - t0
    print(f"  3600s simulation in {elapsed*1000:.1f} ms")
    assert_true(elapsed < 30.0, "Completes in < 30 s")


if __name__ == "__main__":
    tests = [
        test_charge_heats_tank,
        test_discharge_cools_tank,
        test_stratification_during_charge,
        test_no_flow_heat_loss,
        test_temperature_range,
        test_energy_conservation_standby,
        test_charge_discharge_cycle,
        test_uniform_no_stratification,
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
    print(f"EC078 Hot Water Tank F2a -- Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    sys.exit(0 if failed == 0 else 1)
