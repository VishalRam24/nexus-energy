"""
EC079 -- Molten Salt TES -- F2a 1D Transient
Test suite: physics sanity, T-dependent properties, ODE convergence.
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from model import MoltenSaltTES_F2a
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
def test_salt_properties_range():
    print("\n[Test 1] Solar Salt properties in valid ranges")
    m, _ = make_model()
    for T_C in [290, 400, 565]:
        T_K = T_C + 273.15
        rho = m.salt_density(T_K)
        cp = m.salt_cp(T_K)
        k = m.salt_conductivity(T_K)
        assert_true(1700 < rho < 2000, f"rho({T_C}C)={rho:.0f} in [1700,2000] kg/m3")
        assert_true(1400 < cp < 1600, f"cp({T_C}C)={cp:.0f} in [1400,1600] J/(kg.K)")
        assert_true(0.4 < k < 0.6, f"k({T_C}C)={k:.3f} in [0.4,0.6] W/(m.K)")


def test_salt_density_decreases_with_T():
    print("\n[Test 2] Salt density decreases with temperature")
    m, _ = make_model()
    rho_cold = m.salt_density(563.15)
    rho_hot = m.salt_density(838.15)
    assert_true(rho_cold > rho_hot, f"rho(290C)={rho_cold:.0f} > rho(565C)={rho_hot:.0f}")


def test_charge_heats_tank():
    print("\n[Test 3] Charging heats the cold tank")
    m, _ = make_model()
    T_init = m.initial_temperature_profile("cold")
    r = m.simulate(100.0, 838.15, 0.0, 563.15, T_init, 60.0, 7200.0)
    assert_true(r["T_mean"][-1] > 563.15, f"T_mean rises: {r['T_mean'][-1]:.1f} > 563.15 K")
    assert_true(r["E_stored_MWh"][-1] > r["E_stored_MWh"][0],
                f"Energy increases during charge")


def test_discharge_cools_tank():
    print("\n[Test 4] Discharging cools the hot tank")
    m, _ = make_model()
    T_init = m.initial_temperature_profile("hot")
    r = m.simulate(0.0, 838.15, 100.0, 563.15, T_init, 60.0, 7200.0)
    assert_true(r["T_mean"][-1] < 838.15, f"T_mean drops: {r['T_mean'][-1]:.1f} < 838.15 K")


def test_stratification():
    print("\n[Test 5] Stratification during charge: top hotter than bottom")
    m, _ = make_model()
    T_init = m.initial_temperature_profile("cold")
    r = m.simulate(100.0, 838.15, 0.0, 563.15, T_init, 60.0, 3600.0)
    assert_true(r["T_top"][-1] > r["T_bottom"][-1],
                f"T_top={r['T_top'][-1]:.1f} > T_bottom={r['T_bottom'][-1]:.1f}")


def test_temperature_in_operating_range():
    print("\n[Test 6] Temperatures stay in operating range (260-600 C)")
    m, _ = make_model()
    T_init = m.initial_temperature_profile("cold")
    r = m.simulate(100.0, 838.15, 0.0, 563.15, T_init, 60.0, 14400.0)
    T_min = np.min(r["T_profiles"])
    T_max = np.max(r["T_profiles"])
    assert_true(T_min >= 533.15, f"T_min={T_min-273.15:.1f}C >= 260C")
    assert_true(T_max <= 873.15, f"T_max={T_max-273.15:.1f}C <= 600C")


def test_standby_heat_loss():
    print("\n[Test 7] Standby: hot tank loses heat slowly")
    m, _ = make_model()
    T_init = m.initial_temperature_profile("hot")
    r = m.simulate(0.0, 838.15, 0.0, 563.15, T_init, 120.0, 43200.0)  # 12h standby
    T_drop = r["T_mean"][0] - r["T_mean"][-1]
    assert_true(0 < T_drop < 50, f"T drops by {T_drop:.2f} K in 12h (reasonable)")


def test_charge_discharge_cycle():
    print("\n[Test 8] Charge/discharge cycle energy recovery")
    m, _ = make_model()
    T_init = m.initial_temperature_profile("cold")

    def m_ch(t):
        return 100.0 if t < 7200 else 0.0

    def m_dis(t):
        return 0.0 if t < 7200 else 100.0

    r = m.simulate(m_ch, 838.15, m_dis, 563.15, T_init, 60.0, 14400.0)
    idx_mid = np.argmin(np.abs(r["t"] - 7200))
    E_peak = r["E_stored_MWh"][idx_mid]
    E_final = r["E_stored_MWh"][-1]
    assert_true(E_peak > E_final, f"Peak E={E_peak:.2f} > final E={E_final:.2f} MWh")


def test_predict_interface():
    print("\n[Test 9] ComponentModel predict() interface")
    _, cm = make_model()
    r = cm.predict({"duration_s": 600.0, "dt": 60.0})
    for key in ["t", "T_profiles", "T_top", "T_bottom", "T_mean", "E_stored_MWh",
                "stratification_K", "rho_mean", "cp_mean"]:
        assert_true(key in r, f"Key '{key}' in output")


def test_benchmark():
    print("\n[Test 10] Benchmark: 4h sim at dt=60s")
    m, _ = make_model()
    T_init = m.initial_temperature_profile("cold")
    t0 = time.perf_counter()
    m.simulate(100.0, 838.15, 0.0, 563.15, T_init, 60.0, 14400.0)
    elapsed = time.perf_counter() - t0
    print(f"  14400s simulation in {elapsed*1000:.1f} ms")
    assert_true(elapsed < 30.0, "Completes in < 30 s")


if __name__ == "__main__":
    tests = [
        test_salt_properties_range,
        test_salt_density_decreases_with_T,
        test_charge_heats_tank,
        test_discharge_cools_tank,
        test_stratification,
        test_temperature_in_operating_range,
        test_standby_heat_loss,
        test_charge_discharge_cycle,
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
    print(f"EC079 Molten Salt TES F2a -- Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    sys.exit(0 if failed == 0 else 1)
