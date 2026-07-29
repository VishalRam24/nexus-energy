"""
EC146 -- Torrefaction Reactor -- F2a Two-Step Kinetics + Reactor ODE
Test suite: mass conservation, densification, T/time dependence, edge cases.
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from model import TorrefactionF2a
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
def test_mass_conservation():
    print("\n[Test 1] Mass conservation: solid + volatiles == feed")
    m, _ = make_model()
    for T in [240, 280, 310]:
        r = m.simulate(T_set_degC=T, residence_time_min=45, T0_degC=T)
        assert_true(r["mass_balance_residual"] < 1e-9,
                    f"T={T}: residual={r['mass_balance_residual']:.2e} < 1e-9")


def test_mass_yield_below_one():
    print("\n[Test 2] Mass yield < 1 (solid is always lost to volatiles)")
    m, _ = make_model()
    for T in [240, 280, 310]:
        for t in [10, 30, 90]:
            r = m.simulate(T_set_degC=T, residence_time_min=t, T0_degC=T)
            Ym = r["mass_yield_final"]
            assert_true(0.0 < Ym < 1.0, f"T={T},t={t}: Ym={Ym:.3f} in (0,1)")


def test_energy_densification():
    print("\n[Test 3] Energy yield > mass yield (densification)")
    m, _ = make_model()
    for T in [250, 280, 300]:
        for t in [15, 45, 90]:
            r = m.simulate(T_set_degC=T, residence_time_min=t, T0_degC=T)
            Ym, Ye = r["mass_yield_final"], r["energy_yield_final"]
            assert_true(Ye >= Ym - 1e-9,
                        f"T={T},t={t}: Ye={Ye:.3f} >= Ym={Ym:.3f}")


def test_hhv_upgrade():
    print("\n[Test 4] HHV/LHV upgrade >= 1 (solid energy-densifies)")
    m, _ = make_model()
    r = m.simulate(T_set_degC=300, residence_time_min=90, T0_degC=300)
    assert_true(r["hhv_upgrade_final"] >= 1.0,
                f"HHV upgrade={r['hhv_upgrade_final']:.3f} >= 1")
    assert_true(r["LHV_solid_final"] >= m.LHV_raw,
                f"LHV_solid={r['LHV_solid_final']:.2f} >= LHV_raw={m.LHV_raw:.2f}")


def test_temperature_dependence():
    print("\n[Test 5] Mass yield decreases with temperature (fixed time)")
    m, _ = make_model()
    Ym_prev = None
    for T in [230, 250, 270, 290, 310]:
        r = m.simulate(T_set_degC=T, residence_time_min=30, T0_degC=T)
        Ym = r["mass_yield_final"]
        if Ym_prev is not None:
            assert_true(Ym <= Ym_prev + 1e-9,
                        f"T={T}: Ym={Ym:.3f} <= prev={Ym_prev:.3f}")
        Ym_prev = Ym


def test_time_dependence():
    print("\n[Test 6] Mass yield decreases with residence time (fixed T)")
    m, _ = make_model()
    Ym_prev = None
    for t in [5, 15, 30, 60, 120]:
        r = m.simulate(T_set_degC=280, residence_time_min=t, T0_degC=280)
        Ym = r["mass_yield_final"]
        if Ym_prev is not None:
            assert_true(Ym <= Ym_prev + 1e-9,
                        f"t={t}: Ym={Ym:.3f} <= prev={Ym_prev:.3f}")
        Ym_prev = Ym


def test_arrhenius_rates():
    print("\n[Test 7] Arrhenius rate constants increase with temperature")
    m, _ = make_model()
    k1_lo, k1_hi = m.k1(250 + 273.15), m.k1(300 + 273.15)
    k2_lo, k2_hi = m.k2(250 + 273.15), m.k2(300 + 273.15)
    assert_true(k1_hi > k1_lo, f"k1: {k1_hi:.3e} > {k1_lo:.3e}")
    assert_true(k2_hi > k2_lo, f"k2: {k2_hi:.3e} > {k2_lo:.3e}")
    assert_true(k1_lo > 0 and k2_lo > 0, "rates strictly positive")


def test_thermal_ode_heats_up():
    print("\n[Test 8] Reactor ODE: cold feed heats toward wall setpoint")
    m, _ = make_model()
    r = m.simulate(T_set_degC=280, residence_time_min=30, T0_degC=25)
    assert_true(r["temperature_degC"][0] < 30.0, "Starts cold (~25 C)")
    assert_true(r["temperature_final_degC"] > 200.0,
                f"Heats up to {r['temperature_final_degC']:.1f} C")
    # exothermic torrefaction can overshoot wall slightly but stay bounded
    assert_true(r["temperature_final_degC"] < 320.0,
                f"Bounded T_final={r['temperature_final_degC']:.1f} C < 320")


def test_conversion_monotone_in_time():
    print("\n[Test 9] Volatile conversion is monotonically non-decreasing in time")
    m, _ = make_model()
    r = m.simulate(T_set_degC=290, residence_time_min=60, T0_degC=290, dt_s=5)
    conv = r["conversion"]
    diffs = np.diff(conv)
    assert_true(np.all(diffs >= -1e-9), "conversion(t) non-decreasing")
    assert_true(conv[-1] > conv[0], f"net conversion {conv[-1]:.3f} > {conv[0]:.3f}")


def test_zero_time_limit():
    print("\n[Test 10] Zero residence time -> no conversion (Ym ~ 1)")
    m, _ = make_model()
    r = m.simulate(T_set_degC=300, residence_time_min=0.0167, T0_degC=300)  # 1 s
    assert_true(r["mass_yield_final"] > 0.99,
                f"Ym={r['mass_yield_final']:.4f} ~ 1 at near-zero time")
    assert_true(r["conversion_final"] < 0.01,
                f"conversion={r['conversion_final']:.4f} ~ 0")


def test_predict_interface():
    print("\n[Test 11] ComponentModel predict() interface")
    _, cm = make_model()
    r = cm.predict({"T_set_degC": 275, "residence_time_min": 20})
    for key in ["t", "mass_yield", "energy_yield", "hhv_upgrade",
                "conversion", "temperature_degC", "solid_mass"]:
        assert_true(key in r, f"Key '{key}' in output")
    assert_true(len(r["t"]) == len(r["mass_yield"]), "Arrays same length")
    info = cm.get_info()
    assert_true(info["component_id"] == "EC146", "get_info id == EC146")


def test_benchmark():
    print("\n[Test 12] Benchmark: 60-min sim at dt=5s")
    m, _ = make_model()
    t0 = time.perf_counter()
    m.simulate(T_set_degC=280, residence_time_min=60, T0_degC=25, dt_s=5)
    elapsed = time.perf_counter() - t0
    print(f"  60-min torrefaction sim in {elapsed*1000:.1f} ms")
    assert_true(elapsed < 5.0, "Completes in < 5 s")


if __name__ == "__main__":
    tests = [
        test_mass_conservation,
        test_mass_yield_below_one,
        test_energy_densification,
        test_hhv_upgrade,
        test_temperature_dependence,
        test_time_dependence,
        test_arrhenius_rates,
        test_thermal_ode_heats_up,
        test_conversion_monotone_in_time,
        test_zero_time_limit,
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
    print(f"EC146 Torrefaction F2a -- Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    sys.exit(0 if failed == 0 else 1)
