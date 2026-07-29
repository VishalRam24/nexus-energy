"""
EC200 -- Oxy-Fuel Combustion Capture -- F2a Combustion + Recycle Model
Test suite: conservation, monotonicity, known limits, ODE convergence,
edge cases, interface, benchmark. Custom assert_true harness -- NO pytest.
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from model import OxyFuelF2a
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
def test_predict_keys():
    print("\n[Test 1] predict() returns all expected keys")
    _, cm = make_model()
    r = cm.predict({"mdot_fuel": 50.0, "recycle_ratio": 0.6, "duration_s": 60.0})
    for k in ["t", "temperature", "T_steady", "T_adiabatic", "co2_purity_dry",
              "co2_purity_wet", "o2_demand_kgs", "co2_produced_kgs",
              "product_gas_kgs", "recycle_kgs"]:
        assert_true(k in r, f"Key '{k}' present")
    assert_true(len(r["t"]) == len(r["temperature"]), "t and temperature same length")


def test_get_info():
    print("\n[Test 2] get_info() metadata")
    _, cm = make_model()
    info = cm.get_info()
    assert_true(info["component_id"] == "EC200", "component_id == EC200")
    assert_true("F2a" in info["fidelity"], "fidelity contains F2a")


def test_carbon_conservation():
    print("\n[Test 3] Carbon conservation: all fuel C -> CO2")
    m, _ = make_model()
    mdot = 50.0
    s = m.stoichiometry()
    n_C_fuel = m.w_C / m.MW_C * 1000.0          # mol C / kg fuel
    assert_true(abs(s["n_CO2"] - n_C_fuel) < 1e-9, "n_CO2 == n_C (no C lost)")
    co2_kgs = s["n_CO2"] * m.MW_CO2 / 1000.0 * mdot
    c_in = mdot * m.w_C
    c_out = co2_kgs * (m.MW_C / m.MW_CO2)
    assert_true(abs(c_in - c_out) / c_in < 1e-6,
                f"C in fuel ({c_in:.2f}) == C in CO2 ({c_out:.2f}) kg/s")


def test_o2_conservation():
    print("\n[Test 4] O2 balance: supplied O2 >= stoichiometric demand")
    m, _ = make_model()
    s = m.stoichiometry()
    assert_true(m.o2_supplied() > s["n_O2_stoich"],
                "supplied O2 exceeds stoichiometric (excess present)")
    o2_kgs = m.o2_supplied() * m.MW_O2 / 1000.0 * 50.0
    ratio = o2_kgs / 50.0
    assert_true(2.0 < ratio < 3.0, f"O2/fuel mass ratio {ratio:.2f} realistic for coal")


def test_high_co2_purity_dry():
    print("\n[Test 5] Dry-basis CO2 purity > 0.90 (oxy-fuel, Buhre 2005)")
    m, _ = make_model()
    p_dry = m.co2_purity_dry()
    assert_true(p_dry > 0.90, f"dry CO2 purity = {p_dry:.3f} > 0.90")
    assert_true(p_dry < 1.0, f"dry CO2 purity = {p_dry:.3f} < 1.0")


def test_water_knockout_raises_purity():
    print("\n[Test 6] Water knockout: dry purity > wet purity")
    m, _ = make_model()
    assert_true(m.co2_purity_dry() > m.co2_purity_wet(),
                f"dry ({m.co2_purity_dry():.3f}) > wet ({m.co2_purity_wet():.3f})")


def test_recycle_moderates_flame_temp():
    print("\n[Test 7] Recycle ratio MONOTONICALLY lowers adiabatic flame temp")
    m, _ = make_model()
    Rs = [0.0, 0.2, 0.4, 0.6, 0.8]
    T_prev = m.adiabatic_flame_temp(50.0, Rs[0])
    for R in Rs[1:]:
        T = m.adiabatic_flame_temp(50.0, R)
        assert_true(T < T_prev, f"R={R}: T_ad={T:.0f}K < previous {T_prev:.0f}K")
        T_prev = T
    T0 = m.adiabatic_flame_temp(50.0, 0.0)
    T8 = m.adiabatic_flame_temp(50.0, 0.8)
    assert_true(T0 - T8 > 500.0, f"recycle drops T_ad by {T0-T8:.0f} K (significant)")


def test_ode_reaches_steady_state():
    print("\n[Test 8] Furnace ODE converges to steady state")
    _, cm = make_model()
    r = cm.predict({"mdot_fuel": 50.0, "recycle_ratio": 0.6, "duration_s": 300.0, "dt": 1.0})
    T = r["temperature"]
    dT = abs(T[-1] - T[-2])
    assert_true(dT < 0.5, f"near steady: dT = {dT:.4f} K between last two steps")
    assert_true(800.0 < r["T_steady"] < 2000.0,
                f"steady furnace T = {r['T_steady']:.0f} K physically reasonable")


def test_steady_state_energy_balance():
    print("\n[Test 9] Steady-state energy balance (dT/dt ~ 0 at T_steady)")
    m, cm = make_model()
    r = cm.predict({"mdot_fuel": 50.0, "recycle_ratio": 0.6, "duration_s": 400.0, "dt": 1.0})
    rate = m.dTdt(r["T_steady"], 50.0, 0.6)
    assert_true(abs(rate) < 1.0, f"|dT/dt| at steady = {abs(rate):.4f} K/s ~ 0")


def test_flame_above_steady():
    print("\n[Test 10] Adiabatic flame temp > furnace steady temp (wall losses)")
    _, cm = make_model()
    r = cm.predict({"mdot_fuel": 50.0, "recycle_ratio": 0.6, "duration_s": 200.0})
    assert_true(r["T_adiabatic"] > r["T_steady"],
                f"T_ad ({r['T_adiabatic']:.0f}) > T_steady ({r['T_steady']:.0f}) K")


def test_scaling_with_fuel():
    print("\n[Test 11] CO2/O2/gas flows scale linearly with fuel rate")
    _, cm = make_model()
    r1 = cm.predict({"mdot_fuel": 25.0, "recycle_ratio": 0.6, "duration_s": 30.0})
    r2 = cm.predict({"mdot_fuel": 50.0, "recycle_ratio": 0.6, "duration_s": 30.0})
    assert_true(abs(r2["co2_produced_kgs"] / r1["co2_produced_kgs"] - 2.0) < 1e-6,
                "CO2 produced doubles with double fuel")
    assert_true(abs(r2["o2_demand_kgs"] / r1["o2_demand_kgs"] - 2.0) < 1e-6,
                "O2 demand doubles with double fuel")


def test_purity_intensive():
    print("\n[Test 12] CO2 purity is intensive (independent of fuel rate)")
    _, cm = make_model()
    r1 = cm.predict({"mdot_fuel": 10.0, "recycle_ratio": 0.6, "duration_s": 30.0})
    r2 = cm.predict({"mdot_fuel": 80.0, "recycle_ratio": 0.6, "duration_s": 30.0})
    assert_true(abs(r1["co2_purity_dry"] - r2["co2_purity_dry"]) < 1e-9,
                "dry purity independent of fuel rate")


def test_benchmark():
    print("\n[Test 13] Benchmark: 120 s furnace simulation at dt=0.5")
    _, cm = make_model()
    t0 = time.perf_counter()
    cm.predict({"mdot_fuel": 50.0, "recycle_ratio": 0.6, "dt": 0.5, "duration_s": 120.0})
    elapsed = time.perf_counter() - t0
    print(f"  120 s simulation in {elapsed*1000:.1f} ms")
    assert_true(elapsed < 5.0, "completes in < 5 s")


if __name__ == "__main__":
    tests = [
        test_predict_keys,
        test_get_info,
        test_carbon_conservation,
        test_o2_conservation,
        test_high_co2_purity_dry,
        test_water_knockout_raises_purity,
        test_recycle_moderates_flame_temp,
        test_ode_reaches_steady_state,
        test_steady_state_energy_balance,
        test_flame_above_steady,
        test_scaling_with_fuel,
        test_purity_intensive,
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
    print(f"EC200 Oxy-Fuel F2a -- Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    sys.exit(0 if failed == 0 else 1)
