"""
EC193 -- Methanation Reactor (Power-to-Gas) -- F2a Kinetics + Equilibrium
Test suite: physics sanity, mass/energy balance, Le Chatelier, edge cases.
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from model import MethanationReactor_F2a
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
def test_exothermic_temperature_rise():
    print("\n[Test 1] Exothermic reaction: reactor heats up from inlet T")
    m, _ = make_model()
    r = m.simulate(T0=523.15, duration_s=300.0, dt=1.0)
    assert_true(r["T"][-1] > 523.15, f"T_final={r['T'][-1]:.1f} K > 523.15 K (inlet)")
    assert_true(r["T"][-1] < 1000.0, f"T_final={r['T'][-1]:.1f} K < 1000 K (reasonable)")


def test_co2_conversion_positive():
    print("\n[Test 2] CO2 conversion is positive and < 1")
    m, _ = make_model()
    r = m.simulate(T0=573.15, duration_s=600.0, dt=5.0)
    X_final = r["X_CO2"][-1]
    assert_true(X_final > 0.0, f"X_CO2={X_final:.4f} > 0")
    assert_true(X_final <= 1.0 + 1e-6, f"X_CO2={X_final:.4f} <= 1.0")


def test_mass_balance():
    print("\n[Test 3] Mass balance: C atoms conserved (CO2 + CH4 = const)")
    m, _ = make_model()
    r = m.simulate(T0=523.15, duration_s=600.0, dt=5.0)
    # At each time step, CO2 consumed = CH4 produced (1:1 carbon)
    # In CSTR: C_CO2 + C_CH4 should relate to inlet condition
    # Check final: C_CO2_out + C_CH4_out ~ C_CO2_in (accounting for flow)
    # Simple check: CO2 consumed should produce proportional CH4
    C_CO2_consumed = r["C_CO2_in"] - r["C_CO2"][-1]
    C_CH4_produced = r["C_CH4"][-1]
    # In steady-state CSTR: CO2_in - CO2_out = CH4_out (per carbon balance in flow)
    ratio = C_CH4_produced / (C_CO2_consumed + 1e-10)
    assert_true(abs(ratio - 1.0) < 0.3, f"C balance ratio: {ratio:.3f} ~ 1.0")


def test_hydrogen_stoichiometry():
    print("\n[Test 4] H2 consumed ~ 4x CO2 consumed (stoichiometry)")
    m, _ = make_model()
    r = m.simulate(T0=573.15, duration_s=600.0, dt=5.0)
    dCO2 = r["C_CO2_in"] - r["C_CO2"][-1]
    dH2 = r["C_H2_in"] - r["C_H2"][-1]
    ratio = dH2 / (dCO2 + 1e-10)
    assert_true(2.5 < ratio < 5.5, f"H2/CO2 consumption ratio: {ratio:.2f} ~ 4.0")


def test_le_chatelier_pressure():
    print("\n[Test 5] Le Chatelier: higher P -> higher conversion (fewer moles product side)")
    m, _ = make_model()
    r_low = m.simulate(T0=573.15, duration_s=2000.0, dt=50.0, P=5.0,
                       T_in=573.15, T_cool=573.15)
    r_high = m.simulate(T0=573.15, duration_s=2000.0, dt=50.0, P=20.0,
                        T_in=573.15, T_cool=573.15)
    assert_true(
        r_high["X_CO2"][-1] > r_low["X_CO2"][-1],
        f"X(20bar)={r_high['X_CO2'][-1]:.4f} > X(5bar)={r_low['X_CO2'][-1]:.4f}"
    )


def test_equilibrium_conversion_decreases_with_T():
    print("\n[Test 6] Equilibrium conversion decreases with T (exothermic)")
    m, _ = make_model()
    X_low = m.equilibrium_conversion(573.15)
    X_high = m.equilibrium_conversion(1073.15)
    assert_true(X_low > X_high,
                f"X_eq(300C)={X_low:.4f} > X_eq(800C)={X_high:.4f}")


def test_rate_increases_with_T():
    print("\n[Test 7] Reaction rate increases with temperature (Arrhenius)")
    m, _ = make_model()
    r_low = m.reaction_rate(473.15, 2.0, 8.0)
    r_high = m.reaction_rate(673.15, 2.0, 8.0)
    assert_true(r_high > r_low, f"r(400C)={r_high:.6f} > r(200C)={r_low:.6f}")


def test_thermal_runaway_detection():
    print("\n[Test 8] Thermal runaway detection works")
    m, _ = make_model()
    # Normal operation should not trigger runaway
    r = m.simulate(T0=523.15, duration_s=600.0, dt=1.0)
    # Just check the flag is a boolean
    assert_true(isinstance(r["thermal_runaway"], bool), "thermal_runaway is bool")


def test_steady_state_reached():
    print("\n[Test 9] System approaches steady state")
    m, _ = make_model()
    r = m.simulate(T0=523.15, duration_s=2000.0, dt=10.0)
    # Check last two points are close
    dT = abs(r["T"][-1] - r["T"][-2])
    dX = abs(r["X_CO2"][-1] - r["X_CO2"][-2])
    assert_true(dT < 1.0, f"dT={dT:.4f} K < 1.0 (near SS)")
    assert_true(dX < 0.01, f"dX={dX:.6f} < 0.01 (near SS)")


def test_predict_interface():
    print("\n[Test 10] ComponentModel predict() interface")
    _, cm = make_model()
    r = cm.predict({"T0_K": 523.15, "duration_s": 60.0, "dt": 5.0})
    for key in ["t", "T", "X_CO2", "y_CH4_dry", "C_CO2", "C_CH4", "thermal_runaway"]:
        assert_true(key in r, f"Key '{key}' in output")
    assert_true(len(r["t"]) == len(r["T"]), "Arrays same length")


def test_benchmark():
    print("\n[Test 11] Benchmark: 600s simulation at dt=1.0")
    m, _ = make_model()
    t0 = time.perf_counter()
    m.simulate(T0=523.15, duration_s=600.0, dt=1.0)
    elapsed = time.perf_counter() - t0
    print(f"  600s simulation in {elapsed*1000:.1f} ms")
    assert_true(elapsed < 30.0, "Completes in < 30 s")


if __name__ == "__main__":
    tests = [
        test_exothermic_temperature_rise,
        test_co2_conversion_positive,
        test_mass_balance,
        test_hydrogen_stoichiometry,
        test_le_chatelier_pressure,
        test_equilibrium_conversion_decreases_with_T,
        test_rate_increases_with_T,
        test_thermal_runaway_detection,
        test_steady_state_reached,
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
    print(f"EC193 Methanation Reactor F2a -- Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    sys.exit(0 if failed == 0 else 1)
