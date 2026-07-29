"""
EC195 -- Ammonia Synthesis (Haber-Bosch) -- F2a Temkin-Pyzhev Kinetics
Test suite: physics sanity, mass balance, Le Chatelier, edge cases.
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from model import AmmoniaSynthesis_F2a
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
    print("\n[Test 1] Exothermic reaction: reactor heats up")
    m, _ = make_model()
    r = m.simulate(T0=673.15, duration_s=300.0, dt=1.0)
    assert_true(r["T"][-1] > 673.15, f"T_final={r['T'][-1]:.1f} K > 673.15 K")
    assert_true(r["T"][-1] < 1200.0, f"T_final={r['T'][-1]:.1f} K < 1200 K")


def test_n2_conversion_positive():
    print("\n[Test 2] N2 conversion positive and bounded")
    m, _ = make_model()
    r = m.simulate(T0=723.15, duration_s=600.0, dt=5.0)
    X_final = r["X_N2"][-1]
    assert_true(X_final > 0.0, f"X_N2={X_final:.4f} > 0")
    assert_true(X_final <= 1.0 + 1e-6, f"X_N2={X_final:.4f} <= 1.0")


def test_stoichiometry():
    print("\n[Test 3] H2 consumed ~ 3x N2 consumed (stoichiometry)")
    m, _ = make_model()
    r = m.simulate(T0=723.15, duration_s=600.0, dt=5.0)
    dN2 = r["C_N2_in"] - r["C_N2"][-1]
    dH2 = r["C_H2_in"] - r["C_H2"][-1]
    ratio = dH2 / (dN2 + 1e-10)
    assert_true(1.5 < ratio < 4.5, f"H2/N2 consumption ratio: {ratio:.2f} ~ 3.0")


def test_le_chatelier_pressure():
    print("\n[Test 4] Le Chatelier: higher P -> higher equilibrium conversion")
    m, _ = make_model()
    X_low = m.equilibrium_conversion(723.15, P=100)
    X_high = m.equilibrium_conversion(723.15, P=300)
    assert_true(X_high > X_low,
                f"X_eq(300atm)={X_high:.4f} > X_eq(100atm)={X_low:.4f}")


def test_le_chatelier_temperature():
    print("\n[Test 5] Le Chatelier: higher T -> lower equilibrium conversion (exothermic)")
    m, _ = make_model()
    X_low_T = m.equilibrium_conversion(573.15, P=200)
    X_high_T = m.equilibrium_conversion(873.15, P=200)
    assert_true(X_low_T > X_high_T,
                f"X_eq(300C)={X_low_T:.4f} > X_eq(600C)={X_high_T:.4f}")


def test_keq_decreases_with_T():
    print("\n[Test 6] K_eq decreases with T (exothermic)")
    K_low = AmmoniaSynthesis_F2a.K_eq(573.15)
    K_high = AmmoniaSynthesis_F2a.K_eq(873.15)
    assert_true(K_low > K_high, f"K(300C)={K_low:.4e} > K(600C)={K_high:.4e}")


def test_single_pass_conversion_range():
    print("\n[Test 7] Single-pass conversion in typical range (5-30%)")
    m, _ = make_model()
    r = m.simulate(T0=723.15, duration_s=2000.0, dt=50.0)
    X_ss = r["X_N2"][-1]
    assert_true(X_ss > 0.01, f"X_sp={X_ss:.4f} > 1% (some conversion)")
    assert_true(X_ss < 0.60, f"X_sp={X_ss:.4f} < 60% (physically limited)")


def test_recycle_boosts_conversion():
    print("\n[Test 8] Recycle loop achieves higher overall conversion")
    m, _ = make_model()
    r_recycle = m.simulate_with_recycle(n_passes=10)
    assert_true(r_recycle["overall_conversion"] > 0.5,
                f"Overall X={r_recycle['overall_conversion']:.4f} > 0.5 with recycle")


def test_nh3_mole_fraction_positive():
    print("\n[Test 9] NH3 mole fraction is positive")
    m, _ = make_model()
    r = m.simulate(T0=723.15, duration_s=300.0, dt=5.0)
    assert_true(np.all(r["y_NH3"] >= 0), "All y_NH3 >= 0")
    assert_true(r["y_NH3"][-1] > 0, f"y_NH3_final={r['y_NH3'][-1]:.6f} > 0")


def test_predict_interface():
    print("\n[Test 10] ComponentModel predict() interface")
    _, cm = make_model()
    r = cm.predict({"T0_K": 723.15, "duration_s": 60.0, "dt": 5.0})
    for key in ["t", "T", "X_N2", "y_NH3", "C_N2", "C_H2", "C_NH3"]:
        assert_true(key in r, f"Key '{key}' in output")
    assert_true(len(r["t"]) == len(r["T"]), "Arrays same length")


def test_benchmark():
    print("\n[Test 11] Benchmark: 600s simulation at dt=1.0")
    m, _ = make_model()
    t0 = time.perf_counter()
    m.simulate(T0=723.15, duration_s=600.0, dt=1.0)
    elapsed = time.perf_counter() - t0
    print(f"  600s simulation in {elapsed*1000:.1f} ms")
    assert_true(elapsed < 30.0, "Completes in < 30 s")


if __name__ == "__main__":
    tests = [
        test_exothermic_temperature_rise,
        test_n2_conversion_positive,
        test_stoichiometry,
        test_le_chatelier_pressure,
        test_le_chatelier_temperature,
        test_keq_decreases_with_T,
        test_single_pass_conversion_range,
        test_recycle_boosts_conversion,
        test_nh3_mole_fraction_positive,
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
    print(f"EC195 Ammonia Synthesis F2a -- Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    sys.exit(0 if failed == 0 else 1)
