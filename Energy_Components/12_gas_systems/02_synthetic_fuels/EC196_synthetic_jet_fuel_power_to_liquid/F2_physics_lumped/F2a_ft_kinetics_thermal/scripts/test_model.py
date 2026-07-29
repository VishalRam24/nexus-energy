"""
EC196 -- Synthetic Jet Fuel (PtL) -- F2a
Test suite: physics sanity (ASF, conservation, exotherm, efficiency<1),
ODE convergence, edge cases, predict() interface, benchmark timing.
NO pytest -- custom harness, run with system python3.
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from model import FTJetFuelF2a
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
def test_asf_mass_closure():
    print("\n[Test 1] ASF distribution sums to 1 (mass conservation)")
    m, _ = make_model()
    for a in [0.70, 0.85, 0.90, 0.95]:
        n = np.arange(1, 401)
        W = n * (1.0 - a) ** 2 * a ** (n - 1)
        assert_true(abs(np.sum(W) - 1.0) < 1e-3,
                    f"sum(W_n)={np.sum(W):.5f} ~ 1 for alpha={a}")


def test_asf_jet_peak():
    print("\n[Test 2] Jet-cut selectivity is a bounded fraction, peaks at mid-alpha")
    m, _ = make_model()
    S_low = m.asf_selectivity_jet(alpha=0.70)
    S_mid = m.asf_selectivity_jet(alpha=0.88)
    S_high = m.asf_selectivity_jet(alpha=0.98)
    for S in [S_low, S_mid, S_high]:
        assert_true(0.0 <= S <= 1.0, f"S_jet={S:.3f} in [0,1]")
    # Wax-inclusive jet selectivity should be substantial at high alpha
    assert_true(S_mid > 0.3, f"S_jet(0.88)={S_mid:.3f} > 0.3 (jet-tuned)")


def test_alpha_decreases_with_T():
    print("\n[Test 3] Chain-growth alpha decreases with temperature (Dry 2002)")
    m, _ = make_model()
    a_low = m.alpha(200.0)
    a_high = m.alpha(280.0)
    assert_true(a_high < a_low, f"alpha(280C)={a_high:.3f} < alpha(200C)={a_low:.3f}")


def test_arrhenius_monotone():
    print("\n[Test 4] FT rate increases with temperature (Arrhenius)")
    m, _ = make_model()
    r_prev = m.co_consumption_rate(180.0)
    for T in np.linspace(185, 290, 30):
        r = m.co_consumption_rate(T)
        assert_true(r >= r_prev - 1e-15, f"r({T:.0f}C)={r:.4e} >= prev")
        r_prev = r


def test_conversion_bounded():
    print("\n[Test 5] CO conversion in [0,1] and stoichiometry-limited")
    m, _ = make_model()
    for T in [180, 220, 260, 300]:
        X = m.conversion(T)
        assert_true(0.0 <= X <= 1.0, f"X_CO({T}C)={X:.3f} in [0,1]")
    # very high rate -> capped at stoichiometric limit (h2_co/2.1 = 1.0)
    X_hot = m.conversion(300.0)
    assert_true(X_hot <= 1.0 + 1e-9, "X capped at stoichiometric limit")


def test_carbon_balance():
    print("\n[Test 6] Carbon balance: C_to_jet + C_to_other = X*n_CO")
    m, _ = make_model()
    y = m.yields(220.0, n_co_in=1.0)
    lhs = y["C_to_jet_mol_s"] + y["C_to_other_mol_s"]
    rhs = y["co_conversion"] * 1.0
    assert_true(abs(lhs - rhs) < 1e-9,
                f"C_jet+C_other={lhs:.5f} == X*n_CO={rhs:.5f}")
    # jet molar rate consistent with carbon count
    assert_true(abs(y["jet_mol_s"] * m.C_per_jet - y["C_to_jet_mol_s"]) < 1e-12,
                "jet_mol*C_per_jet == C_to_jet")


def test_efficiency_below_one():
    print("\n[Test 7] PtL efficiency strictly in (0,1) (Schmidt 2018)")
    m, _ = make_model()
    for T in [180, 200, 220, 250, 290]:
        eta = m.ptl_efficiency(T)
        assert_true(0.0 < eta < 1.0, f"eta_PtL({T}C)={eta:.3f} in (0,1)")
    # realistic PtL window (state of the art ~0.35-0.55 at design point)
    eta_opt = m.ptl_efficiency(220.0)
    assert_true(0.20 < eta_opt < 0.75,
                f"eta_PtL(220C)={eta_opt:.3f} in realistic PtL band")


def test_exotherm_positive():
    print("\n[Test 8] FT reaction is exothermic: Q_gen > 0 when reacting")
    m, _ = make_model()
    Q = m.heat_released_W(220.0)
    assert_true(Q > 0.0, f"Q_gen(220C)={Q/1000:.1f} kW > 0 (exothermic)")


def test_thermal_steady_state():
    print("\n[Test 9] Lumped thermal ODE reaches steady state")
    m, _ = make_model()
    r = m.simulate(T0_C=205.0, dt=30.0, duration_s=10800.0)
    dT = abs(r["temperature"][-1] - r["temperature"][-2])
    assert_true(dT < 0.05, f"Near SS: dT={dT:.4f} degC between last two steps")
    # energy balance at SS: Q_gen ~ Q_cool
    T_ss = r["temperature"][-1]
    Q_gen = m.heat_released_W(T_ss)
    Q_cool = m.UA * (T_ss - m.T_coolant)
    assert_true(abs(Q_gen - Q_cool) / max(Q_gen, 1.0) < 0.02,
                f"SS energy balance Q_gen~Q_cool ({Q_gen/1000:.1f} vs {Q_cool/1000:.1f} kW)")


def test_thermal_heats_from_cold():
    print("\n[Test 10] Exothermic reactor heats up above coolant from cold start")
    m, _ = make_model()
    r = m.simulate(T0_C=200.0, dt=30.0, duration_s=7200.0)
    assert_true(r["temperature"][-1] > m.T_coolant,
                f"T_final={r['temperature'][-1]:.2f} > T_coolant={m.T_coolant} (self-heating)")
    assert_true(r["temperature"][-1] < 350.0,
                f"T_final={r['temperature'][-1]:.2f} < 350 C (bounded, no runaway)")


def test_predict_interface():
    print("\n[Test 11] ComponentModel predict() interface")
    _, cm = make_model()
    r = cm.predict({"T0_C": 210.0, "dt": 60.0, "duration_s": 1800.0})
    for key in ["t", "temperature", "co_conversion", "selectivity_jet",
                "alpha", "jet_mol_s", "jet_kg_s", "ptl_efficiency",
                "heat_released_kW"]:
        assert_true(key in r, f"Key '{key}' in output")
    assert_true(len(r["t"]) == len(r["temperature"]), "Arrays same length")
    info = cm.get_info()
    assert_true(info["component_id"] == "EC196", "get_info id == EC196")


def test_benchmark():
    print("\n[Test 12] Benchmark: 1 h sim at dt=10 s")
    m, _ = make_model()
    t0 = time.perf_counter()
    m.simulate(T0_C=200.0, dt=10.0, duration_s=3600.0)
    elapsed = time.perf_counter() - t0
    print(f"  3600s simulation in {elapsed*1000:.1f} ms")
    assert_true(elapsed < 5.0, "Completes in < 5 s")


if __name__ == "__main__":
    tests = [
        test_asf_mass_closure,
        test_asf_jet_peak,
        test_alpha_decreases_with_T,
        test_arrhenius_monotone,
        test_conversion_bounded,
        test_carbon_balance,
        test_efficiency_below_one,
        test_exotherm_positive,
        test_thermal_steady_state,
        test_thermal_heats_from_cold,
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
    print(f"EC196 PtL Jet Fuel F2a -- Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    sys.exit(0 if failed == 0 else 1)
