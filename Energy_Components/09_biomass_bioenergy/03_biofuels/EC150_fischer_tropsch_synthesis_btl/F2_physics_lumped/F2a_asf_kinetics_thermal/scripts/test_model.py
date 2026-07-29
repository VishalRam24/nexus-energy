"""
EC150 -- Fischer-Tropsch Synthesis (BTL) -- F2a ASF Kinetics + Thermal ODE
Test suite: ASF/conservation physics, kinetics monotonicity, exothermic thermal
balance, edge cases, predict() interface, benchmark timing. NO pytest.
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from model import FischerTropschF2a, MW_CO, MW_CH2
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
def test_asf_sums_to_one():
    print("\n[Test 1] ASF mole & weight distributions sum to 1")
    m, _ = make_model()
    n = np.arange(1, 400)
    for alpha in [0.70, 0.80, 0.85, 0.92]:
        x_sum = m.asf_mole_fraction(alpha, n).sum()
        W_sum = m.asf_weight_fraction(alpha, n).sum()
        assert_true(abs(x_sum - 1.0) < 1e-3, f"alpha={alpha}: sum x_n={x_sum:.5f}~1")
        assert_true(abs(W_sum - 1.0) < 1e-3, f"alpha={alpha}: sum W_n={W_sum:.5f}~1")


def test_product_cuts_sum_to_one():
    print("\n[Test 2] Lumped product cuts sum to 1 (carbon/weight closure)")
    m, _ = make_model()
    for alpha in [0.75, 0.85, 0.90]:
        cuts = m.product_cuts(alpha)
        tot = sum(cuts.values())
        assert_true(abs(tot - 1.0) < 1e-6, f"alpha={alpha}: cut sum={tot:.6f}=1")


def test_alpha_selectivity_trend():
    print("\n[Test 3] Higher alpha -> heavier slate (more wax, less light gas)")
    m, _ = make_model()
    low = m.product_cuts(0.75)
    high = m.product_cuts(0.90)
    assert_true(high["wax_C21plus"] > low["wax_C21plus"], "wax rises with alpha")
    assert_true(high["light_gas_C1_C4"] < low["light_gas_C1_C4"], "light gas falls with alpha")


def test_conversion_below_one():
    print("\n[Test 4] CO conversion strictly in (0, 1)")
    m, _ = make_model()
    P = m.P_nom
    for T in [470.0, 493.15, 520.0]:
        X = m.co_conversion(T, 40.0, 80.0, P)
        assert_true(0.0 < X < 1.0, f"T={T}K: X={X:.4f} in (0,1)")


def test_arrhenius_monotone():
    print("\n[Test 5] Arrhenius: rate constant increases with T")
    m, _ = make_model()
    ks = [m.rate_constant(T) for T in [470.0, 500.0, 530.0]]
    assert_true(ks[0] < ks[1] < ks[2], f"k(T) increasing: {ks[0]:.3g} < {ks[1]:.3g} < {ks[2]:.3g}")


def test_conversion_increases_with_T():
    print("\n[Test 6] CO conversion increases with temperature (Arrhenius)")
    m, _ = make_model()
    P = m.P_nom
    X_lo = m.co_conversion(470.0, 40.0, 80.0, P)
    X_hi = m.co_conversion(515.0, 40.0, 80.0, P)
    assert_true(X_hi > X_lo, f"X(515K)={X_hi:.4f} > X(470K)={X_lo:.4f}")


def test_carbon_conservation():
    print("\n[Test 7] Carbon conservation: -CH2- mass = (14/28)*CO mass converted")
    m, _ = make_model()
    res = m.steady_products(493.15, 40.0, 80.0, m.P_nom)
    co_mol_s = res["CO_converted_mol_s"]
    co_mass_kg_s = co_mol_s * MW_CO / 1000.0
    ch2_mass_kg_s = res["HC_total_kg_s"]
    expected_ratio = MW_CH2 / MW_CO  # carbon retained as -CH2-
    actual_ratio = ch2_mass_kg_s / co_mass_kg_s if co_mass_kg_s > 0 else 0.0
    assert_true(abs(actual_ratio - expected_ratio) < 1e-6,
                f"mass ratio -CH2-/CO={actual_ratio:.5f} == {expected_ratio:.5f}")
    # carbon atoms conserved: 1 C per converted CO == 1 C per -CH2- unit
    ch2_mol_s = ch2_mass_kg_s * 1000.0 / MW_CH2
    assert_true(abs(ch2_mol_s - co_mol_s) < 1e-6, "C atoms in = C atoms out (CO->CH2)")


def test_exothermic_heats_up():
    print("\n[Test 8] Exothermic ODE: reactor heats up from cold feed start")
    m, _ = make_model()
    r = m.simulate(120.0, 0.40, 470.0, m.P_nom, 50.0, 3000.0)
    assert_true(r["temperature"][-1] > 470.0, f"T_final={r['temperature'][-1]:.2f} > 470 K (exotherm)")
    assert_true(r["temperature"][-1] < 600.0, f"T_final={r['temperature'][-1]:.2f} < 600 K (cooled, no runaway)")
    assert_true(np.all(r["heat_generated_W"] >= 0.0), "Q_gen >= 0 always")


def test_thermal_steady_state():
    print("\n[Test 9] Thermal balance reaches steady state (Q_gen ~ Q_cool)")
    m, _ = make_model()
    r = m.simulate(100.0, 0.40, 490.0, m.P_nom, 50.0, 8000.0)
    dT = abs(r["temperature"][-1] - r["temperature"][-2])
    assert_true(dT < 0.05, f"Near SS: dT={dT:.4f} K between last two steps")
    bal = abs(r["heat_generated_W"][-1] - r["heat_removed_W"][-1])
    scale = max(r["heat_generated_W"][-1], 1.0)
    assert_true(bal / scale < 0.02, f"At SS Q_gen~Q_cool: imbalance={bal/scale*100:.2f}%")


def test_cooling_limits_runaway():
    print("\n[Test 10] Stronger cooling -> lower steady temperature")
    m_weak, _ = make_model()
    m_strong, _ = make_model()
    m_strong.UA = m_weak.UA * 3.0
    r_weak = m_weak.simulate(150.0, 0.40, 490.0, m_weak.P_nom, 50.0, 6000.0)
    r_strong = m_strong.simulate(150.0, 0.40, 490.0, m_strong.P_nom, 50.0, 6000.0)
    assert_true(r_strong["temperature"][-1] < r_weak["temperature"][-1],
                f"T_strong={r_strong['temperature'][-1]:.1f} < T_weak={r_weak['temperature'][-1]:.1f}")


def test_zero_feed_no_reaction():
    print("\n[Test 11] Zero syngas feed -> no conversion, reactor cools to coolant")
    m, _ = make_model()
    r = m.simulate(0.0, 0.40, 520.0, m.P_nom, 50.0, 5000.0)
    assert_true(np.allclose(r["CO_conversion"], 0.0), "X=0 with no CO converted (F_CO0=0)")
    assert_true(np.allclose(r["heat_generated_W"], 0.0), "Q_gen=0 with no feed")
    assert_true(r["temperature"][-1] < 520.0, f"Reactor cools: T_final={r['temperature'][-1]:.2f} < 520 K")
    assert_true(r["temperature"][-1] > m.T_cool - 1.0, "T relaxes toward coolant temperature")


def test_predict_interface():
    print("\n[Test 12] ComponentModel predict() interface keys + array shapes")
    _, cm = make_model()
    r = cm.predict({"syngas_flow_mol_s": 80.0, "CO_fraction": 0.40,
                    "T0_K": 490.0, "dt": 100.0, "duration_s": 1000.0})
    for key in ["t", "temperature", "CO_conversion", "alpha",
                "heat_generated_W", "heat_removed_W",
                "liquid_C5plus_kg_s", "energy_output_MW", "product_cuts"]:
        assert_true(key in r, f"Key '{key}' in output")
    assert_true(len(r["t"]) == len(r["temperature"]) == len(r["CO_conversion"]),
                "Time-series arrays same length")
    assert_true(np.all((r["energy_output_MW"] >= 0.0)), "Energy output >= 0")


def test_benchmark():
    print("\n[Test 13] Benchmark: 3000s reactor transient")
    m, _ = make_model()
    t0 = time.perf_counter()
    m.simulate(100.0, 0.40, 490.0, m.P_nom, 50.0, 3000.0)
    elapsed = time.perf_counter() - t0
    print(f"  3000s transient simulated in {elapsed*1000:.1f} ms")
    assert_true(elapsed < 5.0, "Completes in < 5 s")


if __name__ == "__main__":
    tests = [
        test_asf_sums_to_one,
        test_product_cuts_sum_to_one,
        test_alpha_selectivity_trend,
        test_conversion_below_one,
        test_arrhenius_monotone,
        test_conversion_increases_with_T,
        test_carbon_conservation,
        test_exothermic_heats_up,
        test_thermal_steady_state,
        test_cooling_limits_runaway,
        test_zero_feed_no_reaction,
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
    print(f"EC150 FT F2a -- Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    sys.exit(0 if failed == 0 else 1)
