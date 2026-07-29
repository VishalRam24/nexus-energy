"""
EC143 -- Biomass Gasifier -- F2a Chemical Equilibrium
Test suite: physics sanity, atom balance, Le Chatelier, edge cases.
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from model import BiomassGasifier_F2a
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
def test_atom_balance():
    print("\n[Test 1] Atom balance (C, H, O, N) closes")
    m, _ = make_model()
    r = m.solve_equilibrium(T=1073.15, ER=0.30, moisture=0.15)
    n = r["moles_per_C"]
    x, y, z = m.biomass_formula()
    O2_stoich = m.stoichiometric_air(x, y, z)
    air_m = 0.30 * O2_stoich
    MW_dry = 12.011 + x * 1.008 + y * 15.999 + z * 14.007
    w = (0.15 / 0.85) * MW_dry / 18.015

    # C balance: 1 = CO + CO2 + CH4
    C_bal = n["CO"] + n["CO2"] + n["CH4"]
    assert_true(abs(C_bal - 1.0) < 0.01, f"C balance: {C_bal:.4f} ~ 1.0")

    # H balance: x + 2*w = 2*H2 + 2*H2O + 4*CH4
    H_in = x + 2.0 * w
    H_out = 2 * n["H2"] + 2 * n["H2O"] + 4 * n["CH4"]
    assert_true(abs(H_out - H_in) < 0.01, f"H balance: {H_out:.4f} ~ {H_in:.4f}")

    # O balance: y + w + 2*m = CO + 2*CO2 + H2O
    O_in = y + w + 2 * air_m
    O_out = n["CO"] + 2 * n["CO2"] + n["H2O"]
    assert_true(abs(O_out - O_in) < 0.01, f"O balance: {O_out:.4f} ~ {O_in:.4f}")


def test_composition_sums_to_100():
    print("\n[Test 2] Dry composition sums to ~100%")
    m, _ = make_model()
    r = m.solve_equilibrium()
    total = sum(r["composition_dry_mol_pct"].values())
    assert_true(abs(total - 100.0) < 0.5, f"Total dry = {total:.2f}%")


def test_le_chatelier_temperature():
    print("\n[Test 3] Le Chatelier: Higher T -> more CO, H2 (endothermic rxns favoured)")
    m, _ = make_model()
    r_low = m.solve_equilibrium(T=973.15)
    r_high = m.solve_equilibrium(T=1273.15)
    # CO should increase with T (Boudouard endothermic)
    assert_true(
        r_high["composition_dry_mol_pct"]["CO"] > r_low["composition_dry_mol_pct"]["CO"],
        f"CO: {r_high['composition_dry_mol_pct']['CO']:.2f}% > {r_low['composition_dry_mol_pct']['CO']:.2f}%"
    )
    # CH4 should decrease with T (methanation exothermic)
    assert_true(
        r_high["composition_dry_mol_pct"]["CH4"] < r_low["composition_dry_mol_pct"]["CH4"],
        f"CH4: {r_high['composition_dry_mol_pct']['CH4']:.2f}% < {r_low['composition_dry_mol_pct']['CH4']:.2f}%"
    )


def test_le_chatelier_er():
    print("\n[Test 4] Le Chatelier: Higher ER -> more CO2 (more combustion)")
    m, _ = make_model()
    r_low = m.solve_equilibrium(ER=0.20)
    r_high = m.solve_equilibrium(ER=0.45)
    assert_true(
        r_high["composition_dry_mol_pct"]["CO2"] > r_low["composition_dry_mol_pct"]["CO2"],
        f"CO2: {r_high['composition_dry_mol_pct']['CO2']:.2f}% > {r_low['composition_dry_mol_pct']['CO2']:.2f}%"
    )


def test_lhv_positive():
    print("\n[Test 5] LHV of syngas is positive and reasonable (2-8 MJ/Nm3)")
    _, cm = make_model()
    r = cm.predict({"temperature_K": 1073.15, "equivalence_ratio": 0.30})
    LHV = r["LHV_syngas_MJ_Nm3"]
    assert_true(LHV > 1.0, f"LHV={LHV:.3f} > 1.0 MJ/Nm3")
    assert_true(LHV < 15.0, f"LHV={LHV:.3f} < 15.0 MJ/Nm3")


def test_cge_range():
    print("\n[Test 6] Cold gas efficiency in (0, 1)")
    _, cm = make_model()
    r = cm.predict({"temperature_K": 1073.15, "equivalence_ratio": 0.30})
    CGE = r["cold_gas_efficiency"]
    assert_true(CGE > 0.0, f"CGE={CGE:.3f} > 0")
    assert_true(CGE < 1.5, f"CGE={CGE:.3f} < 1.5 (reasonable)")


def test_h2_co_ratio():
    print("\n[Test 7] H2/CO ratio is positive and reasonable")
    _, cm = make_model()
    r = cm.predict({"temperature_K": 1073.15, "equivalence_ratio": 0.30})
    ratio = r["H2_CO_ratio"]
    assert_true(ratio > 0.0, f"H2/CO={ratio:.3f} > 0")
    assert_true(ratio < 5.0, f"H2/CO={ratio:.3f} < 5")


def test_moisture_increases_h2():
    print("\n[Test 8] Higher moisture content -> more H2 (steam reforming)")
    m, _ = make_model()
    r_dry = m.solve_equilibrium(moisture=0.05)
    r_wet = m.solve_equilibrium(moisture=0.30)
    assert_true(
        r_wet["composition_dry_mol_pct"]["H2"] > r_dry["composition_dry_mol_pct"]["H2"],
        f"H2: {r_wet['composition_dry_mol_pct']['H2']:.2f}% > {r_dry['composition_dry_mol_pct']['H2']:.2f}%"
    )


def test_all_species_non_negative():
    print("\n[Test 9] All species mole fractions non-negative")
    _, cm = make_model()
    for T in [973.15, 1073.15, 1273.15]:
        for ER in [0.20, 0.30, 0.40]:
            r = cm.predict({"temperature_K": T, "equivalence_ratio": ER})
            for sp, val in r["composition_dry_mol_pct"].items():
                assert_true(val >= -0.1, f"T={T:.0f}, ER={ER}: {sp}={val:.3f}% >= 0")


def test_predict_interface():
    print("\n[Test 10] ComponentModel predict() interface")
    _, cm = make_model()
    r = cm.predict({"temperature_K": 1073.15, "equivalence_ratio": 0.30})
    for key in ["composition_dry_mol_pct", "LHV_syngas_MJ_Nm3", "cold_gas_efficiency", "H2_CO_ratio"]:
        assert_true(key in r, f"Key '{key}' in output")


def test_benchmark():
    print("\n[Test 11] Benchmark: 100 equilibrium solves")
    m, _ = make_model()
    t0 = time.perf_counter()
    for _ in range(100):
        m.solve_equilibrium(T=1073.15, ER=0.30)
    elapsed = time.perf_counter() - t0
    print(f"  100 solves in {elapsed*1000:.1f} ms ({elapsed*10:.2f} ms/solve)")
    assert_true(elapsed < 10.0, "Completes in < 10 s")


if __name__ == "__main__":
    tests = [
        test_atom_balance,
        test_composition_sums_to_100,
        test_le_chatelier_temperature,
        test_le_chatelier_er,
        test_lhv_positive,
        test_cge_range,
        test_h2_co_ratio,
        test_moisture_increases_h2,
        test_all_species_non_negative,
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
    print(f"EC143 Biomass Gasifier F2a -- Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    sys.exit(0 if failed == 0 else 1)
