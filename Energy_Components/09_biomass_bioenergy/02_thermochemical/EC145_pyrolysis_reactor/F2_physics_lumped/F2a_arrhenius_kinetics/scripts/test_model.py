"""
EC145 -- Pyrolysis Reactor -- F2a Lumped Arrhenius Kinetics + Energy Balance
Test suite: mass conservation, kinetics monotonicity, yield-temperature dependence,
energy balance, edge cases, predict() interface, benchmark timing.
NO pytest -- custom assert harness, run with: python3 scripts/test_model.py
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from model import PyrolysisReactorF2a, R_GAS
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
def test_rate_arrhenius_increases_with_T():
    print("\n[Test 1] Arrhenius rate constants increase with temperature")
    m, _ = make_model()
    klo = m.rate_constants(600.0)
    khi = m.rate_constants(900.0)
    for a, b, name in zip(klo, khi, ["k_gas", "k_tar", "k_char", "k_crack"]):
        assert_true(b > a, f"{name}: k(900K)={b:.3e} > k(600K)={a:.3e}")


def test_mass_conservation():
    print("\n[Test 2] Mass conservation: y_B+y_G+y_T+y_C = 1 at all times")
    m, _ = make_model()
    r = m.simulate(5000.0, 600.0, dt=0.5, duration_s=120.0)
    assert_true(np.max(r["mass_residual"]) < 1e-6,
                f"max |sum-1| = {np.max(r['mass_residual']):.2e} < 1e-6")


def test_yields_in_unit_interval():
    print("\n[Test 3] All yields and conversion in [0,1]")
    m, _ = make_model()
    r = m.simulate(5000.0, 600.0, dt=1.0, duration_s=120.0)
    for key in ["y_biomass", "y_gas", "y_bio_oil", "y_char", "conversion"]:
        arr = r[key]
        assert_true(np.all(arr >= -1e-9) and np.all(arr <= 1.0 + 1e-9),
                    f"{key} in [0,1] (min={arr.min():.4f}, max={arr.max():.4f})")


def test_conversion_monotone():
    print("\n[Test 4] Conversion is monotonically non-decreasing")
    m, _ = make_model()
    r = m.simulate(5000.0, 650.0, dt=0.5, duration_s=120.0)
    conv = r["conversion"]
    diffs = np.diff(conv)
    assert_true(np.all(diffs >= -1e-6),
                f"conversion non-decreasing (min step = {diffs.min():.2e})")
    assert_true(conv[-1] > conv[0], f"net conversion {conv[-1]:.3f} > {conv[0]:.3f}")


def test_biomass_decays():
    print("\n[Test 5] Residual biomass decays monotonically toward zero")
    m, _ = make_model()
    r = m.simulate(6000.0, 700.0, dt=0.5, duration_s=180.0)
    yB = r["y_biomass"]
    assert_true(np.all(np.diff(yB) <= 1e-6), "y_biomass non-increasing")
    assert_true(yB[-1] < 0.20, f"y_biomass largely consumed: {yB[-1]:.3f} < 0.20")


def test_endothermic_energy_balance():
    print("\n[Test 6] Endothermic reaction retards heating vs no-reaction limit")
    m, _ = make_model()
    # With heating, temperature should rise from a cold-ish start
    r = m.simulate(8000.0, 500.0, dt=0.5, duration_s=120.0)
    assert_true(r["temperature"][-1] > 500.0,
                f"reactor heats up: T_final={r['temperature'][-1]:.1f} > 500 K")
    # No external heat + losses: reactor must cool toward ambient (T_amb=298.15)
    r0 = m.simulate(0.0, 500.0, dt=0.5, duration_s=120.0)
    assert_true(r0["temperature"][-1] < 500.0,
                f"no heat -> cools: T_final={r0['temperature'][-1]:.1f} < 500 K")


def test_fast_pyrolysis_maximizes_bio_oil():
    print("\n[Test 7] Yield-temperature dependence: bio-oil peaks near 500 degC (~773 K)")
    m, _ = make_model()
    temps = [573.15, 673.15, 773.15, 1073.15]  # 300, 400, 500, 800 degC
    oils = {T: m.equilibrium_yields(T)["bio_oil_yield"] for T in temps}
    # Bio-oil at 500C should exceed both low-T (char-dominated) and very-high-T (cracked-to-gas)
    assert_true(oils[773.15] > oils[573.15],
                f"oil(500C)={oils[773.15]:.3f} > oil(300C)={oils[573.15]:.3f}")
    assert_true(oils[773.15] > oils[1073.15],
                f"oil(500C)={oils[773.15]:.3f} > oil(800C)={oils[1073.15]:.3f} (secondary cracking)")


def test_high_T_favors_gas_low_T_favors_char():
    print("\n[Test 8] High T -> more gas; low T -> more char")
    m, _ = make_model()
    lo = m.equilibrium_yields(573.15)   # 300 degC
    hi = m.equilibrium_yields(1073.15)  # 800 degC
    assert_true(hi["gas_yield"] > lo["gas_yield"],
                f"gas(800C)={hi['gas_yield']:.3f} > gas(300C)={lo['gas_yield']:.3f}")
    assert_true(lo["char_yield"] > hi["char_yield"],
                f"char(300C)={lo['char_yield']:.3f} > char(800C)={hi['char_yield']:.3f}")


def test_equilibrium_mass_closes():
    print("\n[Test 9] Isothermal equilibrium yields sum to ~1")
    m, _ = make_model()
    eq = m.equilibrium_yields(773.15)
    total = eq["bio_oil_yield"] + eq["char_yield"] + eq["gas_yield"] + eq["residual_biomass"]
    assert_true(abs(total - 1.0) < 1e-6, f"sum = {total:.6f} ~ 1.0")


def test_predict_interface():
    print("\n[Test 10] ComponentModel predict() interface (dynamic + isothermal)")
    _, cm = make_model()
    r = cm.predict({"Q_ext_W": 4000.0, "T0_K": 600.0, "dt": 1.0, "duration_s": 60.0})
    for key in ["t", "temperature", "y_bio_oil", "y_char", "y_gas",
                "conversion", "mass_residual"]:
        assert_true(key in r, f"key '{key}' in dynamic output")
    assert_true(len(r["t"]) == len(r["temperature"]), "arrays same length")
    iso = cm.predict({"mode": "isothermal", "T_isothermal": 773.15})
    for key in ["bio_oil_yield", "char_yield", "gas_yield"]:
        assert_true(key in iso, f"key '{key}' in isothermal output")
    info = cm.get_info()
    assert_true(info["component_id"] == "EC145", "get_info component_id == EC145")


def test_energy_in_products_positive():
    print("\n[Test 11] Product energy content non-negative and bounded")
    m, _ = make_model()
    r = m.simulate(6000.0, 700.0, dt=1.0, duration_s=120.0)
    for key in ["energy_bio_oil_MJ_kg", "energy_char_MJ_kg", "energy_gas_MJ_kg"]:
        assert_true(np.all(r[key] >= -1e-9), f"{key} >= 0")
    total_e = (r["energy_bio_oil_MJ_kg"][-1] + r["energy_char_MJ_kg"][-1]
               + r["energy_gas_MJ_kg"][-1])
    assert_true(0.0 < total_e < 40.0, f"total product energy {total_e:.1f} MJ/kg in (0,40)")


def test_benchmark():
    print("\n[Test 12] Benchmark: 120 s dynamic sim at dt=0.5")
    m, _ = make_model()
    t0 = time.perf_counter()
    m.simulate(5000.0, 600.0, dt=0.5, duration_s=120.0)
    elapsed = time.perf_counter() - t0
    print(f"  120 s simulation in {elapsed*1000:.1f} ms")
    assert_true(elapsed < 5.0, "completes in < 5 s")


if __name__ == "__main__":
    tests = [
        test_rate_arrhenius_increases_with_T,
        test_mass_conservation,
        test_yields_in_unit_interval,
        test_conversion_monotone,
        test_biomass_decays,
        test_endothermic_energy_balance,
        test_fast_pyrolysis_maximizes_bio_oil,
        test_high_T_favors_gas_low_T_favors_char,
        test_equilibrium_mass_closes,
        test_predict_interface,
        test_energy_in_products_positive,
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
    print(f"EC145 Pyrolysis Reactor F2a -- Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    sys.exit(0 if failed == 0 else 1)
