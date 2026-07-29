"""
EC197 -- DME Synthesis Reactor -- F2a Kinetics + Equilibrium
Test suite: atom/mass conservation, equilibrium limits, exothermic balance,
DME yield vs T/P, edge cases, predict() interface, benchmark timing.
NO pytest -- run as:  python3 scripts/test_model.py
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from model import DMEReactorF2a, IDX, SPECIES
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


def _atoms(n):
    """Return (C, H, O) atom counts for molar vector n (CO,H2,CO2,H2O,MeOH,DME)."""
    C = n[IDX["CO"]] + n[IDX["CO2"]] + n[IDX["CH3OH"]] + 2 * n[IDX["DME"]]
    H = 2 * n[IDX["H2"]] + 2 * n[IDX["H2O"]] + 4 * n[IDX["CH3OH"]] + 6 * n[IDX["DME"]]
    O = n[IDX["CO"]] + 2 * n[IDX["CO2"]] + n[IDX["H2O"]] + n[IDX["CH3OH"]] + n[IDX["DME"]]
    return C, H, O


# ---------------------------------------------------------------------------
def test_atom_conservation():
    print("\n[Test 1] Atom conservation (C, H, O) through reactions")
    m, _ = make_model()
    r = m.simulate(523.15, 40.0, tau_max=4.0, n_eval=80)
    n0 = r["n0"]
    n_end = np.array([r["n_CO"][-1], r["n_H2"][-1], r["n_CO2"][-1],
                      r["n_H2O"][-1], r["n_CH3OH"][-1], r["n_DME"][-1]])
    C0, H0, O0 = _atoms(n0)
    C1, H1, O1 = _atoms(n_end)
    assert_true(abs(C1 - C0) < 1e-6 * max(C0, 1), f"Carbon conserved: dC={C1-C0:.2e}")
    assert_true(abs(H1 - H0) < 1e-6 * max(H0, 1), f"Hydrogen conserved: dH={H1-H0:.2e}")
    assert_true(abs(O1 - O0) < 1e-6 * max(O0, 1), f"Oxygen conserved: dO={O1-O0:.2e}")


def test_equilibrium_constants():
    print("\n[Test 2] Equilibrium constants in physical range / correct trend")
    m, _ = make_model()
    # Methanol dehydration Keq3 ~ 16 (200C) -> 7.5 (300C), decreasing (mildly exothermic)
    K3_lo = m.Keq3(473.15)
    K3_hi = m.Keq3(573.15)
    assert_true(5.0 < K3_hi < K3_lo < 25.0,
                f"Keq3 decreasing 16->7.5: {K3_lo:.2f} (200C) > {K3_hi:.2f} (300C)")
    # MeOH synthesis Keq1 decreases with T (exothermic)
    assert_true(m.Keq1(473.15) > m.Keq1(573.15), "Keq1 decreases with T (exothermic synthesis)")


def test_conversion_bounds():
    print("\n[Test 3] CO conversion and DME selectivity in [0, 1]")
    m, _ = make_model()
    r = m.simulate(523.15, 40.0, tau_max=4.0, n_eval=60)
    assert_true(np.all(r["CO_conversion"] >= -1e-9) and np.all(r["CO_conversion"] <= 1.0 + 1e-9),
                f"X_CO in [0,1], max={r['CO_conversion'].max():.3f}")
    assert_true(np.all(r["DME_selectivity"] >= -1e-9) and np.all(r["DME_selectivity"] <= 1.0 + 1e-9),
                f"DME selectivity in [0,1], max={r['DME_selectivity'].max():.3f}")
    assert_true(np.all(r["DME_yield"] >= -1e-9) and np.all(r["DME_yield"] <= 1.0 + 1e-9),
                "DME yield in [0,1]")


def test_conversion_monotone_in_tau():
    print("\n[Test 4] CO conversion increases monotonically with residence time")
    m, _ = make_model()
    r = m.simulate(523.15, 40.0, tau_max=4.0, n_eval=80)
    X = r["CO_conversion"]
    diffs = np.diff(X)
    assert_true(np.all(diffs >= -1e-6), "X_CO non-decreasing along reactor")
    assert_true(X[-1] > X[0], f"Net conversion grows: {X[0]:.3f} -> {X[-1]:.3f}")


def test_equilibrium_plateau():
    print("\n[Test 5] Conversion plateaus toward equilibrium at long tau")
    m, _ = make_model()
    r = m.simulate(523.15, 40.0, tau_max=8.0, n_eval=120)
    X = r["CO_conversion"]
    early = X[len(X) // 2] - X[len(X) // 4]      # mid-reactor increment
    late = X[-1] - X[-1 - len(X) // 4]            # end increment over same span
    assert_true(late < early + 1e-9,
                f"Rate of conversion slows (equilibrium-limited): late={late:.4f} < early={early:.4f}")


def test_exothermic_heating():
    print("\n[Test 6] Adiabatic operation heats up (exothermic balance)")
    m, _ = make_model()
    r_ad = m.simulate(523.15, 40.0, tau_max=2.0, n_eval=60, adiabatic=True)
    assert_true(r_ad["T"][-1] > r_ad["T"][0] + 10.0,
                f"Adiabatic T rises: {r_ad['T'][0]:.1f} -> {r_ad['T'][-1]:.1f} K")
    assert_true(np.all(r_ad["heat_release_kW"] >= -1e-6),
                "Heat release non-negative (net exothermic)")


def test_cooled_vs_adiabatic():
    print("\n[Test 7] Cooled reactor stays cooler than adiabatic")
    m, _ = make_model()
    r_cool = m.simulate(523.15, 40.0, tau_max=2.0, n_eval=60, adiabatic=False)
    r_ad = m.simulate(523.15, 40.0, tau_max=2.0, n_eval=60, adiabatic=True)
    assert_true(r_cool["T"][-1] < r_ad["T"][-1],
                f"Cooled T_exit={r_cool['T'][-1]:.1f} < adiabatic {r_ad['T'][-1]:.1f} K")


def test_pressure_effect():
    print("\n[Test 8] Higher pressure raises CO conversion (Le Chatelier)")
    m, _ = make_model()
    r_lo = m.simulate(523.15, 20.0, tau_max=2.0, n_eval=40)
    r_hi = m.simulate(523.15, 60.0, tau_max=2.0, n_eval=40)
    assert_true(r_hi["CO_conversion"][-1] > r_lo["CO_conversion"][-1],
                f"X_CO(60bar)={r_hi['CO_conversion'][-1]:.3f} > X_CO(20bar)={r_lo['CO_conversion'][-1]:.3f}")


def test_dme_requires_methanol():
    print("\n[Test 9] DME forms only after methanol is produced")
    m, _ = make_model()
    r = m.simulate(523.15, 40.0, tau_max=2.0, n_eval=80)
    # At inlet (tau=0) no MeOH and no DME
    assert_true(r["n_DME"][0] < 1e-9 and r["n_CH3OH"][0] < 1e-9,
                "No DME/MeOH at reactor inlet")
    # DME appears downstream
    assert_true(r["n_DME"][-1] > 0.0, f"DME produced: {r['n_DME'][-1]:.4f} mol/s")
    # DME requires intermediate methanol -> some MeOH must have existed
    assert_true(np.max(r["n_CH3OH"]) > 0.0, "Methanol intermediate present")


def test_no_reaction_at_low_T():
    print("\n[Test 10] Kinetics frozen at low temperature (Arrhenius)")
    m, _ = make_model()
    r_cold = m.simulate(420.0, 40.0, tau_max=2.0, n_eval=40, adiabatic=True)
    r_hot = m.simulate(540.0, 40.0, tau_max=2.0, n_eval=40, adiabatic=True)
    assert_true(r_cold["CO_conversion"][-1] < r_hot["CO_conversion"][-1],
                f"Cold X_CO={r_cold['CO_conversion'][-1]:.4f} < hot {r_hot['CO_conversion'][-1]:.4f}")


def test_predict_interface():
    print("\n[Test 11] ComponentModel predict() interface + exit summary")
    _, cm = make_model()
    r = cm.predict({"T_in_K": 523.15, "P_bar": 40.0, "tau_max": 2.0})
    for key in ["tau", "T", "CO_conversion", "methanol_conversion",
                "DME_yield", "DME_selectivity", "heat_release_kW", "n_DME", "exit"]:
        assert_true(key in r, f"Key '{key}' in output")
    assert_true(len(r["tau"]) == len(r["CO_conversion"]), "Arrays same length")
    e = r["exit"]
    assert_true(0 <= e["CO_conversion"] <= 1 and 0 <= e["DME_yield"] <= 1,
                "Exit summary scalars in bounds")
    info = cm.get_info()
    assert_true(info["component_id"] == "EC197" and info["version"] == "1.0.0",
                "get_info() metadata correct")


def test_benchmark():
    print("\n[Test 12] Benchmark: reactor integration timing")
    m, _ = make_model()
    t0 = time.perf_counter()
    m.simulate(523.15, 40.0, tau_max=4.0, n_eval=120)
    elapsed = time.perf_counter() - t0
    print(f"  tau_max=4 integration in {elapsed*1000:.1f} ms")
    assert_true(elapsed < 5.0, "Completes in < 5 s")


if __name__ == "__main__":
    tests = [
        test_atom_conservation,
        test_equilibrium_constants,
        test_conversion_bounds,
        test_conversion_monotone_in_tau,
        test_equilibrium_plateau,
        test_exothermic_heating,
        test_cooled_vs_adiabatic,
        test_pressure_effect,
        test_dme_requires_methanol,
        test_no_reaction_at_low_T,
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

    print(f"\n{'='*64}")
    print(f"EC197 DME Synthesis Reactor F2a -- Results: {passed} passed, {failed} failed")
    print(f"{'='*64}")
    sys.exit(0 if failed == 0 else 1)
