"""
EC194 -- Methanol Synthesis Reactor -- F2a Kinetics + Equilibrium
Test suite: physics sanity (mass/atom conservation, equilibrium limits,
exothermic heat balance, T/P yield trends), edge cases, predict() interface,
and a benchmark timing test.
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from model import MethanolReactor_F2a
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
def test_equilibrium_decreases_with_T():
    print("\n[Test 1] Equilibrium conversion decreases with T (exothermic, Le Chatelier)")
    m, _ = make_model()
    Ts = [480.0, 500.0, 520.0, 540.0, 560.0]
    Xeq = [m.equilibrium_conversion(T) for T in Ts]
    for i in range(1, len(Xeq)):
        assert_true(Xeq[i] < Xeq[i - 1],
                    f"Xeq(T={Ts[i]:.0f})={Xeq[i]:.3f} < Xeq(T={Ts[i-1]:.0f})={Xeq[i-1]:.3f}")


def test_equilibrium_increases_with_P():
    print("\n[Test 2] Equilibrium conversion increases with P (mole-reducing reaction)")
    m, _ = make_model()
    Ps = [30.0, 50.0, 80.0, 100.0]
    Xeq = [m.equilibrium_conversion(520.0, P) for P in Ps]
    for i in range(1, len(Xeq)):
        assert_true(Xeq[i] > Xeq[i - 1],
                    f"Xeq(P={Ps[i]:.0f})={Xeq[i]:.3f} > Xeq(P={Ps[i-1]:.0f})={Xeq[i-1]:.3f}")
    assert_true(all(0.0 < x < 1.0 for x in Xeq), "All Xeq in (0,1)")


def test_kinetic_below_equilibrium():
    print("\n[Test 3] Per-pass kinetic conversion never exceeds equilibrium limit")
    m, _ = make_model()
    for T in [500.0, 520.0, 540.0]:
        r = m.simulate(T0=T, T_in=T, T_cool=T, duration_s=600.0, dt=20.0)
        Xeq = m.equilibrium_conversion(T)
        Xk = r["X_C"][-1]
        assert_true(Xk <= Xeq + 1e-3,
                    f"T={T:.0f}: X_kin={Xk:.3f} <= X_eq={Xeq:.3f}")


def test_carbon_atom_conservation():
    print("\n[Test 4] Carbon atom conservation: CO+CO2+CH3OH carbon is conserved")
    m, _ = make_model()
    r = m.simulate(T0=520.0, T_in=520.0, T_cool=520.0, duration_s=400.0, dt=10.0)
    C_in = r["C_CO_in"] + r["C_CO2_in"]
    # carbon-bearing species in the gas: CO, CO2, CH3OH (each 1 C atom)
    C_out = r["C_CO"][-1] + r["C_CO2"][-1] + r["C_CH3OH"][-1]
    rel = abs(C_out - C_in) / C_in
    assert_true(rel < 1e-3, f"Carbon balance closes: in={C_in:.3f}, out={C_out:.3f}, rel_err={rel:.2e}")


def test_oxygen_hydrogen_consistency():
    print("\n[Test 5] H2O produced equals carbon converted (atom balance on the two reactions)")
    # R1 makes 1 H2O per MeOH; R2 makes 1 H2O per CO produced (from CO2).
    # Net: every CO2 consumed produces exactly 1 H2O. So C_H2O == (CO2_in - CO2) net.
    m, _ = make_model()
    r = m.simulate(T0=520.0, T_in=520.0, T_cool=520.0, duration_s=400.0, dt=10.0)
    co2_consumed = r["C_CO2_in"] - r["C_CO2"][-1]
    # CO net change: produced by RWGS. CO2 -> CO via R2; CO2 -> MeOH via R1.
    # H2O = R1_extent + R2_extent = MeOH_made + CO_made(from CO2).
    h2o = r["C_H2O"][-1]
    meoh = r["C_CH3OH"][-1]
    co_made = r["C_CO"][-1] - r["C_CO_in"]
    expected_h2o = meoh + co_made
    rel = abs(h2o - expected_h2o) / (h2o + 1e-9)
    assert_true(rel < 1e-2,
                f"H2O={h2o:.3f} == MeOH+CO_made={expected_h2o:.3f} (rel={rel:.2e})")


def test_exothermic_self_heating():
    print("\n[Test 6] Exothermic: reactor heats above feed inlet under modest cooling")
    m, _ = make_model()
    r = m.simulate(T0=503.15, T_in=503.15, T_cool=503.15, duration_s=600.0, dt=10.0)
    assert_true(r["T"][-1] > 503.15,
                f"T_final={r['T'][-1]:.1f} K > T_in=503.15 K (net exotherm)")
    assert_true(not r["thermal_runaway"], "No thermal runaway under design cooling")


def test_cooling_lowers_temperature():
    print("\n[Test 7] Colder coolant lowers steady-state reactor temperature")
    m, _ = make_model()
    r_cold = m.simulate(T0=503.15, T_in=503.15, T_cool=493.15, duration_s=600.0, dt=10.0)
    r_hot = m.simulate(T0=503.15, T_in=503.15, T_cool=523.15, duration_s=600.0, dt=10.0)
    assert_true(r_cold["T"][-1] < r_hot["T"][-1],
                f"T(cool=493)={r_cold['T'][-1]:.1f} < T(cool=523)={r_hot['T'][-1]:.1f}")


def test_pressure_raises_yield():
    print("\n[Test 8] Higher pressure raises methanol output (high P favours methanol)")
    # Per-pass *fractional* conversion at fixed volumetric GHSV is not a clean
    # function of P (feed molar throughput scales with P), but the absolute
    # methanol production (outlet concentration / partial pressure) must rise
    # with pressure for this mole-reducing, equilibrium-limited reaction.
    m, _ = make_model()
    P_list = [30.0, 50.0, 80.0, 100.0]
    C_meoh = []
    for P in P_list:
        r = m.simulate(T0=515.0, T_in=515.0, T_cool=515.0, P=P,
                       duration_s=600.0, dt=20.0)
        C_meoh.append(r["C_CH3OH"][-1])
    for i in range(1, len(C_meoh)):
        assert_true(C_meoh[i] > C_meoh[i - 1],
                    f"C_MeOH(P={P_list[i]:.0f})={C_meoh[i]:.2f} > "
                    f"C_MeOH(P={P_list[i-1]:.0f})={C_meoh[i-1]:.2f} mol/m3")


def test_conversion_yield_bounds():
    print("\n[Test 9] Conversion, yield, mole fractions stay in physical bounds")
    m, _ = make_model()
    r = m.simulate(duration_s=600.0, dt=10.0)
    assert_true(np.all((r["X_C"] >= 0.0) & (r["X_C"] <= 1.0)), "X_C in [0,1]")
    assert_true(np.all((r["meoh_yield"] >= 0.0) & (r["meoh_yield"] <= 1.0)), "meoh_yield in [0,1]")
    assert_true(np.all((r["y_MeOH_dry"] >= 0.0) & (r["y_MeOH_dry"] <= 1.0)), "y_MeOH_dry in [0,1]")
    assert_true(np.all(r["C_CH3OH"] >= -1e-9), "MeOH concentration non-negative")


def test_conversion_rises_then_saturates():
    print("\n[Test 10] MeOH conversion is monotone non-decreasing toward steady state")
    m, _ = make_model()
    r = m.simulate(T0=520.0, T_in=520.0, T_cool=520.0, duration_s=600.0, dt=10.0)
    X = r["X_C"]
    # allow tiny numerical noise
    diffs = np.diff(X)
    assert_true(np.all(diffs >= -1e-4), "X_C non-decreasing (startup transient)")
    dX_end = abs(X[-1] - X[-2])
    assert_true(dX_end < 1e-2, f"Approaches steady state: dX={dX_end:.2e}")


def test_predict_interface():
    print("\n[Test 11] ComponentModel predict() interface + get_info()")
    _, cm = make_model()
    r = cm.predict({"duration_s": 200.0, "dt": 10.0})
    for key in ["t", "T", "X_C", "meoh_yield", "y_MeOH_dry", "C_CH3OH",
                "T_max", "thermal_runaway", "X_eq_final"]:
        assert_true(key in r, f"Key '{key}' in predict output")
    assert_true(len(r["t"]) == len(r["T"]) == len(r["X_C"]), "Time arrays equal length")
    info = cm.get_info()
    assert_true(info["component_id"] == "EC194", "get_info component_id == EC194")
    assert_true("Vanden Bussche" in info["source"] or "Graaf" in info["source"],
                "Source cites Graaf / Vanden Bussche")


def test_benchmark():
    print("\n[Test 12] Benchmark: 600 s simulation at dt=1 s")
    m, _ = make_model()
    t0 = time.perf_counter()
    m.simulate(duration_s=600.0, dt=1.0)
    elapsed = time.perf_counter() - t0
    print(f"  600s simulation in {elapsed*1000:.1f} ms")
    assert_true(elapsed < 5.0, "Completes in < 5 s")


if __name__ == "__main__":
    tests = [
        test_equilibrium_decreases_with_T,
        test_equilibrium_increases_with_P,
        test_kinetic_below_equilibrium,
        test_carbon_atom_conservation,
        test_oxygen_hydrogen_consistency,
        test_exothermic_self_heating,
        test_cooling_lowers_temperature,
        test_pressure_raises_yield,
        test_conversion_yield_bounds,
        test_conversion_rises_then_saturates,
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

    print(f"\n{'=' * 60}")
    print(f"EC194 Methanol Synthesis F2a -- Results: {passed} passed, {failed} failed")
    print(f"{'=' * 60}")
    sys.exit(0 if failed == 0 else 1)
