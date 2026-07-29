"""
EC014 -- Metal Hydride H2 Storage -- F2a Kinetics + Thermal
Test suite: physics sanity (mass/energy conservation, plateau, monotonicity),
edge cases, predict() interface, benchmark timing.
"""

import sys
import os
import time
import numpy as np
from scipy.integrate import trapezoid

sys.path.insert(0, os.path.dirname(__file__))
from model import MetalHydrideF2a, R_UNIVERSAL, M_H2
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
def test_plateau_vant_hoff():
    print("\n[Test 1] van't Hoff plateau pressure rises with T")
    m, _ = make_model()
    P1 = m.plateau_pressure(293.15, "desorption")
    P2 = m.plateau_pressure(333.15, "desorption")
    assert_true(P2 > P1, f"P_eq(333K)={P2:.3f} > P_eq(293K)={P1:.3f} bar")
    # LaNi5 plateau ~1.7-2 bar at 25C (Sandrock 1999)
    assert_true(1.0 < P1 < 4.0, f"P_eq(293K)={P1:.3f} bar near LaNi5 ~1.7 bar")


def test_hysteresis():
    print("\n[Test 2] Absorption plateau above desorption plateau")
    m, _ = make_model()
    Pa = m.plateau_pressure(303.15, "absorption")
    Pd = m.plateau_pressure(303.15, "desorption")
    assert_true(Pa > Pd, f"P_abs={Pa:.3f} > P_des={Pd:.3f} bar (hysteresis)")
    assert_true(abs(Pa / Pd - m.hysteresis_factor) < 1e-6, "ratio == hysteresis_factor")


def test_absorption_charges():
    print("\n[Test 3] High supply pressure charges the bed (X increases)")
    m, _ = make_model()
    r = m.simulate(15.0, 293.15, X0=0.0, dt=10.0, duration_s=1200.0)
    assert_true(r["HM_ratio"][-1] > r["HM_ratio"][0], "X increased")
    assert_true(r["HM_ratio"][-1] > 1.0, f"X_final={r['HM_ratio'][-1]:.2f} substantially charged")
    assert_true(np.all(np.diff(r["HM_ratio"]) >= -1e-9), "X monotonically non-decreasing")


def test_desorption_discharges():
    print("\n[Test 4] Low supply pressure discharges the bed (X decreases)")
    m, _ = make_model()
    r = m.simulate(0.2, 313.15, X0=5.0, dt=10.0, duration_s=1200.0)
    assert_true(r["HM_ratio"][-1] < r["HM_ratio"][0], "X decreased")
    assert_true(np.all(np.diff(r["HM_ratio"]) <= 1e-9), "X monotonically non-increasing")


def test_mass_bounds():
    print("\n[Test 5] Loading X stays in [0, X_max]")
    m, _ = make_model()
    r = m.simulate(40.0, 293.15, X0=0.0, dt=10.0, duration_s=3600.0)
    assert_true(np.all(r["HM_ratio"] <= m.X_max + 1e-6), f"X <= X_max={m.X_max}")
    assert_true(np.all(r["HM_ratio"] >= -1e-9), "X >= 0")
    assert_true(np.all((r["soc"] >= 0.0) & (r["soc"] <= 1.0)), "SOC in [0,1]")


def test_exothermic_heating():
    print("\n[Test 6] Absorption is exothermic -- bed heats above coolant")
    m, _ = make_model()
    r = m.simulate(20.0, 293.15, X0=0.0, dt=5.0, duration_s=600.0)
    assert_true(np.max(r["temperature"]) > 293.15, "bed temperature rose during absorption")
    assert_true(np.max(r["Q_rxn"]) > 0.0, "Q_rxn positive (heat released) during absorption")


def test_energy_conservation():
    print("\n[Test 7] Energy balance: integral of reaction heat == |dH|*mol_H2")
    m, _ = make_model()
    r = m.simulate(30.0, 293.15, X0=0.0, dt=2.0, duration_s=3600.0)
    # Integrate Q_rxn over time -> total reaction heat released [J]
    Q_total = trapezoid(r["Q_rxn"], r["t"])
    # Moles H2 absorbed from net loading change
    dX = r["HM_ratio"][-1] - r["HM_ratio"][0]
    n_H2 = m.n_formula * 0.5 * dX
    Q_expected = abs(m.delta_H_abs) * n_H2
    rel_err = abs(Q_total - Q_expected) / Q_expected
    print(f"  Q_int={Q_total:.1f} J, Q_expected={Q_expected:.1f} J, rel_err={rel_err*100:.3f}%")
    assert_true(rel_err < 0.01, f"energy conserved within 1% (err={rel_err*100:.3f}%)")


def test_mass_conservation():
    print("\n[Test 8] Stored H2 mass consistent with H/M ratio")
    m, _ = make_model()
    full = m.stored_mass_kg(m.X_max)
    assert_true(abs(full - m.m_H2_max) < 1e-12, "stored_mass(X_max) == m_H2_max")
    # gravimetric wt% at full should match nameplate within reason
    wt = m.gravimetric_wt_pct(m.X_max)
    assert_true(1.0 < wt < 1.7, f"full wt%={wt:.3f} near LaNi5 ~1.4 wt%")


def test_equilibrium_stall():
    print("\n[Test 9] On-plateau supply pressure -> no net reaction")
    m, _ = make_model()
    T = 303.15
    P_mid = 0.5 * (m.plateau_pressure(T, "absorption") + m.plateau_pressure(T, "desorption"))
    rate = m.reaction_rate(2.0, T, P_mid)
    assert_true(abs(rate) < 1e-12, f"dX/dt={rate:.2e} ~ 0 on plateau (equilibrium)")


def test_arrhenius_temperature():
    print("\n[Test 10] Absorption rate constant increases with T (Arrhenius)")
    m, _ = make_model()
    # Compare kinetic prefactor exp(-Ea/RT) at two temps (fixed driving force)
    k_lo = np.exp(-m.Ea_abs / (R_UNIVERSAL * 280.0))
    k_hi = np.exp(-m.Ea_abs / (R_UNIVERSAL * 340.0))
    assert_true(k_hi > k_lo, f"Arrhenius: k(340K)={k_hi:.3e} > k(280K)={k_lo:.3e}")


def test_predict_interface():
    print("\n[Test 11] ComponentModel predict() interface")
    _, cm = make_model()
    r = cm.predict({"P_supply_bar": 12.0, "T_bed_K": 293.15, "dt": 10.0, "duration_s": 300.0})
    for key in ["t", "HM_ratio", "soc", "stored_mass_kg", "temperature",
                "P_supply", "P_eq_abs", "P_eq_des", "Q_rxn", "Q_cool",
                "gravimetric_wt_pct"]:
        assert_true(key in r, f"Key '{key}' in output")
    assert_true(len(r["t"]) == len(r["HM_ratio"]), "Arrays same length")
    info = cm.get_info()
    assert_true(info["component_id"] == "EC014", "get_info id == EC014")


def test_benchmark():
    print("\n[Test 12] Benchmark: 3600s sim at dt=2")
    m, _ = make_model()
    t0 = time.perf_counter()
    m.simulate(20.0, 293.15, X0=0.0, dt=2.0, duration_s=3600.0)
    elapsed = time.perf_counter() - t0
    print(f"  3600s simulation in {elapsed*1000:.1f} ms")
    assert_true(elapsed < 5.0, "Completes in < 5 s")


if __name__ == "__main__":
    tests = [
        test_plateau_vant_hoff,
        test_hysteresis,
        test_absorption_charges,
        test_desorption_discharges,
        test_mass_bounds,
        test_exothermic_heating,
        test_energy_conservation,
        test_mass_conservation,
        test_equilibrium_stall,
        test_arrhenius_temperature,
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
    print(f"EC014 Metal Hydride F2a -- Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    sys.exit(0 if failed == 0 else 1)
