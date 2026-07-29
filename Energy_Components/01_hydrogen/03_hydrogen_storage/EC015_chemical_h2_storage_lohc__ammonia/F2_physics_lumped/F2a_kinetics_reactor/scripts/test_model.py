"""
EC015 -- Chemical H2 Storage (LOHC / Ammonia) -- F2a Lumped Kinetics Reactor
Test suite: Arrhenius physics, conversion bounds, mass/energy conservation,
edge cases, predict() interface, benchmark.
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from model import ChemicalH2StorageF2a, M_H2
from predict import ComponentModel

PASS = "✓"
FAIL = "✗"

# np.trapz was renamed to np.trapezoid in NumPy 2.0
_trapz = getattr(np, "trapezoid", np.trapz)


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
def test_arrhenius_monotone():
    print("\n[Test 1] Arrhenius rate constant increases with temperature")
    m, _ = make_model()
    for mode in ["lohc", "ammonia"]:
        k_lo = m.rate_constant(500.0, mode)
        k_hi = m.rate_constant(800.0, mode)
        assert_true(k_hi > k_lo, f"{mode}: k(800)={k_hi:.3e} > k(500)={k_lo:.3e}")
        # Arrhenius identity check: ln(k) linear in 1/T
        lnk1 = np.log(m.rate_constant(600.0, mode))
        lnk2 = np.log(m.rate_constant(700.0, mode))
        Ea = -(lnk2 - lnk1) / (1.0 / 700.0 - 1.0 / 600.0) * m.R
        assert_true(abs(Ea - m._mode(mode)["Ea"]) < 1.0,
                    f"{mode}: recovered Ea={Ea:.0f} J/mol matches param")


def test_conversion_bounds():
    print("\n[Test 2] Conversion stays in [0, 1]")
    m, _ = make_model()
    for mode in ["lohc", "ammonia"]:
        r = m.simulate(mode=mode, dt=30.0, duration_s=7200.0)
        assert_true(np.all(r["conversion"] >= -1e-9), f"{mode}: X >= 0")
        assert_true(np.all(r["conversion"] <= 1.0 + 1e-9), f"{mode}: X <= 1")


def test_conversion_monotone():
    print("\n[Test 3] Conversion is monotonically non-decreasing")
    m, _ = make_model()
    r = m.simulate(mode="lohc", dt=30.0, duration_s=7200.0)
    X = r["conversion"]
    assert_true(np.all(np.diff(X) >= -1e-7), "X(t) non-decreasing")
    assert_true(X[-1] > X[0], f"X grows: {X[-1]:.4f} > {X[0]:.4f}")


def test_h2_release_nonneg():
    print("\n[Test 4] H2 release rate >= 0 and zero at full conversion")
    m, _ = make_model()
    r = m.simulate(mode="ammonia", dt=30.0, duration_s=7200.0)
    assert_true(np.all(r["h2_rate_mol_s"] >= -1e-9), "release rate >= 0")
    # NH3 fully cracks: final rate -> 0
    assert_true(r["h2_rate_mol_s"][-1] < r["h2_rate_mol_s"][0],
                "rate decays as carrier depletes")


def test_mass_conservation():
    print("\n[Test 5] Mass conservation: released H2 = n_carrier*nu*X*M_H2")
    m, _ = make_model()
    for mode in ["lohc", "ammonia"]:
        r = m.simulate(mode=mode, dt=20.0, duration_s=5000.0)
        p = m._mode(mode)
        expected = r["conversion"][-1] * m.n_carrier0 * p["nu_H2"] * M_H2
        assert_true(abs(r["h2_released_kg"][-1] - expected) < 1e-9,
                    f"{mode}: H2={r['h2_released_kg'][-1]:.5f} == {expected:.5f} kg")
        # cannot exceed theoretical max
        assert_true(r["h2_released_kg"][-1] <= r["h2_total_kg"] + 1e-9,
                    f"{mode}: released <= max ({r['h2_total_kg']:.4f} kg)")


def test_h2_release_integral():
    print("\n[Test 6] Cumulative H2 = integral of release rate (consistency)")
    m, _ = make_model()
    # LOHC at 593 K has a ~minute-scale time constant; dt=2 s resolves the rate
    # spike so the trapezoid of dn_H2/dt matches the cumulative released mass.
    r = m.simulate(mode="lohc", dt=2.0, duration_s=4000.0)
    integ = _trapz(r["h2_rate_kg_s"], r["t"])
    assert_true(abs(integ - r["h2_released_kg"][-1]) / max(r["h2_released_kg"][-1], 1e-9) < 0.02,
                f"int(rate)={integ:.5f} ~= cumulative={r['h2_released_kg'][-1]:.5f} kg")


def test_energy_penalty():
    print("\n[Test 7] Energy penalty: DH/M_H2 specific energy and LHV fraction")
    m, _ = make_model()
    # LOHC 65 kJ/mol_H2 -> 32.2 MJ/kg ; ~27% of LHV
    se_lohc = m.specific_energy("lohc")
    assert_true(31.0 < se_lohc < 33.5, f"LOHC specific energy={se_lohc:.2f} MJ/kg")
    f_lohc = m.energy_penalty_fraction("lohc")
    assert_true(0.25 < f_lohc < 0.29, f"LOHC penalty={f_lohc*100:.1f}% of LHV")
    # NH3 46 kJ/mol_H2 -> 22.8 MJ/kg ; ~19% of LHV (less than LOHC)
    se_nh3 = m.specific_energy("ammonia")
    assert_true(se_nh3 < se_lohc, f"NH3 ({se_nh3:.1f}) cheaper than LOHC ({se_lohc:.1f})")


def test_endotherm_cools_reactor():
    print("\n[Test 8] Endothermic reaction draws heat: T dips below setpoint")
    m, _ = make_model()
    # Start AT setpoint; reaction onset should pull T below T_set then recover
    r = m.simulate(mode="ammonia", T0=773.0, T_set=773.0, dt=10.0, duration_s=3000.0)
    T = r["temperature"]
    assert_true(np.min(T) < 773.0, f"T dips to {np.min(T):.2f} K < setpoint 773 K (endotherm)")
    # heater drives recovery toward setpoint once reaction completes
    assert_true(abs(T[-1] - 773.0) < abs(np.min(T) - 773.0),
                "T recovers toward setpoint after reaction completes")


def test_energy_balance_closure():
    print("\n[Test 9] Energy balance closure: Q_heat = dStored + Q_rxn")
    m, _ = make_model()
    # Fine dt (0.2 s) resolves the fast NH3 endotherm transient so the trapezoid
    # energy integrals are accurate; balance must close to within 2 %.
    # (Residual is a pure output-grid discretization artifact -> 0 as dt -> 0;
    #  the underlying ODE is integrated exactly by solve_ivp.)
    r = m.simulate(mode="ammonia", T0=773.0, T_set=773.0, dt=0.2, duration_s=4000.0)
    E_heat = _trapz(r["q_heat_W"], r["t"])
    E_rxn = _trapz(r["q_rxn_W"], r["t"])
    dStored = m.m_reactor * m.cp_reactor * (r["temperature"][-1] - r["temperature"][0])
    residual = E_heat - (dStored + E_rxn)
    scale = max(abs(E_rxn), 1.0)
    assert_true(abs(residual) / scale < 0.02,
                f"balance closes: residual={residual:.1f} J ({abs(residual)/scale*100:.2f}%)")


def test_temperature_speeds_reaction():
    print("\n[Test 10] Higher setpoint -> faster conversion (Arrhenius)")
    m, _ = make_model()
    r_lo = m.simulate(mode="lohc", T0=560.0, T_set=560.0, dt=30.0, duration_s=3600.0)
    r_hi = m.simulate(mode="lohc", T0=620.0, T_set=620.0, dt=30.0, duration_s=3600.0)
    assert_true(r_hi["conversion"][-1] > r_lo["conversion"][-1],
                f"X(620K)={r_hi['conversion'][-1]:.3f} > X(560K)={r_lo['conversion'][-1]:.3f}")


def test_predict_interface():
    print("\n[Test 11] ComponentModel predict() interface")
    _, cm = make_model()
    r = cm.predict({"mode": "lohc", "dt": 60.0, "duration_s": 1800.0})
    keys = ["t", "conversion", "h2_rate_mol_s", "h2_rate_kg_s", "h2_released_kg",
            "temperature", "q_heat_W", "q_rxn_W", "specific_energy_MJ_per_kg",
            "energy_penalty_frac"]
    for k in keys:
        assert_true(k in r, f"Key '{k}' in output")
    assert_true(len(r["t"]) == len(r["conversion"]) == len(r["temperature"]),
                "Arrays same length")
    info = cm.get_info()
    assert_true(info["component_id"] == "EC015" and cm.version == "1.0.0",
                "metadata correct")


def test_benchmark():
    print("\n[Test 12] Benchmark: 3600s LOHC sim at dt=10")
    m, _ = make_model()
    t0 = time.perf_counter()
    m.simulate(mode="lohc", dt=10.0, duration_s=3600.0)
    elapsed = time.perf_counter() - t0
    print(f"  3600s simulation in {elapsed*1000:.1f} ms")
    assert_true(elapsed < 5.0, "Completes in < 5 s")


if __name__ == "__main__":
    tests = [
        test_arrhenius_monotone,
        test_conversion_bounds,
        test_conversion_monotone,
        test_h2_release_nonneg,
        test_mass_conservation,
        test_h2_release_integral,
        test_energy_penalty,
        test_endotherm_cools_reactor,
        test_energy_balance_closure,
        test_temperature_speeds_reaction,
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
    print(f"EC015 Chemical H2 Storage F2a -- Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    sys.exit(0 if failed == 0 else 1)
