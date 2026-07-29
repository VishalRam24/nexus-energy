"""
EC089 -- Hydrogen Boiler -- F2a Physics-Lumped
Test suite: combustion/mass conservation, energy balance, efficiency bounds,
thermal-ODE behaviour, edge cases, predict() interface, benchmark timing.
Run:  python3 scripts/test_model.py   (system python3, NOT pytest)
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from model import HydrogenBoilerF2a
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
def test_mass_conservation():
    print("\n[Test 1] Mass conservation: fuel + air = flue out")
    m, _ = make_model()
    for phi in [0.25, 0.5, 1.0]:
        m_in, m_out = m.check_mass_balance(phi)
        assert_true(abs(m_in - m_out) < 1e-12 * max(m_in, 1e-12),
                    f"phi={phi}: m_in={m_in:.3e} == m_out={m_out:.3e} kg/s")


def test_stoichiometry():
    print("\n[Test 2] Stoichiometry: 9 kg H2O per kg H2, air = lambda*AFR*m_H2")
    m, _ = make_model()
    m_h2 = m.h2_mass_flow(1.0)
    m_air = m.air_mass_flow(1.0)
    assert_true(abs(m_air - m.lam * m.AFR_s * m_h2) < 1e-15, "air = lambda*AFR_s*m_H2")
    m_flue = m.flue_mass_flow(1.0)
    assert_true(abs(m_flue - (m_h2 + m_air)) < 1e-15,
                "flue = m_H2 + air (mass in = mass out)")
    # Product water mass uses the 9:1 ratio (O2 drawn from the air stream).
    assert_true(abs(m.water_per_fuel * m_h2 - 9.0 * m_h2) < 1e-15,
                "product water = 9 kg per kg H2")


def test_energy_conservation():
    print("\n[Test 3] HHV energy balance closes: residual ~ 0")
    m, _ = make_model()
    for phi in [0.2, 0.5, 0.8, 1.0]:
        res = m.energy_balance_residual(phi)
        Q_hhv = m.fuel_power_HHV(phi)
        assert_true(abs(res) < 1e-6 * Q_hhv,
                    f"phi={phi}: |residual|={abs(res):.3e} W << Q_hhv={Q_hhv:.1f} W")


def test_efficiency_bounds_hhv():
    print("\n[Test 4] HHV efficiency strictly in (0, 1]")
    m, _ = make_model()
    for phi in [0.1, 0.3, 0.6, 1.0]:
        eta = m.efficiency_hhv(phi)
        assert_true(0.0 < eta <= 1.0, f"phi={phi}: eta_HHV={eta:.4f} in (0,1]")


def test_condensing_boosts_efficiency():
    print("\n[Test 5] Condensing mode > non-condensing; LHV-eta can exceed HHV-eta")
    m, _ = make_model()
    eta_cond = m.efficiency_hhv(1.0)
    m.condensing = False
    eta_noncond = m.efficiency_hhv(1.0)
    assert_true(eta_cond > eta_noncond,
                f"eta_cond={eta_cond:.4f} > eta_noncond={eta_noncond:.4f}")
    m.condensing = True
    # LHV-basis efficiency in condensing mode exceeds the HHV-basis value
    assert_true(m.efficiency_lhv(1.0) > m.efficiency_hhv(1.0),
                "eta_LHV > eta_HHV (latent recovery)")


def test_high_condensing_potential():
    print("\n[Test 6] Latent heat is a large share of HHV (high condensing potential)")
    m, _ = make_model()
    frac = (m.HHV - m.LHV) / m.HHV
    # For H2, (HHV-LHV)/HHV ~ 0.154 -- much larger than NG (~0.11)
    assert_true(0.12 < frac < 0.20,
                f"latent share of HHV = {frac:.3f} (~15% for H2)")


def test_flame_temp_and_nox():
    print("\n[Test 7] Higher flame temp & NOx note: excess air lowers both")
    m, _ = make_model()
    T_lean = m.adiabatic_flame_temp(1.25)
    T_leaner = m.adiabatic_flame_temp(2.0)
    assert_true(T_leaner < T_lean, f"T_ad(lambda=2.0)={T_leaner:.0f} < T_ad(1.25)={T_lean:.0f} K")
    nox_lean = m.nox_index(1.25)
    nox_leaner = m.nox_index(2.0)
    assert_true(nox_leaner < nox_lean,
                f"NOx index drops with excess air: {nox_leaner:.3f} < {nox_lean:.3f}")
    # Lumped sensible-balance flame-temp estimate (mean burnt-gas T); high and
    # well above the boiler water temperature -> elevated thermal-NOx note.
    assert_true(T_lean > 1500.0, f"H2 flame temp high ({T_lean:.0f} K, lumped mean)")


def test_thermal_ode_heats_up():
    print("\n[Test 8] Thermal ODE: water heats up from cold start at full fire")
    m, _ = make_model()
    r = m.simulate(1.0, 300.0, 2.0, 600.0)
    assert_true(r["temperature"][-1] > 300.0,
                f"T_final={r['temperature'][-1]:.2f} > 300 K")
    assert_true(np.all(np.diff(r["temperature"]) >= -1e-6),
                "Monotone heating under constant full fire")
    assert_true(r["temperature"][-1] < 373.15,
                f"T_final={r['temperature'][-1]:.2f} < boiling (bounded)")


def test_thermal_steady_state_and_off():
    print("\n[Test 9] Reaches steady state; firing off => cools toward ambient/return")
    m, _ = make_model()
    r = m.simulate(0.5, 330.0, 5.0, 3000.0)
    dT = abs(r["temperature"][-1] - r["temperature"][-2])
    assert_true(dT < 0.05, f"Near steady state: dT={dT:.4f} K/step")
    r_off = m.simulate(0.0, 340.0, 5.0, 600.0)
    assert_true(r_off["temperature"][-1] < 340.0,
                f"Cools when off: {r_off['temperature'][-1]:.2f} < 340 K")


def test_predict_interface():
    print("\n[Test 10] ComponentModel predict() interface + get_info()")
    _, cm = make_model()
    r = cm.predict({"firing_rate": 0.8, "T_water_K": 310.0, "dt": 5.0, "duration_s": 60.0})
    for key in ["t", "temperature", "firing_rate", "heat_to_water_W",
                "efficiency", "efficiency_lhv", "h2_flow_kg_s", "flue_temp_K"]:
        assert_true(key in r, f"Key '{key}' in output")
    assert_true(len(r["t"]) == len(r["temperature"]), "Arrays same length")
    info = cm.get_info()
    assert_true(info["component_id"] == "EC089" and info["version"] == "1.0.0",
                "get_info id/version correct")


def test_step_firing_response():
    print("\n[Test 11] Step firing-rate response: water temp rises faster at high fire")
    m, _ = make_model()
    def step_phi(t):
        return 0.2 if t < 300 else 1.0
    r = m.simulate(step_phi, 310.0, 5.0, 600.0)
    i_lo = np.argmin(np.abs(r["t"] - 280.0))
    i_hi = np.argmin(np.abs(r["t"] - 590.0))
    rate_lo = (r["temperature"][i_lo] - r["temperature"][i_lo - 1])
    rate_hi = (r["temperature"][i_hi] - r["temperature"][i_hi - 1])
    assert_true(rate_hi > rate_lo, "Heating rate increases after firing step up")


def test_benchmark():
    print("\n[Test 12] Benchmark: 1-hour sim at dt=1s")
    m, _ = make_model()
    t0 = time.perf_counter()
    m.simulate(1.0, 300.0, 1.0, 3600.0)
    elapsed = time.perf_counter() - t0
    print(f"  3600s simulation in {elapsed*1000:.1f} ms")
    assert_true(elapsed < 5.0, "Completes in < 5 s")


if __name__ == "__main__":
    tests = [
        test_mass_conservation,
        test_stoichiometry,
        test_energy_conservation,
        test_efficiency_bounds_hhv,
        test_condensing_boosts_efficiency,
        test_high_condensing_potential,
        test_flame_temp_and_nox,
        test_thermal_ode_heats_up,
        test_thermal_steady_state_and_off,
        test_predict_interface,
        test_step_firing_response,
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
    print(f"EC089 Hydrogen Boiler F2a -- Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    sys.exit(0 if failed == 0 else 1)
