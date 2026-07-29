"""
EC123 — CAES (Diabatic) — F2a Cavern Thermodynamics
Test suite: cavern mass/energy conservation, EOS, compressor/turbine physics,
realistic RTE, edge cases, predict() interface, benchmark timing.
"""

import sys
import os
import time
import numpy as np
from scipy.integrate import trapezoid

sys.path.insert(0, os.path.dirname(__file__))
from model import CAESF2a
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
def test_eos_roundtrip():
    print("\n[Test 1] Ideal-gas EOS round-trips (P<->m)")
    m, _ = make_model()
    P = 5.5e6
    T = m.T_rock
    mass = m.mass_from_pressure(P, T)
    P_back = m.pressure(mass, T)
    assert_true(abs(P_back - P) / P < 1e-9, f"P recovered {P_back:.1f} ~ {P:.1f} Pa")


def test_mass_conservation_charge():
    print("\n[Test 2] Charge: cavern mass balance dm = m_dot*duration")
    m, _ = make_model()
    m_dot, dur = 100.0, 1800.0
    r = m.simulate("charge", m_dot, m.T_rock, m.p_min, 60.0, dur)
    dm = r["mass"][-1] - r["mass"][0]
    expected = m_dot * dur
    err = abs(dm - expected) / expected
    assert_true(err < 1e-4, f"dm={dm:.1f} kg vs {expected:.1f} kg (err {err:.2e})")


def test_charge_raises_pressure_and_soc():
    print("\n[Test 3] Charging raises pressure, SOC, and temperature")
    m, _ = make_model()
    r = m.simulate("charge", 100.0, m.T_rock, m.p_min, 60.0, 3600.0)
    assert_true(r["pressure"][-1] > r["pressure"][0], "Pressure increases")
    assert_true(r["soc"][-1] > r["soc"][0], "SOC increases")
    assert_true(r["temperature"][-1] > r["temperature"][0],
                f"T rises {r['temperature'][0]:.1f}->{r['temperature'][-1]:.1f} K (compression heat)")


def test_discharge_drops_pressure():
    print("\n[Test 4] Discharging lowers cavern mass and pressure")
    m, _ = make_model()
    r = m.simulate("discharge", 200.0, m.T_rock, m.p_max, 60.0, 3600.0)
    assert_true(r["mass"][-1] < r["mass"][0], "Mass decreases")
    assert_true(r["pressure"][-1] < r["pressure"][0], "Pressure decreases")


def test_idle_cools_to_rock():
    print("\n[Test 5] Idle hot cavern relaxes toward rock temperature")
    m, _ = make_model()
    T_hot = m.T_rock + 30.0
    r = m.simulate("idle", 0.0, T_hot, m.p_max, 600.0, 5.0e6)
    assert_true(r["temperature"][-1] < T_hot, "Cooled below initial")
    assert_true(r["temperature"][-1] >= m.T_rock - 1e-6,
                f"T_final {r['temperature'][-1]:.2f} >= T_rock {m.T_rock:.2f}")
    assert_true(abs(r["temperature"][-1] - m.T_rock) < 1.0,
                "Approaches rock temperature at long idle")


def test_idle_mass_conserved():
    print("\n[Test 6] Idle: no flow -> mass exactly conserved")
    m, _ = make_model()
    r = m.simulate("idle", 0.0, m.T_rock + 20.0, m.p_max, 600.0, 1.0e5)
    dm = abs(r["mass"][-1] - r["mass"][0])
    assert_true(dm < 1e-3, f"dm={dm:.2e} kg ~ 0")


def test_compressor_work_monotone():
    print("\n[Test 7] Compressor specific work increases with cavern pressure")
    m, _ = make_model()
    w_lo = m.compressor_specific_work(m.p_min)
    w_hi = m.compressor_specific_work(m.p_max)
    assert_true(w_hi > w_lo > 0, f"w({m.p_min/1e5:.0f}bar)={w_lo/1e3:.0f} < w({m.p_max/1e5:.0f}bar)={w_hi/1e3:.0f} kJ/kg")


def test_turbine_work_positive():
    print("\n[Test 8] Turbine specific work positive and rises with pressure ratio")
    m, _ = make_model()
    w_lo = m.turbine_specific_work(m.p_min)
    w_hi = m.turbine_specific_work(m.p_max)
    assert_true(w_lo > 0 and w_hi > w_lo, f"w_turb {w_lo/1e3:.0f} < {w_hi/1e3:.0f} kJ/kg")


def test_rte_realistic():
    print("\n[Test 9] Diabatic RTE realistic; electric RTE > 1-region check")
    m, _ = make_model()
    rte = m.round_trip_efficiency()
    el = m.electric_rte()
    hr = m.heat_rate()
    assert_true(0.30 < rte < 0.60, f"RTE(incl. fuel)={rte:.3f} in [0.30,0.60] (Huntorf/McIntosh class)")
    assert_true(el > rte, f"Electric RTE {el:.3f} > fuel-inclusive RTE {rte:.3f}")
    assert_true(2000.0 < hr < 7000.0, f"Heat rate {hr:.0f} kJ/kWh_e in plant range (Huntorf 5870, McIntosh 4330)")


def test_energy_balance_first_law():
    print("\n[Test 10] First-law check: internal energy change = enthalpy in - heat loss")
    m, _ = make_model()
    # Adiabatic-ish short charge: set UA tiny via direct rhs integral check
    m_dot, dur, dt = 100.0, 600.0, 30.0
    r = m.simulate("charge", m_dot, m.T_rock, m.p_min, dt, dur)
    # Internal energy change of control volume
    U0 = r["mass"][0] * m.cv * r["temperature"][0]
    U1 = r["mass"][-1] * m.cv * r["temperature"][-1]
    dU = U1 - U0
    # Enthalpy injected (constant inlet enthalpy)
    m_in = r["mass"][-1] - r["mass"][0]
    H_in = m_in * m.cp * m.T_intercool
    # Heat loss integral
    Q_loss = trapezoid(m.UA * (r["temperature"] - m.T_rock), r["t"])
    residual = dU - (H_in - Q_loss)
    scale = abs(H_in)
    assert_true(abs(residual) / scale < 1e-2,
                f"dU={dU/1e6:.2f} MJ ~ H_in-Q_loss={(H_in-Q_loss)/1e6:.2f} MJ (resid {residual/scale:.2e})")


def test_predict_interface():
    print("\n[Test 11] ComponentModel predict() interface")
    _, cm = make_model()
    r = cm.predict({"mode": "charge", "m_dot_kg_s": 100.0, "dt": 300.0, "duration_s": 1800.0})
    for key in ["t", "mass", "temperature", "pressure", "soc", "P_elec", "P_fuel",
                "E_elec_J", "round_trip_efficiency", "electric_rte"]:
        assert_true(key in r, f"Key '{key}' in output")
    assert_true(len(r["t"]) == len(r["pressure"]), "Arrays same length")
    assert_true(r["E_elec_J"] > 0, "Positive electrical energy consumed on charge")


def test_discharge_consumes_fuel():
    print("\n[Test 12] Discharge books fuel; charge books none")
    m, _ = make_model()
    rc = m.simulate("charge", 100.0, m.T_rock, m.p_max, 300.0, 1800.0)
    rd = m.simulate("discharge", 200.0, m.T_rock, m.p_max, 300.0, 1800.0)
    assert_true(rc["E_fuel_J"] == 0.0, "No fuel during charge")
    assert_true(rd["E_fuel_J"] > 0.0, f"Fuel burned during discharge: {rd['m_fuel_kg']:.1f} kg gas")


def test_benchmark():
    print("\n[Test 13] Benchmark: 4h charge sim at dt=60s")
    m, _ = make_model()
    t0 = time.perf_counter()
    m.simulate("charge", 100.0, m.T_rock, m.p_min, 60.0, 4 * 3600.0)
    elapsed = time.perf_counter() - t0
    print(f"  4h simulation in {elapsed*1000:.1f} ms")
    assert_true(elapsed < 5.0, "Completes in < 5 s")


if __name__ == "__main__":
    tests = [
        test_eos_roundtrip,
        test_mass_conservation_charge,
        test_charge_raises_pressure_and_soc,
        test_discharge_drops_pressure,
        test_idle_cools_to_rock,
        test_idle_mass_conserved,
        test_compressor_work_monotone,
        test_turbine_work_positive,
        test_rte_realistic,
        test_energy_balance_first_law,
        test_predict_interface,
        test_discharge_consumes_fuel,
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
    print(f"EC123 CAES F2a — Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    sys.exit(0 if failed == 0 else 1)
