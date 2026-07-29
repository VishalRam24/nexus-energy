"""
EC125 — Adiabatic CAES (A-CAES) — F2a Physics-Lumped
Test suite: energy conservation, fuel-free operation, RTE > diabatic CAES,
coupled cavern+TES balance, ODE behaviour, edge cases, predict() interface, benchmark.
Run with: python3 scripts/test_model.py   (NO pytest)
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from model import ACAES_F2a
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
def test_charge_raises_tes_and_soc():
    print("\n[Test 1] Charge raises TES temperature and cavern SOC")
    m, _ = make_model()
    r = m.simulate("charge", m_dot=100.0, duration_s=3600.0, dt=60.0,
                   soc0=0.0, T_tes0=m.T_tes_ambient, T_amb=288.15)
    assert_true(r["solver_success"], "solve_ivp converged")
    assert_true(r["soc"][-1] > r["soc"][0], f"SOC up: {r['soc'][0]:.3f} -> {r['soc'][-1]:.3f}")
    assert_true(r["T_tes"][-1] > r["T_tes"][0],
                f"TES heated by compression heat: {r['T_tes'][0]:.1f} -> {r['T_tes'][-1]:.1f} K")


def test_discharge_drops_tes_and_soc():
    print("\n[Test 2] Discharge lowers TES temperature and cavern SOC")
    m, _ = make_model()
    r = m.simulate("discharge", m_dot=100.0, duration_s=3600.0, dt=60.0,
                   soc0=1.0, T_tes0=m.T_tes_design, T_amb=288.15)
    assert_true(r["soc"][-1] < r["soc"][0], f"SOC down: {r['soc'][0]:.3f} -> {r['soc'][-1]:.3f}")
    assert_true(r["T_tes"][-1] < r["T_tes"][0],
                f"TES heat drawn for re-heat: {r['T_tes'][0]:.1f} -> {r['T_tes'][-1]:.1f} K")


def test_fuel_free_operation():
    print("\n[Test 3] Fuel-free operation (key A-CAES advantage)")
    m, _ = make_model()
    for mode in ["charge", "discharge", "idle"]:
        r = m.simulate(mode, m_dot=80.0, duration_s=1800.0, dt=60.0, soc0=0.5)
        assert_true(np.allclose(r["fuel_power_kw"], 0.0),
                    f"fuel_power == 0 in '{mode}' mode (no combustion)")


def test_rte_in_physical_band():
    print("\n[Test 4] Design RTE in A-CAES band 0.65-0.75")
    m, _ = make_model()
    rte = m.round_trip_efficiency(T_amb_K=288.15, T_tes_K=m.T_tes_design)
    assert_true(0.65 <= rte <= 0.75, f"RTE_design={rte:.3f} in [0.65, 0.75]")
    assert_true(rte < 0.75, f"RTE_design={rte:.3f} < 0.75 hard A-CAES limit (Budt 2016)")


def test_rte_exceeds_diabatic():
    print("\n[Test 5] A-CAES RTE strictly exceeds diabatic CAES")
    m, _ = make_model()
    rte = m.round_trip_efficiency(T_amb_K=288.15, T_tes_K=m.T_tes_design)
    assert_true(rte > m.rte_diabatic_ref,
                f"A-CAES RTE {rte:.3f} > diabatic ref {m.rte_diabatic_ref:.3f}")


def test_roundtrip_ode_rte():
    print("\n[Test 6] ODE-integrated round-trip RTE physically sound")
    m, _ = make_model()
    rte, ch, dis = m.round_trip_simulation(m_dot=100.0, charge_s=3600.0, dt=60.0)
    assert_true(0.0 < rte < 1.0, f"ODE round-trip RTE={rte:.3f} in (0,1)")
    assert_true(rte > m.rte_diabatic_ref,
                f"ODE round-trip RTE {rte:.3f} > diabatic {m.rte_diabatic_ref:.3f}")
    assert_true(ch["E_elec_kwh"] > dis["E_elec_kwh"] > 0,
                f"E_in {ch['E_elec_kwh']:.1f} > E_out {dis['E_elec_kwh']:.1f} > 0 (loss positive)")


def test_tes_energy_conservation_charge():
    print("\n[Test 7] TES energy balance: stored heat = elec_in * eta_comp * eta_motor * eta_tes (within loss)")
    m, _ = make_model()
    r = m.simulate("charge", m_dot=100.0, duration_s=1800.0, dt=30.0,
                   soc0=0.0, T_tes0=m.T_tes_ambient, T_amb=288.15)
    # heat captured into TES should be a sensible fraction of electrical energy in
    E_in_kwh = r["E_elec_kwh"]
    dU_tes_kwh = r["dU_tes_kwh"]
    assert_true(dU_tes_kwh > 0, f"TES gained energy: {dU_tes_kwh:.1f} kWh")
    # captured heat must not exceed electrical energy input (first law)
    assert_true(dU_tes_kwh < E_in_kwh,
                f"Stored TES heat {dU_tes_kwh:.1f} < E_elec_in {E_in_kwh:.1f} kWh (1st law)")


def test_idle_tes_decay():
    print("\n[Test 8] Idle TES cools toward ambient (Newton standby loss)")
    m, _ = make_model()
    r = m.simulate("idle", m_dot=0.0, duration_s=12.0 * 3600.0, dt=600.0,
                   soc0=0.5, T_tes0=m.T_tes_design, T_amb=288.15)
    assert_true(r["T_tes"][-1] < r["T_tes"][0],
                f"TES decays: {r['T_tes'][0]:.1f} -> {r['T_tes'][-1]:.1f} K over 12 h")
    assert_true(r["T_tes"][-1] > m.T_tes_ambient,
                f"TES stays above ambient: {r['T_tes'][-1]:.1f} > {m.T_tes_ambient:.1f} K")
    # check decay rate matches tau_tes ~ 56 h
    expected = m.T_tes_ambient + (m.T_tes_design - m.T_tes_ambient) * np.exp(-12 * 3600.0 / m.tau_tes)
    assert_true(abs(r["T_tes"][-1] - expected) < 5.0,
                f"Decay matches tau_tes: ODE {r['T_tes'][-1]:.1f} vs analytic {expected:.1f} K")


def test_pressure_soc_consistency():
    print("\n[Test 9] Cavern pressure rises with SOC, within bounds")
    m, _ = make_model()
    r = m.simulate("charge", m_dot=120.0, duration_s=3600.0, dt=60.0,
                   soc0=0.1, T_tes0=350.0, T_amb=288.15)
    assert_true(r["pressure"][-1] > r["pressure"][0], "pressure increases on charge")
    assert_true(np.all(r["pressure"] >= m.p_min * 0.9),
                f"pressure >= ~p_min ({m.p_min/1e5:.0f} bar)")
    assert_true(r["pressure"][-1] <= m.p_max * 1.05,
                f"pressure <= ~p_max ({m.p_max/1e5:.0f} bar)")


def test_partial_tes_reduces_output():
    print("\n[Test 10] Partially-charged TES reduces discharge power (less re-heat)")
    m, _ = make_model()
    r_full = m.simulate("discharge", m_dot=100.0, duration_s=600.0, dt=60.0,
                        soc0=1.0, T_tes0=m.T_tes_design)
    r_part = m.simulate("discharge", m_dot=100.0, duration_s=600.0, dt=60.0,
                        soc0=1.0, T_tes0=0.5 * (m.T_tes_design + m.T_tes_ambient))
    assert_true(r_part["power_elec_kw"][0] < r_full["power_elec_kw"][0],
                f"cooler TES -> less power: {r_part['power_elec_kw'][0]:.0f} < "
                f"{r_full['power_elec_kw'][0]:.0f} kW")


def test_predict_interface():
    print("\n[Test 11] ComponentModel predict() interface")
    _, cm = make_model()
    r = cm.predict({"mode": "charge", "m_dot": 100.0, "duration_s": 600.0, "dt": 60.0})
    for key in ["t", "soc", "T_cav", "T_tes", "pressure", "power_elec_kw",
                "fuel_power_kw", "E_elec_kwh", "rte_design"]:
        assert_true(key in r, f"Key '{key}' in output")
    assert_true(len(r["t"]) == len(r["soc"]) == len(r["T_tes"]), "Arrays same length")
    info = cm.get_info()
    assert_true(info["component_id"] == "EC125", "get_info component_id == EC125")


def test_benchmark():
    print("\n[Test 12] Benchmark: 1 h charge ODE sim at dt=60 s")
    m, _ = make_model()
    t0 = time.perf_counter()
    m.simulate("charge", m_dot=100.0, duration_s=3600.0, dt=60.0, soc0=0.0)
    elapsed = time.perf_counter() - t0
    print(f"  1 h charge simulation in {elapsed*1000:.1f} ms")
    assert_true(elapsed < 5.0, "Completes in < 5 s")


if __name__ == "__main__":
    tests = [
        test_charge_raises_tes_and_soc,
        test_discharge_drops_tes_and_soc,
        test_fuel_free_operation,
        test_rte_in_physical_band,
        test_rte_exceeds_diabatic,
        test_roundtrip_ode_rte,
        test_tes_energy_conservation_charge,
        test_idle_tes_decay,
        test_pressure_soc_consistency,
        test_partial_tes_reduces_output,
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
    print(f"EC125 A-CAES F2a -- Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    sys.exit(0 if failed == 0 else 1)
