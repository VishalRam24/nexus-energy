"""
EC205 -- CO2 Electrolyzer (CO2 -> CO/Fuels) -- F2a Electrochemical
Test suite: physics sanity (driven cell, FE<1, Faraday's law, energy
conservation), ODE behaviour, edge cases, predict() interface, benchmark.
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from model import CO2Electrolyzer_F2a
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
def test_driven_above_Erev():
    print("\n[Test 1] Driven cell: V_cell > E_rev for j > 0")
    m, _ = make_model()
    for j in [0.05, 0.2, 0.4, 0.55]:
        V = m.cell_voltage(j, 333.15)
        assert_true(V > m.E_rev, f"V({j})={V:.4f} > E_rev={m.E_rev:.3f}")


def test_voltage_monotone():
    print("\n[Test 2] V_cell increases with j (overpotentials grow)")
    m, _ = make_model()
    j_vals = np.linspace(0.01, 0.58, 50)
    V_prev = m.cell_voltage(j_vals[0], 333.15)
    for j in j_vals[1:]:
        V = m.cell_voltage(j, 333.15)
        assert_true(V >= V_prev - 1e-9, f"V({j:.3f})={V:.4f} >= V_prev={V_prev:.4f}")
        V_prev = V
    print("  All 49 pairs monotone non-decreasing.")


def test_faradaic_below_one():
    print("\n[Test 3] Faradaic efficiency in (0, 1) and FE_CO + FE_H2 = 1")
    m, _ = make_model()
    for j in [0.05, 0.2, 0.4, 0.55]:
        fe = m.faradaic_efficiency(j)
        assert_true(0.0 < fe < 1.0, f"FE_CO({j})={fe:.4f} in (0,1)")
        assert_true(abs(fe + m.fe_h2(j) - 1.0) < 1e-9, "FE_CO + FE_H2 = 1")


def test_faradaic_rolloff():
    print("\n[Test 4] FE_CO falls as j -> j_L (H2 evolution takes over)")
    m, _ = make_model()
    fe_low = m.faradaic_efficiency(0.1 * m.j_L)
    fe_high = m.faradaic_efficiency(0.98 * m.j_L)
    assert_true(fe_high < fe_low, f"FE_CO(0.98 j_L)={fe_high:.3f} < FE_CO(0.1 j_L)={fe_low:.3f}")
    assert_true(fe_low <= m.FE_max + 1e-9, f"FE_CO bounded by FE_max={m.FE_max}")


def test_faradays_law():
    print("\n[Test 5] Product rate follows Faraday's law: n_dot = FE*I/(n_e*F)")
    m, _ = make_model()
    j = 0.3
    I_total = j * m.A_cell * m.N_cells
    expected = m.faradaic_efficiency(j) * I_total / (m.n_e * m.F)
    got = m.co_production_rate(j)
    assert_true(abs(got - expected) < 1e-12, f"n_dot_CO={got:.6e} = FE*I/(nF)={expected:.6e}")
    # Rate scales linearly with current at fixed FE region
    assert_true(m.co_production_rate(0.2) > m.co_production_rate(0.1),
                "Higher current -> higher CO rate")


def test_energy_per_mol_consistency():
    print("\n[Test 6] Energy/mol = n_e*F*V/FE and SEC > 0 (conservation)")
    m, _ = make_model()
    j, T = 0.3, 333.15
    V = m.cell_voltage(j, T)
    fe = m.faradaic_efficiency(j)
    expected = m.n_e * m.F * V / fe
    got = m.energy_per_mol_CO(j, T)
    assert_true(abs(got - expected) < 1e-6, f"E/mol={got:.1f} = nFV/FE={expected:.1f} J/mol")
    sec = m.energy_per_kg_CO_kWh(j, T)
    # Real CO2-to-CO electrolyzers: ~ a few to ~10+ kWh/kg CO -> sanity band
    assert_true(2.0 < sec < 40.0, f"SEC={sec:.2f} kWh/kg CO in plausible band")


def test_energy_increases_with_loss():
    print("\n[Test 7] Energy/mol rises when FE drops or V rises near j_L")
    m, _ = make_model()
    e_mid = m.energy_per_mol_CO(0.3, 333.15)
    e_near_jL = m.energy_per_mol_CO(0.58, 333.15)
    assert_true(e_near_jL > e_mid,
                f"E/mol near j_L ({e_near_jL:.0f}) > mid ({e_mid:.0f}) J/mol")


def test_overpotentials_positive():
    print("\n[Test 8] All overpotentials >= 0 (each is a loss)")
    m, _ = make_model()
    r = m.simulate(0.3, 333.15, 1.0, 10.0)
    for name, arr in r["overpotentials"].items():
        if name == "E_rev":
            continue
        assert_true(np.all(arr >= -1e-9), f"{name} all >= 0")


def test_thermal_ode_heats_up():
    print("\n[Test 9] Thermal ODE: stack self-heats above E_tn dissipation")
    m, _ = make_model()
    r = m.simulate(0.4, 300.0, 0.5, 120.0)
    assert_true(r["temperature"][-1] > 300.0, f"T_final={r['temperature'][-1]:.2f} > 300 K")
    assert_true(r["temperature"][-1] < 400.0, f"T_final={r['temperature'][-1]:.2f} < 400 K (reasonable)")


def test_thermal_steady_state():
    print("\n[Test 10] Thermal reaches approximate steady state")
    m, _ = make_model()
    r = m.simulate(0.3, 313.15, 1.0, 1200.0)
    dT = abs(r["temperature"][-1] - r["temperature"][-2])
    assert_true(dT < 0.1, f"Near SS: dT={dT:.5f} K between last two steps")


def test_predict_interface():
    print("\n[Test 11] ComponentModel predict() interface + metadata")
    _, cm = make_model()
    assert_true(cm.component_id == "EC205", "component_id == EC205")
    assert_true(cm.version == "1.0.0", "version == 1.0.0")
    r = cm.predict({"current_density_A_cm2": 0.3, "dt": 1.0, "duration_s": 5.0})
    for key in ["t", "voltage", "power_density", "temperature",
                "faradaic_efficiency", "co_rate_mol_s", "energy_per_mol_CO",
                "sec_kWh_kg", "overpotentials"]:
        assert_true(key in r, f"Key '{key}' in output")
    assert_true(len(r["t"]) == len(r["voltage"]), "Arrays same length")
    info = cm.get_info()
    assert_true("inputs" in info and "outputs" in info, "get_info has inputs/outputs")


def test_benchmark():
    print("\n[Test 12] Benchmark: 60s sim at dt=0.1")
    m, _ = make_model()
    t0 = time.perf_counter()
    m.simulate(0.3, 313.15, 0.1, 60.0)
    elapsed = time.perf_counter() - t0
    print(f"  60s simulation in {elapsed*1000:.1f} ms")
    assert_true(elapsed < 5.0, "Completes in < 5 s")


if __name__ == "__main__":
    tests = [
        test_driven_above_Erev,
        test_voltage_monotone,
        test_faradaic_below_one,
        test_faradaic_rolloff,
        test_faradays_law,
        test_energy_per_mol_consistency,
        test_energy_increases_with_loss,
        test_overpotentials_positive,
        test_thermal_ode_heats_up,
        test_thermal_steady_state,
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
    print(f"EC205 CO2 Electrolyzer F2a -- Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    sys.exit(0 if failed == 0 else 1)
