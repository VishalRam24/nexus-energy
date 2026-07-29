"""
EC037 -- Zinc-Bromine Flow Battery (ZBFB) -- F2a Physics-Lumped Stack Model
Test suite: physics sanity (Coulomb conservation, V_charge>V_disch, 0<eff<1,
thermal balance, monotonicity), edge cases, predict() interface, benchmark.
Run with system python3 (NO pytest): python3 scripts/test_model.py
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from model import ZnBrFlowF2a
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
def test_nernst_monotone_soc():
    print("\n[Test 1] OCV increases with SOC and with Br2 concentration")
    m, _ = make_model()
    soc_vals = np.linspace(0.1, 0.9, 9)
    prev = m.e_nernst(soc_vals[0], soc_vals[0] * m.c_Br2_max, 298.15)
    for s in soc_vals[1:]:
        E = float(m.e_nernst(s, s * m.c_Br2_max, 298.15))
        assert_true(E > prev - 1e-9, f"OCV({s:.2f})={E:.4f} >= prev={float(prev):.4f}")
        prev = E
    # Br2 dependence at fixed SOC
    E_lo = float(m.e_nernst(0.5, 0.2, 298.15))
    E_hi = float(m.e_nernst(0.5, 0.9, 298.15))
    assert_true(E_hi > E_lo, f"OCV rises with c_Br2: {E_hi:.4f} > {E_lo:.4f}")


def test_ocv_physical_range():
    print("\n[Test 2] Per-cell OCV in a physical ZBFB band")
    m, _ = make_model()
    E = float(m.e_nernst(0.5, 0.5 * m.c_Br2_max, 298.15))
    assert_true(1.5 < E < 2.1, f"OCV(SOC=0.5)={E:.4f} V in [1.5, 2.1]")


def test_v_charge_gt_discharge():
    print("\n[Test 3] V_charge > V_discharge at equal |I| (overpotential sign)")
    m, _ = make_model()
    I = 100.0
    for soc in [0.2, 0.5, 0.8]:
        c = soc * m.c_Br2_max
        V_dis = m.cell_voltage(soc, c, I, 298.15, 2.0)
        V_chg = m.cell_voltage(soc, c, -I, 298.15, 2.0)
        E = float(m.e_nernst(soc, c, 298.15))
        assert_true(V_chg > E > V_dis,
                    f"SOC={soc}: V_chg={V_chg:.4f} > E={E:.4f} > V_dis={V_dis:.4f}")


def test_overpotential_increases_with_current():
    print("\n[Test 4] Discharge voltage drops as |I| rises (more loss)")
    m, _ = make_model()
    soc, c = 0.5, 0.5 * m.c_Br2_max
    prev = m.cell_voltage(soc, c, 1.0, 298.15, 2.0)
    for I in [20.0, 60.0, 120.0, 200.0]:
        V = m.cell_voltage(soc, c, I, 298.15, 2.0)
        assert_true(V <= prev + 1e-9, f"V(I={I})={V:.4f} <= prev={prev:.4f}")
        prev = V


def test_coulombic_efficiency_range():
    print("\n[Test 5] Coulombic efficiency strictly in (0, 1) with shuttle")
    m, _ = make_model()
    for I in [-200.0, -50.0, 50.0, 200.0]:
        for c in [0.1, 0.5, 1.0]:
            eta = m.coulombic_efficiency(I, c)
            assert_true(0.0 < eta < 1.0, f"eta_C(I={I},c={c})={eta:.5f} in (0,1)")
    # higher Br2 -> more shuttle -> lower CE
    e_lo = m.coulombic_efficiency(50.0, 0.1)
    e_hi = m.coulombic_efficiency(50.0, 1.0)
    assert_true(e_hi < e_lo, f"More Br2 lowers CE: {e_hi:.4f} < {e_lo:.4f}")


def test_coulomb_conservation_with_crossover():
    print("\n[Test 6] Coulomb conservation: stored charge < drawn charge (crossover loss)")
    m, _ = make_model()
    I_chg = -100.0
    dur = 1800.0
    r = m.simulate(I_chg, soc0=0.2, T0=298.15, flow_Lpm=2.0, dt=10.0, duration_s=dur)
    dSOC = r["soc"][-1] - r["soc"][0]
    stored_C = dSOC * m.Q_plating                      # actual coulombs stored
    drawn_C = abs(I_chg) * dur                          # coulombs pushed in
    assert_true(stored_C > 0, f"SOC rose on charge: dSOC={dSOC:.4f}")
    assert_true(stored_C < drawn_C,
                f"stored {stored_C:.0f} C < drawn {drawn_C:.0f} C (crossover loss)")
    ce_eff = stored_C / drawn_C
    assert_true(0.0 < ce_eff < 1.0, f"effective CE={ce_eff:.4f} in (0,1)")


def test_plating_limited_capacity():
    print("\n[Test 7] Plating-limited capacity scales with electrode area, not flow")
    m, _ = make_model()
    expected = m.areal_cap_Ah_cm2 * 3600.0 * m.A_cm2 * m.N_cells
    assert_true(abs(m.Q_plating - expected) < 1.0, f"Q_plating={m.Q_plating:.0f} C")
    # SOC swing for fixed charge throughput is independent of flow rate
    r1 = m.simulate(-100.0, soc0=0.3, flow_Lpm=1.0, dt=10.0, duration_s=600.0)
    r2 = m.simulate(-100.0, soc0=0.3, flow_Lpm=6.0, dt=10.0, duration_s=600.0)
    dsoc1 = r1["soc"][-1] - r1["soc"][0]
    dsoc2 = r2["soc"][-1] - r2["soc"][0]
    assert_true(abs(dsoc1 - dsoc2) < 1e-3,
                f"SOC swing flow-independent: {dsoc1:.4f} vs {dsoc2:.4f}")


def test_soc_rails():
    print("\n[Test 8] SOC stays within [0,1] across charge/discharge")
    m, _ = make_model()
    r_chg = m.simulate(-250.0, soc0=0.9, dt=10.0, duration_s=3600.0)
    r_dis = m.simulate(250.0, soc0=0.1, dt=10.0, duration_s=3600.0)
    assert_true(r_chg["soc"].max() <= 1.0 + 1e-6, f"SOC<=1, max={r_chg['soc'].max():.4f}")
    assert_true(r_dis["soc"].min() >= -1e-6, f"SOC>=0, min={r_dis['soc'].min():.4f}")


def test_thermal_balance():
    print("\n[Test 9] Thermal ODE: heats under load, reaches near steady state")
    m, _ = make_model()
    # thermal time constant tau = m*cp/hA ~ 3e4 s -> integrate several tau to steady state
    r = m.simulate(150.0, soc0=0.5, T0=298.15, flow_Lpm=2.0, dt=200.0, duration_s=200000.0)
    assert_true(r["temperature"][-1] > 298.15, f"T rose to {r['temperature'][-1]:.2f} K")
    assert_true(r["temperature"][-1] < 420.0, f"T bounded {r['temperature'][-1]:.2f} K < 420")
    dT = abs(r["temperature"][-1] - r["temperature"][-2])
    assert_true(dT < 0.05, f"near steady state: dT={dT:.5f} K/step")
    # steady-state energy balance: Q_gen ~= Q_loss at the end
    Tend = r["temperature"][-1]
    Q_gen = m.heat_generation(r["soc"][-1], r["c_Br2"][-1], 150.0, Tend, 2.0)
    Q_loss = m.hA * (Tend - m.T_amb)
    assert_true(abs(Q_gen - Q_loss) / max(Q_gen, 1.0) < 0.05,
                f"Q_gen={Q_gen:.1f} ~= Q_loss={Q_loss:.1f} W (balance at steady state)")


def test_self_discharge():
    print("\n[Test 10] Open-circuit (I=0) self-discharges via Br2 shuttle")
    m, _ = make_model()
    r = m.simulate(0.0, soc0=0.8, T0=298.15, dt=60.0, duration_s=86400.0)
    assert_true(r["c_Br2"][-1] < r["c_Br2"][0],
                f"Br2 decays {r['c_Br2'][0]:.4f}->{r['c_Br2'][-1]:.4f} mol/L")
    assert_true(r["soc"][-1] <= r["soc"][0] + 1e-6,
                f"SOC non-increasing at OC: {r['soc'][0]:.4f}->{r['soc'][-1]:.4f}")


def test_predict_interface():
    print("\n[Test 11] ComponentModel predict() interface and shapes")
    _, cm = make_model()
    r = cm.predict({"current_A": -80.0, "soc0": 0.4, "dt": 10.0, "duration_s": 200.0})
    for key in ["t", "soc", "c_Br2", "temperature", "voltage", "ocv",
                "current", "coulombic_efficiency", "shuttle_current", "power"]:
        assert_true(key in r, f"Key '{key}' present")
    n = len(r["t"])
    assert_true(all(len(r[k]) == n for k in
                    ["soc", "voltage", "temperature", "coulombic_efficiency"]),
                "All series same length")
    info = cm.get_info()
    assert_true(info["component_id"] == "EC037" and info["version"] == "1.0.0",
                "get_info() metadata correct")


def test_benchmark():
    print("\n[Test 12] Benchmark: 1 h sim at dt=1 s")
    m, _ = make_model()
    t0 = time.perf_counter()
    m.simulate(100.0, soc0=0.5, T0=298.15, dt=1.0, duration_s=3600.0)
    elapsed = time.perf_counter() - t0
    print(f"  3600 s simulation in {elapsed*1000:.1f} ms")
    assert_true(elapsed < 5.0, "Completes in < 5 s")


if __name__ == "__main__":
    tests = [
        test_nernst_monotone_soc,
        test_ocv_physical_range,
        test_v_charge_gt_discharge,
        test_overpotential_increases_with_current,
        test_coulombic_efficiency_range,
        test_coulomb_conservation_with_crossover,
        test_plating_limited_capacity,
        test_soc_rails,
        test_thermal_balance,
        test_self_discharge,
        test_predict_interface,
        test_benchmark,
    ]
    passed = failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"  ERROR: {e}")

    print(f"\n{'='*60}")
    print(f"EC037 ZBFB F2a -- Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    sys.exit(0 if failed == 0 else 1)
