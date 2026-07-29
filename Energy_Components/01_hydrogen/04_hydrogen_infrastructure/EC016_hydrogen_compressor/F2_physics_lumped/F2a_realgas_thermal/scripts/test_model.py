"""
EC016 -- Hydrogen Compressor -- F2a Real-Gas Multistage Reciprocating
Test suite: real-gas physics, conservation, monotonicity, ODE transient, edges.
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from model import H2CompressorRealGasThermal
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
def test_compressibility_realgas():
    print("\n[Test 1] H2 compressibility Z > 1 and rises with pressure")
    m, _ = make_model()
    Z_low = m.compressibility(298.15, 20.0)
    Z_hi = m.compressibility(298.15, 900.0)
    assert_true(0.99 < Z_low < 1.1, f"Z(20 bar)={Z_low:.4f} ~ 1 (near-ideal)")
    assert_true(Z_hi > Z_low, f"Z(900 bar)={Z_hi:.4f} > Z(20 bar)={Z_low:.4f}")
    assert_true(1.2 < Z_hi < 1.8, f"Z(900 bar)={Z_hi:.4f} in realistic H2 range")


def test_work_positive_and_discharge_hot():
    print("\n[Test 2] Work > 0 and T_discharge > T_inlet every stage")
    m, _ = make_model()
    prof = m.stage_profile(20.0, 900.0)
    assert_true(np.all(prof["w_stage_J_kg"] > 0), "All stage works > 0")
    assert_true(np.all(prof["T_discharge"] > prof["T_in_stage"]),
                "T_discharge > T_inlet for every stage")
    w_tot = m.specific_work(20.0, 900.0)
    assert_true(w_tot > 0, f"Total specific work={w_tot/3.6e6:.3f} kWh/kg > 0")


def test_energy_conservation():
    print("\n[Test 3] Energy balance: dissipated heat <= shaft work")
    m, _ = make_model()
    m_dot = 50.0 / 3600.0
    P_shaft = m.shaft_power_kw(m_dot, 20.0, 900.0)              # kW
    Q_diss = m.dissipated_heat_w(m_dot, 20.0, 900.0) / 1000.0  # kW
    assert_true(Q_diss > 0, f"Dissipated heat={Q_diss:.3f} kW > 0")
    assert_true(Q_diss <= P_shaft + 1e-9,
                f"Q_diss={Q_diss:.3f} <= P_shaft={P_shaft:.3f} kW (no energy created)")
    # reversible work fraction recovered as gas enthalpy must be the remainder
    prof = m.stage_profile(20.0, 900.0)
    w_rev = prof["w_rev_J_kg"].sum() / 3.6e6
    w_tot = m.specific_work(20.0, 900.0) / 3.6e6
    assert_true(w_rev < w_tot, f"reversible {w_rev:.3f} < actual {w_tot:.3f} kWh/kg")


def test_isentropic_efficiency_bounds():
    print("\n[Test 4] Isentropic efficiency in (0,1)")
    m, _ = make_model()
    eta = m.isentropic_efficiency(20.0, 900.0)
    assert_true(0.0 < eta < 1.0, f"eta_isen={eta:.4f} in (0,1)")


def test_volumetric_efficiency_bounds():
    print("\n[Test 5] Volumetric efficiency in (0,1) and falls with PR")
    m, _ = make_model()
    ev_low = m.volumetric_efficiency(1.5)
    ev_hi = m.volumetric_efficiency(5.0)
    assert_true(0.0 < ev_hi < ev_low < 1.0,
                f"eta_vol falls with PR: {ev_low:.3f} (PR=1.5) > {ev_hi:.3f} (PR=5)")


def test_sec_realistic():
    print("\n[Test 6] SEC for 20->900 bar in realistic H2 range")
    m, _ = make_model()
    sec = m.sec_kwh_per_kg(20.0, 900.0)
    assert_true(1.0 < sec < 6.0, f"SEC={sec:.3f} kWh/kg in [1,6] (Sdanghi 2019)")


def test_work_monotone_in_pressure():
    print("\n[Test 7] Specific work rises with discharge pressure")
    m, _ = make_model()
    P_outs = [100.0, 300.0, 500.0, 700.0, 900.0]
    w_prev = -1.0
    for P in P_outs:
        w = m.specific_work(20.0, P)
        assert_true(w > w_prev, f"w(Pout={P})={w/3.6e6:.3f} kWh/kg > prev")
        w_prev = w


def test_thermal_ode_warms_up():
    print("\n[Test 8] Lumped metal ODE warms up from ambient and saturates")
    m, _ = make_model()
    r = m.simulate(50.0 / 3600.0, 20.0, 900.0, dt=30.0, duration_s=3600.0)
    T0 = r["T_metal"][0]
    Tf = r["T_metal"][-1]
    assert_true(Tf > T0, f"Metal heats: {T0:.1f} -> {Tf:.1f} K")
    assert_true(Tf < 600.0, f"Metal T bounded: {Tf:.1f} K < 600 K")
    dT_last = abs(r["T_metal"][-1] - r["T_metal"][-2])
    assert_true(dT_last < 1.0, f"Near steady state: last dT={dT_last:.3f} K")


def test_thermal_steady_state_balance():
    print("\n[Test 9] Steady metal T satisfies analytic heat balance")
    m, _ = make_model()
    m_dot = 50.0 / 3600.0
    Q = m.dissipated_heat_w(m_dot, 20.0, 900.0)
    T_ss_analytic = m.T_amb + m.f_heat * Q / m.hA_amb
    r = m.simulate(m_dot, 20.0, 900.0, dt=60.0, duration_s=10800.0)
    err = abs(r["T_metal"][-1] - T_ss_analytic)
    assert_true(err < 1.0, f"ODE SS={r['T_metal'][-1]:.2f} K vs analytic={T_ss_analytic:.2f} K")


def test_zero_flow_edge():
    print("\n[Test 10] Zero mass flow -> zero power, metal stays ambient")
    m, _ = make_model()
    P = m.shaft_power_kw(0.0, 20.0, 900.0)
    assert_true(abs(P) < 1e-12, f"Power at zero flow = {P:.2e} kW")
    r = m.simulate(0.0, 20.0, 900.0, dt=60.0, duration_s=1800.0)
    assert_true(abs(r["T_metal"][-1] - m.T_amb) < 1e-6, "Metal stays at ambient with no flow")


def test_intercooler_effectiveness():
    print("\n[Test 11] More intercooling lowers work and final discharge T")
    m, _ = make_model()
    w_cool = m.specific_work(20.0, 900.0, eps_ic=0.95)
    w_warm = m.specific_work(20.0, 900.0, eps_ic=0.40)
    assert_true(w_cool < w_warm, f"Better cooling lowers work: {w_cool/3.6e6:.3f} < {w_warm/3.6e6:.3f} kWh/kg")


def test_predict_interface():
    print("\n[Test 12] ComponentModel predict() interface")
    _, cm = make_model()
    r = cm.predict({"P_in_bar": 20.0, "P_out_bar": 500.0, "dt": 60.0, "duration_s": 600.0})
    for key in ["t", "T_metal", "shaft_power_kW", "SEC_kWh_kg",
                "isentropic_efficiency", "T_discharge_final_K", "stage_profile"]:
        assert_true(key in r, f"Key '{key}' in output")
    assert_true(len(r["t"]) == len(r["T_metal"]), "Time arrays same length")
    info = cm.get_info()
    assert_true(info["component_id"] == "EC016", "get_info component_id == EC016")


def test_benchmark():
    print("\n[Test 13] Benchmark: 1 h transient at dt=10 s")
    m, _ = make_model()
    t0 = time.perf_counter()
    m.simulate(50.0 / 3600.0, 20.0, 900.0, dt=10.0, duration_s=3600.0)
    elapsed = time.perf_counter() - t0
    print(f"  3600 s transient in {elapsed*1000:.1f} ms")
    assert_true(elapsed < 5.0, "Completes in < 5 s")


if __name__ == "__main__":
    tests = [
        test_compressibility_realgas,
        test_work_positive_and_discharge_hot,
        test_energy_conservation,
        test_isentropic_efficiency_bounds,
        test_volumetric_efficiency_bounds,
        test_sec_realistic,
        test_work_monotone_in_pressure,
        test_thermal_ode_warms_up,
        test_thermal_steady_state_balance,
        test_zero_flow_edge,
        test_intercooler_effectiveness,
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
    print(f"EC016 H2 Compressor F2a (real-gas thermal) -- Results: {passed} passed, {failed} failed")
    print(f"{'='*64}")
    sys.exit(0 if failed == 0 else 1)
