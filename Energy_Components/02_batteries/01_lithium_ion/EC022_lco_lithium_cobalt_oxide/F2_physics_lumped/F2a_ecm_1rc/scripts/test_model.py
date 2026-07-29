"""
EC022 -- LCO Battery -- F2a Thevenin 1-RC ECM
Test suite: physics sanity (Coulomb conservation, OCV monotonicity,
0<eff<1, thermal balance), edge cases, predict() interface, benchmark.
Run as: python3 scripts/test_model.py   (system python3, NOT pytest)
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from model import LCO_ECM_1RC
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
def test_ocv_monotone():
    print("\n[Test 1] OCV(SOC) strictly monotone increasing, range 3.0-4.2 V")
    m, _ = make_model()
    soc = np.linspace(0.0, 1.0, 501)
    ocv = m.ocv(soc)
    assert_true(np.all(np.diff(ocv) > 0), "OCV strictly increasing in SOC")
    assert_true(2.95 < ocv[0] < 3.1, f"OCV(0)={ocv[0]:.3f} V near 3.0")
    assert_true(4.1 < ocv[-1] < 4.25, f"OCV(1)={ocv[-1]:.3f} V near 4.2")
    assert_true(np.all(m.docv_dsoc(soc) > 0), "Analytic dOCV/dSOC > 0 everywhere")


def test_coulomb_conservation():
    print("\n[Test 2] Coulomb conservation: delta_SOC = -I*t/(3600*Q) on discharge")
    m, _ = make_model()
    I = 2.6  # 1C discharge
    dur = 600.0
    r = m.simulate(I, soc0=0.9, T0=298.15, dt=1.0, duration_s=dur)
    dsoc_expected = -I * dur / (3600.0 * m.Q)
    dsoc_actual = r["soc"][-1] - r["soc"][0]
    err = abs(dsoc_actual - dsoc_expected)
    assert_true(err < 1e-4, f"|delta_SOC err|={err:.2e} (exp {dsoc_expected:.5f}, got {dsoc_actual:.5f})")


def test_coulombic_efficiency_charge():
    print("\n[Test 3] Coulombic efficiency (0<eff_c<1) reduces SOC gain on charge")
    m, _ = make_model()
    I = -2.6  # charge
    dur = 600.0
    r = m.simulate(I, soc0=0.3, T0=298.15, dt=1.0, duration_s=dur)
    dsoc_ideal = -I * dur / (3600.0 * m.Q)          # eta=1
    dsoc_actual = r["soc"][-1] - r["soc"][0]
    assert_true(0.0 < m.eta_c < 1.0, f"eta_c={m.eta_c} in (0,1)")
    assert_true(dsoc_actual < dsoc_ideal, "Charge SOC gain < ideal (eta loss)")
    assert_true(abs(dsoc_actual - m.eta_c * dsoc_ideal) < 1e-4, "Matches eta_c*ideal")


def test_voltage_charge_vs_discharge():
    print("\n[Test 4] V_term < OCV on discharge, > OCV on charge")
    m, _ = make_model()
    soc, T = 0.6, 298.15
    ocv = float(m.ocv(soc))
    v_dis = m.terminal_voltage(soc, 0.0, 2.6, T)    # discharge
    v_chg = m.terminal_voltage(soc, 0.0, -2.6, T)   # charge
    assert_true(v_dis < ocv, f"V_discharge={v_dis:.3f} < OCV={ocv:.3f}")
    assert_true(v_chg > ocv, f"V_charge={v_chg:.3f} > OCV={ocv:.3f}")


def test_arrhenius_resistance():
    print("\n[Test 5] Arrhenius: R0,R1 increase as T decreases")
    m, _ = make_model()
    assert_true(m.R0(278.15) > m.R0(298.15) > m.R0(313.15),
                "R0 monotonically rises with falling T")
    assert_true(m.R1(278.15) > m.R1(298.15) > m.R1(313.15),
                "R1 monotonically rises with falling T")
    assert_true(abs(m.R0(298.15) - m.R0_ref) < 1e-12, "R0(T_ref)=R0_ref")


def test_thermal_heats_up():
    print("\n[Test 6] Thermal ODE: cell self-heats under high-rate discharge")
    m, _ = make_model()
    r = m.simulate(7.8, soc0=0.95, T0=298.15, dt=1.0, duration_s=400.0)  # ~3C
    assert_true(r["temperature"][-1] > 298.15, f"T_final={r['temperature'][-1]:.2f} > T0")
    assert_true(r["temperature"][-1] < 333.15, f"T_final={r['temperature'][-1]:.2f} < 60 C (bounded)")


def test_thermal_balance_rest():
    print("\n[Test 7] Thermal balance: at I=0 from ambient, T stays at ambient")
    m, _ = make_model()
    r = m.simulate(0.0, soc0=0.5, T0=298.15, dt=5.0, duration_s=600.0)
    dT = abs(r["temperature"][-1] - m.T_amb)
    assert_true(dT < 1e-6, f"No current, no heat: |T-T_amb|={dT:.2e} K")


def test_thermal_relaxation():
    print("\n[Test 8] Newton cooling: hot cell relaxes toward ambient")
    m, _ = make_model()
    r = m.simulate(0.0, soc0=0.5, T0=313.15, dt=5.0, duration_s=2000.0)
    assert_true(r["temperature"][-1] < 313.15, "Hot cell cools")
    assert_true(r["temperature"][-1] > m.T_amb - 0.5, "Approaches ambient from above")
    # heat balance sign: Q_gen<=0 (only -Q_cool when I=0), so monotone decrease
    assert_true(r["temperature"][0] > r["temperature"][-1], "Monotone cooling")


def test_efficiency_bounded():
    print("\n[Test 9] Voltaic efficiency strictly in (0,1)")
    m, _ = make_model()
    r = m.simulate(2.6, soc0=0.9, T0=298.15, dt=2.0, duration_s=600.0)
    eff = r["efficiency"]
    assert_true(np.all(eff > 0.0) and np.all(eff < 1.0),
                f"all eff in (0,1): min={eff.min():.4f}, max={eff.max():.4f}")


def test_rc_relaxation():
    print("\n[Test 10] RC branch relaxes (V_rc decays) after current cutoff")
    m, _ = make_model()
    def pulse(t):
        return 5.0 if t < 100.0 else 0.0
    r = m.simulate(pulse, soc0=0.8, T0=298.15, dt=1.0, duration_s=400.0)
    i_on = np.argmin(np.abs(r["t"] - 99.0))
    i_off = np.argmin(np.abs(r["t"] - 350.0))
    assert_true(abs(r["v_rc"][i_on]) > abs(r["v_rc"][i_off]),
                f"|V_rc| decays: {abs(r['v_rc'][i_on]):.4f} -> {abs(r['v_rc'][i_off]):.4f}")
    assert_true(abs(r["v_rc"][i_off]) < 5e-3, "V_rc near zero after long rest")


def test_predict_interface():
    print("\n[Test 11] ComponentModel predict()/get_info() interface")
    _, cm = make_model()
    info = cm.get_info()
    for k in ["component_id", "component_name", "fidelity", "version", "inputs", "outputs"]:
        assert_true(k in info, f"get_info has '{k}'")
    assert_true(cm.component_id == "EC022", "component_id == EC022")
    r = cm.predict({"current_A": 2.6, "soc0": 0.9, "dt": 5.0, "duration_s": 100.0})
    for key in ["t", "soc", "voltage", "current", "power", "v_rc",
                "temperature", "heat_gen", "R0", "R1", "efficiency"]:
        assert_true(key in r, f"output key '{key}' present")
    assert_true(len(r["t"]) == len(r["voltage"]), "arrays same length")


def test_benchmark():
    print("\n[Test 12] Benchmark: 1 h sim at dt=1 s")
    m, _ = make_model()
    t0 = time.perf_counter()
    m.simulate(2.6, soc0=0.95, T0=298.15, dt=1.0, duration_s=3600.0)
    elapsed = time.perf_counter() - t0
    print(f"  3600 s simulation in {elapsed*1000:.1f} ms")
    assert_true(elapsed < 5.0, "Completes in < 5 s")


if __name__ == "__main__":
    tests = [
        test_ocv_monotone,
        test_coulomb_conservation,
        test_coulombic_efficiency_charge,
        test_voltage_charge_vs_discharge,
        test_arrhenius_resistance,
        test_thermal_heats_up,
        test_thermal_balance_rest,
        test_thermal_relaxation,
        test_efficiency_bounded,
        test_rc_relaxation,
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
    print(f"EC022 LCO F2a 1-RC ECM -- Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    sys.exit(0 if failed == 0 else 1)
