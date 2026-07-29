"""
EC029 -- NiMH Battery -- F2a Thevenin 2-RC Electrothermal
Test suite: physics sanity (Coulomb conservation, OCV monotonicity, thermal
balance, efficiency bounds), overcharge exotherm, edge cases, predict() interface,
and a benchmark timing test. Custom harness, NO pytest.
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from model import NiMH_F2a
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
def test_ocv_range_and_plateau():
    print("\n[Test 1] OCV in NiMH plateau range (~1.2 V)")
    m, _ = make_model()
    for s in [0.1, 0.3, 0.5, 0.7, 0.9]:
        v = m.ocv(s, m.T_ref)
        assert_true(1.0 < v < 1.45, f"OCV(SOC={s})={v:.4f} V in (1.0, 1.45)")
    # plateau: mid-SOC band should be fairly flat (much flatter than Li-ion's
    # ~0.5 V swing). NiMH plateau spread over SOC 0.2-0.8 is ~0.15 V.
    band = [m.ocv(s, m.T_ref) for s in np.linspace(0.2, 0.8, 13)]
    spread = max(band) - min(band)
    assert_true(spread < 0.20, f"Mid-SOC OCV plateau spread={spread:.4f} V (fairly flat)")


def test_ocv_monotone_increasing():
    print("\n[Test 2] OCV monotonically increases with SOC")
    m, _ = make_model()
    s = np.linspace(0.02, 0.98, 60)
    v = m.ocv(s, m.T_ref)
    diffs = np.diff(v)
    assert_true(np.all(diffs > -1e-6), "OCV non-decreasing across full SOC sweep")
    assert_true(v[-1] > v[0], f"OCV(0.98)={v[-1]:.4f} > OCV(0.02)={v[0]:.4f}")


def test_coulomb_conservation():
    print("\n[Test 3] Coulomb counting conserves charge (no self-discharge, no overcharge)")
    m, _ = make_model()
    # kill leak terms to test pure Coulomb counting
    m.k_sd = 0.0
    I = 2.0          # A discharge
    dur = 300.0      # s
    r = m.simulate(I, soc0=0.9, T0=m.T_ref, dt=5.0, duration_s=dur)
    dSOC_expected = -I * dur / (3600.0 * m.Q_ref)
    dSOC_actual = r["soc"][-1] - r["soc"][0]
    err = abs(dSOC_actual - dSOC_expected)
    assert_true(err < 1e-3, f"dSOC actual={dSOC_actual:.5f} vs expected={dSOC_expected:.5f} (err={err:.2e})")


def test_charge_discharge_symmetry():
    print("\n[Test 4] Charge raises SOC, discharge lowers SOC")
    m, _ = make_model()
    m.k_sd = 0.0
    rc = m.simulate(-2.0, soc0=0.3, T0=m.T_ref, dt=5.0, duration_s=200.0)
    rd = m.simulate(2.0, soc0=0.7, T0=m.T_ref, dt=5.0, duration_s=200.0)
    assert_true(rc["soc"][-1] > rc["soc"][0], f"charge: SOC {rc['soc'][0]:.3f}->{rc['soc'][-1]:.3f}")
    assert_true(rd["soc"][-1] < rd["soc"][0], f"discharge: SOC {rd['soc'][0]:.3f}->{rd['soc'][-1]:.3f}")


def test_voltage_sign_convention():
    print("\n[Test 5] Discharge V < OCV, charge V > OCV")
    m, _ = make_model()
    soc = 0.5
    V_dis = m.terminal_voltage(soc, 3.0, m.T_ref)   # discharge
    V_chg = m.terminal_voltage(soc, -3.0, m.T_ref)  # charge
    ocv = m.ocv(soc, m.T_ref)
    assert_true(V_dis < ocv, f"V_discharge={V_dis:.4f} < OCV={ocv:.4f}")
    assert_true(V_chg > ocv, f"V_charge={V_chg:.4f} > OCV={ocv:.4f}")


def test_arrhenius_resistance():
    print("\n[Test 6] Resistance rises as temperature falls (Arrhenius)")
    m, _ = make_model()
    R_cold = m.R0(263.15)
    R_ref = m.R0(m.T_ref)
    R_hot = m.R0(323.15)
    assert_true(R_cold > R_ref > R_hot, f"R(cold)={R_cold:.4f} > R(ref)={R_ref:.4f} > R(hot)={R_hot:.4f}")


def test_efficiency_bounds():
    print("\n[Test 7] Efficiency strictly in (0, 1)")
    m, _ = make_model()
    for I in [3.0, -3.0, 0.5]:
        r = m.simulate(I, soc0=0.6, T0=m.T_ref, dt=5.0, duration_s=120.0)
        eta = r["efficiency"]
        assert_true(np.all(eta > 0.0) and np.all(eta < 1.0),
                    f"I={I}: eta in ({eta.min():.4f}, {eta.max():.4f}) subset (0,1)")


def test_thermal_balance_steady():
    print("\n[Test 8] Thermal: heating then approach to balance; bounded T")
    m, _ = make_model()
    r = m.simulate(5.0, soc0=0.7, T0=m.T_ref, dt=5.0, duration_s=1800.0)
    assert_true(r["temperature"][-1] > m.T_ref, f"cell heated: T_final={r['temperature'][-1]:.2f} > {m.T_ref:.2f}")
    assert_true(r["temperature"][-1] < 360.0, f"T_final={r['temperature'][-1]:.2f} K bounded < 360 K")
    # near balance: small dT between last two output steps
    dT = abs(r["temperature"][-1] - r["temperature"][-2])
    assert_true(dT < 0.5, f"approaching thermal balance: dT={dT:.4f} K/step")


def test_overcharge_exotherm():
    print("\n[Test 9] Overcharge: SOC saturates and cell strongly heats (O2 recombination)")
    m, _ = make_model()
    # charge hard starting near full
    r = m.simulate(-4.0, soc0=0.93, T0=m.T_ref, dt=5.0, duration_s=900.0)
    soc_final = r["soc"][-1]
    dT = r["temperature"][-1] - r["temperature"][0]
    f_oc_final = r["overcharge_fraction"][-1]
    Q_recomb_max = r["heat"]["recombination"].max()
    assert_true(soc_final <= 1.0 + 1e-6, f"SOC clamps at full: {soc_final:.4f} <= 1.0")
    assert_true(f_oc_final > 0.3, f"oxygen recombination active: f_oc={f_oc_final:.3f}")
    assert_true(Q_recomb_max > 0.0, f"recombination heat > 0: max={Q_recomb_max:.3f} W")
    assert_true(dT > 3.0, f"strongly exothermic overcharge: dT={dT:.2f} K")


def test_self_discharge():
    print("\n[Test 10] Self-discharge lowers SOC at rest (I=0)")
    m, _ = make_model()
    # long rest, exaggerate via high temperature to make leak visible quickly
    r = m.simulate(0.0, soc0=0.9, T0=323.15, dt=60.0, duration_s=36000.0)
    assert_true(r["soc"][-1] < r["soc"][0], f"rest SOC {r['soc'][0]:.4f} -> {r['soc'][-1]:.4f} (leaks)")
    assert_true(r["soc"][-1] > 0.0, "SOC stays physical (>0)")


def test_predict_interface():
    print("\n[Test 11] ComponentModel predict() interface + get_info()")
    _, cm = make_model()
    info = cm.get_info()
    for k in ["component_id", "component_name", "fidelity", "version", "inputs", "outputs"]:
        assert_true(k in info, f"get_info has '{k}'")
    assert_true(info["component_id"] == "EC029", "component_id == EC029")
    r = cm.predict({"current_A": 2.0, "soc0": 0.6, "dt": 5.0, "duration_s": 60.0})
    for key in ["t", "soc", "voltage", "current", "power", "temperature", "efficiency", "heat"]:
        assert_true(key in r, f"Output key '{key}' present")
    assert_true(len(r["t"]) == len(r["voltage"]) == len(r["temperature"]), "Arrays same length")


def test_benchmark():
    print("\n[Test 12] Benchmark: 1 h sim at dt=1 s")
    m, _ = make_model()
    t0 = time.perf_counter()
    m.simulate(2.0, soc0=0.8, T0=m.T_ref, dt=1.0, duration_s=3600.0)
    elapsed = time.perf_counter() - t0
    print(f"  3600 s simulation in {elapsed*1000:.1f} ms")
    assert_true(elapsed < 5.0, "Completes in < 5 s")


if __name__ == "__main__":
    tests = [
        test_ocv_range_and_plateau,
        test_ocv_monotone_increasing,
        test_coulomb_conservation,
        test_charge_discharge_symmetry,
        test_voltage_sign_convention,
        test_arrhenius_resistance,
        test_efficiency_bounds,
        test_thermal_balance_steady,
        test_overcharge_exotherm,
        test_self_discharge,
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
    print(f"EC029 NiMH F2a (Thevenin 2-RC electrothermal) -- {passed} passed, {failed} failed")
    print(f"{'='*60}")
    sys.exit(0 if failed == 0 else 1)
