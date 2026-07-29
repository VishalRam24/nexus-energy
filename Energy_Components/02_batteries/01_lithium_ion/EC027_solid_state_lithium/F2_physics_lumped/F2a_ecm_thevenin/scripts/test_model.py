"""
EC027 -- Solid-State Lithium Battery -- F2a Thevenin ECM
Test suite: Coulomb conservation, OCV monotonicity, Arrhenius R(T) sensitivity,
efficiency bounds, thermal ODE, predict() interface, benchmark.
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from model import SolidStateLiECM_F2a
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
    print("\n[Test 1] OCV(SOC) monotonically increasing on [0,1]")
    m, _ = make_model()
    soc = np.linspace(0.0, 1.0, 200)
    ocv = np.array([float(m.ocv(s)) for s in soc])
    diffs = np.diff(ocv)
    assert_true(np.all(diffs > -1e-9), "OCV non-decreasing across full SOC range")
    assert_true(ocv[-1] > ocv[0], f"OCV(1)={ocv[-1]:.3f} > OCV(0)={ocv[0]:.3f}")


def test_ocv_endpoints():
    print("\n[Test 2] OCV endpoints near voltage window")
    m, _ = make_model()
    assert_true(2.9 < float(m.ocv(0.0)) < 3.2, f"OCV(0)={float(m.ocv(0.0)):.3f} ~ V_min")
    assert_true(4.0 < float(m.ocv(1.0)) < 4.4, f"OCV(1)={float(m.ocv(1.0)):.3f} ~ V_max")


def test_coulomb_conservation():
    print("\n[Test 3] Coulomb conservation: dSOC matches integral of current")
    m, _ = make_model()
    I = 4.0  # A discharge
    dur = 1800.0  # 0.5 h
    r = m.simulate(I, soc0=0.9, T0=298.15, dt=10.0, duration_s=dur)
    dsoc_expected = I * dur / 3600.0 / m.C_cap   # Ah drawn / Ah capacity
    dsoc_actual = r["soc"][0] - r["soc"][-1]
    assert_true(abs(dsoc_actual - dsoc_expected) < 1e-3,
                f"dSOC actual={dsoc_actual:.5f} vs expected={dsoc_expected:.5f}")
    # charge then discharge same Coulombs -> SOC returns to start
    def cyc(t):
        return -4.0 if t < 900 else 4.0
    r2 = m.simulate(cyc, soc0=0.5, T0=298.15, dt=5.0, duration_s=1800.0)
    assert_true(abs(r2["soc"][-1] - 0.5) < 2e-3,
                f"SOC round-trips: end={r2['soc'][-1]:.5f} ~ 0.5")


def test_arrhenius_R_sensitivity():
    print("\n[Test 4] Strong R0(T) Arrhenius sensitivity (very poor cold)")
    m, _ = make_model()
    R_cold = float(m.R0(263.15))   # -10 C
    R_ref = float(m.R0(298.15))    # 25 C
    R_hot = float(m.R0(333.15))    # 60 C
    assert_true(R_cold > R_ref > R_hot, "R0 decreases with temperature (Arrhenius)")
    ratio = R_cold / R_hot
    assert_true(ratio > 3.0, f"R0(-10C)/R0(60C)={ratio:.2f} >> 1 (strong T-dependence)")
    # E_a must exceed a liquid-electrolyte scale (~20 kJ/mol)
    assert_true(m.E_a_R0 > 25000.0, f"E_a_R0={m.E_a_R0:.0f} J/mol > liquid-electrolyte scale")


def test_R0_dominant():
    print("\n[Test 5] Series SE ionic resistance dominates interfacial")
    m, _ = make_model()
    assert_true(m.R0_ref > m.R1_ref, f"R0_ref={m.R0_ref} > R1_ref={m.R1_ref} (SE ionic dominant)")


def test_voltage_below_ocv_on_discharge():
    print("\n[Test 6] Discharge terminal V < OCV; charge V > OCV")
    m, _ = make_model()
    r = m.simulate(4.0, soc0=0.7, T0=298.15, dt=2.0, duration_s=200.0)
    # mid-sim point (after RC settles)
    k = len(r["t"]) // 2
    assert_true(r["voltage"][k] < r["ocv"][k], "discharge: V < OCV")
    rc = m.simulate(-4.0, soc0=0.5, T0=298.15, dt=2.0, duration_s=200.0)
    k2 = len(rc["t"]) // 2
    assert_true(rc["voltage"][k2] > rc["ocv"][k2], "charge: V > OCV")


def test_efficiency_bounds():
    print("\n[Test 7] Coulombic/energy efficiency in (0,1)")
    m, _ = make_model()
    for I, T in [(2.0, 298.15), (8.0, 298.15), (4.0, 273.15)]:
        r = m.simulate(I, soc0=0.8, T0=T, dt=5.0, duration_s=600.0)
        eff = r["coulombic_efficiency"]
        assert_true(0.0 < eff < 1.0, f"I={I}A T={T}K eff={eff:.4f} in (0,1)")


def test_cold_penalty():
    print("\n[Test 8] Cold cell delivers less power (higher losses)")
    m, _ = make_model()
    r_cold = m.simulate(8.0, soc0=0.8, T0=263.15, T_amb=263.15, dt=2.0, duration_s=120.0)
    r_warm = m.simulate(8.0, soc0=0.8, T0=313.15, T_amb=313.15, dt=2.0, duration_s=120.0)
    k = 30
    assert_true(r_cold["voltage"][k] < r_warm["voltage"][k],
                f"V_cold={r_cold['voltage'][k]:.3f} < V_warm={r_warm['voltage'][k]:.3f}")
    assert_true(r_cold["coulombic_efficiency"] < r_warm["coulombic_efficiency"],
                "cold efficiency lower")


def test_thermal_ode_heats_on_discharge():
    print("\n[Test 9] Thermal ODE: high-rate discharge heats the cell")
    m, _ = make_model()
    r = m.simulate(15.0, soc0=0.9, T0=298.15, T_amb=298.15, dt=2.0, duration_s=300.0)
    assert_true(r["temperature"][-1] > 298.15, f"T_final={r['temperature'][-1]:.2f} > T0")
    assert_true(r["temperature"][-1] < 360.0, f"T_final={r['temperature'][-1]:.2f} bounded")


def test_rc_dynamics():
    print("\n[Test 10] RC branch relaxation: voltage recovers after current cutoff")
    m, _ = make_model()
    def pulse(t):
        return 10.0 if t < 60 else 0.0
    r = m.simulate(pulse, soc0=0.7, T0=298.15, dt=1.0, duration_s=180.0)
    i_load = np.argmin(np.abs(r["t"] - 59.0))
    i_rest = np.argmin(np.abs(r["t"] - 170.0))
    assert_true(r["voltage"][i_rest] > r["voltage"][i_load],
                "terminal V recovers during rest (RC relaxation)")


def test_predict_interface():
    print("\n[Test 11] ComponentModel predict() interface")
    _, cm = make_model()
    r = cm.predict({"current_A": 4.0, "soc0": 0.8, "dt": 5.0, "duration_s": 100.0})
    for key in ["t", "soc", "voltage", "current", "power", "temperature",
                "R0", "ocv", "coulombic_efficiency"]:
        assert_true(key in r, f"Key '{key}' in output")
    assert_true(len(r["t"]) == len(r["voltage"]) == len(r["soc"]), "arrays same length")
    info = cm.get_info()
    assert_true(info["component_id"] == "EC027", "get_info component_id")


def test_benchmark():
    print("\n[Test 12] Benchmark: 1 h sim at dt=1 s")
    m, _ = make_model()
    t0 = time.perf_counter()
    m.simulate(4.0, soc0=0.9, T0=298.15, dt=1.0, duration_s=3600.0)
    elapsed = time.perf_counter() - t0
    print(f"  3600 s simulation in {elapsed*1000:.1f} ms")
    assert_true(elapsed < 5.0, "completes in < 5 s")


if __name__ == "__main__":
    tests = [
        test_ocv_monotone,
        test_ocv_endpoints,
        test_coulomb_conservation,
        test_arrhenius_R_sensitivity,
        test_R0_dominant,
        test_voltage_below_ocv_on_discharge,
        test_efficiency_bounds,
        test_cold_penalty,
        test_thermal_ode_heats_on_discharge,
        test_rc_dynamics,
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
    print(f"EC027 Solid-State Li F2a Thevenin ECM -- Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    sys.exit(0 if failed == 0 else 1)
