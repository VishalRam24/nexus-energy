"""
EC021 -- LTO Battery -- F2a Thevenin 1-RC ECM + Thermal
Test suite: physics sanity (coulomb conservation, monotonicity, OCV plateau),
RC dynamics, thermal balance, edge cases, predict() interface, benchmark.
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from model import LTO_ECM_F2a
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
def test_ocv_flat_plateau():
    print("\n[Test 1] LTO OCV: flat plateau + monotone increasing in SOC")
    m, _ = make_model()
    socs = np.linspace(0.1, 0.9, 9)
    v = m.ocv(socs)
    # All within LTO voltage window
    assert_true(np.all(v > 1.5) and np.all(v < 2.7), f"OCV in (1.5,2.7): [{v.min():.3f},{v.max():.3f}]")
    # Monotone non-decreasing (charge sense)
    assert_true(np.all(np.diff(v) > -1e-6), "OCV non-decreasing with SOC")
    # Flat plateau: variation across 20-80% SOC is small (< 0.25 V)
    mid = m.ocv(np.linspace(0.2, 0.8, 13))
    spread = float(np.ptp(mid))
    assert_true(spread < 0.25, f"Flat plateau: OCV spread over 20-80% = {spread:.3f} V < 0.25")


def test_coulomb_conservation():
    print("\n[Test 2] Coulomb counting conserves charge")
    m, _ = make_model()
    I = 2.9  # 1C discharge
    dur = 1200.0
    r = m.simulate(I, soc0=0.9, T0=298.15, dt=2.0, duration_s=dur)
    # Expected dSOC = I*dt / (3600*Q_eff); use mean T capacity
    Q_eff = float(m.effective_capacity(np.mean(r["temperature"])))
    expected_dsoc = I * dur / (3600.0 * Q_eff)
    actual_dsoc = r["soc"][0] - r["soc"][-1]
    assert_true(abs(actual_dsoc - expected_dsoc) < 1e-3,
                f"dSOC actual={actual_dsoc:.5f} vs expected={expected_dsoc:.5f}")


def test_soc_decreases_on_discharge():
    print("\n[Test 3] SOC decreases on discharge, increases on charge")
    m, _ = make_model()
    rd = m.simulate(2.9, soc0=0.5, dt=5.0, duration_s=300.0)
    rc = m.simulate(-2.9, soc0=0.5, dt=5.0, duration_s=300.0)
    assert_true(rd["soc"][-1] < rd["soc"][0], f"discharge SOC {rd['soc'][0]:.3f}->{rd['soc'][-1]:.3f}")
    assert_true(rc["soc"][-1] > rc["soc"][0], f"charge SOC {rc['soc'][0]:.3f}->{rc['soc'][-1]:.3f}")


def test_voltage_below_ocv_on_discharge():
    print("\n[Test 4] Terminal V < OCV on discharge, V > OCV on charge")
    m, _ = make_model()
    rd = m.simulate(5.0, soc0=0.5, dt=2.0, duration_s=120.0)
    # compare at the same instantaneous SOC
    ocv_d = m.ocv(rd["soc"])
    assert_true(np.all(rd["voltage"] <= ocv_d + 1e-6), "discharge: V_t <= OCV everywhere")
    rc = m.simulate(-5.0, soc0=0.5, dt=2.0, duration_s=120.0)
    ocv_c = m.ocv(rc["soc"])
    assert_true(np.all(rc["voltage"] >= ocv_c - 1e-6), "charge: V_t >= OCV everywhere")


def test_rc_relaxation():
    print("\n[Test 5] RC branch: V_RC relaxes to ~I*R1 then to 0 at rest")
    m, _ = make_model()
    # Load then rest: I=5A for 100s, then 0 for 200s
    def prof(t):
        return 5.0 if t < 100.0 else 0.0
    r = m.simulate(prof, soc0=0.6, dt=1.0, duration_s=300.0)
    idx_load = np.argmin(np.abs(r["t"] - 99.0))
    # During load, V_RC should approach steady I*R1
    vrc_ss = 5.0 * float(m.R1(r["temperature"][idx_load]))
    assert_true(abs(r["v_rc"][idx_load] - vrc_ss) < 0.3 * vrc_ss,
                f"V_RC under load {r['v_rc'][idx_load]:.5f} ~ I*R1 {vrc_ss:.5f}")
    # After rest, V_RC decays toward 0
    assert_true(abs(r["v_rc"][-1]) < abs(r["v_rc"][idx_load]),
                f"V_RC relaxes: rest {r['v_rc'][-1]:.5f} < load {r['v_rc'][idx_load]:.5f}")


def test_time_constant_order():
    print("\n[Test 6] RC time constant in physical range (1-100 s)")
    m, _ = make_model()
    tau = float(m.tau1(298.15))
    assert_true(1.0 < tau < 100.0, f"tau1 = {tau:.2f} s in (1,100)")


def test_thermal_heats_under_load():
    print("\n[Test 7] Thermal ODE: cell heats up under high-rate load")
    m, _ = make_model()
    r = m.simulate(20.0, soc0=0.9, T0=298.15, dt=2.0, duration_s=600.0)
    assert_true(r["temperature"][-1] > 298.15, f"T heats: {r['temperature'][-1]:.3f} > 298.15 K")
    assert_true(r["temperature"][-1] < 360.0, f"T bounded: {r['temperature'][-1]:.3f} < 360 K")


def test_thermal_equilibrium_at_rest():
    print("\n[Test 8] At zero current, cell relaxes to ambient")
    m, _ = make_model()
    r = m.simulate(0.0, soc0=0.5, T0=320.0, dt=10.0, duration_s=5000.0)
    assert_true(abs(r["temperature"][-1] - m.T_amb) < 1.0,
                f"T -> ambient: {r['temperature'][-1]:.3f} ~ {m.T_amb:.2f} K")
    # Zero current, started at rest: V_RC stays ~0
    assert_true(abs(r["v_rc"][-1]) < 1e-6, f"V_RC stays 0 at rest: {r['v_rc'][-1]:.2e}")


def test_arrhenius_resistance():
    print("\n[Test 9] Resistance increases as temperature drops (Arrhenius)")
    m, _ = make_model()
    R_cold = float(m.R0(253.15))   # -20 C
    R_ref = float(m.R0(298.15))    # 25 C
    R_hot = float(m.R0(323.15))    # 50 C
    assert_true(R_cold > R_ref > R_hot, f"R0: cold {R_cold:.4f} > ref {R_ref:.4f} > hot {R_hot:.4f}")


def test_predict_interface():
    print("\n[Test 10] ComponentModel predict() interface")
    _, cm = make_model()
    r = cm.predict({"current_A": 2.9, "soc0": 0.8, "dt": 5.0, "duration_s": 60.0})
    for key in ["t", "soc", "voltage", "v_rc", "current", "temperature", "power", "heat_gen"]:
        assert_true(key in r, f"Key '{key}' in output")
    assert_true(len(r["t"]) == len(r["voltage"]) == len(r["soc"]), "Arrays same length")
    info = cm.get_info()
    assert_true(info["component_id"] == "EC021" and info["version"] == "1.0.0",
                "Metadata id/version correct")


def test_high_rate_capability():
    print("\n[Test 11] LTO high C-rate: 10C discharge stays above v_min")
    m, _ = make_model()
    # 10C ~ 29 A; short burst since LTO tolerates high rates
    r = m.simulate(29.0, soc0=0.9, T0=298.15, dt=1.0, duration_s=30.0)
    assert_true(np.all(r["voltage"] >= m.v_min), f"V_min held under 10C: min={r['voltage'].min():.3f} V")
    assert_true(r["voltage"].min() > 1.5, "Voltage stays in usable LTO window under 10C")


def test_benchmark():
    print("\n[Test 12] Benchmark: 3600 s sim at dt=1.0")
    m, _ = make_model()
    t0 = time.perf_counter()
    m.simulate(2.9, soc0=0.95, T0=298.15, dt=1.0, duration_s=3600.0)
    elapsed = time.perf_counter() - t0
    print(f"  3600s simulation in {elapsed*1000:.1f} ms")
    assert_true(elapsed < 5.0, "Completes in < 5 s")


if __name__ == "__main__":
    tests = [
        test_ocv_flat_plateau,
        test_coulomb_conservation,
        test_soc_decreases_on_discharge,
        test_voltage_below_ocv_on_discharge,
        test_rc_relaxation,
        test_time_constant_order,
        test_thermal_heats_under_load,
        test_thermal_equilibrium_at_rest,
        test_arrhenius_resistance,
        test_predict_interface,
        test_high_rate_capability,
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
    print(f"EC021 LTO F2a (1-RC ECM + thermal) -- Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    sys.exit(0 if failed == 0 else 1)
