"""
EC030 -- Nickel-Cadmium Battery (NiCd) -- F2a Thevenin ECM
Test suite: Coulomb conservation, OCV monotonicity/plateau, efficiency bounds,
thermal balance, RC dynamics, Arrhenius R(T), predict() interface, benchmark.
Custom harness (NO pytest). Run: python3 scripts/test_model.py
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from model import NiCdTheveninF2a
from predict import ComponentModel

PASS = "✓"
FAIL = "✗"


def assert_true(cond, msg):
    if cond:
        print(f"  {PASS}  {msg}")
    else:
        print(f"  {FAIL}  FAILED: {msg}")
        raise AssertionError(msg)


def make_model(n_rc=2):
    cm = ComponentModel(n_rc=n_rc)
    return cm._model, cm


# ---------------------------------------------------------------------------
def test_ocv_monotonic_and_plateau():
    print("\n[Test 1] OCV monotonically increasing in SOC + NiCd ~1.2 V plateau")
    m, _ = make_model()
    soc = np.linspace(0.0, 1.0, 200)
    v = m.ocv(soc)
    diffs = np.diff(v)
    assert_true(np.all(diffs >= -1e-9), "OCV non-decreasing with SOC")
    # NiCd flat plateau ~1.2 V; mid-SOC OCV stays in a narrow ~1.05-1.30 V band
    # (reused EC030 F1b polynomial). Span across the plateau is small vs. the
    # 0.9->1.45 V terminal range, confirming the characteristic flatness.
    plateau = m.ocv(np.array([0.3, 0.5, 0.7]))
    assert_true(np.all((plateau > 1.05) & (plateau < 1.30)),
                f"Mid-SOC OCV in flat ~1.2 V plateau: {plateau.round(3)}")
    span = m.ocv(np.array([0.2, 0.8]))
    assert_true(abs(span[1] - span[0]) < 0.25,
                f"Plateau flatness: OCV span 0.2->0.8 = {abs(span[1]-span[0]):.3f} V")


def test_coulomb_conservation():
    print("\n[Test 2] Coulomb conservation: dSOC matches integrated charge")
    m, _ = make_model()
    I = 10.0  # 1C discharge
    dur = 1200.0
    r = m.simulate(I, soc0=1.0, dt=2.0, duration_s=dur)
    # Expected dSOC = integral(I/C_eff/3600). C_eff varies with T slightly;
    # use mean effective capacity over the run.
    C_eff = m.effective_capacity(r["temperature"]).mean()
    expected_dsoc = I * dur / (C_eff * 3600.0)
    actual_dsoc = r["soc"][0] - r["soc"][-1]
    rel = abs(actual_dsoc - expected_dsoc) / expected_dsoc
    assert_true(rel < 0.02, f"dSOC actual={actual_dsoc:.4f} vs expected={expected_dsoc:.4f} (rel {rel:.3%})")


def test_coulombic_efficiency_charge():
    print("\n[Test 3] Charge less effective than discharge (eta_c < 1)")
    m, _ = make_model()
    # Discharge then equal-Ah charge should NOT fully recover SOC.
    ddis = m.dSOC_dt(10.0, 298.15)   # discharge
    dcha = m.dSOC_dt(-10.0, 298.15)  # charge
    # |dcha| should be eta_c * |ddis|
    assert_true(abs(dcha) < abs(ddis), f"|dSOC charge|={abs(dcha):.3e} < |dSOC dis|={abs(ddis):.3e}")
    ratio = abs(dcha) / abs(ddis)
    assert_true(abs(ratio - m.eta_c) < 1e-6, f"charge/discharge ratio = eta_c = {ratio:.3f}")


def test_efficiency_bounds():
    print("\n[Test 4] Voltage efficiency strictly in (0,1) under load")
    m, _ = make_model()
    r = m.simulate(20.0, soc0=0.9, dt=2.0, duration_s=300.0)
    eff = r["efficiency"]
    assert_true(np.all((eff > 0.0) & (eff < 1.0)), f"all eff in (0,1); min={eff.min():.3f} max={eff.max():.3f}")


def test_terminal_below_ocv_discharge():
    print("\n[Test 5] Terminal V < OCV on discharge, > OCV on charge")
    m, _ = make_model()
    r_dis = m.simulate(15.0, soc0=0.8, dt=2.0, duration_s=200.0)
    assert_true(np.all(r_dis["voltage"] <= r_dis["ocv"] + 1e-6), "V_t <= OCV during discharge")
    r_cha = m.simulate(-15.0, soc0=0.3, dt=2.0, duration_s=200.0)
    # ignore clipping at v_max
    mask = r_cha["voltage"] < m.v_max - 1e-6
    assert_true(np.all(r_cha["voltage"][mask] >= r_cha["ocv"][mask] - 1e-6), "V_t >= OCV during charge")


def test_arrhenius_resistance():
    print("\n[Test 6] Arrhenius R(T): resistance falls as T rises")
    m, _ = make_model()
    r_cold = m.R0(258.15)
    r_ref = m.R0(298.15)
    r_hot = m.R0(323.15)
    assert_true(r_cold > r_ref > r_hot, f"R0: cold={r_cold:.4f} > ref={r_ref:.4f} > hot={r_hot:.4f}")
    assert_true(abs(r_ref - m.R0_ref) < 1e-9, "R0(T_ref) == R0_ref")


def test_thermal_balance_discharge_heats():
    print("\n[Test 7] Thermal ODE: high-rate discharge heats cell, bounded")
    m, _ = make_model()
    r = m.simulate(40.0, soc0=1.0, T0=298.15, dt=1.0, duration_s=300.0)
    assert_true(r["temperature"][-1] > 298.15, f"T rose to {r['temperature'][-1]:.2f} K (entropic+Joule)")
    assert_true(r["temperature"][-1] < 400.0, f"T bounded {r['temperature'][-1]:.2f} K < 400 K")


def test_thermal_steady_state_rest():
    print("\n[Test 8] Thermal: hot cell at rest relaxes toward ambient")
    m, _ = make_model()
    r = m.simulate(0.0, soc0=0.5, T0=330.0, dt=5.0, duration_s=3000.0)
    assert_true(r["temperature"][-1] < 330.0, "Resting hot cell cools")
    assert_true(abs(r["temperature"][-1] - m.T_amb) < 2.0,
                f"Approaches ambient: {r['temperature'][-1]:.2f} vs {m.T_amb:.2f} K")


def test_rc_relaxation():
    print("\n[Test 9] RC overpotential relaxes to zero after current removed")
    m, _ = make_model()
    def load(t):
        return 20.0 if t < 100.0 else 0.0
    r = m.simulate(load, soc0=0.8, dt=1.0, duration_s=600.0)
    # at end (500 s rest, >> tau2 ~120 s) RC voltages should be small
    assert_true(abs(r["V_rc1"][-1]) < 1e-3, f"V_rc1 relaxed: {r['V_rc1'][-1]:.2e} V")
    assert_true(abs(r["V_rc2"][-1]) < 1e-2, f"V_rc2 relaxed: {r['V_rc2'][-1]:.2e} V")
    # voltage recovers toward OCV after load removed
    assert_true(r["voltage"][-1] > r["voltage"][95], "Terminal V recovers after load removed")


def test_1rc_vs_2rc():
    print("\n[Test 10] 1-RC and 2-RC both run; 2-RC has extra dynamics")
    m1, _ = make_model(n_rc=1)
    m2, _ = make_model(n_rc=2)
    r1 = m1.simulate(20.0, soc0=0.9, dt=2.0, duration_s=200.0)
    r2 = m2.simulate(20.0, soc0=0.9, dt=2.0, duration_s=200.0)
    assert_true(np.all(r1["V_rc2"] == 0.0), "1-RC model has no second RC branch")
    assert_true(np.any(np.abs(r2["V_rc2"]) > 1e-4), "2-RC model has active diffusion branch")


def test_predict_interface():
    print("\n[Test 11] ComponentModel predict() interface + array consistency")
    _, cm = make_model()
    r = cm.predict({"current_A": 5.0, "soc0": 1.0, "dt": 2.0, "duration_s": 60.0})
    for key in ["t", "soc", "voltage", "current", "power", "efficiency",
                "temperature", "ocv", "V_rc1", "V_rc2", "heat_W"]:
        assert_true(key in r, f"Key '{key}' present")
    n = len(r["t"])
    assert_true(all(len(r[k]) == n for k in ["soc", "voltage", "temperature"]),
                "All time-series arrays same length")
    info = cm.get_info()
    assert_true(info["component_id"] == "EC030", "get_info component_id == EC030")


def test_benchmark():
    print("\n[Test 12] Benchmark: 1 h (3600 s) sim at dt=1.0")
    m, _ = make_model()
    t0 = time.perf_counter()
    m.simulate(10.0, soc0=1.0, dt=1.0, duration_s=3600.0)
    elapsed = time.perf_counter() - t0
    print(f"  3600 s simulation in {elapsed*1000:.1f} ms")
    assert_true(elapsed < 5.0, "Completes in < 5 s")


if __name__ == "__main__":
    tests = [
        test_ocv_monotonic_and_plateau,
        test_coulomb_conservation,
        test_coulombic_efficiency_charge,
        test_efficiency_bounds,
        test_terminal_below_ocv_discharge,
        test_arrhenius_resistance,
        test_thermal_balance_discharge_heats,
        test_thermal_steady_state_rest,
        test_rc_relaxation,
        test_1rc_vs_2rc,
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
    print(f"EC030 NiCd F2a Thevenin ECM -- Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    sys.exit(0 if failed == 0 else 1)
