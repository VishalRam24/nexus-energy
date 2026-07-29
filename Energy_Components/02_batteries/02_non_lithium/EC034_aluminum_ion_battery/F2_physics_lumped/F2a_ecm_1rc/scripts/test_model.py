"""
EC034 -- Aluminum-Ion Battery -- F2a Thevenin ECM
Test suite: physics sanity (Coulomb conservation, OCV monotonicity,
efficiency bounds, thermal balance), edge cases, predict() interface, benchmark.
Run:  python3 scripts/test_model.py   (system python3, NOT pytest)
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from model import AluminumIonECM_F2a
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
    print("\n[Test 1] OCV(SOC) monotonically increasing")
    m, _ = make_model()
    soc = np.linspace(0.0, 1.0, 200)
    v = m.ocv(soc)
    diffs = np.diff(v)
    assert_true(np.all(diffs >= -1e-9), "OCV non-decreasing over full SOC range")
    assert_true(v[-1] > v[0], f"OCV(1)={v[-1]:.3f} > OCV(0)={v[0]:.3f}")


def test_ocv_plateau_range():
    print("\n[Test 2] Al-ion OCV in plateau band over working SOC (~1.5-2.45 V)")
    m, _ = make_model()
    # Working SOC window where the staging plateaus live; below ~0.2 the OCV
    # drops steeply toward the lower cutoff (steep low-SOC knee, Lin 2015).
    v = m.ocv(np.linspace(0.3, 0.95, 50))
    assert_true(np.all(v > 1.5) and np.all(v < m.v_max + 1e-6),
                f"OCV in [{v.min():.3f}, {v.max():.3f}] V within Al-ion plateaus")


def test_coulomb_conservation():
    print("\n[Test 3] Coulomb counting conserves charge (discharge)")
    m, _ = make_model()
    I = 1.0  # A discharge, capacity 1 Ah -> 1C
    dur = 1800.0  # 0.5 h
    r = m.simulate(I, soc0=0.95, dt=5.0, duration_s=dur)
    # Expected dSOC = -I*dur/(C_eff*3600); use isothermal-ish near T_ref
    C_eff = m.effective_capacity(r["temperature"].mean())
    expected = -I * dur / (C_eff * 3600.0)
    actual = r["soc"][-1] - r["soc"][0]
    assert_true(abs(actual - expected) < 5e-3,
                f"dSOC actual={actual:.4f} vs expected={expected:.4f}")


def test_coulomb_efficiency_charge():
    print("\n[Test 4] Charge stores less SOC than ideal (0<eta_I<1)")
    m, _ = make_model()
    assert_true(0.0 < m.coulomb_eff < 1.0, f"eta_I={m.coulomb_eff} in (0,1)")
    # charge same |I| and time; |dSOC_charge| should be eta_I * |dSOC_discharge|
    dur = 600.0
    rd = m.simulate(1.0, soc0=0.5, dt=5.0, duration_s=dur)
    rc = m.simulate(-1.0, soc0=0.5, dt=5.0, duration_s=dur)
    d_dis = abs(rd["soc"][-1] - rd["soc"][0])
    d_chg = abs(rc["soc"][-1] - rc["soc"][0])
    ratio = d_chg / d_dis
    assert_true(abs(ratio - m.coulomb_eff) < 0.02,
                f"charge/discharge SOC ratio={ratio:.4f} ~ eta_I={m.coulomb_eff}")


def test_charge_raises_soc():
    print("\n[Test 5] Negative current (charge) raises SOC")
    m, _ = make_model()
    r = m.simulate(-2.0, soc0=0.3, dt=5.0, duration_s=600.0)
    assert_true(r["soc"][-1] > r["soc"][0],
                f"SOC {r['soc'][0]:.3f} -> {r['soc'][-1]:.3f} on charge")


def test_voltage_polarization():
    print("\n[Test 6] Discharge V < OCV, charge V > OCV (polarization sign)")
    m, _ = make_model()
    rd = m.simulate(3.0, soc0=0.7, dt=2.0, duration_s=120.0)
    rc = m.simulate(-3.0, soc0=0.7, dt=2.0, duration_s=120.0)
    # after RC settle, compare last sample
    assert_true(rd["voltage"][-1] < rd["ocv"][-1] + 1e-6,
                f"discharge V={rd['voltage'][-1]:.3f} < OCV={rd['ocv'][-1]:.3f}")
    assert_true(rc["voltage"][-1] > rc["ocv"][-1] - 1e-6,
                f"charge V={rc['voltage'][-1]:.3f} > OCV={rc['ocv'][-1]:.3f}")


def test_efficiency_bounds():
    print("\n[Test 7] Efficiency strictly in (0, 1) under load")
    m, _ = make_model()
    for I in [1.0, 5.0, -1.0, -5.0]:
        r = m.simulate(I, soc0=0.6, dt=5.0, duration_s=300.0)
        eff = r["efficiency"]
        # ignore the t=0 transient where RC=0 may give eff=1 exactly
        eff_load = eff[1:]
        assert_true(np.all(eff_load > 0.0) and np.all(eff_load < 1.0),
                    f"I={I}: eff in ({eff_load.min():.4f}, {eff_load.max():.4f})")


def test_arrhenius_resistance():
    print("\n[Test 8] Arrhenius: R increases as T drops")
    m, _ = make_model()
    R_cold = m.R0(273.15)
    R_ref = m.R0(m.T_ref)
    R_hot = m.R0(323.15)
    assert_true(R_cold > R_ref > R_hot,
                f"R0: cold={R_cold:.4f} > ref={R_ref:.4f} > hot={R_hot:.4f} Ohm")


def test_thermal_balance():
    print("\n[Test 9] Thermal ODE: heats under load, bounded, approaches balance")
    m, _ = make_model()
    r = m.simulate(8.0, soc0=0.9, dt=2.0, duration_s=600.0)  # ~8C high rate
    T0, Tf = r["temperature"][0], r["temperature"][-1]
    assert_true(Tf > T0, f"T rises under load: {T0:.2f} -> {Tf:.2f} K")
    assert_true(Tf < 400.0, f"T bounded (no runaway): {Tf:.2f} K")
    # at steady balance Q_gen ~ hA*(T-Tamb); check heat_gen positive & finite
    assert_true(np.all(np.isfinite(r["heat_gen"])) and r["heat_gen"][-1] > 0,
                f"heat_gen finite & positive (={r['heat_gen'][-1]:.3f} W)")


def test_high_rate_capability():
    print("\n[Test 10] Very high rate (20C) completes without instability")
    m, _ = make_model()
    r = m.simulate(20.0, soc0=0.95, dt=0.5, duration_s=120.0)
    assert_true(np.all(np.isfinite(r["voltage"])), "voltage finite at 20C")
    assert_true(np.all(r["soc"] >= 0.0) and np.all(r["soc"] <= 1.0),
                "SOC stays in [0,1] at high rate")


def test_two_rc_mode():
    print("\n[Test 11] 2-RC mode runs and differs from 1-RC transient")
    m, _ = make_model()
    r1 = m.simulate(5.0, soc0=0.8, dt=1.0, duration_s=60.0)
    m.n_rc = 2
    r2 = m.simulate(5.0, soc0=0.8, dt=1.0, duration_s=60.0)
    m.n_rc = 1
    assert_true(np.all(np.isfinite(r2["voltage"])), "2-RC voltage finite")
    # extra RC branch adds polarization -> lower discharge voltage early on
    assert_true(r2["voltage"][3] <= r1["voltage"][3] + 1e-6,
                "2-RC adds polarization (V2 <= V1 in transient)")


def test_predict_interface():
    print("\n[Test 12] ComponentModel predict() interface + get_info()")
    _, cm = make_model()
    info = cm.get_info()
    for k in ["component_id", "component_name", "fidelity", "version", "inputs", "outputs"]:
        assert_true(k in info, f"get_info has '{k}'")
    assert_true(info["component_id"] == "EC034", "component_id == EC034")
    r = cm.predict({"current_A": 1.5, "soc0": 0.8, "dt": 5.0, "duration_s": 60.0})
    for key in ["t", "soc", "voltage", "ocv", "current", "power",
                "temperature", "efficiency", "heat_gen", "v_rc"]:
        assert_true(key in r, f"output key '{key}' present")
    assert_true(len(r["t"]) == len(r["voltage"]) == len(r["soc"]),
                "output arrays same length")


def test_benchmark():
    print("\n[Test 13] Benchmark: 600s sim at dt=1.0")
    m, _ = make_model()
    t0 = time.perf_counter()
    m.simulate(2.0, soc0=0.9, dt=1.0, duration_s=600.0)
    elapsed = time.perf_counter() - t0
    print(f"  600s simulation in {elapsed*1000:.1f} ms")
    assert_true(elapsed < 5.0, "completes in < 5 s")


if __name__ == "__main__":
    tests = [
        test_ocv_monotone,
        test_ocv_plateau_range,
        test_coulomb_conservation,
        test_coulomb_efficiency_charge,
        test_charge_raises_soc,
        test_voltage_polarization,
        test_efficiency_bounds,
        test_arrhenius_resistance,
        test_thermal_balance,
        test_high_rate_capability,
        test_two_rc_mode,
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
    print(f"EC034 Al-ion F2a ECM -- Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    sys.exit(0 if failed == 0 else 1)
