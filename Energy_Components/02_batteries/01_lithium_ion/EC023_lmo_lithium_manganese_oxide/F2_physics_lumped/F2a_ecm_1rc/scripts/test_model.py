"""
EC023 -- LMO Battery -- F2a Thevenin ECM
Test suite: physics sanity (Coulomb conservation, OCV monotonicity, efficiency
bounds, thermal balance), edge cases, predict() interface, benchmark timing.
Run:  python3 scripts/test_model.py   (system python3, NOT pytest)
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from model import LMO_Thevenin_F2a
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
    print("\n[Test 1] OCV(SOC) strictly increasing on [0,1] (LMO dual-plateau)")
    m, _ = make_model()
    socs = np.linspace(0.0, 1.0, 200)
    ocv = m.ocv(socs)
    diffs = np.diff(ocv)
    assert_true(np.all(diffs > 0), "OCV strictly increases with SOC")
    assert_true(2.9 < ocv[0] < 3.2, f"OCV(0)={ocv[0]:.3f} V near v_min")
    assert_true(4.0 < ocv[-1] < 4.3, f"OCV(1)={ocv[-1]:.3f} V near 4.2 V")


def test_ocv_range_spinel():
    print("\n[Test 2] OCV stays in spinel window ~3.0-4.2 V")
    m, _ = make_model()
    ocv = m.ocv(np.linspace(0, 1, 100))
    assert_true(ocv.min() >= 2.9 and ocv.max() <= 4.25,
                f"OCV in [{ocv.min():.3f}, {ocv.max():.3f}] within spinel range")


def test_arrhenius_resistance():
    print("\n[Test 3] Arrhenius R(T): resistance rises as cell cools")
    m, _ = make_model()
    R_cold = m.R0(263.15)
    R_ref = m.R0(298.15)
    R_hot = m.R0(323.15)
    assert_true(R_cold > R_ref > R_hot, f"R(-10C)={R_cold:.4f} > R(25C)={R_ref:.4f} > R(50C)={R_hot:.4f}")
    assert_true(abs(R_ref - m.R0_ref) < 1e-9, "R0(T_ref) == R0_ref")


def test_voltage_below_ocv_on_discharge():
    print("\n[Test 4] Terminal V < OCV on discharge, V > OCV on charge")
    m, _ = make_model()
    soc, T = 0.6, 298.15
    # steady-state RC drops for a constant current
    I = 5.0
    vrc1 = I * m.R1(T)
    vrc2 = I * m.R2(T)
    v_dis = m.terminal_voltage(soc, I, T, vrc1, vrc2)
    v_chg = m.terminal_voltage(soc, -I, T, -vrc1, -vrc2)
    ocv = m.ocv(soc)
    assert_true(v_dis < ocv, f"V_dis={v_dis:.4f} < OCV={ocv:.4f}")
    assert_true(v_chg > ocv, f"V_chg={v_chg:.4f} > OCV={ocv:.4f}")


def test_coulomb_conservation():
    print("\n[Test 5] Coulomb counting: dSOC matches integrated charge")
    m, _ = make_model()
    # discharge 3 A for 600 s, no thermal drift influence on SOC
    r = m.simulate(3.0, soc0=0.9, T0=298.15, dt=1.0, duration_s=600.0)
    dsoc = r["soc"][0] - r["soc"][-1]
    expected = 3.0 * 600.0 / (m.Q_cap * 3600.0)   # discharge: eta=1
    assert_true(abs(dsoc - expected) < 1e-3,
                f"dSOC={dsoc:.5f} matches Q/(Cap)={expected:.5f}")


def test_coulombic_efficiency_loss():
    print("\n[Test 6] Charge/discharge round-trip loses (1-eta_c) of charge")
    m, _ = make_model()
    eta = m.eta_c
    assert_true(0.0 < eta < 1.0, f"0 < eta_c={eta} < 1")
    # charge 3 A for 100 s then discharge 3 A for 100 s -> net SOC deficit
    def prof(t):
        return -3.0 if t < 100.0 else 3.0
    r = m.simulate(prof, soc0=0.5, T0=298.15, dt=0.5, duration_s=200.0)
    # charge added eta*3*100, discharge removed 3*100 -> net loss
    net = r["soc"][-1] - r["soc"][0]
    assert_true(net < 0, f"Net SOC change {net:.5f} < 0 (round-trip loss)")


def test_efficiency_bounds():
    print("\n[Test 7] Instantaneous efficiency in (0, 1)")
    _, cm = make_model()
    for I in [5.0, -5.0, 2.0]:
        r = cm.predict({"current_A": I, "soc0": 0.6, "duration_s": 60.0, "dt": 2.0})
        eff = r["efficiency"]
        assert_true(np.all((eff > 0) & (eff < 1.0)),
                    f"eff in (0,1) for I={I} (min={eff.min():.4f}, max={eff.max():.4f})")


def test_thermal_heats_on_load():
    print("\n[Test 8] Thermal ODE: cell warms under high discharge")
    m, _ = make_model()
    r = m.simulate(12.0, soc0=0.8, T0=298.15, dt=1.0, duration_s=300.0)
    assert_true(r["temperature"][-1] > 298.15, f"T_final={r['temperature'][-1]:.2f} > T0")
    assert_true(r["temperature"][-1] < 333.15, f"T_final={r['temperature'][-1]:.2f} < 60C")
    assert_true(np.all(r["heat_generation"] > 0), "Q_gen > 0 under discharge")


def test_thermal_balance_rest():
    print("\n[Test 9] At rest (I=0) cell relaxes to ambient, no heating")
    m, _ = make_model()
    r = m.simulate(0.0, soc0=0.7, T0=310.0, dt=2.0, duration_s=1200.0)
    assert_true(abs(r["temperature"][-1] - m.T_amb) < 1.0,
                f"T_final={r['temperature'][-1]:.2f} -> ambient {m.T_amb:.2f} K")
    assert_true(np.allclose(r["heat_generation"], 0.0, atol=1e-9),
                "Q_gen == 0 at I=0 (no Joule, no entropic)")
    assert_true(abs(r["soc"][-1] - 0.7) < 1e-9, "SOC unchanged at rest")


def test_rc_relaxation():
    print("\n[Test 10] RC voltage relaxes to I*R at steady state, decays at rest")
    m, _ = make_model()
    T = 298.15
    r = m.simulate(5.0, soc0=0.8, T0=T, dt=1.0, duration_s=2000.0)
    # after ~2000 s (>> tau2~240s), v_rc1 ~ I*R1, v_rc2 ~ I*R2
    assert_true(abs(r["v_rc1"][-1] - 5.0 * m.R1(r["temperature"][-1])) < 5e-3,
                "v_rc1 -> I*R1 at steady state")
    assert_true(abs(r["v_rc2"][-1] - 5.0 * m.R2(r["temperature"][-1])) < 5e-3,
                "v_rc2 -> I*R2 at steady state")


def test_predict_interface():
    print("\n[Test 11] ComponentModel predict()/get_info() interface")
    _, cm = make_model()
    info = cm.get_info()
    for k in ["component_id", "component_name", "fidelity", "version", "inputs", "outputs"]:
        assert_true(k in info, f"get_info has '{k}'")
    assert_true(info["component_id"] == "EC023", "component_id == EC023")
    r = cm.predict({"current_A": 3.0, "soc0": 0.85, "duration_s": 100.0, "dt": 5.0})
    for key in ["t", "soc", "voltage", "current", "power", "efficiency",
                "temperature", "heat_generation", "v_rc1", "v_rc2"]:
        assert_true(key in r, f"output key '{key}' present")
    assert_true(len(r["t"]) == len(r["voltage"]) == len(r["soc"]), "arrays same length")


def test_benchmark():
    print("\n[Test 12] Benchmark: 3600 s (1 h) sim at dt=1.0")
    m, _ = make_model()
    t0 = time.perf_counter()
    m.simulate(3.0, soc0=0.95, T0=298.15, dt=1.0, duration_s=3600.0)
    elapsed = time.perf_counter() - t0
    print(f"  3600 s simulation in {elapsed*1000:.1f} ms")
    assert_true(elapsed < 5.0, "Completes in < 5 s")


if __name__ == "__main__":
    tests = [
        test_ocv_monotone,
        test_ocv_range_spinel,
        test_arrhenius_resistance,
        test_voltage_below_ocv_on_discharge,
        test_coulomb_conservation,
        test_coulombic_efficiency_loss,
        test_efficiency_bounds,
        test_thermal_heats_on_load,
        test_thermal_balance_rest,
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
    print(f"EC023 LMO F2a Thevenin ECM -- Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    sys.exit(0 if failed == 0 else 1)
