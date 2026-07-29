"""
EC080 -- Phase Change Material (PCM) Storage -- F2a Enthalpy Method
Test suite: physics sanity, ODE convergence, edge cases.
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from model import PCMStorage_F2a
from predict import ComponentModel

PASS = "\u2713"
FAIL = "\u2717"


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
def test_enthalpy_temperature_roundtrip():
    print("\n[Test 1] Enthalpy-temperature roundtrip consistency")
    m, _ = make_model()
    # Solid phase
    T_solid = 320.0
    h = m.temperature_to_enthalpy(T_solid)
    T_back = m.enthalpy_to_temperature(h)
    assert_true(abs(T_back - T_solid) < 0.01, f"Solid roundtrip: {T_solid} -> h={h:.0f} -> {T_back:.2f}")
    # Liquid phase
    T_liquid = 350.0
    h = m.temperature_to_enthalpy(T_liquid)
    T_back = m.enthalpy_to_temperature(h)
    assert_true(abs(T_back - T_liquid) < 0.01, f"Liquid roundtrip: {T_liquid} -> h={h:.0f} -> {T_back:.2f}")


def test_phase_change_plateau():
    print("\n[Test 2] Temperature plateau during phase change")
    m, _ = make_model()
    # Enthalpies in the mushy zone should all give T_melt
    for h_frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
        h = h_frac * m.L_f
        T = m.enthalpy_to_temperature(h)
        assert_true(abs(T - m.T_melt) < 0.01, f"h={h:.0f} J/kg -> T={T:.2f} K == T_melt={m.T_melt:.2f}")


def test_liquid_fraction_range():
    print("\n[Test 3] Liquid fraction in [0, 1]")
    m, _ = make_model()
    assert_true(m.liquid_fraction(-1000) == 0.0, "Solid: lf=0")
    assert_true(m.liquid_fraction(m.L_f / 2) == 0.5, "Half melted: lf=0.5")
    assert_true(m.liquid_fraction(m.L_f + 1000) == 1.0, "Liquid: lf=1")


def test_charge_heats_up():
    print("\n[Test 4] Charging from cold: PCM heats up")
    m, _ = make_model()
    r = m.simulate(353.15, 0.5, 293.15, 10.0, 600.0, mode="charge")
    assert_true(r["T_mean"][-1] > r["T_mean"][0], f"T_mean increases: {r['T_mean'][0]:.1f} -> {r['T_mean'][-1]:.1f}")
    assert_true(r["E_stored_J"][-1] > r["E_stored_J"][0], "Energy stored increases during charge")


def test_discharge_cools_down():
    print("\n[Test 5] Discharging from hot: PCM cools down")
    m, _ = make_model()
    # Start fully liquid (above melt)
    r = m.simulate(313.15, 0.5, 350.0, 10.0, 600.0, mode="discharge")
    assert_true(r["T_mean"][-1] < r["T_mean"][0], f"T_mean decreases: {r['T_mean'][0]:.1f} -> {r['T_mean'][-1]:.1f}")


def test_energy_conservation():
    print("\n[Test 6] Energy conservation (approximate)")
    m, _ = make_model()
    r = m.simulate(353.15, 0.5, 293.15, 5.0, 300.0, mode="charge")
    # Energy stored should be positive and bounded by HTF capacity
    E_final = r["E_stored_J"][-1]
    E_init = r["E_stored_J"][0]
    delta_E = E_final - E_init
    # Max possible: m_dot * cp * dT * t
    Q_max = 0.5 * 4186.0 * (353.15 - 293.15) * 300.0
    assert_true(delta_E > 0, f"Energy gained: {delta_E/1e6:.2f} MJ")
    assert_true(delta_E < Q_max, f"Energy < max HTF input: {delta_E/1e6:.2f} < {Q_max/1e6:.2f} MJ")


def test_no_flow_heat_loss():
    print("\n[Test 7] No HTF flow: PCM loses heat to ambient")
    m, _ = make_model()
    r = m.simulate(293.15, 0.0, 350.0, 10.0, 600.0, mode="discharge")
    assert_true(r["T_mean"][-1] < 350.0, f"Cools toward ambient: T_final={r['T_mean'][-1]:.1f}")
    assert_true(r["T_mean"][-1] > m.T_amb, f"Stays above ambient: T_final={r['T_mean'][-1]:.1f} > {m.T_amb}")


def test_full_charge_liquid_fraction():
    print("\n[Test 8] Long charge: liquid fraction approaches 1")
    m, _ = make_model()
    r = m.simulate(363.15, 1.0, 293.15, 30.0, 7200.0, mode="charge")
    assert_true(r["lf_mean"][-1] > 0.8, f"lf_mean={r['lf_mean'][-1]:.3f} > 0.8 after long charge")


def test_htf_outlet_temperature():
    print("\n[Test 9] HTF outlet < inlet during charge")
    m, _ = make_model()
    r = m.simulate(353.15, 0.5, 293.15, 10.0, 300.0, mode="charge")
    # During charge, HTF gives up heat, so outlet < inlet
    for k in range(1, len(r["t"])):
        assert_true(r["T_htf_out"][k] <= 353.15 + 0.1,
                    f"T_htf_out[{k}]={r['T_htf_out'][k]:.1f} <= 353.15")


def test_predict_interface():
    print("\n[Test 10] ComponentModel predict() interface")
    _, cm = make_model()
    r = cm.predict({"T_htf_K": 353.15, "dt": 10.0, "duration_s": 60.0})
    for key in ["t", "T_nodes", "T_mean", "lf_mean", "E_stored_J", "Q_rate_W"]:
        assert_true(key in r, f"Key '{key}' in output")
    assert_true(len(r["t"]) == len(r["T_mean"]), "Arrays same length")


def test_benchmark():
    print("\n[Test 11] Benchmark: 3600s sim at dt=10")
    m, _ = make_model()
    t0 = time.perf_counter()
    m.simulate(353.15, 0.5, 293.15, 10.0, 3600.0)
    elapsed = time.perf_counter() - t0
    print(f"  3600s simulation in {elapsed*1000:.1f} ms")
    assert_true(elapsed < 30.0, "Completes in < 30 s")


if __name__ == "__main__":
    tests = [
        test_enthalpy_temperature_roundtrip,
        test_phase_change_plateau,
        test_liquid_fraction_range,
        test_charge_heats_up,
        test_discharge_cools_down,
        test_energy_conservation,
        test_no_flow_heat_loss,
        test_full_charge_liquid_fraction,
        test_htf_outlet_temperature,
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
    print(f"EC080 PCM Storage F2a -- Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    sys.exit(0 if failed == 0 else 1)
