"""
EC126 -- Flywheel Energy Storage -- F2a Dynamic
Test suite: physics sanity, ODE convergence, edge cases.
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from model import FlywheelStorage_F2a
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


def test_energy_formula():
    print("\n[Test 1] E = 0.5 * J * omega^2")
    m, _ = make_model()
    omega = 3000.0
    E = m.stored_energy(omega)
    E_expected = 0.5 * m.J * omega**2
    assert_true(abs(E - E_expected) < 1.0, f"E={E:.0f} J == 0.5*J*w^2={E_expected:.0f}")


def test_soc_at_limits():
    print("\n[Test 2] SOC = 0 at omega_min, SOC = 1 at omega_max")
    m, _ = make_model()
    soc_max = m.soc(m.omega_max)
    soc_min = m.soc(m.omega_min)
    assert_true(abs(soc_max - 1.0) < 0.01, f"SOC(omega_max)={soc_max:.4f}")
    assert_true(abs(soc_min - 0.0) < 0.01, f"SOC(omega_min)={soc_min:.4f}")


def test_discharge_reduces_speed():
    print("\n[Test 3] Discharge reduces angular speed")
    m, _ = make_model()
    r = m.simulate(-200000, omega0=3665.0, dt=1.0, duration_s=60.0)
    assert_true(r["omega"][-1] < r["omega"][0],
                f"omega: {r['omega'][0]:.1f} -> {r['omega'][-1]:.1f}")


def test_charge_increases_speed():
    print("\n[Test 4] Charge increases angular speed")
    m, _ = make_model()
    r = m.simulate(200000, omega0=2750.0, dt=1.0, duration_s=60.0)
    assert_true(r["omega"][-1] > r["omega"][0],
                f"omega: {r['omega'][0]:.1f} -> {r['omega'][-1]:.1f}")


def test_self_discharge():
    print("\n[Test 5] Self-discharge (idle) -- speed decreases from friction")
    m, _ = make_model()
    r = m.simulate(0.0, omega0=3665.0, dt=1.0, duration_s=600.0)
    assert_true(r["omega"][-1] < r["omega"][0],
                f"Self-discharge: {r['omega'][0]:.1f} -> {r['omega'][-1]:.1f}")
    # Should not lose too much in 10 min
    soc_loss = r["SOC"][0] - r["SOC"][-1]
    assert_true(soc_loss < 0.1, f"SOC loss in 10 min: {soc_loss:.4f} (< 0.1)")


def test_energy_conservation_charge():
    print("\n[Test 6] Energy conservation during charge")
    m, _ = make_model()
    r = m.simulate(200000, omega0=2750.0, dt=0.5, duration_s=120.0)
    dE = r["E_stored"][-1] - r["E_stored"][0]
    E_input = 200000 * 120.0  # Approximate electrical energy in
    # Stored energy gain should be less than input (due to losses)
    assert_true(dE > 0, f"Energy gained: {dE/1e6:.2f} MJ")
    assert_true(dE < E_input, f"dE={dE:.0f} < E_input={E_input:.0f} (losses present)")


def test_friction_positive():
    print("\n[Test 7] Friction torque always positive")
    m, _ = make_model()
    for omega in [0, 100, 1000, 3665]:
        T_f = m.friction_torque(omega)
        assert_true(T_f >= 0, f"T_friction({omega})={T_f:.4f} >= 0")


def test_soc_bounded():
    print("\n[Test 8] SOC stays in [0, 1] during long discharge")
    m, _ = make_model()
    r = m.simulate(-250000, omega0=3665.0, dt=1.0, duration_s=1800.0)
    assert_true(np.all(r["SOC"] >= -0.01) and np.all(r["SOC"] <= 1.01),
                f"SOC range: [{r['SOC'].min():.4f}, {r['SOC'].max():.4f}]")


def test_predict_interface():
    print("\n[Test 9] ComponentModel predict() interface")
    _, cm = make_model()
    r = cm.predict({"P_command_W": -100000, "dt": 1.0, "duration_s": 10.0})
    for key in ["t", "omega", "E_stored", "SOC", "P_command", "P_loss", "efficiency"]:
        assert_true(key in r, f"Key '{key}' in output")
    assert_true(len(r["t"]) == len(r["omega"]), "Arrays same length")


def test_benchmark():
    print("\n[Test 10] Benchmark: 3600s sim at dt=0.1")
    m, _ = make_model()
    t0 = time.perf_counter()
    m.simulate(-200000, omega0=3665.0, dt=0.1, duration_s=3600.0)
    elapsed = time.perf_counter() - t0
    print(f"  3600s simulation in {elapsed*1000:.1f} ms")
    assert_true(elapsed < 30.0, "Completes in < 30 s")


if __name__ == "__main__":
    tests = [
        test_energy_formula,
        test_soc_at_limits,
        test_discharge_reduces_speed,
        test_charge_increases_speed,
        test_self_discharge,
        test_energy_conservation_charge,
        test_friction_positive,
        test_soc_bounded,
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
    print(f"EC126 Flywheel F2a -- Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    sys.exit(0 if failed == 0 else 1)
