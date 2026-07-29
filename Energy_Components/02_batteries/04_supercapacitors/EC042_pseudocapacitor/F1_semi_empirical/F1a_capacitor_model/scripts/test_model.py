"""EC042 -- Pseudocapacitor -- F1a Capacitor Model -- Test Suite"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(__file__))
from predict import ComponentModel

PASS, FAIL = "\u2713", "\u2717"


def assert_true(condition, message):
    if condition:
        print(f"  {PASS}  {message}")
    else:
        print(f"  {FAIL}  FAILED: {message}")
        raise AssertionError(message)


def test_instantiation():
    print("Test: instantiation")
    m = ComponentModel()
    assert_true(m is not None, "ComponentModel instantiates")
    info = m.get_info()
    assert_true(info["ec_id"] == "EC042", "ec_id == EC042")
    assert_true(info["fidelity"] == "F1a", "fidelity == F1a")


def test_output_keys():
    print("Test: output keys")
    m = ComponentModel()
    r = m.predict({"V0": 2.7, "current": 100.0})
    for k in ["V_terminal", "V_new", "E_stored_J", "E_stored_Wh", "P_output", "SOC", "efficiency"]:
        assert_true(k in r, f"key '{k}' in output")


def test_esr_voltage_drop():
    print("Test: V_terminal = V0 - I*R_esr")
    m = ComponentModel()
    V0 = 2.7
    I = 100.0
    R_esr = 0.005
    r = m.predict({"V0": V0, "current": I})
    expected = V0 - I * R_esr
    assert_true(abs(r["V_terminal"] - expected) < 1e-6,
                f"V_terminal = V0 - I*R_esr = {expected:.6f}V (got {r['V_terminal']:.6f})")


def test_energy_at_full_charge():
    print("Test: E_stored = 0.5*C*V^2 at full charge")
    m = ComponentModel()
    C = 500.0
    V = 2.7
    I = 0.0
    r = m.predict({"V0": V, "current": I})
    expected_J = 0.5 * C * V ** 2
    assert_true(abs(r["E_stored_J"] - expected_J) < 1.0,
                f"E_stored_J at V_max ~ {expected_J:.1f}J (got {r['E_stored_J']:.1f})")


def test_soc_at_full_voltage():
    print("Test: SOC ~ 1.0 at V_max")
    m = ComponentModel()
    r = m.predict({"V0": 2.7, "current": 0.0})
    assert_true(r["SOC"] > 0.99, f"SOC ~ 1.0 at V_max (got {r['SOC']:.4f})")


def test_voltage_decreases_with_discharge():
    print("Test: V_new < V0 after discharge time step")
    m = ComponentModel()
    r = m.predict({"V0": 2.7, "current": 100.0, "dt": 10.0})
    assert_true(r["V_new"] < 2.7, f"V_new < 2.7 after discharge (got {r['V_new']:.4f})")


def test_power_positive_discharge():
    print("Test: P_output > 0 during discharge")
    m = ComponentModel()
    r = m.predict({"V0": 2.7, "current": 100.0})
    assert_true(r["P_output"] > 0, f"P_output > 0 (got {r['P_output']:.2f})")


def test_efficiency_below_one():
    print("Test: efficiency < 1 with ESR losses")
    m = ComponentModel()
    r = m.predict({"V0": 2.7, "current": 100.0})
    assert_true(r["efficiency"] < 1.0, f"efficiency < 1 with ESR (got {r['efficiency']:.4f})")
    assert_true(r["efficiency"] > 0.0, "efficiency > 0")


def test_zero_current_no_esr_drop():
    print("Test: zero current -> no ESR drop")
    m = ComponentModel()
    r = m.predict({"V0": 2.0, "current": 0.0})
    assert_true(abs(r["V_terminal"] - 2.0) < 1e-9, "V_terminal = V0 at I=0")


def test_energy_in_wh():
    print("Test: E_stored_Wh = E_stored_J / 3600")
    m = ComponentModel()
    r = m.predict({"V0": 2.0, "current": 0.0})
    # Allow 1e-6 tolerance due to rounding in individual fields
    assert_true(abs(r["E_stored_Wh"] - r["E_stored_J"] / 3600.0) < 1e-5,
                "E_Wh = E_J / 3600")


def test_benchmark():
    print("Test: benchmark 1000 predictions < 1s")
    m = ComponentModel()
    start = time.perf_counter()
    for i in range(1000):
        V0 = 0.5 + (i % 220) * 0.01
        m.predict({"V0": V0, "current": float(i % 200), "dt": 1.0})
    elapsed = time.perf_counter() - start
    print(f"    1000 predictions in {elapsed*1000:.1f} ms")
    assert_true(elapsed < 1.0, "1000 predictions complete in < 1s")


if __name__ == "__main__":
    tests = [
        test_instantiation,
        test_output_keys,
        test_esr_voltage_drop,
        test_energy_at_full_charge,
        test_soc_at_full_voltage,
        test_voltage_decreases_with_discharge,
        test_power_positive_discharge,
        test_efficiency_below_one,
        test_zero_current_no_esr_drop,
        test_energy_in_wh,
        test_benchmark,
    ]
    p = f = 0
    for t in tests:
        try:
            t()
            p += 1
        except Exception:
            f += 1
    print(f"\nEC042 F1a -- {p} passed, {f} failed")
    sys.exit(0 if f == 0 else 1)
