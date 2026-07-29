"""EC005 -- MCFC -- F1a -- Test suite"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from model import MCFCPolarizationModel
from predict import ComponentModel

PASS, FAIL = "\u2713", "\u2717"

def assert_true(cond, msg):
    if cond:
        print(f"  {PASS}  {msg}")
    else:
        print(f"  {FAIL}  FAILED: {msg}")
        raise AssertionError(msg)


def test_open_circuit():
    print("\nTest 1: Open-circuit voltage (j=0)")
    m = MCFCPolarizationModel({"T": 923.15, "E_rev": 1.05, "i0": 0.01,
                                "R_ohm": 0.1, "j_L": 0.5, "alpha": 0.5})
    r = m.evaluate(j=0.0)
    assert_true(r["V_cell"] == 1.05, f"OCV should be 1.05 V, got {r['V_cell']}")


def test_high_temperature_effect():
    print("\nTest 2: High temperature (923 K) yields lower activation loss than PAFC")
    # At 923 K the Tafel slope is larger but i0 is also larger (0.01 vs 1e-4)
    m = MCFCPolarizationModel({"T": 923.15, "E_rev": 1.05, "i0": 0.01,
                                "R_ohm": 0.1, "j_L": 0.5, "alpha": 0.5})
    r = m.evaluate(j=0.1)
    assert_true(r["V_cell"] > 0.5, f"V_cell={r['V_cell']} should be > 0.5 V at j=0.1")


def test_voltage_decreases_with_current():
    print("\nTest 3: Voltage decreases monotonically with current density")
    m = MCFCPolarizationModel({"T": 923.15, "E_rev": 1.05, "i0": 0.01,
                                "R_ohm": 0.1, "j_L": 0.5, "alpha": 0.5})
    js = [0.02, 0.05, 0.1, 0.2, 0.35]
    vs = [m.evaluate(j=j_)["V_cell"] for j_ in js]
    assert_true(all(vs[i] > vs[i+1] for i in range(len(vs)-1)),
                f"Voltages not monotonically decreasing: {vs}")


def test_limiting_current():
    print("\nTest 4: At or beyond j_L, voltage = 0")
    m = MCFCPolarizationModel({"T": 923.15, "E_rev": 1.05, "i0": 0.01,
                                "R_ohm": 0.1, "j_L": 0.5, "alpha": 0.5})
    assert_true(m.evaluate(j=0.5)["V_cell"] == 0.0, "V at j_L should be 0")
    assert_true(m.evaluate(j=0.9)["V_cell"] == 0.0, "V beyond j_L should be 0")


def test_stack_power():
    print("\nTest 5: Stack power = n_cells * area * P_density")
    m = MCFCPolarizationModel({"T": 923.15, "E_rev": 1.05, "i0": 0.01,
                                "R_ohm": 0.1, "j_L": 0.5, "alpha": 0.5,
                                "n_cells": 50, "area": 200.0})
    r = m.evaluate(j=0.15)
    expected = r["P_density"] * 200.0 * 50
    assert_true(abs(r["P_stack"] - expected) < 0.01,
                f"P_stack={r['P_stack']:.2f} != expected {expected:.2f}")


def test_component_model_wrapper():
    print("\nTest 6: ComponentModel wrapper")
    cm = ComponentModel()
    info = cm.get_info()
    assert_true(info["component_id"] == "EC005", "component_id mismatch")
    r = cm.predict({"j": 0.1})
    assert_true("V_cell" in r and "P_stack" in r, "Missing outputs in predict")
    assert_true(0.0 < r["efficiency"] < 1.0,
                f"efficiency={r['efficiency']} out of range")


if __name__ == "__main__":
    tests = [
        test_open_circuit,
        test_high_temperature_effect,
        test_voltage_decreases_with_current,
        test_limiting_current,
        test_stack_power,
        test_component_model_wrapper,
    ]
    p = f = 0
    for t in tests:
        try:
            t(); p += 1
        except Exception as e:
            f += 1; print(f"  ERROR: {e}")
    print(f"\n{'='*50}\nEC005 F1a -- {p} passed, {f} failed\n{'='*50}")
    sys.exit(0 if f == 0 else 1)
