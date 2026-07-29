"""EC004 -- PAFC -- F1a -- Test suite"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from model import PAFCPolarizationModel
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
    m = PAFCPolarizationModel({"T": 423.15, "E_rev": 1.1, "i0": 1e-4,
                                "R_ohm": 0.5, "j_L": 0.8, "alpha": 0.5})
    r = m.evaluate(j=0.0)
    assert_true(r["V_cell"] == 1.1, f"OCV should be E_rev=1.1, got {r['V_cell']}")
    assert_true(r["P_density"] == 0.0, f"Power density at OCV should be 0, got {r['P_density']}")


def test_mid_current():
    print("\nTest 2: Mid-range current density (j=0.3 A/cm^2)")
    m = PAFCPolarizationModel({"T": 423.15, "E_rev": 1.1, "i0": 1e-4,
                                "R_ohm": 0.5, "j_L": 0.8, "alpha": 0.5})
    r = m.evaluate(j=0.3)
    assert_true(0.4 < r["V_cell"] < 1.1, f"V_cell={r['V_cell']} should be in (0.4, 1.1)")
    assert_true(r["P_density"] > 0, f"P_density should be > 0, got {r['P_density']}")


def test_voltage_decreases_with_current():
    print("\nTest 3: Voltage decreases monotonically with current density")
    m = PAFCPolarizationModel({"T": 423.15, "E_rev": 1.1, "i0": 1e-4,
                                "R_ohm": 0.5, "j_L": 0.8, "alpha": 0.5})
    voltages = [m.evaluate(j=j_)["V_cell"] for j_ in [0.05, 0.1, 0.2, 0.4, 0.6]]
    assert_true(all(voltages[i] > voltages[i+1] for i in range(len(voltages)-1)),
                f"Voltages not monotonically decreasing: {voltages}")


def test_near_limiting_current():
    print("\nTest 4: Near-limiting current returns zero (j >= j_L)")
    m = PAFCPolarizationModel({"T": 423.15, "E_rev": 1.1, "i0": 1e-4,
                                "R_ohm": 0.5, "j_L": 0.8, "alpha": 0.5})
    r = m.evaluate(j=0.8)
    assert_true(r["V_cell"] == 0.0, f"V at j_L should be 0, got {r['V_cell']}")
    r2 = m.evaluate(j=1.0)
    assert_true(r2["V_cell"] == 0.0, f"V beyond j_L should be 0, got {r2['V_cell']}")


def test_stack_scaling():
    print("\nTest 5: Stack voltage scales with n_cells")
    m_single = PAFCPolarizationModel({"T": 423.15, "E_rev": 1.1, "i0": 1e-4,
                                       "R_ohm": 0.5, "j_L": 0.8, "alpha": 0.5,
                                       "n_cells": 1, "area": 100.0})
    m_stack  = PAFCPolarizationModel({"T": 423.15, "E_rev": 1.1, "i0": 1e-4,
                                       "R_ohm": 0.5, "j_L": 0.8, "alpha": 0.5,
                                       "n_cells": 10, "area": 100.0})
    r1 = m_single.evaluate(j=0.3)
    r10 = m_stack.evaluate(j=0.3)
    assert_true(abs(r10["V_stack"] - 10 * r1["V_cell"]) < 1e-9,
                f"Stack voltage should be 10x cell voltage")
    assert_true(abs(r10["P_stack"] - 10 * r1["P_stack"]) < 1e-6,
                f"Stack power should be 10x single-cell power")


def test_component_model_wrapper():
    print("\nTest 6: ComponentModel wrapper")
    cm = ComponentModel()
    info = cm.get_info()
    assert_true(info["component_id"] == "EC004", "component_id mismatch")
    assert_true(info["fidelity"].startswith("F1a"), "fidelity should start with F1a")
    r = cm.predict({"j": 0.2})
    assert_true("V_cell" in r, "predict output missing V_cell")
    assert_true("efficiency" in r, "predict output missing efficiency")
    assert_true(0.0 < r["efficiency"] < 1.0, f"efficiency={r['efficiency']} should be in (0,1)")


if __name__ == "__main__":
    tests = [
        test_open_circuit,
        test_mid_current,
        test_voltage_decreases_with_current,
        test_near_limiting_current,
        test_stack_scaling,
        test_component_model_wrapper,
    ]
    p = f = 0
    for t in tests:
        try:
            t()
            p += 1
        except Exception as e:
            f += 1
            print(f"  ERROR: {e}")
    print(f"\n{'='*50}")
    print(f"EC004 F1a -- {p} passed, {f} failed")
    print(f"{'='*50}")
    sys.exit(0 if f == 0 else 1)
