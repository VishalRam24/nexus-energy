"""EC040 -- Hydrogen-Bromine Flow Battery -- F1a SOC-only -- Test Suite"""

import sys
import os
import time
import math

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
    assert_true(info["ec_id"] == "EC040", "ec_id == EC040")
    assert_true(info["fidelity"] == "F1a", "fidelity == F1a")


def test_output_keys():
    print("Test: output keys")
    m = ComponentModel()
    r = m.predict({"soc": 0.5, "current": 30.0})
    for k in ["V_cell_ocv", "V_stack_ocv", "V_stack_terminal", "P_stack", "SOC_new", "efficiency"]:
        assert_true(k in r, f"key '{k}' in output")


def test_nernst_midpoint():
    print("Test: Nernst OCV at SOC=0.5 equals E0=1.09V")
    m = ComponentModel()
    r = m.predict({"soc": 0.5, "current": 0.0})
    assert_true(abs(r["V_cell_ocv"] - 1.09) < 1e-5,
                f"V_cell_ocv@SOC=0.5 = 1.09V (got {r['V_cell_ocv']:.6f})")


def test_stack_voltage_proportional():
    print("Test: V_stack_ocv = 30 * V_cell_ocv")
    m = ComponentModel()
    r = m.predict({"soc": 0.5, "current": 0.0})
    assert_true(abs(r["V_stack_ocv"] - 30 * r["V_cell_ocv"]) < 1e-4,
                "V_stack_ocv = 30*V_cell_ocv")


def test_nernst_increases_with_soc():
    print("Test: Nernst OCV increases with SOC")
    m = ComponentModel()
    r_lo = m.predict({"soc": 0.2, "current": 0.0})
    r_hi = m.predict({"soc": 0.8, "current": 0.0})
    assert_true(r_hi["V_cell_ocv"] > r_lo["V_cell_ocv"], "OCV increases with SOC")


def test_n2_smaller_nernst_slope():
    """n=2 gives half the Nernst slope compared to n=1."""
    print("Test: n=2 gives smaller OCV swing than n=1 Fe-Cr")
    m = ComponentModel()
    # Swing SOC=0.2->0.8: 2 * RT/(2F)*ln(0.8/0.2) = RT/F*ln(4)
    # (symmetric around SOC=0.5, so full swing = 2 * one-sided term)
    RT_over_nF = 8.314 * 298.15 / (2 * 96485.0)
    expected_delta = 2.0 * RT_over_nF * math.log(0.8 / 0.2)
    r_lo = m.predict({"soc": 0.2, "current": 0.0})
    r_hi = m.predict({"soc": 0.8, "current": 0.0})
    actual_delta = r_hi["V_cell_ocv"] - r_lo["V_cell_ocv"]
    assert_true(abs(actual_delta - expected_delta) < 1e-6,
                f"OCV swing = 2*RT/(2F)*ln(4) = {expected_delta:.6f}V (got {actual_delta:.6f})")


def test_voltage_drops_under_current():
    print("Test: terminal voltage drops under discharge current")
    m = ComponentModel()
    r0 = m.predict({"soc": 0.5, "current": 0.0})
    r1 = m.predict({"soc": 0.5, "current": 50.0})
    assert_true(r1["V_stack_terminal"] < r0["V_stack_ocv"],
                "V_terminal < V_ocv under discharge")


def test_power_positive_discharge():
    print("Test: power positive during discharge")
    m = ComponentModel()
    r = m.predict({"soc": 0.5, "current": 30.0})
    assert_true(r["P_stack"] > 0, f"P_stack > 0 (got {r['P_stack']:.2f})")


def test_efficiency_bounded():
    print("Test: efficiency in (0, 1]")
    m = ComponentModel()
    r = m.predict({"soc": 0.5, "current": 30.0})
    assert_true(0 < r["efficiency"] <= 1.0, f"efficiency in (0,1] (got {r['efficiency']:.4f})")


def test_soc_update():
    print("Test: SOC decreases during discharge with dt>0")
    m = ComponentModel()
    r = m.predict({"soc": 0.5, "current": 30.0, "dt": 3600.0})
    assert_true(r["SOC_new"] < 0.5, f"SOC_new < 0.5 after 1h discharge (got {r['SOC_new']:.4f})")


def test_benchmark():
    print("Test: benchmark 1000 predictions < 1s")
    m = ComponentModel()
    start = time.perf_counter()
    for i in range(1000):
        soc = 0.1 + (i % 80) * 0.01
        m.predict({"soc": soc, "current": float(i % 80)})
    elapsed = time.perf_counter() - start
    print(f"    1000 predictions in {elapsed*1000:.1f} ms")
    assert_true(elapsed < 1.0, "1000 predictions complete in < 1s")


if __name__ == "__main__":
    tests = [
        test_instantiation,
        test_output_keys,
        test_nernst_midpoint,
        test_stack_voltage_proportional,
        test_nernst_increases_with_soc,
        test_n2_smaller_nernst_slope,
        test_voltage_drops_under_current,
        test_power_positive_discharge,
        test_efficiency_bounded,
        test_soc_update,
        test_benchmark,
    ]
    p = f = 0
    for t in tests:
        try:
            t()
            p += 1
        except Exception:
            f += 1
    print(f"\nEC040 F1a -- {p} passed, {f} failed")
    sys.exit(0 if f == 0 else 1)
