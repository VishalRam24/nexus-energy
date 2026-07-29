"""
EC086 — Electric Boiler / Resistance Heater — F1a Constant Efficiency
Test suite: physics sanity checks, edge cases, benchmark.
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from model import ElectricBoilerModel
from predict import ComponentModel

DEFAULT_PARAMS = {
    "P_rated_kw":   30.0,
    "eta_nom":      0.99,
    "P_standby_kw": 0.02,
    "PLR_min":      0.0,
}

PASS = "\u2713"
FAIL = "\u2717"


def assert_true(cond, msg):
    if cond:
        print(f"  {PASS}  {msg}")
    else:
        print(f"  {FAIL}  FAILED: {msg}")
        raise AssertionError(msg)


def make_model():
    return ElectricBoilerModel(DEFAULT_PARAMS)


def test_eta_below_unity():
    print("\n[Test 1] Efficiency <= 1.0 for all PLR")
    m = make_model()
    for PLR in np.linspace(0.0, 1.0, 50):
        eta = m.efficiency(PLR)
        assert_true(eta <= 1.0, f"eta={eta:.4f} <= 1.0 at PLR={PLR:.2f}")


def test_eta_at_full_load():
    print("\n[Test 2] Efficiency ~ eta_nom at PLR=1")
    m = make_model()
    eta = m.efficiency(1.0)
    assert_true(abs(eta - m.eta_nom) < 1e-9,
                f"eta(PLR=1)={eta:.6f} == eta_nom={m.eta_nom}")


def test_thermal_le_electrical():
    print("\n[Test 3] Q_out <= P_elec (energy conservation)")
    m = make_model()
    for PLR in np.linspace(0.0, 1.0, 30):
        Q = m.thermal_output_kw(PLR)
        P = m.electrical_input_kw(PLR)
        assert_true(Q <= P + 1e-9, f"Q={Q:.4f} <= P={P:.4f} at PLR={PLR:.2f}")


def test_linearity_in_plr():
    print("\n[Test 4] Q_out linear in PLR (above standby)")
    m = make_model()
    Q1 = m.thermal_output_kw(0.5) - m.eta_nom * m.P_standby
    Q2 = m.thermal_output_kw(1.0) - m.eta_nom * m.P_standby
    assert_true(abs(Q2 / Q1 - 2.0) < 1e-9,
                f"Q(PLR=1)/Q(PLR=0.5) (load only) = {Q2/Q1:.10f} == 2.0")


def test_zero_co2():
    print("\n[Test 5] Point-of-use CO2 == 0")
    cm = ComponentModel()
    out = cm.predict({"part_load_ratio": 0.7})
    assert_true(out["co2_emissions_g_per_kwh_th"] == 0.0,
                "CO2 emissions = 0 g/kWh_th at point of use")


def test_standby_load():
    print("\n[Test 6] At PLR=0, only standby is drawn")
    m = make_model()
    P0 = m.electrical_input_kw(0.0)
    assert_true(abs(P0 - m.P_standby) < 1e-12,
                f"P_in(PLR=0)={P0:.6f} == P_standby={m.P_standby}")


def test_predict_interface():
    print("\n[Test 7] ComponentModel predict() interface")
    cm = ComponentModel()
    out = cm.predict({"part_load_ratio": 0.6})
    required = [
        "thermal_output_kw", "electrical_input_kw", "efficiency",
        "PLR_effective", "co2_emissions_g_per_kwh_th",
    ]
    for k in required:
        assert_true(k in out, f"Output key '{k}' present")
    assert_true(out["efficiency"] <= 1.0, "efficiency <= 1.0")
    assert_true(out["thermal_output_kw"] <= out["electrical_input_kw"],
                "Q_out <= P_in")


def test_get_info():
    print("\n[Test 8] get_info() completeness")
    cm = ComponentModel()
    info = cm.get_info()
    for k in ["component_id", "component_name", "fidelity",
              "inputs", "outputs", "source"]:
        assert_true(k in info, f"'{k}' in get_info()")
    assert_true(info["component_id"] == "EC086", "EC ID is EC086")


def test_invalid_plr():
    print("\n[Test 9] PLR outside [0, 1] raises ValueError")
    m = make_model()
    try:
        m.evaluate(1.2)
        assert_true(False, "Should raise on PLR > 1")
    except ValueError:
        assert_true(True, "ValueError raised on PLR > 1")


def test_benchmark():
    print("\n[Test 10] Benchmark: 10,000 evaluate() calls")
    m = make_model()
    plr_vals = np.linspace(0.0, 1.0, 10000)
    t0 = time.perf_counter()
    for plr in plr_vals:
        m.evaluate(float(plr))
    elapsed = time.perf_counter() - t0
    print(f"  10,000 evaluations in {elapsed*1000:.1f} ms  "
          f"({elapsed/10000*1e6:.2f} µs/call)")
    assert_true(elapsed < 5.0, "10,000 calls complete in < 5 s")


if __name__ == "__main__":
    tests = [
        test_eta_below_unity,
        test_eta_at_full_load,
        test_thermal_le_electrical,
        test_linearity_in_plr,
        test_zero_co2,
        test_standby_load,
        test_predict_interface,
        test_get_info,
        test_invalid_plr,
        test_benchmark,
    ]
    passed = failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except AssertionError as e:
            failed += 1
            print(f"  ASSERTION ERROR: {e}")
        except Exception as e:
            failed += 1
            print(f"  UNEXPECTED ERROR in {t.__name__}: {e}")

    print(f"\n{'='*60}")
    print(f"EC086 Electric Boiler — Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    sys.exit(0 if failed == 0 else 1)
