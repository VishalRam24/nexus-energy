"""
EC085 — Natural Gas Boiler — F1a Part-Load Efficiency
Test suite: physics sanity checks, edge cases, benchmark.
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from model import NaturalGasBoilerModel
from predict import ComponentModel

DEFAULT_PARAMS = {
    "Q_rated":  50.0,    # kW
    "eta_nom":  0.95,    # condensing boiler
    "a0":       0.1,
    "a1":       0.9,
    "a2":       0.0,
    "PLR_min":  0.1,
    "LHV_gas":  36.6,    # MJ/m³
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
    return NaturalGasBoilerModel(DEFAULT_PARAMS)


# ---------------------------------------------------------------------------
# Test 1 — Polynomial constraint validation
# ---------------------------------------------------------------------------

def test_polynomial_constraint():
    print("\n[Test 1] Polynomial constraint a0+a1+a2=1")
    # Valid params
    m = make_model()
    assert_true(abs(m.a0 + m.a1 + m.a2 - 1.0) < 1e-9, "a0+a1+a2=1 for default params")

    # Invalid params should raise
    bad_params = dict(DEFAULT_PARAMS)
    bad_params["a2"] = 0.1   # now sum = 1.1
    try:
        NaturalGasBoilerModel(bad_params)
        assert_true(False, "Should have raised ValueError for bad coefficients")
    except ValueError:
        assert_true(True, "ValueError raised for a0+a1+a2 != 1.0")


# ---------------------------------------------------------------------------
# Test 2 — Efficiency at full load equals eta_nom
# ---------------------------------------------------------------------------

def test_eta_at_full_load():
    print("\n[Test 2] Efficiency = eta_nom at PLR=1 (no temp correction)")
    m = make_model()
    eta = m.efficiency(1.0)
    assert_true(abs(eta - m.eta_nom) < 1e-9,
                f"eta(PLR=1)={eta:.6f} == eta_nom={m.eta_nom}")


# ---------------------------------------------------------------------------
# Test 3 — Efficiency <= 1.0 at all PLR
# ---------------------------------------------------------------------------

def test_eta_below_unity():
    print("\n[Test 3] Efficiency <= 1.0 for all PLR")
    m = make_model()
    for PLR in np.linspace(0.0, 1.0, 50):
        eta = m.efficiency(PLR)
        assert_true(eta <= 1.0, f"eta={eta:.4f} <= 1.0 at PLR={PLR:.2f}")


# ---------------------------------------------------------------------------
# Test 4 — Fuel >= thermal output (energy conservation)
# ---------------------------------------------------------------------------

def test_fuel_geq_thermal():
    print("\n[Test 4] Fuel input >= thermal output (energy conservation)")
    m = make_model()
    for PLR in np.linspace(0.1, 1.0, 20):
        Q_out  = m.thermal_output_kw(PLR)
        Q_fuel = m.fuel_input_kw(PLR)
        assert_true(Q_fuel >= Q_out - 1e-9,
                    f"Q_fuel ({Q_fuel:.2f}) >= Q_out ({Q_out:.2f}) at PLR={PLR:.2f}")


# ---------------------------------------------------------------------------
# Test 5 — Efficiency drops at low PLR (compared to full-load)
# ---------------------------------------------------------------------------

def test_eta_drops_at_low_plr():
    print("\n[Test 5] Efficiency lower at PLR=0.1 than PLR=1.0")
    m = make_model()
    eta_min = m.efficiency(0.1)
    eta_full = m.efficiency(1.0)
    assert_true(eta_min < eta_full,
                f"eta(PLR=0.1)={eta_min:.4f} < eta(PLR=1)={eta_full:.4f}")


# ---------------------------------------------------------------------------
# Test 6 — Gas consumption proportional to fuel input
# ---------------------------------------------------------------------------

def test_gas_consumption_scaling():
    print("\n[Test 6] Gas consumption proportional to fuel input")
    m = make_model()
    PLR1, PLR2 = 0.5, 1.0
    V1 = m.gas_consumption_m3h(PLR1)
    V2 = m.gas_consumption_m3h(PLR2)
    Q1 = m.fuel_input_kw(PLR1)
    Q2 = m.fuel_input_kw(PLR2)
    # Ratio of gas = ratio of fuel input
    assert_true(abs(V2 / V1 - Q2 / Q1) < 1e-9,
                f"Gas ratio ({V2/V1:.6f}) == fuel ratio ({Q2/Q1:.6f})")


# ---------------------------------------------------------------------------
# Test 7 — Condensing correction: lower T → higher efficiency
# ---------------------------------------------------------------------------

def test_condensing_correction():
    print("\n[Test 7] Condensing correction: lower T -> higher efficiency")
    m = make_model()
    c_low  = m.condensing_correction(30.0)
    c_mid  = m.condensing_correction(55.0)
    c_high = m.condensing_correction(80.0)
    assert_true(c_low > c_mid,  f"correction(30°C)={c_low:.3f} > correction(55°C)={c_mid:.3f}")
    assert_true(c_mid > c_high, f"correction(55°C)={c_mid:.3f} > correction(80°C)={c_high:.3f}")
    assert_true(abs(c_mid - 1.0) < 1e-9, f"correction(55°C)=1.0 (reference point)")


# ---------------------------------------------------------------------------
# Test 8 — PLR_min clamping
# ---------------------------------------------------------------------------

def test_plr_min_clamping():
    print("\n[Test 8] PLR below PLR_min is clamped to PLR_min")
    m = make_model()
    # eta(0.05) should equal eta(PLR_min=0.1) since modulation clamps
    eta_005 = m.efficiency(0.05)
    eta_010 = m.efficiency(0.10)
    assert_true(abs(eta_005 - eta_010) < 1e-9,
                f"eta(0.05)={eta_005:.6f} == eta(0.10)={eta_010:.6f}")


# ---------------------------------------------------------------------------
# Test 9 — Thermal output linear in PLR
# ---------------------------------------------------------------------------

def test_thermal_output_linear():
    print("\n[Test 9] Thermal output linear in PLR")
    m = make_model()
    Q1 = m.thermal_output_kw(0.5)
    Q2 = m.thermal_output_kw(1.0)
    assert_true(abs(Q2 / Q1 - 2.0) < 1e-9,
                f"Q(PLR=1)/Q(PLR=0.5) = {Q2/Q1:.10f} == 2.0")


# ---------------------------------------------------------------------------
# Test 10 — ComponentModel predict() interface
# ---------------------------------------------------------------------------

def test_predict_interface():
    print("\n[Test 10] ComponentModel predict() interface")
    cm = ComponentModel()
    out = cm.predict({"part_load_ratio": 0.6, "supply_temp": 55.0})
    required = [
        "thermal_output_kw", "fuel_input_kw", "efficiency",
        "gas_consumption_m3h", "PLR_effective", "condensing_factor"
    ]
    for k in required:
        assert_true(k in out, f"Output key '{k}' present")
    assert_true(out["efficiency"] <= 1.0,           "efficiency <= 1.0")
    assert_true(out["fuel_input_kw"] >= out["thermal_output_kw"],
                "fuel >= thermal output")
    assert_true(out["gas_consumption_m3h"] > 0,     "gas consumption > 0")


# ---------------------------------------------------------------------------
# Test 11 — get_info() completeness
# ---------------------------------------------------------------------------

def test_get_info():
    print("\n[Test 11] get_info() completeness")
    cm = ComponentModel()
    info = cm.get_info()
    for k in ["component_id", "component_name", "fidelity", "inputs", "outputs", "source"]:
        assert_true(k in info, f"'{k}' in get_info()")


# ---------------------------------------------------------------------------
# Test 12 — Benchmark
# ---------------------------------------------------------------------------

def test_benchmark():
    print("\n[Test 12] Benchmark: 10,000 evaluate() calls")
    m = make_model()
    plr_vals = np.linspace(0.1, 1.0, 10000)
    t0 = time.perf_counter()
    for plr in plr_vals:
        m.evaluate(float(plr), 60.0)
    elapsed = time.perf_counter() - t0
    print(f"  10,000 evaluations in {elapsed*1000:.1f} ms  ({elapsed/10000*1e6:.2f} µs/call)")
    assert_true(elapsed < 5.0, "10,000 calls complete in < 5 s")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        test_polynomial_constraint,
        test_eta_at_full_load,
        test_eta_below_unity,
        test_fuel_geq_thermal,
        test_eta_drops_at_low_plr,
        test_gas_consumption_scaling,
        test_condensing_correction,
        test_plr_min_clamping,
        test_thermal_output_linear,
        test_predict_interface,
        test_get_info,
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

    print(f"\n{'='*50}")
    print(f"EC085 Natural Gas Boiler — Results: {passed} passed, {failed} failed")
    print(f"{'='*50}")
    sys.exit(0 if failed == 0 else 1)
