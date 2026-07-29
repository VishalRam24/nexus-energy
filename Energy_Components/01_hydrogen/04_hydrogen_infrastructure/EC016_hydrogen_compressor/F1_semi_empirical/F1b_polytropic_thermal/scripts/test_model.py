"""
EC016 -- H2 Compressor -- F1b Polytropic Thermal
Test suite: physics sanity checks.

Critical test rule: tests must FAIL the model, not accommodate it.
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from model import H2CompressorThermalModel
from predict import ComponentModel

DEFAULT_PARAMS = {
    "compressor": {
        "n_stages": {"value": 4},
        "polytropic_index": {"value": 1.38},
        "eta_polytropic": {"value": 0.75},
        "eta_mech": {"value": 0.92},
        "T_inlet": {"value": 298.15},
        "P_inlet": {"value": 20.0},
        "P_outlet_max": {"value": 900.0},
        "intercooled": {"value": True},
        "intercooler_effectiveness": {"value": 0.85},
        "T_coolant": {"value": 298.15},
    },
    "hydrogen": {
        "molar_mass": {"value": 0.002016},
        "R_specific": {"value": 4124.2},
        "gamma": {"value": 1.41},
        "LHV": {"value": 120.0},
        "cp": {"value": 14307.0},
    },
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
    return H2CompressorThermalModel(DEFAULT_PARAMS)


# ---------------------------------------------------------------------------
# Test 1 -- Compressor work increases with T_inlet
# Physics: w_stage = (n/(n-1))*R_s*T_in*... -> linearly proportional to T_in
# Ref: Sdanghi et al. (2019): "inlet temperature directly proportional to work"
# ---------------------------------------------------------------------------

def test_work_increases_with_T_inlet():
    print("\n[Test 1] Shaft work increases with T_inlet")
    m = make_model()
    w_cold = m.specific_work(20.0, 900.0, T_inlet=273.15)
    w_warm = m.specific_work(20.0, 900.0, T_inlet=298.15)
    w_hot  = m.specific_work(20.0, 900.0, T_inlet=323.15)
    assert_true(w_warm > w_cold, f"w(298K)={w_warm/1e6:.4f} MJ/kg > w(273K)={w_cold/1e6:.4f}")
    assert_true(w_hot > w_warm,  f"w(323K)={w_hot/1e6:.4f} MJ/kg > w(298K)={w_warm/1e6:.4f}")


# ---------------------------------------------------------------------------
# Test 2 -- SEC increases with T_inlet
# ---------------------------------------------------------------------------

def test_sec_increases_with_T_inlet():
    print("\n[Test 2] SEC (kWh/kg) increases with T_inlet")
    m = make_model()
    sec_cold = m.sec_kwh_per_kg(20.0, 900.0, T_inlet=273.15)
    sec_hot  = m.sec_kwh_per_kg(20.0, 900.0, T_inlet=323.15)
    assert_true(sec_hot > sec_cold, f"SEC(323K)={sec_hot:.4f} > SEC(273K)={sec_cold:.4f}")


# ---------------------------------------------------------------------------
# Test 3 -- Discharge temperature increases with T_inlet
# ---------------------------------------------------------------------------

def test_T_discharge_increases_with_T_inlet():
    print("\n[Test 3] Final discharge T increases with T_inlet")
    m = make_model()
    T_out_cold = m.final_discharge_temperature(20.0, 900.0, T_inlet=273.15)
    T_out_hot  = m.final_discharge_temperature(20.0, 900.0, T_inlet=323.15)
    assert_true(T_out_hot > T_out_cold,
                f"T_out(323K)={T_out_hot:.2f} K > T_out(273K)={T_out_cold:.2f} K")


# ---------------------------------------------------------------------------
# Test 4 -- Stage discharge T > stage inlet T (compression heats the gas)
# ---------------------------------------------------------------------------

def test_stage_temperatures_increase():
    print("\n[Test 4] Each stage discharges hotter than it receives")
    m = make_model()
    prof = m.stage_temperature_profile(20.0, 900.0)
    T_in   = prof["T_in_stage"]
    T_disc = prof["T_discharge"]
    for k in range(m.N):
        assert_true(T_disc[k] > T_in[k],
                    f"Stage {k+1}: T_disc={T_disc[k]:.2f} > T_in={T_in[k]:.2f}")


# ---------------------------------------------------------------------------
# Test 5 -- After-intercooler T is between T_discharge and T_coolant
# ---------------------------------------------------------------------------

def test_intercooler_reduces_temperature():
    print("\n[Test 5] Intercooler reduces temperature (T_cool <= T_ic <= T_disc)")
    m = make_model()
    prof = m.stage_temperature_profile(20.0, 900.0)
    T_disc = prof["T_discharge"]
    T_ic   = prof["T_after_ic"]
    T_cool = m.T_cool
    for k in range(m.N - 1):  # last stage no intercooler
        assert_true(T_cool <= T_ic[k] <= T_disc[k] + 0.01,
                    f"Stage {k+1}: {T_cool:.1f} <= T_ic={T_ic[k]:.2f} <= T_disc={T_disc[k]:.2f}")


# ---------------------------------------------------------------------------
# Test 6 -- Perfect intercooler (eps=1) reduces work versus no intercooling (eps=0)
# Physics: perfect intercooling -> all stages start at T_inlet -> minimum work
# ---------------------------------------------------------------------------

def test_perfect_intercooler_minimum_work():
    print("\n[Test 6] Perfect intercooling (eps=1) gives lower work than no intercooling (eps=0)")
    m = make_model()
    w_perfect = m.specific_work(20.0, 900.0, eps_ic=1.0)
    w_none    = m.specific_work(20.0, 900.0, eps_ic=0.0)
    assert_true(w_perfect < w_none,
                f"w(eps=1)={w_perfect/1e6:.4f} MJ/kg < w(eps=0)={w_none/1e6:.4f} MJ/kg")


# ---------------------------------------------------------------------------
# Test 7 -- Compression efficiency < 1
# ---------------------------------------------------------------------------

def test_efficiency_below_unity():
    print("\n[Test 7] Compression efficiency < 1")
    m = make_model()
    for T in [273.15, 298.15, 323.15]:
        eta = m.compression_efficiency(20.0, 900.0, T_inlet=T)
        assert_true(0.0 < eta < 1.0, f"eta({T})={eta:.4f} in (0, 1)")


# ---------------------------------------------------------------------------
# Test 8 -- SEC in physically realistic range for H2 compression
#           Ref: Sdanghi et al. (2019): 20->900 bar ~1.5-3 kWh/kg for 4-stage
# ---------------------------------------------------------------------------

def test_sec_realistic_range():
    print("\n[Test 8] SEC in realistic range [1.0, 5.0] kWh/kg (20-900 bar, 4-stage)")
    m = make_model()
    sec = m.sec_kwh_per_kg(20.0, 900.0)
    # RATIONALE: Sdanghi et al. (2019) Table 1: mechanical compression 20->700 bar ~1.5-2.5 kWh/kg
    # for 4-stage reciprocating. 900 bar is slightly higher. Bossel (2006) gives ~3 kWh/kg for
    # full chain. With eta_p=0.75, eta_mech=0.92, 4 stages, expect ~2-4 kWh/kg.
    assert_true(1.0 <= sec <= 5.0, f"SEC={sec:.3f} kWh/kg in [1, 5]")


# ---------------------------------------------------------------------------
# Test 9 -- Heat rejected is positive and increases with lower intercooler eps
#           (paradox note: eps=0 means no heat is rejected; eps=1 means max heat removed)
# ---------------------------------------------------------------------------

def test_heat_rejected_increases_with_eps():
    print("\n[Test 9] Heat rejected increases with intercooler effectiveness")
    m = make_model()
    Q_low  = m.heat_rejected_kw(0.014, 20.0, 900.0, eps_ic=0.3)
    Q_high = m.heat_rejected_kw(0.014, 20.0, 900.0, eps_ic=0.9)
    assert_true(Q_high > Q_low,
                f"Q_rej(eps=0.9)={Q_high:.3f} kW > Q_rej(eps=0.3)={Q_low:.3f} kW")


# ---------------------------------------------------------------------------
# Test 10 -- Work proportional to mass flow
# ---------------------------------------------------------------------------

def test_power_proportional_to_mass_flow():
    print("\n[Test 10] Shaft power proportional to mass flow")
    m = make_model()
    P1 = m.shaft_power_kw(0.007, 20.0, 900.0)
    P2 = m.shaft_power_kw(0.014, 20.0, 900.0)
    assert_true(abs(P2 / P1 - 2.0) < 1e-9, f"P(0.014)/P(0.007)={P2/P1:.6f} = 2.0")


# ---------------------------------------------------------------------------
# Test 11 -- ComponentModel predict() interface
# ---------------------------------------------------------------------------

def test_predict_interface():
    print("\n[Test 11] ComponentModel predict() interface")
    cm = ComponentModel()
    out = cm.predict({"mass_flow": 0.014, "P_inlet": 20.0, "P_outlet": 900.0,
                      "T_inlet": 298.15})
    required = [
        "shaft_power_kW", "SEC_kWh_kg", "efficiency", "heat_rejected_kW",
        "T_discharge_final_K", "stage_T_in_K", "stage_T_discharge_K", "stage_T_after_ic_K",
    ]
    for k in required:
        assert_true(k in out, f"Output key '{k}' present")
    assert_true(out["shaft_power_kW"] > 0, "shaft_power > 0")
    assert_true(len(out["stage_T_in_K"]) == 4, "stage arrays length 4")

    # Physics: hot inlet increases power
    out_hot = cm.predict({"mass_flow": 0.014, "P_inlet": 20.0, "P_outlet": 900.0,
                          "T_inlet": 323.15})
    assert_true(out_hot["shaft_power_kW"] > out["shaft_power_kW"],
                "Hot inlet gives higher shaft power")


# ---------------------------------------------------------------------------
# Test 12 -- Benchmark
# ---------------------------------------------------------------------------

def test_benchmark():
    print("\n[Test 12] Benchmark: 1000 evaluate() calls")
    m = make_model()
    T_arr = np.linspace(273.15, 323.15, 1000)
    t0 = time.perf_counter()
    for T in T_arr:
        m.evaluate(0.014, 20.0, 900.0, T_inlet=float(T))
    elapsed = time.perf_counter() - t0
    print(f"  1000 evaluations in {elapsed*1000:.1f} ms")
    assert_true(elapsed < 5.0, "Completes in < 5 s")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        test_work_increases_with_T_inlet,
        test_sec_increases_with_T_inlet,
        test_T_discharge_increases_with_T_inlet,
        test_stage_temperatures_increase,
        test_intercooler_reduces_temperature,
        test_perfect_intercooler_minimum_work,
        test_efficiency_below_unity,
        test_sec_realistic_range,
        test_heat_rejected_increases_with_eps,
        test_power_proportional_to_mass_flow,
        test_predict_interface,
        test_benchmark,
    ]
    passed = failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except AssertionError:
            failed += 1
        except Exception as e:
            failed += 1
            print(f"  UNEXPECTED ERROR in {t.__name__}: {e}")

    print(f"\n{'='*60}")
    print(f"EC016 H2 Compressor F1b Thermal -- Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    sys.exit(0 if failed == 0 else 1)
