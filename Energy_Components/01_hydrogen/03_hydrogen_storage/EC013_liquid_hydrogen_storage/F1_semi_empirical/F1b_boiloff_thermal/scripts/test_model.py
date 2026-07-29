"""
EC013 -- LH2 Storage -- F1b Boil-Off Thermal
Test suite: physics sanity checks.

Critical test rule: tests must FAIL the model, not accommodate it.
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from model import LH2ThermalModel
from predict import ComponentModel

DEFAULT_PARAMS = {
    "component": "EC013",
    "fidelity": "F1b",
    "description": "test",
    "source": "test",
    "tank": {
        "volume":          {"value": 1.0},
        "surface_area":    {"value": 5.0},
        "mass_empty":      {"value": 250.0},
        "U_ref":           {"value": 0.0015},
        "MLI_layers":      {"value": 30},
        "MLI_thickness":   {"value": 0.03},
        "max_pressure":    {"value": 6.0},
        "vent_pressure":   {"value": 5.5},
        "fill_fraction_max": {"value": 0.95},
    },
    "hydrogen": {
        "T_sat_1atm":  {"value": 20.28},
        "rho_liquid":  {"value": 70.85},
        "rho_vapor":   {"value": 1.34},
        "h_vap":       {"value": 445.6},
        "cp_liquid":   {"value": 9.69},
        "LHV":         {"value": 120.0},
        "R_H2":        {"value": 4124.2},
        "Z_vapor":     {"value": 1.0},
        "dTsat_dP":    {"value": 0.154},
    },
    "ambient": {
        "T_ambient_default": {"value": 298.15},
        "T_ambient_min":     {"value": 233.15},
        "T_ambient_max":     {"value": 333.15},
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
    return LH2ThermalModel(DEFAULT_PARAMS)


# ---------------------------------------------------------------------------
# Test 1 -- Boil-off rate increases with ambient temperature
# Physics: higher T_amb -> larger dT -> more heat leak -> more boil-off
# ---------------------------------------------------------------------------

def test_bor_increases_with_T_amb():
    print("\n[Test 1] Boil-off rate increases with ambient temperature")
    m = make_model()
    bor_cold = float(m.boiloff_rate_percent_day(0.8, 253.15))
    bor_warm = float(m.boiloff_rate_percent_day(0.8, 298.15))
    bor_hot  = float(m.boiloff_rate_percent_day(0.8, 333.15))
    assert_true(bor_warm > bor_cold,
                f"BOR(298K)={bor_warm:.4f} > BOR(253K)={bor_cold:.4f}")
    assert_true(bor_hot > bor_warm,
                f"BOR(333K)={bor_hot:.4f} > BOR(298K)={bor_warm:.4f}")


# ---------------------------------------------------------------------------
# Test 2 -- MLI effective U increases with ambient temperature
# Physics: k_mli(T) increases -> U_eff increases
# ---------------------------------------------------------------------------

def test_u_increases_with_T_amb():
    print("\n[Test 2] U_eff increases with ambient temperature (MLI T-dependence)")
    m = make_model()
    U_cold = float(m.u_eff(253.15))
    U_warm = float(m.u_eff(298.15))
    U_hot  = float(m.u_eff(333.15))
    assert_true(U_warm > U_cold, f"U(298K)={U_warm:.6f} > U(253K)={U_cold:.6f}")
    assert_true(U_hot > U_warm,  f"U(333K)={U_hot:.6f} > U(298K)={U_warm:.6f}")


# ---------------------------------------------------------------------------
# Test 3 -- Saturation temperature increases with pressure (Clausius-Clapeyron)
# ---------------------------------------------------------------------------

def test_t_sat_increases_with_pressure():
    print("\n[Test 3] T_sat increases with tank pressure (Clausius-Clapeyron)")
    m = make_model()
    T_1bar = float(m.t_sat(1.01325))
    T_3bar = float(m.t_sat(3.0))
    T_5bar = float(m.t_sat(5.0))
    assert_true(T_3bar > T_1bar, f"T_sat(3 bar)={T_3bar:.3f} > T_sat(1 bar)={T_1bar:.3f}")
    assert_true(T_5bar > T_3bar, f"T_sat(5 bar)={T_5bar:.3f} > T_sat(3 bar)={T_3bar:.3f}")


# ---------------------------------------------------------------------------
# Test 4 -- Higher pressure reduces effective dT (T_amb - T_sat), so heat leak
#           and BOR decrease when pressure rises (dormancy self-pressurization effect)
# ---------------------------------------------------------------------------

def test_higher_pressure_reduces_bor():
    print("\n[Test 4] Higher P_tank reduces heat leak (smaller dT)")
    m = make_model()
    Q_1atm = float(m.heat_leak(298.15, 1.01325))
    Q_3bar = float(m.heat_leak(298.15, 3.0))
    Q_5bar = float(m.heat_leak(298.15, 5.0))
    assert_true(Q_3bar < Q_1atm, f"Q(3 bar)={Q_3bar:.4f} < Q(1 atm)={Q_1atm:.4f}")
    assert_true(Q_5bar < Q_3bar, f"Q(5 bar)={Q_5bar:.4f} < Q(3 bar)={Q_3bar:.4f}")


# ---------------------------------------------------------------------------
# Test 5 -- Stored mass proportional to fill fraction
# ---------------------------------------------------------------------------

def test_stored_mass_proportional():
    print("\n[Test 5] Stored mass proportional to fill fraction")
    m = make_model()
    m1 = float(m.stored_mass(0.4))
    m2 = float(m.stored_mass(0.8))
    # RATIONALE: linear relationship from simple rho_L * V * f
    assert_true(abs(m2 / m1 - 2.0) < 1e-9, f"m(0.8)/m(0.4)={m2/m1:.6f} = 2.0")


# ---------------------------------------------------------------------------
# Test 6 -- Pressurization transient: pressure rises in closed-vent dormancy
# ---------------------------------------------------------------------------

def test_pressurization_rises():
    print("\n[Test 6] Closed-vent dormancy: pressure rises over time")
    m = make_model()
    t, P_arr, m_liq, bor_arr = m.pressurization_transient(
        fill_fraction=0.9, T_amb_K=298.15, P0_bar=1.0,
        t_span=(0, 86400), n_steps=100,
    )
    assert_true(P_arr[-1] > P_arr[0], f"P_final={P_arr[-1]:.3f} > P_init={P_arr[0]:.3f}")


# ---------------------------------------------------------------------------
# Test 7 -- Pressure never exceeds vent set-point
# ---------------------------------------------------------------------------

def test_pressure_capped_at_vent():
    print("\n[Test 7] Pressure capped at vent set-point during long dormancy")
    m = make_model()
    t, P_arr, m_liq, _ = m.pressurization_transient(
        fill_fraction=0.9, T_amb_K=310.0, P0_bar=1.0,
        t_span=(0, 7 * 86400), n_steps=200,
    )
    assert_true(float(P_arr.max()) <= m.P_vent + 0.05,
                f"P_max={float(P_arr.max()):.3f} <= P_vent={m.P_vent}")


# ---------------------------------------------------------------------------
# Test 8 -- Liquid mass decreases over time (boil-off depletes tank)
# ---------------------------------------------------------------------------

def test_liquid_mass_decreases():
    print("\n[Test 8] Liquid mass decreases over time during dormancy")
    m = make_model()
    t, P_arr, m_liq, _ = m.pressurization_transient(
        fill_fraction=0.8, T_amb_K=298.15, P0_bar=1.0,
        t_span=(0, 30 * 86400), n_steps=100,
    )
    assert_true(float(m_liq[-1]) < float(m_liq[0]),
                f"m_liq_final={float(m_liq[-1]):.2f} < m_liq_init={float(m_liq[0]):.2f}")


# ---------------------------------------------------------------------------
# Test 9 -- BOR at 298 K is in physically realistic range (0.01 – 1 %/day)
#           Ref: NREL HTAC targets 0.1 %/day for large tank; single m3 ~0.1-0.3
# ---------------------------------------------------------------------------

def test_bor_realistic_range():
    print("\n[Test 9] BOR in realistic range [0.01, 1.0] %/day at 298 K")
    m = make_model()
    bor = float(m.boiloff_rate_percent_day(0.8, 298.15))
    # RATIONALE: DOE target for road vehicle is <0.1 %/day; stationary ~0.1-0.5 %/day
    # for 1 m3 class tanks with 30-layer MLI. Johnson (2010), Petitpas (2018).
    assert_true(0.01 <= bor <= 1.0, f"BOR={bor:.4f} %/day in [0.01, 1.0]")


# ---------------------------------------------------------------------------
# Test 10 -- ComponentModel predict() interface
# ---------------------------------------------------------------------------

def test_predict_interface():
    print("\n[Test 10] ComponentModel predict() interface")
    cm = ComponentModel()
    out = cm.predict({"fill_fraction": 0.8, "T_ambient": 298.15})
    required = [
        "stored_mass_kg", "energy_stored_MJ", "heat_leak_W",
        "boiloff_rate_kg_s", "BOR_pct_day", "U_eff_W_m2_K", "T_sat_K",
    ]
    for k in required:
        assert_true(k in out, f"Output key '{k}' present")
    assert_true(out["BOR_pct_day"] > 0, "BOR > 0")
    assert_true(out["T_sat_K"] > 15.0, "T_sat > 15 K")


# ---------------------------------------------------------------------------
# Test 11 -- Benchmark
# ---------------------------------------------------------------------------

def test_benchmark():
    print("\n[Test 11] Benchmark: 10 000 evaluations")
    m = make_model()
    f_arr = np.linspace(0.1, 0.9, 10000)
    t0 = time.perf_counter()
    m.evaluate(f_arr, 298.15)
    elapsed = time.perf_counter() - t0
    print(f"  10 000 evaluations (vectorized) in {elapsed*1000:.1f} ms")
    assert_true(elapsed < 5.0, "Completes in < 5 s")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        test_bor_increases_with_T_amb,
        test_u_increases_with_T_amb,
        test_t_sat_increases_with_pressure,
        test_higher_pressure_reduces_bor,
        test_stored_mass_proportional,
        test_pressurization_rises,
        test_pressure_capped_at_vent,
        test_liquid_mass_decreases,
        test_bor_realistic_range,
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
    print(f"EC013 LH2 F1b Thermal -- Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    sys.exit(0 if failed == 0 else 1)
