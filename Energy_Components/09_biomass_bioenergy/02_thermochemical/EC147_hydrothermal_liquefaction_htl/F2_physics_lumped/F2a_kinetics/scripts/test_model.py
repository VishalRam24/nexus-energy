"""
EC147 -- Hydrothermal Liquefaction (HTL) -- F2a Physics-Lumped Kinetics
Test suite: mass conservation, Arrhenius monotonicity, intermediate-severity
biocrude peak, energy balance, subcritical-water guard, predict() interface.

Run:  python3 scripts/test_model.py   (system python3, NOT pytest)
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from model import HTL_F2a, water_saturation_pressure_MPa
from predict import ComponentModel

PASS = "✓"
FAIL = "✗"


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
def test_mass_conservation():
    print("\n[Test 1] Mass conservation: lumps sum to biomass0 at all times")
    m, _ = make_model()
    r = m.simulate(T_setpoint_C=350.0, residence_min=45.0, T0_C=200.0, biomass0=1.0)
    tot = r["mass_total"]
    assert_true(np.allclose(tot, 1.0, atol=1e-6),
                f"sum(lumps) == 1 (max dev {np.max(np.abs(tot-1.0)):.2e})")
    # also at a non-unit initial loading
    r2 = m.simulate(biomass0=0.2, residence_min=30.0)
    assert_true(np.allclose(r2["mass_total"], 0.2, atol=1e-6),
                "sum(lumps) == biomass0 for partial loading")


def test_yields_in_unit_interval():
    print("\n[Test 2] All product yields in [0, 1] and conversion in [0, 1]")
    m, _ = make_model()
    r = m.simulate(T_setpoint_C=350.0, residence_min=60.0)
    for key in ["biocrude_yield", "aqueous_yield", "gas_yield", "solid_yield", "conversion"]:
        arr = r[key]
        assert_true(np.all(arr >= -1e-9) and np.all(arr <= 1.0 + 1e-9),
                    f"{key} in [0,1]")


def test_arrhenius_monotone():
    print("\n[Test 3] Arrhenius rate constants increase with temperature")
    m, _ = make_model()
    k_low = m.rate_constants(550.0)   # ~277 C
    k_high = m.rate_constants(640.0)  # ~367 C
    for name in k_low:
        assert_true(k_high[name] > k_low[name],
                    f"{name}: k(640K)={k_high[name]:.3e} > k(550K)={k_low[name]:.3e}")


def test_conversion_increases_with_severity():
    print("\n[Test 4] Conversion increases with temperature and time")
    m, _ = make_model()
    c_lowT = m.simulate(T_setpoint_C=280.0, residence_min=30.0)["final"]["conversion"]
    c_highT = m.simulate(T_setpoint_C=360.0, residence_min=30.0)["final"]["conversion"]
    assert_true(c_highT > c_lowT, f"conv(360C)={c_highT:.3f} > conv(280C)={c_lowT:.3f}")
    c_short = m.simulate(T_setpoint_C=340.0, residence_min=5.0)["final"]["conversion"]
    c_long = m.simulate(T_setpoint_C=340.0, residence_min=60.0)["final"]["conversion"]
    assert_true(c_long > c_short, f"conv(60min)={c_long:.3f} > conv(5min)={c_short:.3f}")


def test_biocrude_peaks_at_intermediate_severity():
    print("\n[Test 5] Biocrude yield peaks at intermediate severity (not monotone)")
    m, _ = make_model()
    # Sweep residence time at high temperature where secondary cracking matters.
    times = np.linspace(2.0, 120.0, 25)
    yields = np.array([m.biocrude_yield_at(360.0, t, T0_C=360.0) for t in times])
    i_peak = int(np.argmax(yields))
    assert_true(0 < i_peak < len(times) - 1,
                f"peak at interior index {i_peak}/{len(times)-1} (t={times[i_peak]:.1f} min)")
    assert_true(yields[-1] < yields[i_peak] - 1e-3,
                f"biocrude declines after peak: y_end={yields[-1]:.3f} < y_peak={yields[i_peak]:.3f}")


def test_biocrude_positive_and_significant():
    print("\n[Test 6] Biocrude is produced (yield clearly > 0) at HTL conditions")
    m, _ = make_model()
    y = m.simulate(T_setpoint_C=350.0, residence_min=30.0, T0_C=300.0)["final"]["biocrude_yield"]
    assert_true(0.05 < y < 0.95, f"biocrude_yield={y:.3f} is a physical HTL value")


def test_energy_balance_heats_to_setpoint():
    print("\n[Test 7] Energy balance: reactor heats from T0 toward setpoint")
    m, _ = make_model()
    r = m.simulate(T_setpoint_C=350.0, residence_min=90.0, T0_C=150.0)
    T = r["temperature_C"]
    assert_true(T[0] < T[-1], f"reactor heats up: T0={T[0]:.1f}C -> Tf={T[-1]:.1f}C")
    assert_true(T[-1] <= 360.0,
                f"T stays at/below setpoint band: Tf={T[-1]:.1f}C (set 350C)")
    assert_true(abs(T[-1] - 350.0) < 5.0, f"approaches setpoint: Tf={T[-1]:.1f}C ~ 350C")


def test_subcritical_guard():
    print("\n[Test 8] Subcritical-water guard (T<Tc, P<Pc, P>Psat)")
    m, _ = make_model()
    # Typical HTL: 320 C, 18 MPa -> subcritical compressed liquid.
    assert_true(m.is_subcritical(320.0, 18.0), "320C/18MPa is subcritical liquid")
    # Below saturation pressure at 350 C (~16.5 MPa) -> water would boil.
    assert_true(not m.is_subcritical(350.0, 12.0), "350C/12MPa < Psat -> not liquid")
    # Above critical temperature -> not subcritical.
    assert_true(not m.is_subcritical(380.0, 25.0), "380C is supercritical")
    # Psat at 350 C must be near the IAPWS value ~16.5 MPa.
    psat = water_saturation_pressure_MPa(350.0)
    assert_true(14.0 < psat < 18.5, f"Psat(350C)={psat:.2f} MPa near IAPWS 16.5")


def test_zero_time_and_zero_biomass():
    print("\n[Test 9] Edge cases: zero residence time and zero biomass")
    m, _ = make_model()
    r0 = m.simulate(residence_min=0.0, biomass0=1.0)
    assert_true(abs(r0["final"]["conversion"]) < 1e-9, "t=0 -> no conversion")
    assert_true(abs(r0["final"]["biocrude_yield"]) < 1e-9, "t=0 -> no biocrude")
    rb = m.simulate(biomass0=0.0, residence_min=30.0)
    assert_true(abs(rb["final"]["biocrude_yield"]) < 1e-12, "no biomass -> no biocrude")


def test_predict_interface():
    print("\n[Test 10] ComponentModel predict()/get_info() interface")
    _, cm = make_model()
    info = cm.get_info()
    for k in ["component_id", "component_name", "fidelity", "version", "inputs", "outputs"]:
        assert_true(k in info, f"get_info has '{k}'")
    assert_true(info["component_id"] == "EC147", "component_id == EC147")
    r = cm.predict({"T_setpoint_C": 340.0, "residence_min": 20.0})
    for key in ["t_min", "biocrude_yield", "conversion", "temperature_C", "final", "subcritical"]:
        assert_true(key in r, f"predict output has '{key}'")
    assert_true(len(r["t_min"]) == len(r["biocrude_yield"]), "time arrays aligned")


def test_benchmark():
    print("\n[Test 11] Benchmark: full 60 min HTL simulation")
    m, _ = make_model()
    t0 = time.perf_counter()
    m.simulate(T_setpoint_C=350.0, residence_min=60.0, T0_C=200.0, n_out=200)
    elapsed = time.perf_counter() - t0
    print(f"  60 min simulation in {elapsed*1000:.1f} ms")
    assert_true(elapsed < 5.0, "completes in < 5 s")


if __name__ == "__main__":
    tests = [
        test_mass_conservation,
        test_yields_in_unit_interval,
        test_arrhenius_monotone,
        test_conversion_increases_with_severity,
        test_biocrude_peaks_at_intermediate_severity,
        test_biocrude_positive_and_significant,
        test_energy_balance_heats_to_setpoint,
        test_subcritical_guard,
        test_zero_time_and_zero_biomass,
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
    print(f"EC147 HTL F2a -- Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    sys.exit(0 if failed == 0 else 1)
