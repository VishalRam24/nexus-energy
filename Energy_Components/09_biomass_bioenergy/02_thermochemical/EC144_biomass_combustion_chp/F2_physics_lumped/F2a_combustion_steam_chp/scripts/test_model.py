"""
EC144 -- Biomass Combustion CHP -- F2a Combustion + Steam-Cycle
Test suite: combustion/CHP physics sanity, conservation, ODE convergence, edges.
Run as: python3 scripts/test_model.py   (system python3, NOT pytest)
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from model import BiomassCombustionCHP_F2a
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
def test_lhv_moisture_monotone():
    print("\n[Test 1] Effective LHV decreases monotonically with moisture")
    m, _ = make_model()
    prev = m.lhv_effective(0.0)
    for M in [0.1, 0.2, 0.3, 0.4, 0.5]:
        cur = m.lhv_effective(M)
        assert_true(cur < prev, f"LHV_eff(M={M})={cur:.0f} < prev={prev:.0f} kJ/kg")
        prev = cur


def test_efficiency_bounds():
    print("\n[Test 2] Boiler efficiency in (0, 1)")
    m, _ = make_model()
    for PLR in [0.3, 0.6, 1.0]:
        for M in [0.0, 0.2, 0.4]:
            eta = m.boiler_efficiency(PLR, M)
            assert_true(0.0 < eta < 1.0, f"eta_boiler(PLR={PLR},M={M})={eta:.3f} in (0,1)")


def test_total_gt_electrical_lt_one():
    print("\n[Test 3] eta_electrical < eta_total < 1  (CHP ordering)")
    m, _ = make_model()
    for PLR in [0.4, 0.7, 1.0]:
        r = m.predict_steady(PLR, 0.2)
        assert_true(r["eta_electrical"] < r["eta_total_chp"],
                    f"eta_el={r['eta_electrical']:.3f} < eta_total={r['eta_total_chp']:.3f}")
        assert_true(r["eta_total_chp"] < 1.0,
                    f"eta_total={r['eta_total_chp']:.3f} < 1")
        assert_true(r["eta_electrical"] > 0.0, "eta_el > 0")


def test_electrical_below_carnot():
    print("\n[Test 4] Electrical conversion of useful heat < Carnot limit")
    m, _ = make_model()
    r = m.predict_steady(1.0, 0.2)
    # shaft-work conversion of useful heat must respect Carnot
    w_useful = r["P_electrical_kw"] / m.eta_mg / max(r["useful_heat_kw"], 1e-9)
    assert_true(w_useful < r["eta_carnot"],
                f"useful->shaft frac={w_useful:.3f} < Carnot={r['eta_carnot']:.3f}")
    assert_true(r["eta_electrical"] < r["eta_carnot"],
                f"eta_el={r['eta_electrical']:.3f} < Carnot={r['eta_carnot']:.3f}")


def test_energy_conservation():
    print("\n[Test 5] Energy balance closes: P_el + Q_th + flue + rad = fuel")
    m, _ = make_model()
    for PLR in [0.5, 1.0]:
        r = m.predict_steady(PLR, 0.25)
        total = (r["P_electrical_kw"] + r["Q_thermal_kw"]
                 + r["flue_loss_kw"] + r["radiation_loss_kw"])
        # gen/mech loss accounts for the small remainder; balance within it
        gen_loss = r["P_electrical_kw"] * (1.0 / m.eta_mg - 1.0)
        residual = abs(r["fuel_input_kw"] - (total + gen_loss))
        assert_true(residual < 1e-6 * max(r["fuel_input_kw"], 1.0),
                    f"PLR={PLR}: residual={residual:.3e} kW ~ 0")


def test_moisture_reduces_output():
    print("\n[Test 6] Higher moisture reduces electrical output & efficiency")
    m, _ = make_model()
    dry = m.predict_steady(1.0, 0.05)
    wet = m.predict_steady(1.0, 0.45)
    assert_true(wet["eta_total_chp"] < dry["eta_total_chp"],
                f"eta_total wet={wet['eta_total_chp']:.3f} < dry={dry['eta_total_chp']:.3f}")
    assert_true(wet["P_electrical_kw"] < dry["P_electrical_kw"],
                f"P_el wet={wet['P_electrical_kw']:.1f} < dry={dry['P_electrical_kw']:.1f}")


def test_power_to_heat_ratio():
    print("\n[Test 7] Power-to-heat ratio physical (back-pressure CHP, < 1)")
    m, _ = make_model()
    r = m.predict_steady(1.0, 0.2)
    assert_true(0.0 < r["power_to_heat_ratio"] < 1.0,
                f"P/H={r['power_to_heat_ratio']:.3f} in (0,1) for back-pressure CHP")


def test_thermal_ode_heats_up():
    print("\n[Test 8] Lumped boiler ODE: heats from cold start, bounded")
    m, _ = make_model()
    therm = m.simulate_thermal(1.0, 0.2, T0_K=288.15, dt=20.0, duration_s=7200.0)
    T = therm["T_boiler_K"]
    assert_true(T[-1] > T[0] + 50.0, f"Heats up: {T[0]:.1f} -> {T[-1]:.1f} K")
    assert_true(T[-1] < 1200.0, f"Bounded: T_final={T[-1]:.1f} K < 1200 K")
    assert_true(np.all(np.diff(T) > -1e-6), "Monotone rise toward steady state")


def test_thermal_steady_state():
    print("\n[Test 9] Boiler ODE approaches steady state")
    m, _ = make_model()
    therm = m.simulate_thermal(1.0, 0.2, T0_K=288.15, dt=20.0, duration_s=14400.0)
    T = therm["T_boiler_K"]
    dT = abs(T[-1] - T[-2])
    assert_true(dT < 0.05, f"Near SS: dT={dT:.4f} K between last two steps")


def test_zero_load_edge():
    print("\n[Test 10] Zero load -> zero outputs, finite efficiencies")
    m, _ = make_model()
    r = m.predict_steady(0.0, 0.2)
    assert_true(abs(r["fuel_input_kw"]) < 1e-9, "fuel=0 at PLR=0")
    assert_true(abs(r["P_electrical_kw"]) < 1e-9, "P_el=0 at PLR=0")
    assert_true(np.isfinite(r["eta_carnot"]), "Carnot finite")


def test_predict_interface():
    print("\n[Test 11] ComponentModel predict()/get_info() interface")
    _, cm = make_model()
    info = cm.get_info()
    assert_true(info["component_id"] == "EC144", "component_id EC144")
    r = cm.predict({"PLR": 0.8, "moisture_fraction": 0.2,
                    "duration_s": 600.0, "dt": 30.0})
    for key in ["P_electrical_kw", "Q_thermal_kw", "eta_total_chp",
                "power_to_heat_ratio", "t", "T_boiler_K"]:
        assert_true(key in r, f"Key '{key}' in output")
    assert_true(len(r["t"]) == len(r["T_boiler_K"]), "ODE arrays same length")


def test_benchmark():
    print("\n[Test 12] Benchmark: 1h transient at dt=10 s")
    m, _ = make_model()
    t0 = time.perf_counter()
    m.simulate_thermal(1.0, 0.2, T0_K=288.15, dt=10.0, duration_s=3600.0)
    elapsed = time.perf_counter() - t0
    print(f"  1h simulation in {elapsed*1000:.1f} ms")
    assert_true(elapsed < 5.0, "Completes in < 5 s")


if __name__ == "__main__":
    tests = [
        test_lhv_moisture_monotone,
        test_efficiency_bounds,
        test_total_gt_electrical_lt_one,
        test_electrical_below_carnot,
        test_energy_conservation,
        test_moisture_reduces_output,
        test_power_to_heat_ratio,
        test_thermal_ode_heats_up,
        test_thermal_steady_state,
        test_zero_load_edge,
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
    print(f"EC144 Biomass Combustion CHP F2a -- Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    sys.exit(0 if failed == 0 else 1)
