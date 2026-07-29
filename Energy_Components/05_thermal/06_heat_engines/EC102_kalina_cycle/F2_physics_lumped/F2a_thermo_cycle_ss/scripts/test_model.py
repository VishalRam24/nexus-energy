"""
EC102 -- Kalina Cycle -- F2a Physics-Lumped
Test suite: thermodynamic sanity, temperature-glide behaviour, conservation,
separator split, ODE convergence, edge cases, predict() interface, benchmark.
Run with system python3 (NOT pytest):  python3 scripts/test_model.py
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from model import KalinaCycleF2a
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
def test_efficiency_below_carnot():
    print("\n[Test 1] eta_thermal < eta_Carnot (2nd law)")
    m, _ = make_model()
    for Tsrc in [100.0, 130.0, 160.0, 200.0]:
        r = m.solve_cycle(T_source_c=Tsrc, T_sink_c=25.0)
        assert_true(0.0 < r["eta_thermal"] < r["eta_carnot"],
                    f"T={Tsrc}C: eta={r['eta_thermal']:.4f} < Carnot={r['eta_carnot']:.4f}")


def test_efficiency_monotone_source():
    print("\n[Test 2] eta increases with source temperature")
    m, _ = make_model()
    Ts = np.linspace(90.0, 210.0, 20)
    eta_prev = m.solve_cycle(T_source_c=Ts[0])["eta_thermal"]
    for T in Ts[1:]:
        eta = m.solve_cycle(T_source_c=T)["eta_thermal"]
        assert_true(eta >= eta_prev - 1e-9, f"eta({T:.0f}C)={eta:.4f} >= {eta_prev:.4f}")
        eta_prev = eta
    print("  All 19 pairs monotone non-decreasing.")


def test_temperature_glide():
    print("\n[Test 3] Temperature GLIDE: zeotropic dew > bubble, peaks mid-composition")
    m, _ = make_model()
    # glide vanishes at pure ends
    g_nh3 = m.glide_width(30.0, 0.999)
    g_h2o = m.glide_width(30.0, 0.001)
    g_mid = m.glide_width(30.0, 0.5)
    assert_true(g_nh3 < 1.0 and g_h2o < 1.0, f"glide ~0 at pure ends: NH3={g_nh3:.3f}, H2O={g_h2o:.3f}")
    assert_true(g_mid > 20.0, f"glide large mid-composition: {g_mid:.2f} K")
    # dew strictly above bubble for the working fluid
    Tb = m.bubble_temp(30.0, 0.82)
    Td = m.dew_temp(30.0, 0.82)
    assert_true(Td > Tb, f"dew {Td:.1f}K > bubble {Tb:.1f}K -> glide {Td-Tb:.1f}K")


def test_glide_in_cycle_output():
    print("\n[Test 4] Cycle reports non-zero hot & cold glide (Kalina advantage)")
    m, _ = make_model()
    r = m.solve_cycle(T_source_c=150.0, T_sink_c=25.0)
    assert_true(r["glide_hot_K"] > 5.0, f"hot glide {r['glide_hot_K']:.1f} K > 5")
    assert_true(r["glide_cold_K"] > 5.0, f"cold glide {r['glide_cold_K']:.1f} K > 5")


def test_separator_split():
    print("\n[Test 5] Separator: vapor NH3-enriched, liquid NH3-lean, mass balance")
    m, _ = make_model()
    r = m.solve_cycle(T_source_c=150.0)
    w_b = m.x_basic
    yv, wv, wl = r["vapor_fraction"], r["w_NH3_vapor"], r["w_NH3_liquid"]
    assert_true(wv > w_b > wl, f"w_vap {wv:.3f} > feed {w_b:.3f} > w_liq {wl:.3f}")
    assert_true(0.0 <= yv <= 1.0, f"vapor fraction {yv:.3f} in [0,1]")
    # NH3 mass balance:  feed ~= yv*wv + (1-yv)*wl
    recomb = yv * wv + (1 - yv) * wl
    assert_true(abs(recomb - w_b) < 0.06, f"NH3 balance: recomb={recomb:.3f} ~= feed={w_b:.3f}")


def test_energy_conservation():
    print("\n[Test 6] Energy conservation: Q_in = P_net + Q_out")
    m, _ = make_model()
    for Tsrc, Q in [(120.0, 800.0), (150.0, 1000.0), (190.0, 1500.0)]:
        r = m.solve_cycle(T_source_c=Tsrc, Q_in_kw=Q)
        resid = r["Q_in_kW"] - (r["P_net_kW"] + r["Q_out_kW"])
        assert_true(abs(resid) < 1e-6, f"T={Tsrc}: |Q_in-P-Q_out|={abs(resid):.2e} kW")
        assert_true(r["P_net_kW"] > 0 and r["Q_out_kW"] > 0, "positive power & rejection")


def test_turbine_exceeds_pump():
    print("\n[Test 7] Turbine work > pump work (net positive cycle)")
    m, _ = make_model()
    r = m.solve_cycle(T_source_c=150.0)
    assert_true(r["w_turbine_spec_J_kg"] > r["w_pump_spec_J_kg"],
                f"w_turb={r['w_turbine_spec_J_kg']:.0f} > w_pump={r['w_pump_spec_J_kg']:.0f} J/kg")
    assert_true(r["w_net_spec_J_kg"] > 0, f"w_net={r['w_net_spec_J_kg']:.0f} J/kg > 0")


def test_antoine_saturation():
    print("\n[Test 8] Pure-component saturation temperatures physical")
    m, _ = make_model()
    # NH3 boils far below water at the same pressure
    assert_true(m.Tsat_NH3(30.0) < m.Tsat_H2O(30.0),
                f"Tsat_NH3 {m.Tsat_NH3(30.0)-273.15:.1f}C < Tsat_H2O {m.Tsat_H2O(30.0)-273.15:.1f}C")
    # NH3 ~ -33C at 1 atm (1.013 bar)
    t_nh3_1atm = m.Tsat_NH3(1.013) - 273.15
    assert_true(-45 < t_nh3_1atm < -20, f"NH3 NBP ~ {t_nh3_1atm:.1f}C (expect ~ -33C)")
    # H2O ~ 100C at 1 atm
    t_h2o_1atm = m.Tsat_H2O(1.013) - 273.15
    assert_true(90 < t_h2o_1atm < 110, f"H2O NBP ~ {t_h2o_1atm:.1f}C (expect ~ 100C)")


def test_transient_ode():
    print("\n[Test 9] Transient drum ODE converges to a steady operating point")
    m, _ = make_model()
    # drum time constant ~ C/(k_cycle+UA) ~ 2350 s; integrate several tau to settle
    tr = m.simulate_transient(q_source_func=1.0e6, T0_K=380.0, duration_s=12000.0)
    assert_true(tr["success"], "solve_ivp succeeded")
    dT = abs(tr["T_drum_K"][-1] - tr["T_drum_K"][-2])
    assert_true(dT < 0.5, f"near steady state: last dT={dT:.4f} K")
    assert_true(np.all(np.isfinite(tr["P_net_kW"])), "P_net finite over transient")


def test_composition_effect():
    print("\n[Test 10] Ammonia fraction changes the split & glide (zeotropic)")
    m, _ = make_model()
    r_lean = m.solve_cycle(T_source_c=150.0, w_basic=0.50)
    r_rich = m.solve_cycle(T_source_c=150.0, w_basic=0.90)
    # richer feed -> higher vapor NH3 fraction
    assert_true(r_rich["w_NH3_vapor"] > r_lean["w_NH3_vapor"],
                f"rich vapor {r_rich['w_NH3_vapor']:.3f} > lean {r_lean['w_NH3_vapor']:.3f}")
    assert_true(r_lean["glide_hot_K"] > 5.0 and r_rich["glide_hot_K"] >= 0.0,
                "glide present across compositions")


def test_predict_interface():
    print("\n[Test 11] ComponentModel predict() interface + get_info")
    _, cm = make_model()
    r = cm.predict({"T_source_c": 150.0, "T_sink_c": 25.0})
    for key in ["P_net_kW", "Q_in_kW", "Q_out_kW", "eta_thermal", "eta_carnot",
                "vapor_fraction", "glide_hot_K", "glide_cold_K"]:
        assert_true(key in r, f"Key '{key}' in output")
    info = cm.get_info()
    assert_true(info["component_id"] == "EC102", "component_id == EC102")
    assert_true(info["version"] == "1.0.0", "version 1.0.0")
    # transient path through predict
    rt = cm.predict({"T_source_c": 150.0, "transient": True, "duration_s": 300.0})
    assert_true("transient" in rt and len(rt["transient"]["t"]) > 10,
                "transient predict returns time series")


def test_benchmark():
    print("\n[Test 12] Benchmark: steady solve + 600s transient")
    m, _ = make_model()
    t0 = time.perf_counter()
    for _ in range(1000):
        m.solve_cycle(T_source_c=150.0)
    t_ss = time.perf_counter() - t0
    print(f"  1000 steady solves in {t_ss*1000:.1f} ms")
    t1 = time.perf_counter()
    m.simulate_transient(q_source_func=1.0e6, duration_s=600.0)
    t_tr = time.perf_counter() - t1
    print(f"  600s transient in {t_tr*1000:.1f} ms")
    assert_true(t_tr < 5.0, "transient completes in < 5 s")


if __name__ == "__main__":
    tests = [
        test_efficiency_below_carnot,
        test_efficiency_monotone_source,
        test_temperature_glide,
        test_glide_in_cycle_output,
        test_separator_split,
        test_energy_conservation,
        test_turbine_exceeds_pump,
        test_antoine_saturation,
        test_transient_ode,
        test_composition_effect,
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
    print(f"EC102 Kalina Cycle F2a -- Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    sys.exit(0 if failed == 0 else 1)
