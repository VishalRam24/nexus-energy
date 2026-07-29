"""
EC105 -- Gas Turbine CHP -- F2a Physics-Lumped Thermo Cycle
Test suite: thermodynamic sanity, energy conservation, Carnot/CHP bounds,
HRSG transient, predict() interface, benchmark.
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from model import GasTurbineCHP_F2a
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
def test_brayton_temperatures():
    print("\n[Test 1] Brayton temperatures ordered T1<T2<T4<T3, real>isentropic")
    m, _ = make_model()
    s = m.cycle_state(1.0)
    assert_true(s["T1_K"] < s["T2_K"], f"T1={s['T1_K']:.0f} < T2={s['T2_K']:.0f}")
    assert_true(s["T2_K"] < s["T3_K"], f"T2={s['T2_K']:.0f} < T3(TIT)={s['T3_K']:.0f}")
    assert_true(s["T4_K"] < s["T3_K"], f"T4(exh)={s['T4_K']:.0f} < T3={s['T3_K']:.0f}")
    assert_true(s["T4_K"] > s["T1_K"], f"T4={s['T4_K']:.0f} > T1={s['T1_K']:.0f} (waste heat exists)")
    # real compressor exit hotter than isentropic exit (eta_c < 1)
    T2s = s["T1_K"] * m.rp ** ((m.g_a - 1.0) / m.g_a)
    assert_true(s["T2_K"] > T2s, f"real T2={s['T2_K']:.1f} > isentropic {T2s:.1f}")


def test_electrical_efficiency_below_carnot():
    print("\n[Test 2] Power-cycle efficiency below Carnot bound")
    m, _ = make_model()
    s = m.cycle_state(1.0)
    assert_true(s["eta_electrical"] < s["eta_carnot"],
                f"eta_el={s['eta_electrical']*100:.1f}% < Carnot={s['eta_carnot']*100:.1f}%")
    assert_true(0.20 < s["eta_electrical"] < 0.45,
                f"eta_el={s['eta_electrical']*100:.1f}% in plausible GT range")


def test_chp_total_efficiency_bounds():
    print("\n[Test 3] total_eff > electrical_eff and total_eff < 1")
    m, _ = make_model()
    for plr in [0.4, 0.6, 0.8, 1.0]:
        s = m.cycle_state(plr)
        assert_true(s["eta_total"] > s["eta_electrical"],
                    f"PLR={plr}: eta_total={s['eta_total']*100:.1f}% > eta_el={s['eta_electrical']*100:.1f}%")
        assert_true(s["eta_total"] < 1.0,
                    f"PLR={plr}: eta_total={s['eta_total']*100:.1f}% < 100%")


def test_thermal_efficiency_positive():
    print("\n[Test 4] HRSG recovers useful heat (eta_th > 0, HPR > 0)")
    m, _ = make_model()
    s = m.cycle_state(1.0)
    assert_true(s["eta_thermal"] > 0.0, f"eta_th={s['eta_thermal']*100:.1f}% > 0")
    assert_true(s["heat_to_power_ratio"] > 0.0, f"HPR={s['heat_to_power_ratio']:.2f} > 0")
    assert_true(0.5 < s["heat_to_power_ratio"] < 3.0,
                f"HPR={s['heat_to_power_ratio']:.2f} in typical GT-CHP range")


def test_energy_conservation():
    print("\n[Test 5] Energy conservation: P_el + Q_th + losses == Q_fuel")
    m, _ = make_model()
    s = m.cycle_state(1.0)
    Q_fuel = s["fuel_power_w"]
    P_el = s["electrical_power_w"]
    Q_th = s["thermal_power_w"]
    # losses = generator/mech loss + unrecovered exhaust + stack + combustor loss
    losses = Q_fuel - P_el - Q_th
    assert_true(losses >= 0.0, f"non-negative losses = {losses/1e3:.1f} kW")
    closure = (P_el + Q_th + losses) / Q_fuel
    assert_true(abs(closure - 1.0) < 1e-9, f"energy balance closes: {closure:.9f}")
    # recovered heat cannot exceed available exhaust enthalpy
    assert_true(Q_th <= s["exhaust_available_w"] + 1e-6,
                f"Q_th={Q_th/1e3:.0f} <= available exhaust {s['exhaust_available_w']/1e3:.0f} kW")


def test_pressure_ratio_effect():
    print("\n[Test 6] Higher pressure ratio raises electrical efficiency")
    m, _ = make_model()
    s_low = m.cycle_state(1.0, rp=8.0)
    s_high = m.cycle_state(1.0, rp=20.0)
    assert_true(s_high["eta_electrical"] > s_low["eta_electrical"],
                f"eta_el rp=20 ({s_high['eta_electrical']*100:.1f}%) > rp=8 ({s_low['eta_electrical']*100:.1f}%)")


def test_ambient_derate():
    print("\n[Test 7] Hotter ambient lowers net specific work / efficiency")
    m, _ = make_model()
    s_cold = m.cycle_state(1.0, T_amb=278.15)
    s_hot = m.cycle_state(1.0, T_amb=313.15)
    assert_true(s_hot["w_net_jkg"] < s_cold["w_net_jkg"],
                f"w_net hot ({s_hot['w_net_jkg']:.0f}) < cold ({s_cold['w_net_jkg']:.0f}) J/kg")


def test_hrsg_transient_warms_and_settles():
    print("\n[Test 8] HRSG thermal ODE: cold start warms toward steady T_m")
    m, _ = make_model()
    sim = m.simulate(1.0, dt=2.0, duration_s=1200.0)
    assert_true(sim["solver_success"], "solve_ivp succeeded")
    T = sim["T_hrsg_K"]
    assert_true(T[-1] > T[0], f"HRSG warms up: {T[0]:.1f} -> {T[-1]:.1f} K")
    assert_true(abs(T[-1] - sim["T_hrsg_steady_K"]) < 5.0,
                f"settles near steady {sim['T_hrsg_steady_K']:.1f} K (final {T[-1]:.1f} K)")
    # steady HRSG temp must lie between ambient and exhaust inlet
    assert_true(m.T_amb < sim["T_hrsg_steady_K"] < sim["T_exhaust_K"][0],
                f"T_amb < T_m_ss={sim['T_hrsg_steady_K']:.0f} < T_exh={sim['T_exhaust_K'][0]:.0f}")


def test_transient_heat_monotone():
    print("\n[Test 9] Delivered useful heat rises monotonically during warm-up")
    m, _ = make_model()
    sim = m.simulate(1.0, dt=2.0, duration_s=900.0)
    Q = sim["thermal_power_w"]
    diffs = np.diff(Q)
    # Allow tiny solver jitter on the saturated plateau (~1e-6 of full duty)
    tol = 1e-6 * max(abs(Q).max(), 1.0)
    assert_true(np.all(diffs >= -tol), "thermal power non-decreasing during cold-start warm-up")
    assert_true(Q[0] < Q[-1], f"Q rises {Q[0]/1e3:.0f} -> {Q[-1]/1e3:.0f} kW")


def test_partload_efficiency():
    print("\n[Test 10] Part-load electrical efficiency stays in (0,1) and below full load Carnot")
    m, _ = make_model()
    for plr in [0.4, 0.5, 0.7, 1.0]:
        s = m.cycle_state(plr)
        assert_true(0.0 < s["eta_electrical"] < 1.0, f"PLR={plr}: 0<eta_el<1")
        assert_true(s["eta_electrical"] < s["eta_carnot"], f"PLR={plr}: eta_el<Carnot")


def test_predict_interface():
    print("\n[Test 11] ComponentModel predict() interface")
    _, cm = make_model()
    r = cm.predict({"part_load_ratio": 0.8, "dt": 5.0, "duration_s": 200.0})
    for key in ["electrical_power_kw", "thermal_power_kw", "eta_electrical",
                "eta_thermal", "eta_total", "heat_to_power_ratio", "t", "T_hrsg_K"]:
        assert_true(key in r, f"Key '{key}' in output")
    assert_true(len(r["t"]) == len(r["T_hrsg_K"]), "Transient arrays same length")
    assert_true(r["eta_total"] > r["eta_electrical"], "total > electrical via interface")


def test_benchmark():
    print("\n[Test 12] Benchmark: 600 s HRSG transient at dt=1")
    m, _ = make_model()
    t0 = time.perf_counter()
    m.simulate(1.0, dt=1.0, duration_s=600.0)
    elapsed = time.perf_counter() - t0
    print(f"  600 s HRSG transient in {elapsed*1000:.1f} ms")
    assert_true(elapsed < 5.0, "Completes in < 5 s")


if __name__ == "__main__":
    tests = [
        test_brayton_temperatures,
        test_electrical_efficiency_below_carnot,
        test_chp_total_efficiency_bounds,
        test_thermal_efficiency_positive,
        test_energy_conservation,
        test_pressure_ratio_effect,
        test_ambient_derate,
        test_hrsg_transient_warms_and_settles,
        test_transient_heat_monotone,
        test_partload_efficiency,
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
    print(f"EC105 Gas Turbine CHP F2a -- Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    sys.exit(0 if failed == 0 else 1)
