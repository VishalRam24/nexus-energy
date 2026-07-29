"""
EC114 -- Supercritical / Ultra-Supercritical Coal Plant -- F2a Physics-Lumped
Test suite: thermodynamic sanity, conservation laws, Carnot bound,
SC>subcritical, realistic band, evaporator ODE, predict() interface, benchmark.

Run:  python3 scripts/test_model.py   (system python3, NOT pytest)
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from model import SupercriticalCoalF2a, WaterSteam
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
def test_steam_properties():
    print("\n[Test 1] Steam-table correlations near IAPWS anchor points")
    W = WaterSteam
    h = W.h_superheated(250.0, 600.0)        # IAPWS ~3491 kJ/kg
    assert_true(3300 < h < 3650, f"h(250bar,600C)={h:.0f} kJ/kg ~ IAPWS 3491")
    h2 = W.h_superheated(55.0, 600.0)        # IAPWS ~3666
    assert_true(3450 < h2 < 3800, f"h(55bar,600C)={h2:.0f} kJ/kg ~ IAPWS 3666")
    Ts = W.Tsat(0.05)                        # ~32.9 C
    assert_true(28 < Ts < 38, f"Tsat(0.05bar)={Ts:.1f}C ~ 32.9C")
    assert_true(W.P_CRIT > 220.0, f"critical pressure {W.P_CRIT} bar > 220.6")


def test_supercritical_pressure():
    print("\n[Test 2] Boiler operates above the water critical pressure")
    m, _ = make_model()
    assert_true(m.P_boiler > WaterSteam.P_CRIT,
                f"P_boiler={m.P_boiler} bar > critical {WaterSteam.P_CRIT} bar (supercritical)")
    assert_true(m.T_main >= 540.0, f"T_main={m.T_main}C in SC/USC range (>=540C)")


def test_efficiency_below_carnot():
    print("\n[Test 3] Cycle efficiency < Carnot bound")
    m, _ = make_model()
    c = m.compute_cycle(1.0)
    assert_true(0.0 < c["eta_cycle"] < c["eta_carnot"],
                f"eta_cycle={c['eta_cycle']:.4f} < Carnot={c['eta_carnot']:.4f}")
    assert_true(c["eta_net"] < c["eta_cycle"],
                f"eta_net={c['eta_net']:.4f} < eta_cycle={c['eta_cycle']:.4f} (losses)")


def test_net_above_subcritical():
    print("\n[Test 4] Net efficiency exceeds subcritical reference")
    m, _ = make_model()
    c = m.compute_cycle(1.0)
    assert_true(c["eta_net"] > m.eta_subcritical_ref,
                f"eta_net={c['eta_net']:.4f} > subcritical ref={m.eta_subcritical_ref:.3f}")


def test_realistic_efficiency_band():
    print("\n[Test 5] Net efficiency in realistic SC/USC band (0.40-0.50)")
    m, _ = make_model()
    c = m.compute_cycle(1.0)
    assert_true(0.40 <= c["eta_net"] <= 0.50,
                f"eta_net={c['eta_net']:.4f} in [0.40, 0.50] (USC ~44-47%)")
    co2 = c["co2_intensity_g_per_kwh"]
    assert_true(680 <= co2 <= 900,
                f"CO2 intensity={co2:.0f} g/kWh in [680,900] (SC/USC realistic)")


def test_energy_conservation():
    print("\n[Test 6] Cycle energy balance: w_net + q_out = q_in")
    m, _ = make_model()
    c = m.compute_cycle(1.0)
    lhs = c["w_net"] + c["q_out"]
    rhs = c["q_in"]
    rel = abs(lhs - rhs) / rhs
    assert_true(rel < 0.02,
                f"|(w_net+q_out)-q_in|/q_in = {rel:.4f} < 2% "
                f"(w_net={c['w_net']:.0f}, q_out={c['q_out']:.0f}, q_in={c['q_in']:.0f})")


def test_mass_energy_plant():
    print("\n[Test 7] Plant fuel/mass balance: P_net = eta_net * Q_fuel")
    m, _ = make_model()
    c = m.compute_cycle(1.0)
    P_from_fuel = c["eta_net"] * c["Q_fuel_MW"]
    rel = abs(P_from_fuel - c["P_net_MW"]) / c["P_net_MW"]
    assert_true(rel < 1e-6, f"P_net={c['P_net_MW']:.1f} = eta*Q_fuel ({rel:.2e})")
    co2_check = c["coal_rate_kgs"] * m.CO2_per_kg
    assert_true(abs(co2_check - c["co2_rate_kgs"]) < 1e-6,
                "CO2 rate = coal_rate * CO2_per_kg (mass conservation)")


def test_partload_monotone():
    print("\n[Test 8] Net power monotone in PLR; efficiency degrades at part-load")
    m, _ = make_model()
    plrs = [0.3, 0.5, 0.7, 1.0]
    P_prev, eta_full = -1.0, m.compute_cycle(1.0)["eta_net"]
    for plr in plrs:
        c = m.compute_cycle(plr)
        assert_true(c["P_net_MW"] > P_prev, f"P({plr})={c['P_net_MW']:.0f} MW rising")
        P_prev = c["P_net_MW"]
        assert_true(c["eta_net"] <= eta_full + 1e-9,
                    f"eta({plr})={c['eta_net']:.4f} <= full-load {eta_full:.4f}")


def test_evaporator_ode():
    print("\n[Test 9] Lumped evaporator ODE: bounded, responds to load drop")
    m, _ = make_model()
    # step DOWN in load -> firing falls, evaporator cools toward new balance
    def load(t):
        return 1.0 if t < 300 else 0.5
    r = m.simulate(load, dt=15.0, duration_s=1200.0)
    T0, Tend = r["T_evap"][0], r["T_evap"][-1]
    assert_true(np.all(r["T_evap"] > m.T_feedwater),
                f"T_evap stays above feedwater T ({m.T_feedwater}C)")
    assert_true(np.all(r["T_evap"] < m.T_flame),
                f"T_evap stays below flame T ({m.T_flame}C)")
    assert_true(Tend < T0 + 1.0,
                f"T_evap responds to load drop: {T0:.1f} -> {Tend:.1f} C")


def test_evaporator_heat_balance():
    print("\n[Test 10] Evaporator: furnace heat ~ steam heat at steady load")
    m, _ = make_model()
    r = m.simulate(1.0, dt=30.0, duration_s=3600.0)
    Qf, Qs = r["Q_furnace_MW"][-1], r["Q_steam_MW"][-1]
    rel = abs(Qf - Qs) / Qs
    assert_true(rel < 0.05,
                f"Q_furnace={Qf:.0f} ~ Q_steam={Qs:.0f} MW at SS (rel {rel:.3f})")


def test_predict_interface():
    print("\n[Test 11] ComponentModel predict() interface (cycle + transient)")
    _, cm = make_model()
    c = cm.predict({"plr": 0.8})
    for key in ["eta_net", "eta_cycle", "eta_carnot", "P_net_MW",
                "coal_rate_kgs", "co2_rate_kgs", "state_points"]:
        assert_true(key in c, f"cycle key '{key}' present")
    assert_true(len(c["state_points"]["h"]) == 9, "9 state-point enthalpies")
    rt = cm.predict({"mode": "transient", "plr": [1.0, 0.7],
                     "dt": 30.0, "duration_s": 600.0})
    assert_true(len(rt["t"]) == len(rt["T_evap"]), "transient arrays aligned")
    info = cm.get_info()
    assert_true(info["component_id"] == "EC114" and info["version"] == "1.0.0",
                "get_info() id/version correct")


def test_benchmark():
    print("\n[Test 12] Benchmark: 1800 s transient at dt=10 s")
    m, _ = make_model()
    t0 = time.perf_counter()
    m.simulate(lambda t: 1.0 if t < 900 else 0.6, dt=10.0, duration_s=1800.0)
    elapsed = time.perf_counter() - t0
    print(f"  1800 s evaporator transient in {elapsed*1000:.1f} ms")
    assert_true(elapsed < 5.0, "Completes in < 5 s")


if __name__ == "__main__":
    tests = [
        test_steam_properties,
        test_supercritical_pressure,
        test_efficiency_below_carnot,
        test_net_above_subcritical,
        test_realistic_efficiency_band,
        test_energy_conservation,
        test_mass_energy_plant,
        test_partload_monotone,
        test_evaporator_ode,
        test_evaporator_heat_balance,
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

    print(f"\n{'='*64}")
    print(f"EC114 SC/USC Coal F2a -- Results: {passed} passed, {failed} failed")
    print(f"{'='*64}")
    sys.exit(0 if failed == 0 else 1)
