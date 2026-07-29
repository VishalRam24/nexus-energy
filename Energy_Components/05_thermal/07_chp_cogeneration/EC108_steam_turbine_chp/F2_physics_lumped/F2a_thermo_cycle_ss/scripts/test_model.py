"""
EC108 -- Steam Turbine CHP -- F2a Physics-Lumped Thermo-Cycle
Test suite: steam-property sanity, cycle conservation, CHP-efficiency
bounds, Carnot bound, thermal ODE transient, predict() interface.
Run:  python3 scripts/test_model.py   (system python3, NOT pytest)
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from model import SteamTurbineCHPF2a
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
def test_steam_property_sanity():
    print("\n[Test 1] Steam properties (IF97-simplified) physical")
    m, _ = make_model()
    # Tsat: 1 bar ~ 100 degC, higher P -> higher Tsat
    assert_true(abs(m.Tsat(1.0) - 100.0) < 3.0, f"Tsat(1 bar)={m.Tsat(1.0):.1f} ~ 100 degC")
    assert_true(m.Tsat(60.0) > m.Tsat(4.0), "Tsat rises with pressure")
    # h_g > h_f, latent heat positive and decreasing with pressure
    assert_true(m.h_g(10.0) > m.h_f(10.0), "h_g > h_f")
    assert_true(m.h_fg(4.0) > m.h_fg(60.0) > 0, "h_fg positive, falls with P")
    # superheated enthalpy exceeds saturated vapour
    assert_true(m.h_superheat(60.0, 480.0) > m.h_g(60.0), "h_superheat > h_g")
    # rough magnitude: live steam ~3300-3400 kJ/kg at 60 bar/480 C
    h1 = m.h_superheat(60.0, 480.0)
    assert_true(3200 < h1 < 3500, f"h1={h1:.0f} kJ/kg near steam-table value")


def test_entropy_monotone():
    print("\n[Test 2] Entropy rises with superheat; s_g > s_f")
    m, _ = make_model()
    assert_true(m.s_g(60.0) > m.s_f(60.0), "s_g > s_f")
    assert_true(m.s_superheat(60.0, 480.0) > m.s_g(60.0), "superheat raises entropy")


def test_isentropic_drop_positive():
    print("\n[Test 3] Isentropic expansion produces positive work, exit enthalpy lower")
    m, _ = make_model()
    ss = m.steady_state(1.0)
    assert_true(ss["w_isentropic_kj_kg"] > 0, f"w_is={ss['w_isentropic_kj_kg']:.1f} > 0")
    assert_true(ss["h2s_kj_kg"] < ss["h1_kj_kg"], "h2s < h1 (expansion)")
    # actual work less than isentropic ideal (eta_is < 1)
    assert_true(ss["w_actual_kj_kg"] < ss["w_isentropic_kj_kg"],
                "actual work < isentropic work")
    assert_true(ss["h2_kj_kg"] > ss["h2s_kj_kg"], "real exit enthalpy > isentropic exit")


def test_energy_conservation():
    print("\n[Test 4] Energy balance: P_el + Q_useful <= Q_fuel")
    m, _ = make_model()
    for plr in [0.3, 0.5, 0.75, 1.0]:
        ss = m.steady_state(plr)
        lhs = ss["P_el_kw"] + ss["Q_useful_kw"]
        assert_true(lhs <= ss["Q_fuel_kw"] + 1e-6,
                    f"PLR={plr}: P_el+Q_use={lhs:.0f} <= Q_fuel={ss['Q_fuel_kw']:.0f}")


def test_chp_efficiency_bounds():
    print("\n[Test 5] CHP efficiency bounds: eta_el < eta_total < 1")
    m, _ = make_model()
    for plr in [0.3, 0.6, 1.0]:
        ss = m.steady_state(plr)
        assert_true(0 < ss["eta_el"] < ss["eta_total"],
                    f"PLR={plr}: eta_el={ss['eta_el']:.3f} < eta_total={ss['eta_total']:.3f}")
        assert_true(ss["eta_total"] < 1.0,
                    f"PLR={plr}: eta_total={ss['eta_total']:.3f} < 1")
        assert_true(ss["eta_total"] > 0.7,
                    f"PLR={plr}: eta_total={ss['eta_total']:.3f} > 0.7 (CHP regime)")


def test_carnot_bound():
    print("\n[Test 6] Electrical efficiency below Carnot for the power conversion")
    m, _ = make_model()
    ss = m.steady_state(1.0)
    # back-pressure CHP: power conversion (work / heat-to-work share) must
    # respect Carnot between live steam and heat-rejection temperatures
    eta_power = ss["w_actual_kj_kg"] / (ss["h1_kj_kg"] -
                                        m.CP_LIQ * m.T_return)
    assert_true(eta_power < ss["eta_carnot"],
                f"power conversion {eta_power:.3f} < Carnot {ss['eta_carnot']:.3f}")
    assert_true(ss["eta_el"] < ss["eta_carnot"],
                f"eta_el {ss['eta_el']:.3f} < Carnot {ss['eta_carnot']:.3f}")


def test_power_to_heat_ratio():
    print("\n[Test 7] Power-to-heat ratio in steam-CHP range (~0.1-0.5)")
    m, _ = make_model()
    ss = m.steady_state(1.0)
    pth = ss["power_to_heat"]
    assert_true(0.05 < pth < 0.8, f"power-to-heat={pth:.3f} in plausible range")
    assert_true(abs(ss["HPR"] * pth - 1.0) < 1e-6, "HPR = 1/power_to_heat")


def test_partload_monotone():
    print("\n[Test 8] Outputs scale up with PLR")
    m, _ = make_model()
    s_lo = m.steady_state(0.4)
    s_hi = m.steady_state(1.0)
    assert_true(s_hi["P_el_kw"] > s_lo["P_el_kw"], "P_el rises with PLR")
    assert_true(s_hi["Q_useful_kw"] > s_lo["Q_useful_kw"], "Q_useful rises with PLR")


def test_thermal_ode_warmup():
    print("\n[Test 9] Lumped thermal ODE: boiler warms from cold start")
    m, _ = make_model()
    r = m.simulate(1.0, T0_C=80.0, duration_s=1800.0, dt=10.0)
    assert_true(r["success"], "solve_ivp succeeded")
    assert_true(r["T_boiler_C"][-1] > r["T_boiler_C"][0],
                f"T_boiler {r['T_boiler_C'][0]:.0f} -> {r['T_boiler_C'][-1]:.0f} degC warms up")
    T_op = m.Tsat(m.P_boiler)
    assert_true(r["T_boiler_C"][-1] < T_op + 50.0,
                f"T_final={r['T_boiler_C'][-1]:.0f} bounded near Tsat={T_op:.0f}")
    # electrical output increases as boiler reaches readiness
    assert_true(r["P_el_kw"][-1] > r["P_el_kw"][0] - 1e-6, "P_el ramps with readiness")


def test_thermal_ode_steady():
    print("\n[Test 10] Thermal ODE approaches steady state")
    m, _ = make_model()
    r = m.simulate(1.0, T0_C=None, duration_s=3600.0, dt=20.0)
    dT = abs(r["T_boiler_C"][-1] - r["T_boiler_C"][-2])
    assert_true(dT < 0.5, f"near steady state: dT={dT:.4f} K between last steps")


def test_predict_interface():
    print("\n[Test 11] ComponentModel predict()/get_info() interface")
    _, cm = make_model()
    info = cm.get_info()
    for k in ["component_id", "component_name", "fidelity", "version", "inputs", "outputs"]:
        assert_true(k in info, f"get_info has '{k}'")
    assert_true(info["component_id"] == "EC108", "component_id == EC108")
    r = cm.predict({"PLR": 0.8})
    for key in ["P_el_kw", "Q_useful_kw", "Q_fuel_kw", "eta_el", "eta_th",
                "eta_total", "power_to_heat", "eta_carnot"]:
        assert_true(key in r, f"predict output has '{key}'")
    rt = cm.predict({"PLR": 1.0, "transient": True, "duration_s": 600.0, "dt": 20.0})
    assert_true("transient" in rt and "t" in rt["transient"],
                "transient time series returned")


def test_benchmark():
    print("\n[Test 12] Benchmark: 1800 s transient + 100 steady evals")
    m, _ = make_model()
    t0 = time.perf_counter()
    m.simulate(1.0, duration_s=1800.0, dt=5.0)
    for plr in np.linspace(0.3, 1.0, 100):
        m.steady_state(plr)
    elapsed = time.perf_counter() - t0
    print(f"  completed in {elapsed*1000:.1f} ms")
    assert_true(elapsed < 5.0, "completes in < 5 s")


if __name__ == "__main__":
    tests = [
        test_steam_property_sanity,
        test_entropy_monotone,
        test_isentropic_drop_positive,
        test_energy_conservation,
        test_chp_efficiency_bounds,
        test_carnot_bound,
        test_power_to_heat_ratio,
        test_partload_monotone,
        test_thermal_ode_warmup,
        test_thermal_ode_steady,
        test_predict_interface,
        test_benchmark,
    ]
    passed = failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"  ERROR: {e}")

    print(f"\n{'='*60}")
    print(f"EC108 Steam Turbine CHP F2a -- Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    sys.exit(0 if failed == 0 else 1)
