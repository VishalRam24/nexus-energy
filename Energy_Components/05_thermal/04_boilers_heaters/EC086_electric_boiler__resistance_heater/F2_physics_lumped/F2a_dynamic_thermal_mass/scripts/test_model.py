"""
EC086 -- Electric Boiler / Resistance Heater -- F2a Dynamic Thermal Mass
Test suite: energy conservation, monotonicity, control, edge cases.
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from model import ElectricBoilerF2a
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
def test_heats_up():
    print("\n[Test 1] Cold start: water heats toward setpoint")
    m, _ = make_model()
    r = m.simulate(293.15, 0.0, dt=5.0, duration_s=3600.0)
    assert_true(r["temperature"][-1] > 293.15 + 30.0,
                f"T rises from 20C to {r['temperature'][-1]-273.15:.1f}C")
    assert_true(r["temperature"][-1] <= m.T_set + m.T_db,
                f"T does not overshoot setpoint+db: {r['temperature'][-1]-273.15:.1f}C")


def test_energy_conservation():
    print("\n[Test 2] Energy conservation: in = loss + load + stored")
    m, _ = make_model()
    r = m.simulate(293.15, 0.03, dt=2.0, duration_s=7200.0, control="onoff")
    e = r["energy"]
    rel = abs(e["E_residual_J"]) / max(abs(e["E_thermal_in_J"]), 1.0)
    assert_true(rel < 1e-3,
                f"residual/in = {rel:.2e} (E_in={e['E_thermal_in_J']:.3e} J)")


def test_efficiency_le_one():
    print("\n[Test 3] Instantaneous efficiency <= 1 everywhere")
    m, _ = make_model()
    r = m.simulate(293.15, 0.05, dt=5.0, duration_s=3600.0)
    assert_true(np.all(r["efficiency"] <= 1.0 + 1e-9),
                f"max efficiency = {r['efficiency'].max():.4f}")
    assert_true(np.all(r["efficiency"] >= 0.0),
                "efficiency >= 0 everywhere")


def test_overall_efficiency_le_one():
    print("\n[Test 4] Overall delivered/electrical energy <= eta_elec")
    m, _ = make_model()
    # warm vessel, steady draw so load dominates
    r = m.simulate(343.15, 0.05, dt=5.0, duration_s=7200.0)
    e = r["energy"]
    overall = e["E_load_J"] / max(e["E_elec_J"], 1.0)
    assert_true(overall <= m.eta_elec + 1e-6,
                f"overall load/elec = {overall:.4f} <= eta_elec={m.eta_elec}")


def test_temperature_rise_under_power():
    print("\n[Test 5] Forced power raises T (dT/dt > 0 at start, no draw)")
    m, _ = make_model()
    r = m.simulate(293.15, 0.0, dt=1.0, duration_s=60.0,
                   P_input=m.P_rated)
    assert_true(r["temperature"][5] > r["temperature"][0],
                "T increasing under forced full power")
    # analytic check: dT/dt ~ eta*P / C_th initially
    dTdt_expect = m.eta_elec * m.P_rated / m.C_th
    dTdt_num = (r["temperature"][1] - r["temperature"][0]) / (r["t"][1] - r["t"][0])
    assert_true(abs(dTdt_num - dTdt_expect) / dTdt_expect < 0.05,
                f"dT/dt num={dTdt_num:.4e} vs analytic={dTdt_expect:.4e} K/s")


def test_steady_state_bound():
    print("\n[Test 6] Numeric SS approaches analytic steady_temperature")
    m, _ = make_model()
    # finite draw so full-on SS is bounded; force full power via P_input.
    mdot = 0.03
    r = m.simulate(293.15, mdot, dt=10.0, duration_s=40000.0,
                   P_input=m.P_rated)
    T_ss_analytic = m.steady_temperature(mdot)
    assert_true(abs(r["temperature"][-1] - T_ss_analytic) < 1.0,
                f"T_num={r['temperature'][-1]-273.15:.2f}C vs "
                f"T_analytic={T_ss_analytic-273.15:.2f}C")


def test_load_draw_cools():
    print("\n[Test 7] Large cold draw lowers T below setpoint (load > capacity)")
    m, _ = make_model()
    # draw large enough that eta*P < mdot*cp*(T_set - T_inlet)
    big = (m.eta_elec * m.P_rated) / (m.cp_water * (m.T_set - m.T_inlet)) * 2.0
    r = m.simulate(m.T_set, big, dt=2.0, duration_s=1200.0)
    assert_true(r["temperature"][-1] < m.T_set,
                f"Heavy draw pulls T below setpoint: {r['temperature'][-1]-273.15:.1f}C")


def test_standby_loss_cools():
    print("\n[Test 8] No power, no draw: vessel cools toward ambient")
    m, _ = make_model()
    r = m.simulate(343.15, 0.0, dt=10.0, duration_s=20000.0,
                   P_input=0.0)
    assert_true(r["temperature"][-1] < 343.15,
                f"Cools from 70C to {r['temperature'][-1]-273.15:.1f}C")
    assert_true(r["temperature"][-1] > m.T_ambient - 0.5,
                "Does not cool below ambient")


def test_thermostat_cycles():
    print("\n[Test 9] On/off thermostat cycles (firing fraction toggles)")
    m, _ = make_model()
    r = m.simulate(m.T_set, 0.02, dt=2.0, duration_s=3600.0, control="onoff")
    u = r["firing_fraction"]
    assert_true(set(np.unique(np.round(u))) <= {0.0, 1.0},
                "on/off fractions are bang-bang {0,1}")
    assert_true(u.max() == 1.0 and u.min() == 0.0,
                "thermostat both fires and cuts out over the hour")


def test_modulating_holds_setpoint():
    print("\n[Test 10] Modulating control holds T near setpoint")
    m, _ = make_model()
    r = m.simulate(293.15, 0.02, dt=5.0, duration_s=7200.0,
                   control="modulating")
    err = abs(r["temperature"][-1] - m.T_set)
    assert_true(err < 2.0,
                f"|T_final - T_set| = {err:.2f} K under modulating control")
    assert_true(np.all((r["firing_fraction"] >= 0.0) &
                       (r["firing_fraction"] <= 1.0)),
                "modulating firing fraction in [0,1]")


def test_predict_interface():
    print("\n[Test 11] ComponentModel predict() interface")
    _, cm = make_model()
    r = cm.predict({"T_init_K": 293.15, "mdot_kg_s": 0.03,
                    "dt": 5.0, "duration_s": 600.0})
    for key in ["t", "temperature", "firing_fraction", "P_elec_W",
                "Q_loss_W", "Q_load_W", "efficiency", "energy"]:
        assert_true(key in r, f"Key '{key}' in output")
    assert_true(len(r["t"]) == len(r["temperature"]), "Arrays same length")
    info = cm.get_info()
    assert_true(info["component_id"] == "EC086", "get_info id == EC086")


def test_benchmark():
    print("\n[Test 12] Benchmark: 1 h sim at dt=1 s")
    m, _ = make_model()
    t0 = time.perf_counter()
    m.simulate(293.15, 0.03, dt=1.0, duration_s=3600.0)
    elapsed = time.perf_counter() - t0
    print(f"  3600 s simulation in {elapsed*1000:.1f} ms")
    assert_true(elapsed < 5.0, "Completes in < 5 s")


if __name__ == "__main__":
    tests = [
        test_heats_up,
        test_energy_conservation,
        test_efficiency_le_one,
        test_overall_efficiency_le_one,
        test_temperature_rise_under_power,
        test_steady_state_bound,
        test_load_draw_cools,
        test_standby_loss_cools,
        test_thermostat_cycles,
        test_modulating_holds_setpoint,
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
    print(f"EC086 Electric Boiler F2a -- Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    sys.exit(0 if failed == 0 else 1)
