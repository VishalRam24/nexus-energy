"""
EC112 -- Micro Gas Turbine -- F2a Recuperated Brayton Cycle
Test suite: thermodynamic sanity (eff<Carnot, energy conservation,
recuperator gain), part-load, transients, edge cases, predict() interface.
NO pytest -- custom assert harness, run as `python3 scripts/test_model.py`.
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from model import MicroGasTurbineF2a
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
    print("\n[Test 1] Cycle efficiency < Carnot bound (2nd law)")
    m, _ = make_model()
    c = m.cycle()
    assert_true(0.0 < c["eta_thermal"] < c["eta_carnot"],
                f"eta_th={c['eta_thermal']*100:.1f}% < Carnot={c['eta_carnot']*100:.1f}%")
    assert_true(0.0 < c["eta_electrical"] < c["eta_thermal"],
                f"eta_el={c['eta_electrical']*100:.1f}% < eta_th (generator loss)")


def test_efficiency_realistic():
    print("\n[Test 2] Full-load efficiency in micro-turbine range (~28-33%)")
    m, _ = make_model()
    c = m.cycle()
    assert_true(0.27 < c["eta_electrical"] < 0.34,
                f"eta_el={c['eta_electrical']*100:.1f}% matches Capstone C200 class")


def test_recuperator_raises_efficiency():
    print("\n[Test 3] Recuperator raises efficiency (key MGT physics)")
    m, _ = make_model()
    c = m.cycle()
    nr = m.cycle_no_recuperator()
    assert_true(c["eta_electrical"] > nr["eta_electrical"] + 0.05,
                f"with recup {c['eta_electrical']*100:.1f}% >> without {nr['eta_electrical']*100:.1f}%")
    assert_true(nr["eta_electrical"] < 0.22,
                f"non-recuperated low-PR cycle is poor: {nr['eta_electrical']*100:.1f}% (~17% expected)")


def test_energy_conservation():
    print("\n[Test 4] Energy balance: w_net = w_turb - w_comp; q_in = w_net + q_rejected")
    m, _ = make_model()
    c = m.cycle()
    assert_true(abs(c["w_net"] - (c["w_turb"] - c["w_comp"])) < 1e-6,
                "w_net == w_turb - w_comp")
    # first law over the air stream: q_in (per kg) must exceed net work
    assert_true(c["q_in"] > c["w_net"] > 0,
                f"q_in={c['q_in']:.0f} > w_net={c['w_net']:.0f} > 0 J/kg")
    # electrical = shaft work * generator efficiency, then * mdot
    P_check = c["w_net"] * m.mdot_air_rated * m.eta_gen
    assert_true(abs(P_check - c["P_el_W"]) / c["P_el_W"] < 1e-6,
                "P_el == w_net * mdot * eta_gen")


def test_fuel_energy_closure():
    print("\n[Test 5] Fuel mass-flow closes the LHV heat balance")
    m, _ = make_model()
    c = m.cycle()
    Q_from_fuel = c["mdot_fuel_kgs"] * m.LHV
    assert_true(abs(Q_from_fuel - c["Q_fuel_W"]) / c["Q_fuel_W"] < 1e-6,
                "mdot_fuel * LHV == Q_fuel")
    eta_check = c["P_el_W"] / c["Q_fuel_W"]
    assert_true(abs(eta_check - c["eta_electrical"]) < 1e-3,
                f"P_el/Q_fuel={eta_check*100:.1f}% == eta_el")


def test_station_temperature_order():
    print("\n[Test 6] Station temperatures ordered physically")
    m, _ = make_model()
    c = m.cycle()
    assert_true(c["T1"] < c["T2"], f"compressor heats air T1={c['T1']:.0f}<T2={c['T2']:.0f}")
    assert_true(c["T2"] < c["T2r"] < c["T3"],
                f"recup preheat T2<{c['T2r']:.0f}<TIT={c['T3']:.0f}")
    assert_true(c["T4"] < c["T3"], f"turbine cools gas T4={c['T4']:.0f}<TIT")
    assert_true(c["T_exhaust_K"] < c["T4"],
                f"recup cools exhaust T5={c['T_exhaust_K']:.0f}<T4={c['T4']:.0f}")


def test_pressure_ratio_optimum():
    print("\n[Test 7] Recuperated cycle favours LOW pressure ratio")
    m, _ = make_model()
    eta_lo = m.cycle(rp=3.5)["eta_electrical"]
    eta_hi = m.cycle(rp=10.0)["eta_electrical"]
    # in a recuperated cycle, raising rp too far hurts efficiency (recup deltaT shrinks)
    assert_true(eta_lo > eta_hi,
                f"eta(rp=3.5)={eta_lo*100:.1f}% > eta(rp=10)={eta_hi*100:.1f}% (recuperation physics)")


def test_part_load():
    print("\n[Test 8] Part-load: lower PLR -> lower power & speed")
    _, cm = make_model()
    full = cm.predict({"PLR": 1.0})
    half = cm.predict({"PLR": 0.5})
    assert_true(half["P_el_W"] < full["P_el_W"],
                f"P(50%)={half['P_el_W']/1e3:.0f}kW < P(100%)={full['P_el_W']/1e3:.0f}kW")
    assert_true(half["speed_fraction"] < full["speed_fraction"] + 1e-6,
                f"speed drops at part load: {half['speed_fraction']*100:.0f}%")
    assert_true(abs(half["P_el_W"]/1e3 - 100.0) < 5.0,
                f"50% load tracks ~100 kW: {half['P_el_W']/1e3:.1f} kW")


def test_ambient_derate():
    print("\n[Test 9] Hot ambient lowers efficiency & power")
    m, _ = make_model()
    cold = m.cycle(T_amb=278.15)   # 5 C
    hot = m.cycle(T_amb=313.15)    # 40 C
    assert_true(hot["eta_electrical"] < cold["eta_electrical"],
                f"hot eta {hot['eta_electrical']*100:.1f}% < cold {cold['eta_electrical']*100:.1f}%")
    assert_true(hot["w_net"] < cold["w_net"],
                "hot ambient reduces specific net work")


def test_transient_recuperator_warmup():
    print("\n[Test 10] Transient ODE: recuperator metal warms from cold start")
    m, _ = make_model()
    r = m.simulate(fuel_fraction=1.0, T_rec0=288.15, duration_s=120.0, dt=2.0)
    assert_true(r["T_recup"][-1] > r["T_recup"][0] + 100.0,
                f"T_recup {r['T_recup'][0]:.0f}->{r['T_recup'][-1]:.0f} K")
    assert_true(np.all(r["P_el_kw"] > 0),
                "electrical power positive throughout transient")
    # approach to steady state
    dT = abs(r["T_recup"][-1] - r["T_recup"][-2])
    assert_true(dT < 1.0, f"near thermal steady state: dT={dT:.3f} K/step")


def test_transient_speed_stable():
    print("\n[Test 11] Shaft ODE stays near rated speed at full fuel")
    m, _ = make_model()
    r = m.simulate(fuel_fraction=1.0, duration_s=60.0, dt=2.0)
    rpm = r["speed_rpm"]
    assert_true(np.all(rpm > 30000) and np.all(rpm < 60000),
                f"spool speed bounded {rpm.min():.0f}-{rpm.max():.0f} rpm")
    # efficiency stays below Carnot at every instant
    assert_true(np.all(r["eta_electrical"] < r["eta_carnot"]),
                "eta_el < Carnot at every time step")


def test_predict_interface():
    print("\n[Test 12] ComponentModel predict() interface (steady + transient)")
    m, cm = make_model()
    s = cm.predict({"mode": "steady", "PLR": 1.0})
    for k in ["eta_electrical", "eta_carnot", "P_el_W", "T_exhaust_K", "w_net", "q_in"]:
        assert_true(k in s, f"steady key '{k}'")
    t = cm.predict({"mode": "transient", "duration_s": 10.0, "dt": 2.0})
    for k in ["t", "omega", "speed_rpm", "T_recup", "P_el_kw", "eta_electrical"]:
        assert_true(k in t, f"transient key '{k}'")
    assert_true(len(t["t"]) == len(t["P_el_kw"]), "arrays same length")
    info = cm.get_info()
    assert_true(info["component_id"] == "EC112", "component_id == EC112")


def test_benchmark():
    print("\n[Test 13] Benchmark: 120 s transient at dt=0.5")
    m, _ = make_model()
    t0 = time.perf_counter()
    m.simulate(fuel_fraction=1.0, duration_s=120.0, dt=0.5)
    elapsed = time.perf_counter() - t0
    print(f"  120 s simulation in {elapsed*1000:.1f} ms")
    assert_true(elapsed < 5.0, "completes in < 5 s")


if __name__ == "__main__":
    tests = [
        test_efficiency_below_carnot,
        test_efficiency_realistic,
        test_recuperator_raises_efficiency,
        test_energy_conservation,
        test_fuel_energy_closure,
        test_station_temperature_order,
        test_pressure_ratio_optimum,
        test_part_load,
        test_ambient_derate,
        test_transient_recuperator_warmup,
        test_transient_speed_stable,
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
    print(f"EC112 Micro Gas Turbine F2a -- Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    sys.exit(0 if failed == 0 else 1)
