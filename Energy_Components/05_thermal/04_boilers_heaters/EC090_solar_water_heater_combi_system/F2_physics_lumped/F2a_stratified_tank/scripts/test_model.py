"""
EC090 -- Solar Water Heater Combi System -- F2a Stratified Tank
Test suite: physics sanity (conservation, stratification, solar fraction),
edge cases (night, no-load charge), predict() interface, benchmark timing.
Run with system python3 (NOT pytest):  python3 scripts/test_model.py
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from model import SolarCombiF2a
from predict import (
    ComponentModel,
    default_irradiance,
    default_ambient,
    default_dhw_load,
    default_space_load,
)

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
def test_collector_hottel_whillier():
    print("\n[Test 1] Collector Hottel-Whillier: gain rises with G, falls with T_in")
    m, _ = make_model()
    Q_lowG = m.collector_useful_gain(300.0, 313.15, 293.15)
    Q_highG = m.collector_useful_gain(900.0, 313.15, 293.15)
    assert_true(Q_highG > Q_lowG, f"Q(900)={Q_highG:.0f} > Q(300)={Q_lowG:.0f} W")
    Q_hotIn = m.collector_useful_gain(900.0, 353.15, 293.15)
    assert_true(Q_hotIn < Q_highG, f"hotter inlet lowers gain: {Q_hotIn:.0f} < {Q_highG:.0f}")
    # intercept: at T_in=T_amb gain = A*FR(ta)*G
    Q_int = m.collector_useful_gain(800.0, 293.15, 293.15)
    expect = m.A_c * m.FR_ta * 800.0
    assert_true(abs(Q_int - expect) < 1e-6, f"intercept matches A*FR(ta)*G = {expect:.0f} W")


def test_collector_zero_at_night():
    print("\n[Test 2] Q_collector = 0 at night (G=0)")
    m, _ = make_model()
    for T_in in [283.15, 313.15, 343.15]:
        Q = m.collector_useful_gain(0.0, T_in, 283.15)
        assert_true(Q == 0.0, f"Q(G=0, T_in={T_in:.1f})={Q:.4f} == 0")
    # and the pump must be off with no sun
    assert_true(not m.pump_on(0.0, 283.15, 300.0, prev_on=True), "pump OFF at night")


def test_no_negative_gain():
    print("\n[Test 3] No negative collector gain (loss-dominated clamp)")
    m, _ = make_model()
    # weak sun, very hot tank bottom -> losses exceed optical gain
    Q = m.collector_useful_gain(50.0, 353.15, 273.15)
    assert_true(Q >= 0.0, f"Q={Q:.4f} clamped >= 0")


def test_stratification_maintained():
    print("\n[Test 4] Stratification: top node hotter than bottom over the day")
    _, cm = make_model()
    r = cm.predict({"dt": 300.0, "duration_s": 86400.0})
    # top should be >= bottom (within small conduction/numerical tolerance) at all times
    viol = np.sum(r["T_top"] < r["T_bottom"] - 0.5)
    assert_true(viol == 0, f"top>=bottom at all {len(r['T_top'])} steps (viol={viol})")
    assert_true(r["T_top"].mean() > r["T_bottom"].mean(),
                f"mean top {r['T_top'].mean()-273.15:.1f}C > bottom {r['T_bottom'].mean()-273.15:.1f}C")


def test_monotone_stratification_profile():
    print("\n[Test 5] Node profile monotone non-increasing top->bottom (mid-afternoon)")
    _, cm = make_model()
    r = cm.predict({"dt": 600.0, "duration_s": 86400.0})
    # pick 14:00 index
    k = int(np.argmin(np.abs(r["t"] - 14 * 3600.0)))
    prof = r["T_nodes"][:, k]
    # allow tiny tolerance for conduction smoothing
    ok = np.all(np.diff(prof) <= 0.5)
    assert_true(ok, f"profile non-increasing top->bottom: {np.round(prof-273.15,1)} C")


def test_solar_fraction_bounds():
    print("\n[Test 6] solar_fraction in [0,1]")
    _, cm = make_model()
    for Gp in [0.0, 400.0, 900.0]:
        r = cm.predict({"G_peak": Gp, "dt": 600.0, "duration_s": 86400.0})
        f = r["solar_fraction"]
        assert_true(0.0 <= f <= 1.0, f"G_peak={Gp}: f_solar={f:.3f} in [0,1]")


def test_solar_fraction_increases_with_sun():
    print("\n[Test 7] solar_fraction increases with irradiance, zero at night")
    _, cm = make_model()
    f0 = cm.predict({"G_peak": 0.0, "dt": 600.0, "duration_s": 86400.0})["solar_fraction"]
    f_sun = cm.predict({"G_peak": 900.0, "dt": 600.0, "duration_s": 86400.0})["solar_fraction"]
    assert_true(abs(f0) < 1e-9, f"f_solar(no sun)={f0:.4f} == 0")
    assert_true(f_sun > f0, f"f_solar(sun)={f_sun:.3f} > f_solar(dark)={f0:.3f}")


def test_energy_conservation_no_load():
    print("\n[Test 8] Energy conservation: closed tank, no draw, no aux, no sun")
    m, _ = make_model()
    # disable aux; set tank-loss ambient == tank temp so wall losses vanish;
    # no sun, no load -> truly closed adiabatic store -> stored energy constant.
    m.Q_aux_max = 0.0
    m.T_amb_tank = 323.15
    N = m.N
    T0 = np.full(N, 323.15)  # uniform 50C
    r = m.simulate(lambda t: 0.0, lambda t: 323.15, lambda t: 0.0,
                   lambda t: 0.0, T_init=T0, dt=600.0, duration_s=21600.0)
    # ambient == tank temp -> no driving dT -> energy ~ constant
    E0 = m.m_node * m.cp * T0.sum()
    Ef = m.m_node * m.cp * r["T_nodes"][:, -1].sum()
    rel = abs(Ef - E0) / E0
    assert_true(rel < 1e-3, f"stored energy conserved (rel drift {rel:.2e}) with T_amb=T_tank")


def test_charge_when_sunny_no_load():
    print("\n[Test 9] Sunny day, no draw -> tank charges (mean temp rises)")
    m, _ = make_model()
    N = m.N
    T0 = np.full(N, 298.15)  # cool 25C start
    r = m.simulate(default_irradiance(900.0), default_ambient(),
                   lambda t: 0.0, lambda t: 0.0,
                   T_init=T0, dt=600.0, duration_s=86400.0)
    assert_true(r["T_mean"].max() > 298.15 + 5.0,
                f"tank warms: peak mean {r['T_mean'].max()-273.15:.1f}C > 30C")
    assert_true(r["E_solar_J"] > 0.0, f"E_solar={r['E_solar_J']/3.6e6:.2f} kWh > 0")


def test_aux_holds_setpoint():
    print("\n[Test 10] Auxiliary backup holds top node near setpoint in the dark")
    m, _ = make_model()
    N = m.N
    T0 = np.full(N, 293.15)  # cold start, no sun -> aux must lift top node
    r = m.simulate(lambda t: 0.0, lambda t: 288.15, lambda t: 0.0,
                   lambda t: 0.0, T_init=T0, dt=300.0, duration_s=43200.0)
    T_top_final = r["T_top"][-1]
    assert_true(T_top_final > m.T_set_aux - m.T_db - 1.0,
                f"top node {T_top_final-273.15:.1f}C reaches aux band (~{m.T_set_aux-273.15:.0f}C)")
    assert_true(T_top_final <= m.T_set_aux + 0.5,
                f"aux does not overshoot setpoint: {T_top_final-273.15:.1f}C")


def test_predict_interface():
    print("\n[Test 11] ComponentModel predict() interface keys + array shapes")
    _, cm = make_model()
    r = cm.predict({"dt": 600.0, "duration_s": 43200.0})
    for key in ["t", "T_nodes", "T_top", "T_bottom", "Q_solar", "Q_aux_fuel",
                "Q_load", "pump_on", "E_solar_J", "solar_fraction"]:
        assert_true(key in r, f"Key '{key}' in output")
    M = len(r["t"])
    assert_true(r["T_nodes"].shape == (cm._model.N, M), "T_nodes shape (N, M)")
    assert_true(len(r["T_top"]) == M and len(r["pump_on"]) == M, "series aligned to t")


def test_benchmark():
    print("\n[Test 12] Benchmark: full 24h day simulation timing")
    _, cm = make_model()
    t0 = time.perf_counter()
    cm.predict({"dt": 300.0, "duration_s": 86400.0})
    elapsed = time.perf_counter() - t0
    print(f"  24h day (dt=300s) simulated in {elapsed*1000:.1f} ms")
    assert_true(elapsed < 5.0, "Completes in < 5 s")


if __name__ == "__main__":
    tests = [
        test_collector_hottel_whillier,
        test_collector_zero_at_night,
        test_no_negative_gain,
        test_stratification_maintained,
        test_monotone_stratification_profile,
        test_solar_fraction_bounds,
        test_solar_fraction_increases_with_sun,
        test_energy_conservation_no_load,
        test_charge_when_sunny_no_load,
        test_aux_holds_setpoint,
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
    print(f"EC090 Solar Combi F2a -- Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    sys.exit(0 if failed == 0 else 1)
