"""
EC155 -- Geothermal District Heating -- F2a Lumped Network Thermal Transient
Test suite: energy conservation, thermal monotonicity, known limits, edge
cases, predict() interface, benchmark timing.
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from model import GeothermalDH_F2a
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


# --------------------------------------------------------------------------- #
def test_supply_above_return():
    print("\n[Test 1] T_supply > T_return throughout (heating loop)")
    m, _ = make_model()
    r = m.simulate(Q_load_kW=6000.0, dt=300.0, duration_s=86400.0)
    assert_true(np.all(r["T_supply"] > r["T_return"] - 1e-6),
                "T_supply >= T_return at every step")
    assert_true(r["T_supply"][-1] > r["T_return"][-1] + 1.0,
                f"Final dT={r['T_supply'][-1]-r['T_return'][-1]:.2f} C > 1 C")


def test_energy_conservation():
    print("\n[Test 2] Lumped energy balance closes (dStored = in - out)")
    m, _ = make_model()
    r = m.simulate(Q_load_kW=6000.0, dt=120.0, duration_s=86400.0)
    # Net energy injected minus delivered minus losses = stored change
    t = r["t"]
    Q_in = (r["Q_geo_kW"] + r["Q_boiler_kW"]) * 1e3          # W
    Q_out = (r["Q_load_kW"] + r["Q_loss_kW"]) * 1e3          # W
    E_net = np.trapezoid(Q_in - Q_out, t)                    # J over horizon
    dE_store = (m.C_s * (r["T_supply"][-1] - r["T_supply"][0]) +
                m.C_r * (r["T_return"][-1] - r["T_return"][0]))
    rel = abs(E_net - dE_store) / (abs(dE_store) + 1e3)
    assert_true(rel < 0.02,
                f"Energy balance closes: dStore={dE_store/1e9:.3f} GJ vs "
                f"net={E_net/1e9:.3f} GJ (rel err {rel*100:.2f}%)")


def test_delivered_tracks_load_steady():
    print("\n[Test 3] At steady state delivered heat tracks the load")
    m, _ = make_model()
    s = m.steady_state(Q_load_kW=5000.0)
    # Return-side steady state implies advected heat = load + return-side loss
    return_loss = m.UA_r * (s["T_return"] - m.T_ground) / 1e3
    assert_true(abs(s["Q_delivered_kW"] - (s["Q_load_kW"] + return_loss)) < 5.0,
                f"Q_delivered={s['Q_delivered_kW']:.1f} ~ load+return_loss="
                f"{s['Q_load_kW']+return_loss:.1f} kW")


def test_reinject_below_source_above_min():
    print("\n[Test 4] T_reinject_min <= T_reinject < T_geo_source")
    m, _ = make_model()
    r = m.simulate(Q_load_kW=6000.0, dt=300.0, duration_s=86400.0)
    assert_true(np.all(r["T_reinject"] < r["T_geo_source"] + 1e-6),
                "Reinjection cooler than wellhead")
    assert_true(np.all(r["T_reinject"] >= m.T_reinject_min - 1e-6),
                f"Reinjection >= T_reinject_min ({m.T_reinject_min} C)")


def test_higher_load_lower_temp():
    print("\n[Test 5] Higher demand -> lower steady supply temperature")
    m, _ = make_model()
    s_lo = m.steady_state(Q_load_kW=3000.0, boiler_on=False)
    s_hi = m.steady_state(Q_load_kW=9000.0, boiler_on=False)
    assert_true(s_hi["T_supply"] < s_lo["T_supply"],
                f"T_supply(9MW)={s_hi['T_supply']:.2f} < "
                f"T_supply(3MW)={s_lo['T_supply']:.2f} C")


def test_boiler_fires_under_high_load():
    print("\n[Test 6] Peak boiler engages only when geo cannot meet the load")
    m, _ = make_model()
    s_lo = m.steady_state(Q_load_kW=2000.0, boiler_on=True)
    s_hi = m.steady_state(Q_load_kW=10000.0, boiler_on=True)
    assert_true(s_lo["Q_boiler_kW"] < 1.0,
                f"Low load -> boiler off ({s_lo['Q_boiler_kW']:.2f} kW)")
    assert_true(s_hi["Q_boiler_kW"] > 0.0,
                f"High load -> boiler on ({s_hi['Q_boiler_kW']:.1f} kW)")
    assert_true(s_hi["Q_boiler_kW"] <= m.Q_boiler_max / 1e3 + 1e-6,
                "Boiler respects capacity cap")


def test_geo_heat_positive_and_monotone_in_dT():
    print("\n[Test 7] HX heat >= 0 and increases with source-return gap")
    m, _ = make_model()
    q_cold_net = m.hx_heat_to_network(T_r=30.0)
    q_warm_net = m.hx_heat_to_network(T_r=60.0)
    assert_true(q_cold_net >= q_warm_net - 1e-6,
                f"Colder return extracts more heat: {q_cold_net/1e3:.1f} >= "
                f"{q_warm_net/1e3:.1f} kW")
    assert_true(m.hx_heat_to_network(T_r=200.0) == 0.0,
                "No heat when return hotter than source")


def test_cascade_recovers_residual():
    print("\n[Test 8] Cascade heat >= 0 and zero at minimum reinjection")
    m, _ = make_model()
    qc_hot = m.cascade_heat(T_reinject=50.0)
    qc_min = m.cascade_heat(T_reinject=m.T_reinject_min)
    assert_true(qc_hot > 0.0, f"Cascade recovers residual heat ({qc_hot/1e3:.1f} kW)")
    assert_true(abs(qc_min) < 1e-9, "No cascade heat at minimum reinjection T")


def test_zero_load_warms_to_source():
    print("\n[Test 9] Zero load + no boiler: network warms toward source")
    m, _ = make_model()
    r = m.simulate(Q_load_kW=0.0, T_s0=40.0, T_r0=35.0,
                   dt=300.0, duration_s=172800.0, boiler_on=False)
    assert_true(r["T_supply"][-1] > r["T_supply"][0],
                f"Supply warms {r['T_supply'][0]:.1f}->{r['T_supply'][-1]:.1f} C")
    assert_true(r["T_supply"][-1] < m.T_geo_source + 1e-6,
                "Supply stays below geothermal source temperature")


def test_steady_state_settles():
    print("\n[Test 10] Solution reaches an approximate steady state")
    m, _ = make_model()
    r = m.simulate(Q_load_kW=6000.0, dt=600.0, duration_s=172800.0)
    dTs = abs(r["T_supply"][-1] - r["T_supply"][-2])
    dTr = abs(r["T_return"][-1] - r["T_return"][-2])
    assert_true(dTs < 0.05 and dTr < 0.05,
                f"Near SS: dT_s={dTs:.4f}, dT_r={dTr:.4f} C/step")
    assert_true(r["success"], "solve_ivp reported success")


def test_predict_interface():
    print("\n[Test 11] ComponentModel predict()/get_info() interface")
    _, cm = make_model()
    info = cm.get_info()
    assert_true(info["component_id"] == "EC155", "component_id EC155")
    r = cm.predict({"Q_load_kW": 5000.0, "dt": 600.0, "duration_s": 21600.0})
    for key in ["t", "T_supply", "T_return", "T_reinject", "Q_geo_kW",
                "Q_boiler_kW", "Q_load_kW", "Q_cascade_kW", "Q_loss_kW",
                "Q_delivered_kW"]:
        assert_true(key in r, f"Key '{key}' in output")
    assert_true(len(r["t"]) == len(r["T_supply"]) == len(r["Q_geo_kW"]),
                "All output arrays same length")


def test_time_varying_load():
    print("\n[Test 12] Time-varying load callable accepted")
    _, cm = make_model()
    load = lambda t: 4000.0 + 3000.0 * np.sin(2 * np.pi * t / 86400.0)
    r = cm.predict({"Q_load_kW": load, "dt": 600.0, "duration_s": 86400.0})
    assert_true(r["Q_load_kW"].max() > r["Q_load_kW"].min() + 1000.0,
                "Load varies over the day")
    assert_true(np.all(np.isfinite(r["T_supply"])), "Supply stays finite")
    assert_true(np.all(r["T_supply"] > r["T_return"] - 1e-6),
                "T_supply > T_return under varying load")


def test_benchmark():
    print("\n[Test 13] Benchmark: 1-day sim at dt=60 s")
    m, _ = make_model()
    t0 = time.perf_counter()
    m.simulate(Q_load_kW=6000.0, dt=60.0, duration_s=86400.0)
    elapsed = time.perf_counter() - t0
    print(f"  1-day simulation in {elapsed*1000:.1f} ms")
    assert_true(elapsed < 5.0, "Completes in < 5 s")


if __name__ == "__main__":
    tests = [
        test_supply_above_return,
        test_energy_conservation,
        test_delivered_tracks_load_steady,
        test_reinject_below_source_above_min,
        test_higher_load_lower_temp,
        test_boiler_fires_under_high_load,
        test_geo_heat_positive_and_monotone_in_dT,
        test_cascade_recovers_residual,
        test_zero_load_warms_to_source,
        test_steady_state_settles,
        test_predict_interface,
        test_time_varying_load,
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
    print(f"EC155 Geothermal DH F2a -- Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    sys.exit(0 if failed == 0 else 1)
