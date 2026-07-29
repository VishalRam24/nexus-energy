"""
EC055 -- Solar Tower / Central Receiver CSP -- F2a Physics-Lumped
Test suite: optics, lumped-receiver ODE, energy conservation, T^4 law,
efficiency-drop-at-high-T, P=0 at DNI=0, predict() interface, benchmark.
NO pytest -- run as:  python3 scripts/test_model.py
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from model import SolarTowerF2a
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
def test_field_efficiency_range():
    print("\n[Test 1] Field optical efficiency in (0,1) and peaks at zenith=0")
    m, _ = make_model()
    e0 = m.field_efficiency(0.0)
    e45 = m.field_efficiency(45.0)
    e80 = m.field_efficiency(80.0)
    assert_true(0.0 < e0 <= 1.0, f"eta_field(0)={e0:.4f} in (0,1]")
    assert_true(e45 < e0, f"eta_field(45)={e45:.4f} < eta_field(0)={e0:.4f}")
    assert_true(e80 < e45, f"eta_field(80)={e80:.4f} < eta_field(45)={e45:.4f}")


def test_radiative_T4_law():
    print("\n[Test 2] Radiative loss scales as (T^4 - T_amb^4)")
    m, _ = make_model()
    T_amb = 298.15
    T1, T2 = 800.0, 900.0
    q1 = m.Q_rad(T1, T_amb)
    q2 = m.Q_rad(T2, T_amb)
    expected_ratio = (T2 ** 4 - T_amb ** 4) / (T1 ** 4 - T_amb ** 4)
    actual_ratio = q2 / q1
    assert_true(abs(actual_ratio - expected_ratio) < 1e-9,
                f"Q_rad ratio {actual_ratio:.4f} matches T^4 law {expected_ratio:.4f}")
    assert_true(abs(m.Q_rad(T_amb, T_amb)) < 1e-9, "Q_rad=0 when T=T_amb")


def test_p_zero_at_dni_zero():
    print("\n[Test 3] DNI=0 => Q_field=0, no useful heat, P_block=0")
    m, _ = make_model()
    r = m.simulate(dni=0.0, zenith_deg=30.0, T0_C=290.0, duration_s=1800.0, dt=60.0)
    assert_true(np.allclose(r["Q_field_W"], 0.0), "Q_field == 0 at DNI=0")
    assert_true(np.allclose(r["P_electric_W"], 0.0), "P_electric == 0 at DNI=0")
    assert_true(np.all(r["Q_thermal_to_PB_W"] >= -1e-6), "Q_to_PB >= 0")


def test_receiver_heats_up():
    print("\n[Test 4] Receiver heats up under high DNI from cold start")
    m, _ = make_model()
    r = m.simulate(dni=950.0, zenith_deg=20.0, T0_C=290.0, mdot_salt=50.0,
                   duration_s=3600.0, dt=30.0)
    assert_true(r["T_receiver_C"][-1] > r["T_receiver_C"][0],
                f"T rose {r['T_receiver_C'][0]:.1f} -> {r['T_receiver_C'][-1]:.1f} degC")
    assert_true(r["T_receiver_C"][-1] < 1200.0,
                f"T_final={r['T_receiver_C'][-1]:.1f} degC stays physical (<1200)")


def test_energy_conservation():
    print("\n[Test 5] Energy balance: Q_abs = dE/dt + Q_loss + Q_htf")
    m, _ = make_model()
    r = m.simulate(dni=900.0, zenith_deg=25.0, T0_C=400.0, wind_speed=4.0,
                   mdot_salt=120.0, duration_s=3000.0, dt=20.0)
    # check at interior points (avoid gradient end effects)
    lhs = r["Q_absorbed_W"]
    rhs = r["dE_stored_dt_W"] + r["Q_loss_W"] + r["Q_thermal_to_PB_W"]
    rel_err = np.abs(lhs[5:-5] - rhs[5:-5]) / np.maximum(np.abs(lhs[5:-5]), 1.0)
    max_err = float(np.max(rel_err))
    assert_true(max_err < 0.02, f"max energy-balance residual {max_err*100:.3f}% < 2%")


def test_steady_state_balance():
    print("\n[Test 6] Reaches steady state: dT/dt -> 0")
    m, _ = make_model()
    r = m.simulate(dni=900.0, zenith_deg=20.0, T0_C=290.0, mdot_salt=150.0,
                   duration_s=7200.0, dt=60.0)
    dT = abs(r["T_receiver_C"][-1] - r["T_receiver_C"][-2])
    assert_true(dT < 0.5, f"Near SS: dT={dT:.4f} degC over last step")


def test_efficiency_drops_at_high_T():
    print("\n[Test 7] Receiver efficiency drops as receiver runs hotter")
    m, _ = make_model()
    Tamb = 298.15
    eff_cool = m.receiver_efficiency(950.0, 20.0, 500.0 + 273.15, Tamb)
    eff_hot = m.receiver_efficiency(950.0, 20.0, 800.0 + 273.15, Tamb)
    assert_true(0.0 < eff_hot < eff_cool < 1.0,
                f"eta_recv(800C)={eff_hot:.3f} < eta_recv(500C)={eff_cool:.3f} < 1")


def test_higher_flow_lowers_temperature():
    print("\n[Test 8] Higher salt flow => lower steady receiver temperature")
    m, _ = make_model()
    r_lo = m.simulate(dni=900.0, zenith_deg=20.0, T0_C=400.0, mdot_salt=80.0,
                      duration_s=5400.0, dt=60.0)
    r_hi = m.simulate(dni=900.0, zenith_deg=20.0, T0_C=400.0, mdot_salt=250.0,
                      duration_s=5400.0, dt=60.0)
    assert_true(r_hi["T_receiver_C"][-1] < r_lo["T_receiver_C"][-1],
                f"T(mdot=250)={r_hi['T_receiver_C'][-1]:.1f} < "
                f"T(mdot=80)={r_lo['T_receiver_C'][-1]:.1f} degC")


def test_loss_terms_positive_when_hot():
    print("\n[Test 9] Rad + conv losses > 0 when receiver above ambient")
    m, _ = make_model()
    r = m.simulate(dni=900.0, zenith_deg=20.0, T0_C=565.0, mdot_salt=150.0,
                   duration_s=600.0, dt=60.0)
    assert_true(np.all(r["Q_rad_loss_W"] > 0), "Q_rad > 0 when T > T_amb")
    assert_true(np.all(r["Q_conv_loss_W"] > 0), "Q_conv > 0 when T > T_amb")


def test_diurnal_profile_power_tracks_dni():
    print("\n[Test 10] Diurnal DNI profile => power follows, zero at night")
    m, _ = make_model()
    dur = 24 * 3600.0
    dt = 600.0
    n = int(dur / dt) + 1
    t = np.linspace(0, dur, n)
    hour = t / 3600.0
    # bell-shaped DNI peaking at noon, zero before 6h and after 18h
    dni = np.where((hour > 6) & (hour < 18),
                   950.0 * np.maximum(0.0, np.sin(np.pi * (hour - 6) / 12.0)), 0.0)
    zen = np.clip(np.abs(hour - 12) * 7.0, 0.0, 85.0)
    r = m.simulate(dni=dni, zenith_deg=zen, T0_C=290.0, mdot_salt=150.0,
                   duration_s=dur, dt=dt)
    night = (hour < 5.5)
    assert_true(np.allclose(r["Q_field_W"][night], 0.0), "Q_field=0 at night")
    assert_true(np.allclose(r["P_electric_W"][night], 0.0), "P_elec=0 at night")
    noon_idx = int(np.argmin(np.abs(hour - 12)))
    assert_true(r["Q_thermal_to_PB_W"][noon_idx] > 0, "Q_to_PB > 0 at midday")


def test_predict_interface():
    print("\n[Test 11] ComponentModel predict() interface keys + shapes")
    _, cm = make_model()
    r = cm.predict({"dni": 900.0, "solar_zenith": 25.0, "duration_s": 1200.0, "dt": 60.0})
    for key in ["t", "T_receiver_C", "Q_field_W", "Q_rad_loss_W",
                "Q_thermal_to_PB_MWth", "P_electric_MWe",
                "field_efficiency", "receiver_efficiency", "overall_efficiency"]:
        assert_true(key in r, f"Key '{key}' in output")
    assert_true(len(r["t"]) == len(r["T_receiver_C"]), "Arrays same length")
    info = cm.get_info()
    assert_true(info["component_id"] == "EC055", "get_info id == EC055")


def test_benchmark():
    print("\n[Test 12] Benchmark: 1h sim at dt=10s")
    m, _ = make_model()
    t0 = time.perf_counter()
    m.simulate(dni=950.0, zenith_deg=25.0, T0_C=290.0, duration_s=3600.0, dt=10.0)
    elapsed = time.perf_counter() - t0
    print(f"  1h simulation in {elapsed*1000:.1f} ms")
    assert_true(elapsed < 5.0, "Completes in < 5 s")


if __name__ == "__main__":
    tests = [
        test_field_efficiency_range,
        test_radiative_T4_law,
        test_p_zero_at_dni_zero,
        test_receiver_heats_up,
        test_energy_conservation,
        test_steady_state_balance,
        test_efficiency_drops_at_high_T,
        test_higher_flow_lowers_temperature,
        test_loss_terms_positive_when_hot,
        test_diurnal_profile_power_tracks_dni,
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
    print(f"EC055 Solar Tower F2a -- Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    sys.exit(0 if failed == 0 else 1)
