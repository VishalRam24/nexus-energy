"""
EC088 -- Oil-Fired Boiler -- F2a Dynamic Thermal Mass
Test suite: combustion mass/energy conservation, bounded efficiency,
thermal-ODE behaviour, edge cases, predict() interface, benchmark.
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from model import OilBoilerF2a
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
def test_mass_conservation():
    print("\n[Test 1] Combustion mass balance: m_flue = m_fuel + m_air")
    m, _ = make_model()
    for plr in [0.3, 0.6, 1.0]:
        m_fuel = m.fuel_mass_flow(plr)
        m_flue = m.flue_mass_flow(plr)
        m_air = m.lambda_a * m.AFR * m_fuel
        assert_true(abs(m_flue - (m_fuel + m_air)) < 1e-12 * max(m_flue, 1e-12),
                    f"plr={plr}: m_flue={m_flue:.6e} = m_fuel+m_air")
        assert_true(m_flue > m_fuel, f"plr={plr}: flue > fuel (air added)")


def test_energy_conservation():
    print("\n[Test 2] Energy balance: Q_fuel = Q_useful + Q_sensible + Q_latent")
    m, _ = make_model()
    for plr in [0.2, 0.5, 1.0]:
        Q_fuel = m.fuel_input(plr)
        Q_use = m.useful_heat(plr)
        Q_sens = m.sensible_loss(plr)
        Q_lat = m.latent_loss(plr)
        resid = abs(Q_fuel - (Q_use + Q_sens + Q_lat))
        assert_true(resid < 1e-6 * Q_fuel,
                    f"plr={plr}: residual={resid:.3e} W ~ 0")


def test_efficiency_bounds():
    print("\n[Test 3] Combustion efficiency in (0,1)")
    m, _ = make_model()
    for plr in np.linspace(0.15, 1.0, 12):
        eta = m.combustion_efficiency(plr)
        assert_true(0.0 < eta < 1.0, f"plr={plr:.2f}: eta_comb={eta:.4f} in (0,1)")
    # oil boiler should land in a realistic band at high fire
    eta_full = m.combustion_efficiency(1.0)
    assert_true(0.80 < eta_full < 0.97, f"full-fire eta_comb={eta_full:.4f} realistic")


def test_burner_off_below_turndown():
    print("\n[Test 4] Burner off below turndown (PLR_min)")
    m, _ = make_model()
    assert_true(m.fuel_input(0.05) == 0.0, "Q_fuel=0 below PLR_min")
    assert_true(m.useful_heat(0.05) == 0.0, "Q_useful=0 below PLR_min")
    assert_true(m.fuel_input(0.5) > 0.0, "Q_fuel>0 above PLR_min")


def test_thermal_ode_heats_up():
    print("\n[Test 5] Thermal ODE: boiler heats from cold start")
    m, _ = make_model()
    r = m.simulate(0.9, T_water_init=20.0, T_return=20.0, T_ambient=20.0,
                   dt=10.0, duration_s=1800.0)
    assert_true(r["T_water_C"][-1] > 20.0, f"T_final={r['T_water_C'][-1]:.2f} > 20 C")
    assert_true(r["T_water_C"][-1] < 200.0, f"T_final={r['T_water_C'][-1]:.2f} < 200 C")
    assert_true(np.all(np.diff(r["T_water_C"]) >= -1e-6),
                "T rises monotonically with constant fire from cold")


def test_thermal_steady_state():
    print("\n[Test 6] Thermal ODE approaches steady state")
    m, _ = make_model()
    r = m.simulate(0.6, T_water_init=70.0, T_return=60.0, T_ambient=20.0,
                   dt=15.0, duration_s=7200.0)
    dT = abs(r["T_water_C"][-1] - r["T_water_C"][-2])
    assert_true(dT < 0.05, f"Near SS: dT={dT:.5f} C between last two steps")


def test_standby_loss_positive():
    print("\n[Test 7] Standby/casing loss > 0 when T_water > T_ambient")
    m, _ = make_model()
    r = m.simulate(0.7, T_water_init=80.0, T_return=70.0, T_ambient=10.0,
                   dt=20.0, duration_s=600.0)
    assert_true(np.all(r["Q_standby_loss_W"] > 0.0), "standby loss positive")
    assert_true(np.all(r["Q_latent_loss_W"] > 0.0), "latent stack loss positive")


def test_flue_temp_monotone():
    print("\n[Test 8] Flue temperature rises with firing rate")
    m, _ = make_model()
    T_low = m.flue_temp(0.3)
    T_high = m.flue_temp(1.0)
    assert_true(T_high > T_low, f"T_flue(1.0)={T_high:.1f} > T_flue(0.3)={T_low:.1f}")
    assert_true(T_low > m.T_air, "flue hotter than combustion air")


def test_excess_air_increases_stack_loss():
    print("\n[Test 9] More excess air -> more sensible stack loss -> lower eta")
    m, _ = make_model()
    eta_lean = m.combustion_efficiency(0.8, excess_air=1.10)
    eta_rich_air = m.combustion_efficiency(0.8, excess_air=1.50)
    assert_true(eta_lean > eta_rich_air,
                f"eta(lambda=1.10)={eta_lean:.4f} > eta(lambda=1.50)={eta_rich_air:.4f}")


def test_predict_interface():
    print("\n[Test 10] ComponentModel predict() interface")
    _, cm = make_model()
    r = cm.predict({"firing_rate": 0.7, "dt": 20.0, "duration_s": 600.0})
    for key in ["t", "T_water_C", "Q_fuel_W", "Q_useful_W", "Q_load_W",
                "eta_combustion", "eta_overall", "T_flue_C"]:
        assert_true(key in r, f"Key '{key}' in output")
    assert_true(len(r["t"]) == len(r["T_water_C"]), "Arrays same length")
    info = cm.get_info()
    assert_true(info["component_id"] == "EC088", "get_info id == EC088")


def test_callable_firing_rate():
    print("\n[Test 11] Time-varying (callable) firing rate accepted")
    m, _ = make_model()
    def ramp(t):
        return 0.3 if t < 300 else 0.95
    r = m.simulate(ramp, T_water_init=30.0, T_return=30.0, T_ambient=20.0,
                   dt=10.0, duration_s=900.0)
    idx_lo = np.argmin(np.abs(r["t"] - 250.0))
    idx_hi = np.argmin(np.abs(r["t"] - 600.0))
    assert_true(r["Q_fuel_W"][idx_hi] > r["Q_fuel_W"][idx_lo],
                "Fuel input rises after firing-rate step up")


def test_benchmark():
    print("\n[Test 12] Benchmark: 1-hour sim at dt=5 s")
    m, _ = make_model()
    t0 = time.perf_counter()
    m.simulate(0.8, T_water_init=30.0, T_return=60.0, T_ambient=20.0,
               dt=5.0, duration_s=3600.0)
    elapsed = time.perf_counter() - t0
    print(f"  3600s simulation in {elapsed*1000:.1f} ms")
    assert_true(elapsed < 5.0, "Completes in < 5 s")


if __name__ == "__main__":
    tests = [
        test_mass_conservation,
        test_energy_conservation,
        test_efficiency_bounds,
        test_burner_off_below_turndown,
        test_thermal_ode_heats_up,
        test_thermal_steady_state,
        test_standby_loss_positive,
        test_flue_temp_monotone,
        test_excess_air_increases_stack_loss,
        test_predict_interface,
        test_callable_firing_rate,
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
    print(f"EC088 Oil Boiler F2a -- Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    sys.exit(0 if failed == 0 else 1)
