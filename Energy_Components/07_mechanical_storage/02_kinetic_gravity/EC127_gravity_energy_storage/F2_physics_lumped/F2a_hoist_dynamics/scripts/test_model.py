"""
EC127 -- Gravity Energy Storage -- F2a Hoist Dynamics
Test suite: physics sanity, energy conservation, ODE behaviour, edge cases.
Run:  python3 scripts/test_model.py   (system python3, NOT pytest)
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from model import GravityHoistF2a
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
def test_energy_conservation_mgh():
    print("\n[Test 1] Stored energy obeys E = m g h")
    m, _ = make_model()
    x = 250.0
    E_J = m.stored_energy_J(x)
    E_expected = m.m * m.g * (m.h_min + x)
    assert_true(abs(E_J - E_expected) < 1e-3,
                f"E={E_J:.3e} J == m g h = {E_expected:.3e} J")
    # capacity = m g h_usable
    cap = m.energy_capacity_kwh()
    cap_exp = m.m * m.g * m.h_usable / 3.6e6
    assert_true(abs(cap - cap_exp) < 1e-6, f"capacity={cap:.1f} kWh == m g h/3.6e6")


def test_charge_lifts_mass():
    print("\n[Test 2] Charge stroke lifts mass (x and SOC increase)")
    m, _ = make_model()
    r = m.simulate(mode="charge", dt=20.0)
    assert_true(r["x"][-1] > r["x"][0] + 1.0, f"x rose {r['x'][0]:.1f}->{r['x'][-1]:.1f} m")
    assert_true(r["soc"][-1] > r["soc"][0], "SOC increased over charge")
    assert_true(r["E_stored_kwh"][-1] > r["E_stored_kwh"][0],
                "Stored PE increased during lift")


def test_discharge_lowers_mass():
    print("\n[Test 3] Discharge stroke lowers mass (x and SOC decrease)")
    m, _ = make_model()
    r = m.simulate(mode="discharge", dt=20.0)
    assert_true(r["x"][-1] < r["x"][0] - 1.0, f"x fell {r['x'][0]:.1f}->{r['x'][-1]:.1f} m")
    assert_true(r["E_stored_kwh"][-1] < r["E_stored_kwh"][0],
                "Stored PE decreased during lower")


def test_newton_velocity_capped():
    print("\n[Test 4] Newton ODE velocity respects line-speed and power limits")
    m, _ = make_model()
    v_pow = m.cruise_speed("charge")   # power-limited cruise for this mass
    r = m.simulate(mode="charge", dt=20.0)
    assert_true(np.max(np.abs(r["v"])) <= m.v_max + 1e-6,
                f"max|v|={np.max(np.abs(r['v'])):.4f} <= v_max={m.v_max}")
    assert_true(np.max(np.abs(r["v"])) <= v_pow * 1.05 + 1e-6,
                f"max|v|<= power-limited cruise {v_pow:.4f} m/s")
    # velocity converges toward the achievable command (steady cruise reached)
    v_cruise = np.median(np.abs(r["v"][len(r["v"])//2:]))
    assert_true(v_cruise > 0.5 * v_pow, f"cruise |v|={v_cruise:.4f} near target {v_pow:.4f}")


def test_power_limited_by_rating():
    print("\n[Test 5] Electrical power bounded by hoist rating")
    m, _ = make_model()
    r = m.simulate(mode="charge", dt=20.0)
    Pmax = np.max(np.abs(r["P_elec"]))
    # allow charge division by efficiency (~1/0.91) margin above P_rated
    assert_true(Pmax <= m.P_rated / 0.85,
                f"max|P_elec|={Pmax/1e3:.0f} kW within rating margin")


def test_charge_costs_more_than_pe():
    print("\n[Test 6] Charge electrical input exceeds PE gained (losses)")
    m, _ = make_model()
    r = m.simulate(mode="charge", dt=20.0)
    dPE = r["E_stored_kwh"][-1] - r["E_stored_kwh"][0]
    E_in = abs(r["E_elec_kwh"])
    assert_true(E_in > dPE > 0, f"E_in={E_in:.1f} > dPE={dPE:.1f} kWh > 0")


def test_discharge_returns_less_than_pe():
    print("\n[Test 7] Discharge electrical output less than PE released")
    m, _ = make_model()
    r = m.simulate(mode="discharge", dt=20.0)
    dPE = abs(r["E_stored_kwh"][0] - r["E_stored_kwh"][-1])
    E_out = abs(r["E_elec_kwh"])
    assert_true(0 < E_out < dPE, f"0 < E_out={E_out:.1f} < dPE={dPE:.1f} kWh")


def test_round_trip_efficiency_bounded():
    print("\n[Test 8] 0 < round-trip efficiency < 1")
    m, _ = make_model()
    eta = m.round_trip_efficiency(dt=20.0)
    assert_true(0.0 < eta < 1.0, f"eta_RT={eta:.4f} in (0,1)")
    # solid-mass gravity storage typically 0.7-0.9
    assert_true(0.6 < eta < 0.95, f"eta_RT={eta:.4f} in plausible 0.6-0.95 band")


def test_machine_efficiency_partload():
    print("\n[Test 9] Part-load machine efficiency monotone, <= rated")
    m, _ = make_model()
    eta_full = m.motor_efficiency(1.0)
    eta_half = m.motor_efficiency(0.5)
    eta_low = m.motor_efficiency(0.1)
    assert_true(eta_full <= m.eta_motor_rated + 1e-9, "eta(1) <= rated")
    assert_true(eta_full > eta_half > eta_low, "efficiency drops at lower PLF")
    assert_true(eta_low > 0.0, "efficiency stays positive")


def test_friction_drag_oppose_motion():
    print("\n[Test 10] Friction & drag forces oppose velocity")
    m, _ = make_model()
    assert_true(m.friction_force(2.0) > 0 and m.friction_force(-2.0) < 0,
                "Coulomb friction sign tracks velocity")
    assert_true(m.drag_force(3.0) > 0 and m.drag_force(-3.0) < 0,
                "Drag sign tracks velocity")
    assert_true(m.drag_force(3.0) > m.drag_force(1.0),
                "Drag grows with speed (quadratic)")


def test_predict_interface():
    print("\n[Test 11] ComponentModel predict() interface")
    _, cm = make_model()
    r = cm.predict({"mode": "charge", "dt": 20.0})
    for key in ["t", "x", "v", "height", "soc", "F_cable",
                "P_mech", "P_elec", "E_stored_kwh", "E_elec_kwh"]:
        assert_true(key in r, f"Key '{key}' in output")
    assert_true(len(r["t"]) == len(r["v"]), "Arrays same length")
    info = cm.get_info()
    assert_true(info["component_id"] == "EC127", "component_id == EC127")


def test_benchmark():
    print("\n[Test 12] Benchmark: full charge stroke simulation")
    m, _ = make_model()
    t0 = time.perf_counter()
    m.simulate(mode="charge", dt=10.0)
    elapsed = time.perf_counter() - t0
    print(f"  full lift stroke in {elapsed*1000:.1f} ms")
    assert_true(elapsed < 5.0, "Completes in < 5 s")


if __name__ == "__main__":
    tests = [
        test_energy_conservation_mgh,
        test_charge_lifts_mass,
        test_discharge_lowers_mass,
        test_newton_velocity_capped,
        test_power_limited_by_rating,
        test_charge_costs_more_than_pe,
        test_discharge_returns_less_than_pe,
        test_round_trip_efficiency_bounded,
        test_machine_efficiency_partload,
        test_friction_drag_oppose_motion,
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
    print(f"EC127 Gravity Storage F2a -- Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    sys.exit(0 if failed == 0 else 1)
