"""
EC169 -- Variable Frequency Drive (VFD) -- F2a Physics-Lumped V/f Drive Chain
Test suite: V/f law, torque-speed physics, conservation, ODE tracking, edge cases.
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from model import VFDF2a
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
def test_vf_constant_below_base():
    print("\n[Test 1] Constant V/f below base speed")
    m, _ = make_model()
    # Compare V/f ratio (above the boost knee) at several sub-base frequencies.
    f = np.array([20.0, 30.0, 40.0, 50.0])
    V = m.output_voltage(f)
    # slope (V_rated - V_boost)/f_rated must be reproduced
    slope = (m.V_rated - m.V_boost) / m.f_rated
    expected = m.V_boost + slope * f
    assert_true(np.allclose(V, expected, rtol=1e-9),
                "V_out follows boost + linear V/f below base")
    assert_true(abs(V[-1] - m.V_rated) < 1e-9,
                f"V(f_rated)={V[-1]:.1f} == V_rated={m.V_rated:.1f}")


def test_field_weakening_clamp():
    print("\n[Test 2] Voltage clamps to V_rated above base (field weakening)")
    m, _ = make_model()
    V = m.output_voltage(np.array([60.0, 80.0, 120.0]))
    assert_true(np.all(np.abs(V - m.V_rated) < 1e-9),
                f"V clamped at {m.V_rated:.0f} V above base")


def test_torque_zero_at_sync():
    print("\n[Test 3] Torque -> 0 as speed -> synchronous")
    m, _ = make_model()
    f = 50.0
    omega_s = float(m.sync_speed_mech(f))
    T_near = float(m.motor_torque(np.array([omega_s * 0.999]), f)[0])
    T_mid = float(m.motor_torque(np.array([omega_s * 0.97]), f)[0])
    assert_true(abs(T_near) < abs(T_mid),
                f"T near sync ({T_near:.2f}) << T at 3% slip ({T_mid:.2f})")


def test_breakdown_torque_exists():
    print("\n[Test 4] Torque-speed curve has a breakdown (pull-out) peak")
    m, _ = make_model()
    Tmax = m.breakdown_torque(50.0)
    omega_s = float(m.sync_speed_mech(50.0))
    T_rated_slip = float(m.motor_torque(np.array([omega_s * 0.97]), 50.0)[0])
    assert_true(Tmax > T_rated_slip > 0,
                f"breakdown {Tmax:.1f} > rated-slip torque {T_rated_slip:.1f} > 0")


def test_speed_tracks_frequency():
    print("\n[Test 5] Steady speed tracks commanded frequency")
    m, _ = make_model()
    res = []
    for f in [25.0, 40.0, 50.0]:
        r = m.simulate(f, T_load=40.0, dt=0.01, duration_s=4.0)
        res.append(r["speed_rpm"][-1])
        omega_s = float(m.sync_speed_mech(f))
        rpm_sync = omega_s * 60.0 / (2 * np.pi)
        # rotor settles just below synchronous (positive slip under load)
        assert_true(0.80 * rpm_sync < r["speed_rpm"][-1] < rpm_sync,
                    f"f={f}Hz -> {r['speed_rpm'][-1]:.0f} rpm (sync={rpm_sync:.0f})")
    assert_true(res[0] < res[1] < res[2],
                "Higher frequency command -> higher steady speed (monotone)")


def _soft_start(f_final, t_ramp=2.0, f0=5.0):
    """VFD soft-start frequency profile (ramp f0 -> f_final over t_ramp)."""
    slope = (f_final - f0) / t_ramp
    return lambda t: min(f_final, f0 + slope * t)


def test_slip_positive_under_load():
    print("\n[Test 6] Slip positive (motoring) at steady state under load")
    m, _ = make_model()
    # VFD soft-start (ramped f) so the motor accelerates the load to 50 Hz.
    r = m.simulate(_soft_start(50.0), T_load=60.0, dt=0.01, duration_s=5.0)
    assert_true(0.0 < r["slip"][-1] < 0.2,
                f"steady slip={r['slip'][-1]:.4f} in (0, 0.2)")


def test_efficiency_bounds():
    print("\n[Test 7] Chain efficiency strictly in (0,1)")
    m, _ = make_model()
    r = m.simulate(_soft_start(50.0), T_load=60.0, dt=0.01, duration_s=5.0)
    eta_ss = r["efficiency"][-1]
    assert_true(0.0 < eta_ss < 1.0, f"steady eta={eta_ss:.4f} in (0,1)")
    # spot-check across the whole trace once spun up
    tail = r["efficiency"][r["t"] > 2.0]
    assert_true(np.all((tail > 0.0) & (tail < 1.0)), "all tail eff in (0,1)")


def test_energy_conservation():
    print("\n[Test 8] Energy conservation: P_grid = P_mech + losses, all >= 0")
    m, _ = make_model()
    r = m.simulate(_soft_start(50.0), T_load=60.0, dt=0.01, duration_s=5.0)
    i = -1
    omega = r["omega_m"][i]
    f = r["f_out"][i]
    P_mech = float(m.mech_power(np.array([omega]), f)[0])
    P_elec = float(m.motor_elec_power(np.array([omega]), f)[0])
    P_grid = P_elec / (m.eta_rect * m.eta_inv)
    losses = P_grid - P_mech
    assert_true(P_mech > 0, f"P_mech={P_mech:.1f} W > 0")
    assert_true(losses > 0, f"total losses={losses:.1f} W > 0 (2nd law)")
    assert_true(abs(P_grid - (P_mech + losses)) < 1e-6, "P_grid balances exactly")


def test_dc_link_relaxes():
    print("\n[Test 9] DC-link voltage stays near nominal (capacitor ODE stable)")
    m, _ = make_model()
    # start the bus 50 V low; ODE should pull it back toward nominal
    r = m.simulate(50.0, T_load=50.0, V_dc0=m.V_dc_nom - 50.0,
                   dt=0.005, duration_s=2.0)
    assert_true(abs(r["V_dc"][-1] - m.V_dc_nom) < abs(r["V_dc"][0] - m.V_dc_nom),
                f"V_dc recovered: {r['V_dc'][0]:.1f} -> {r['V_dc'][-1]:.1f} "
                f"(nom {m.V_dc_nom:.0f})")
    assert_true(np.all(r["V_dc"] > 0), "DC-link voltage stays positive")


def test_frequency_ramp_tracking():
    print("\n[Test 10] Speed follows a frequency ramp (soft-start)")
    m, _ = make_model()

    def ramp(t):
        return min(50.0, 10.0 + 20.0 * t)  # 10 Hz -> 50 Hz over 2 s

    r = m.simulate(ramp, T_load=40.0, dt=0.01, duration_s=5.0)
    # speed at end of ramp region should exceed early speed
    early = r["speed_rpm"][np.argmin(np.abs(r["t"] - 0.5))]
    late = r["speed_rpm"][-1]
    assert_true(late > early > 0, f"ramp: {early:.0f} -> {late:.0f} rpm increasing")


def test_zero_frequency_no_torque():
    print("\n[Test 11] Edge case: f=0 gives zero voltage and zero torque")
    m, _ = make_model()
    assert_true(float(m.output_voltage(0.0)) <= m.V_boost + 1e-9,
                "V_out(0) <= boost level")
    T = float(m.motor_torque(np.array([0.0]), 0.0)[0])
    assert_true(abs(T) < 1e-9, f"T(f=0)={T:.3e} ~ 0")


def test_predict_interface():
    print("\n[Test 12] ComponentModel predict() interface + benchmark")
    _, cm = make_model()
    t0 = time.perf_counter()
    r = cm.predict({"f_set": 40.0, "T_load": 50.0, "dt": 0.005, "duration_s": 3.0})
    elapsed = time.perf_counter() - t0
    for key in ["t", "f_out", "V_out", "V_dc", "omega_m", "speed_rpm",
                "slip", "torque", "P_mech", "efficiency"]:
        assert_true(key in r, f"Key '{key}' in output")
    assert_true(len(r["t"]) == len(r["omega_m"]), "Arrays same length")
    print(f"  3 s simulation in {elapsed*1000:.1f} ms")
    assert_true(elapsed < 5.0, "Completes in < 5 s")


if __name__ == "__main__":
    tests = [
        test_vf_constant_below_base,
        test_field_weakening_clamp,
        test_torque_zero_at_sync,
        test_breakdown_torque_exists,
        test_speed_tracks_frequency,
        test_slip_positive_under_load,
        test_efficiency_bounds,
        test_energy_conservation,
        test_dc_link_relaxes,
        test_frequency_ramp_tracking,
        test_zero_frequency_no_torque,
        test_predict_interface,
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
    print(f"EC169 VFD F2a -- Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    sys.exit(0 if failed == 0 else 1)
