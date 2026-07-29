"""
EC177 -- Brushless DC Motor (BLDC) -- F2a dq/phase-domain
Test suite: physics sanity (torque∝current, EMF∝speed, energy conservation),
speed-torque behavior, edge cases, predict() interface, benchmark timing.
NO pytest -- run with: python3 scripts/test_model.py
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from model import BLDC_F2a
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
def test_torque_proportional_current():
    print("\n[Test 1] Electromagnetic torque proportional to current (flat-top)")
    m, _ = make_model()
    theta = np.pi / 3.0  # inside +1 flat-top region -> f_trap = +1
    for i in [1.0, 2.0, 5.0]:
        T = m.torque_e(i, theta)
        assert_true(abs(T - m.Kt_eq * i) < 1e-12, f"T_e({i})={T:.4f} == Kt*i")
    # doubling current doubles torque
    assert_true(abs(m.torque_e(4.0, theta) - 2 * m.torque_e(2.0, theta)) < 1e-12,
                "Torque scales linearly with current")


def test_backemf_proportional_speed():
    print("\n[Test 2] Back-EMF proportional to speed (e = Ke*omega)")
    m, _ = make_model()
    theta = np.pi / 3.0  # flat-top f_trap=+1
    e1 = m.back_emf(100.0, theta)
    e2 = m.back_emf(200.0, theta)
    assert_true(abs(e1 - m.Ke_eq * 100.0) < 1e-12, f"e(100)={e1:.3f} == Ke*omega")
    assert_true(abs(e2 - 2 * e1) < 1e-12, "Doubling speed doubles back-EMF")
    assert_true(abs(m.back_emf(0.0, theta)) < 1e-12, "Zero speed -> zero EMF")


def test_trapezoid_shape():
    print("\n[Test 3] Trapezoidal back-EMF shape bounded in [-1, 1]")
    m, _ = make_model()
    for th in np.linspace(0, 4 * np.pi, 200):
        f = m.f_trap(th)
        assert_true(-1.0 - 1e-9 <= f <= 1.0 + 1e-9, f"f_trap({th:.2f})={f:.3f} in [-1,1]")
    # flat-top regions
    assert_true(abs(m.f_trap(np.pi / 3) - 1.0) < 1e-9, "Positive flat top = +1")
    assert_true(abs(m.f_trap(4 * np.pi / 3) + 1.0) < 1e-9, "Negative flat top = -1")


def test_motor_spins_up():
    print("\n[Test 4] Motor spins up from rest under applied Vdc")
    m, _ = make_model()
    r = m.simulate(T_load=0.1, dt=2e-4, duration_s=0.4)
    assert_true(r["omega_final"] > 0.0, f"omega_final={r['omega_final']:.1f} > 0")
    assert_true(r["speed_rpm"][-1] < 5000.0, f"speed={r['speed_rpm'][-1]:.0f} rpm reasonable")


def test_no_load_speed_analytic():
    print("\n[Test 5] No-load speed approaches Vdc/Ke")
    m, _ = make_model()
    w0 = m.no_load_speed()
    assert_true(abs(w0 - m.Vdc / m.Ke_eq) < 1e-9, f"omega_0={w0:.1f} == Vdc/Ke")
    # simulate near no-load (tiny load) -> approaches w0 from below
    r = m.simulate(T_load=0.005, dt=2e-4, duration_s=1.0)
    assert_true(r["omega_final"] < w0 + 1e-6,
                f"loaded omega={r['omega_final']:.1f} <= no-load {w0:.1f}")
    assert_true(r["omega_final"] > 0.6 * w0,
                f"reaches >60% of no-load speed ({r['omega_final']:.1f}/{w0:.1f})")


def test_speed_torque_monotone():
    print("\n[Test 6] Speed decreases as load torque increases")
    m, _ = make_model()
    speeds = []
    for T in [0.0, 0.5, 1.0, 1.5]:
        r = m.simulate(T_load=T, dt=2e-4, duration_s=1.2)
        speeds.append(r["omega_final"])
    for k in range(1, len(speeds)):
        assert_true(speeds[k] < speeds[k - 1] + 1e-6,
                    f"omega(T_{k}) {speeds[k]:.1f} < omega(T_{k-1}) {speeds[k-1]:.1f}")


def test_stall_torque():
    print("\n[Test 7] Stall torque = Kt*Vdc/R_eq and exceeds rated load")
    m, _ = make_model()
    Ts = m.stall_torque()
    assert_true(abs(Ts - m.Kt_eq * m.Vdc / m.R_eq) < 1e-9, f"T_stall={Ts:.2f} Nm analytic")
    # current at stall = Vdc/R_eq, torque should be largest at omega=0
    assert_true(Ts > 3.0, f"Stall torque {Ts:.2f} Nm >> rated (~3.2 Nm)")


def test_energy_conservation():
    print("\n[Test 8] Energy balance: P_elec = P_mech + P_cu + inertial/friction")
    m, _ = make_model()
    r = m.simulate(T_load=0.5, dt=1e-4, duration_s=0.6)
    # At steady state (tail), electrical in = air-gap mech + copper loss
    N = len(r["t"])
    tail = slice(int(0.85 * N), N)
    P_in = np.mean(r["P_elec"][tail])
    P_mech = np.mean(r["P_mech"][tail])
    P_cu = np.mean(r["P_cu"][tail])
    resid = P_in - (P_mech + P_cu)
    # residual should be small vs input (di/dt and domega/dt ~ 0 at SS)
    assert_true(abs(resid) < 0.06 * abs(P_in) + 1.0,
                f"SS balance: P_in={P_in:.1f} ~ P_mech={P_mech:.1f}+P_cu={P_cu:.1f} "
                f"(resid={resid:.2f} W)")


def test_efficiency_range():
    print("\n[Test 9] Efficiency strictly in (0, 1)")
    m, _ = make_model()
    for T in [0.2, 0.5, 1.0, 2.0]:
        r = m.simulate(T_load=T, dt=2e-4, duration_s=0.8)
        eta = r["efficiency"]
        assert_true(0.0 < eta < 1.0, f"T={T}: eff={eta:.3f} in (0,1)")


def test_higher_voltage_higher_speed():
    print("\n[Test 10] Higher Vdc -> higher steady speed")
    _, cm = make_model()
    r24 = cm.predict({"Vdc": 24.0, "T_load_Nm": 0.3, "duration_s": 1.0})
    cm2 = ComponentModel()
    r48 = cm2.predict({"Vdc": 48.0, "T_load_Nm": 0.3, "duration_s": 1.0})
    assert_true(r48["omega_final"] > r24["omega_final"],
                f"omega(48V)={r48['omega_final']:.0f} > omega(24V)={r24['omega_final']:.0f}")


def test_predict_interface():
    print("\n[Test 11] ComponentModel predict()/get_info() interface")
    _, cm = make_model()
    info = cm.get_info()
    assert_true(info["component_id"] == "EC177", "component_id EC177")
    assert_true(cm.version == "1.0.0", "version 1.0.0")
    r = cm.predict({"T_load_Nm": 0.4, "dt": 2e-4, "duration_s": 0.3})
    for key in ["t", "current", "omega", "speed_rpm", "back_emf",
                "torque_e", "P_mech", "P_elec", "efficiency"]:
        assert_true(key in r, f"Key '{key}' in output")
    assert_true(len(r["t"]) == len(r["current"]) == len(r["omega"]),
                "Time-series arrays same length")


def test_benchmark():
    print("\n[Test 12] Benchmark: 0.4 s transient at dt=2e-4")
    m, _ = make_model()
    t0 = time.perf_counter()
    m.simulate(T_load=0.5, dt=2e-4, duration_s=0.4)
    elapsed = time.perf_counter() - t0
    print(f"  0.4s transient in {elapsed*1000:.1f} ms")
    assert_true(elapsed < 5.0, "Completes in < 5 s")


if __name__ == "__main__":
    tests = [
        test_torque_proportional_current,
        test_backemf_proportional_speed,
        test_trapezoid_shape,
        test_motor_spins_up,
        test_no_load_speed_analytic,
        test_speed_torque_monotone,
        test_stall_torque,
        test_energy_conservation,
        test_efficiency_range,
        test_higher_voltage_higher_speed,
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
    print(f"EC177 BLDC F2a -- Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    sys.exit(0 if failed == 0 else 1)
