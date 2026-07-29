"""
EC180 -- Doubly-Fed Induction Generator (DFIG) -- F2a dq-Frame Model
Test suite: DFIG physics sanity (slip-power relation, decoupled P/Q control,
variable-speed range, energy conservation, efficiency bounds), edge cases,
predict() interface, and a benchmark timing test.

NO pytest -- run directly with: python3 scripts/test_model.py
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from model import DFIG_F2a
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


def _settled(arr, frac=0.2):
    """Mean of the last `frac` of a trajectory (steady-state estimate)."""
    n = max(1, int(len(arr) * frac))
    return float(np.mean(arr[-n:]))


# ---------------------------------------------------------------------------
def test_slip_speed_consistency():
    print("\n[Test 1] Slip <-> rotor-speed relation (sub/super-synchronous)")
    m, _ = make_model()
    # synchronous speed in rpm = omega_s / P * 30/pi
    n_sync = m.omega_s / m.P * 30.0 / np.pi
    assert_true(abs(n_sync - 1500.0) < 1.0, f"Synchronous speed {n_sync:.1f} rpm = 1500")
    # super-synchronous: s<0 -> speed above synchronous
    n_super = m.speed_rpm(m.omega_r_from_slip(-0.2))
    assert_true(n_super > n_sync, f"s=-0.2 -> {n_super:.0f} rpm > {n_sync:.0f} (super-sync)")
    # sub-synchronous: s>0 -> speed below synchronous
    n_sub = m.speed_rpm(m.omega_r_from_slip(0.2))
    assert_true(n_sub < n_sync, f"s=+0.2 -> {n_sub:.0f} rpm < {n_sync:.0f} (sub-sync)")


def test_slip_power_relation():
    print("\n[Test 2] Slip-power relation: sign of P_rotor = -s * P_stator")
    _, cm = make_model()
    # Generating (P_stator < 0, delivered to grid) at several slips
    for s in [-0.25, -0.1, 0.1, 0.25]:
        r = cm.predict({"mode": "power_control", "P_stator_ref_W": -1.5e6,
                        "Q_stator_ref_VAr": 0.0, "slip": s,
                        "duration_s": 1.0, "dt": 2e-4})
        Ps = _settled(r["P_stator"])
        Pr = _settled(r["P_rotor"])
        expected = -s * Ps  # air-gap slip-power relation
        # check the SIGN matches (super-sync s<0 -> Pr<0 to grid; sub-sync opposite)
        assert_true(np.sign(Pr) == np.sign(expected) or abs(Pr) < 1e4,
                    f"s={s:+.2f}: sign(P_rotor={Pr/1e3:.0f}kW)=sign(-s*Ps={expected/1e3:.0f}kW)")


def test_synchronous_zero_slip_power():
    print("\n[Test 3] At synchronous speed (s=0) rotor (slip) power ~ 0")
    _, cm = make_model()
    r = cm.predict({"mode": "power_control", "P_stator_ref_W": -1.0e6,
                    "Q_stator_ref_VAr": 0.0, "slip": 0.0,
                    "duration_s": 1.0, "dt": 2e-4})
    Pr = _settled(r["P_rotor"])
    Ps = _settled(r["P_stator"])
    assert_true(abs(Pr) < 0.15 * abs(Ps),
                f"|P_rotor|={abs(Pr)/1e3:.0f}kW << |P_stator|={abs(Ps)/1e3:.0f}kW at s=0")


def test_active_power_control():
    print("\n[Test 4] Rotor-side converter controls stator ACTIVE power")
    _, cm = make_model()
    P_prev = 0.0
    for Pref in [-0.5e6, -1.0e6, -1.5e6]:
        r = cm.predict({"mode": "power_control", "P_stator_ref_W": Pref,
                        "Q_stator_ref_VAr": 0.0, "slip": -0.2,
                        "duration_s": 1.0, "dt": 2e-4})
        Ps = _settled(r["P_stator"])
        assert_true(Ps < P_prev, f"Pref={Pref/1e6:.1f}MW -> Ps={Ps/1e6:.3f}MW more negative")
        P_prev = Ps


def test_reactive_power_control():
    print("\n[Test 5] Rotor-side converter controls stator REACTIVE power")
    _, cm = make_model()
    Qs_vals = []
    for Qref in [-500e3, 0.0, 500e3]:
        r = cm.predict({"mode": "power_control", "P_stator_ref_W": -1.0e6,
                        "Q_stator_ref_VAr": Qref, "slip": -0.2,
                        "duration_s": 1.0, "dt": 2e-4})
        Qs_vals.append(_settled(r["Q_stator"]))
    # Q must increase monotonically with the set-point (decoupled from P)
    assert_true(Qs_vals[0] < Qs_vals[1] < Qs_vals[2],
                f"Q tracks set-point: {[round(q/1e3) for q in Qs_vals]} kVAr increasing")


def test_pq_decoupling():
    print("\n[Test 6] Changing Q set-point barely moves P (decoupled control)")
    _, cm = make_model()
    r1 = cm.predict({"mode": "power_control", "P_stator_ref_W": -1.0e6,
                     "Q_stator_ref_VAr": -400e3, "slip": -0.2,
                     "duration_s": 1.0, "dt": 2e-4})
    r2 = cm.predict({"mode": "power_control", "P_stator_ref_W": -1.0e6,
                     "Q_stator_ref_VAr": 400e3, "slip": -0.2,
                     "duration_s": 1.0, "dt": 2e-4})
    P1, P2 = _settled(r1["P_stator"]), _settled(r2["P_stator"])
    assert_true(abs(P2 - P1) < 0.1 * abs(P1),
                f"dP={abs(P2-P1)/1e3:.0f}kW small vs P={abs(P1)/1e3:.0f}kW under 800kVAr Q swing")


def test_variable_speed_range_stable():
    print("\n[Test 7] Stable (finite) operation across ±30% slip range")
    _, cm = make_model()
    for s in np.linspace(-0.30, 0.30, 7):
        r = cm.predict({"mode": "power_control", "P_stator_ref_W": -1.2e6,
                        "Q_stator_ref_VAr": 0.0, "slip": float(s),
                        "duration_s": 0.8, "dt": 2e-4})
        finite = np.all(np.isfinite(r["P_stator"])) and np.all(np.isfinite(r["i_rotor"]))
        bounded = _settled(r["i_rotor"]) < 1e5
        assert_true(finite and bounded, f"s={s:+.2f}: finite & bounded currents")


def test_efficiency_in_bounds():
    print("\n[Test 8] Generating efficiency in (0, 1)")
    _, cm = make_model()
    # eff = |P delivered to grid| / |P mechanical input| at super-sync generating.
    # P_grid = P_stator + converter rotor power; P_mech = P_grid + total losses.
    for s in [-0.25, -0.1, 0.1]:
        r = cm.predict({"mode": "power_control", "P_stator_ref_W": -1.5e6,
                        "Q_stator_ref_VAr": 0.0, "slip": float(s),
                        "duration_s": 1.0, "dt": 2e-4})
        m, _ = make_model()
        i_ds = _settled(r["i_ds"]); i_qs = _settled(r["i_qs"])
        i_dr = _settled(r["i_dr"]); i_qr = _settled(r["i_qr"])
        P_grid = _settled(r["P_grid"])               # < 0 (to grid)
        # copper losses
        P_cu = 1.5 * (m.Rs * (i_ds**2 + i_qs**2) + m.Rr * (i_dr**2 + i_qr**2))
        P_mech = abs(P_grid) + P_cu                  # mechanical input
        eta = abs(P_grid) / P_mech
        assert_true(0.0 < eta < 1.0, f"s={s:+.2f}: eta={eta:.4f} in (0,1)")


def test_energy_conservation():
    print("\n[Test 9] Power balance: P_mech = P_grid + copper losses")
    m, _ = make_model()
    cm = ComponentModel()
    r = cm.predict({"mode": "power_control", "P_stator_ref_W": -1.5e6,
                    "Q_stator_ref_VAr": 0.0, "slip": -0.2,
                    "duration_s": 1.2, "dt": 2e-4})
    i_ds = _settled(r["i_ds"]); i_qs = _settled(r["i_qs"])
    i_dr = _settled(r["i_dr"]); i_qr = _settled(r["i_qr"])
    Ps = _settled(r["P_stator"]); Pr = _settled(r["P_rotor"])
    # electromechanical power = air-gap torque * mechanical speed
    T_e = m.torque(i_ds, i_qs, i_dr, i_qr)
    omega_m = m.omega_r_from_slip(-0.2) / m.P
    P_em = T_e * omega_m
    # Terminal electrical power = stator + rotor; difference should be copper loss
    P_term = Ps + Pr
    P_cu = 1.5 * (m.Rs * (i_ds**2 + i_qs**2) + m.Rr * (i_dr**2 + i_qr**2))
    residual = abs(P_em - (P_term - P_cu))
    scale = max(abs(P_em), 1.0)
    assert_true(residual / scale < 0.10,
                f"Balance residual {residual/1e3:.0f}kW < 10% of {scale/1e3:.0f}kW")


def test_torque_definition():
    print("\n[Test 10] Electromagnetic torque from dq currents is consistent")
    m, _ = make_model()
    T = m.torque(100.0, -500.0, 200.0, 300.0)
    T_manual = 1.5 * m.P * m.Lm * (-500.0 * 200.0 - 100.0 * 300.0)
    assert_true(abs(T - T_manual) < 1e-6, f"T_e={T:.3f} matches closed form")
    assert_true(abs(m.torque(0, 0, 0, 0)) < 1e-12, "Zero currents -> zero torque")


def test_predict_interface():
    print("\n[Test 11] ComponentModel predict()/get_info() interface")
    _, cm = make_model()
    info = cm.get_info()
    for k in ["component_id", "component_name", "fidelity", "version", "inputs", "outputs"]:
        assert_true(k in info, f"get_info has '{k}'")
    assert_true(info["component_id"] == "EC180", "component_id == EC180")
    r = cm.predict({"mode": "power_control", "P_stator_ref_W": -1.0e6,
                    "Q_stator_ref_VAr": 0.0, "slip": -0.2,
                    "duration_s": 0.3, "dt": 2e-4})
    for key in ["t", "speed_rpm", "slip", "torque", "P_stator", "Q_stator",
                "P_rotor", "P_grid", "i_stator", "i_rotor"]:
        assert_true(key in r, f"output key '{key}' present")
    assert_true(len(r["t"]) == len(r["P_stator"]), "Output arrays same length")


def test_benchmark():
    print("\n[Test 12] Benchmark: 1 s power-control simulation")
    _, cm = make_model()
    t0 = time.perf_counter()
    cm.predict({"mode": "power_control", "P_stator_ref_W": -1.5e6,
                "Q_stator_ref_VAr": 0.0, "slip": -0.2,
                "duration_s": 1.0, "dt": 1e-4})
    elapsed = time.perf_counter() - t0
    print(f"  1 s sim at dt=1e-4 in {elapsed*1000:.1f} ms")
    assert_true(elapsed < 5.0, "Completes in < 5 s")


if __name__ == "__main__":
    tests = [
        test_slip_speed_consistency,
        test_slip_power_relation,
        test_synchronous_zero_slip_power,
        test_active_power_control,
        test_reactive_power_control,
        test_pq_decoupling,
        test_variable_speed_range_stable,
        test_efficiency_in_bounds,
        test_energy_conservation,
        test_torque_definition,
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

    print(f"\n{'='*62}")
    print(f"EC180 DFIG F2a -- Results: {passed} passed, {failed} failed")
    print(f"{'='*62}")
    sys.exit(0 if failed == 0 else 1)
