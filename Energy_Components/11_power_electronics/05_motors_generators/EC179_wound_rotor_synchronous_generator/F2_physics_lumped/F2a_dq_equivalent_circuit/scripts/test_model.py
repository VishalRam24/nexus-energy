"""
EC179 -- Wound Rotor Synchronous Generator -- F2a dq-frame
Test suite: power-angle physics, excitation/reactive-power coupling, swing-equation
energy/stability behaviour, capability curve, AVR, predict() interface, benchmark.
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from model import WRSyncGenF2a
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
def test_power_angle_law():
    print("\n[Test 1] P = Ef*Vt/Xs * sin(delta) enforced exactly")
    m, _ = make_model()
    Ef, Vt = 1.7, 1.0
    for d_deg in [10, 30, 45, 60, 80]:
        d = np.radians(d_deg)
        P = m.active_power(Ef, d, Vt)
        expected = (Ef * Vt / m.Xs) * np.sin(d)
        assert_true(abs(P - expected) < 1e-12, f"delta={d_deg}deg: P={P:.4f} == Ef*Vt/Xs*sin(d)")


def test_power_monotone_and_pmax():
    print("\n[Test 2] P rises with delta up to 90deg; Pmax = Ef*Vt/Xs")
    m, _ = make_model()
    Ef, Vt = 1.5, 1.0
    deltas = np.radians(np.linspace(0, 90, 50))
    P = m.active_power(Ef, deltas, Vt)
    assert_true(np.all(np.diff(P) >= -1e-12), "P monotonically increases to 90deg")
    assert_true(abs(P[-1] - m.pmax(Ef, Vt)) < 1e-9, f"P(90deg)={P[-1]:.4f} == Pmax={m.pmax(Ef,Vt):.4f}")


def test_excitation_controls_Q():
    print("\n[Test 3] Excitation controls reactive power (over/under-excited)")
    m, _ = make_model()
    Vt, delta = 1.0, np.radians(20)
    Q_over = m.reactive_power(1.8, delta, Vt)   # Ef > Vt
    Q_under = m.reactive_power(0.8, delta, Vt)  # Ef < Vt
    assert_true(Q_over > 0, f"Over-excited (Ef=1.8): Q={Q_over:.4f} > 0 (delivers VARs)")
    assert_true(Q_under < 0, f"Under-excited (Ef=0.8): Q={Q_under:.4f} < 0 (absorbs VARs)")
    assert_true(Q_over > Q_under, "More excitation -> more reactive power")


def test_emf_from_field():
    print("\n[Test 4] EMF rises linearly with field current; invertible")
    m, _ = make_model()
    Ef = m.emf_from_field(500.0)
    assert_true(Ef > 0, f"Ef({500} A) = {Ef:.4f} pu > 0")
    assert_true(m.emf_from_field(600.0) > m.emf_from_field(400.0), "Ef monotone in If")
    If_back = m.field_from_emf(Ef)
    assert_true(abs(If_back - 500.0) < 1e-6, f"Inverse field_from_emf -> {If_back:.2f} A")


def test_operating_point_consistency():
    print("\n[Test 5] operating_point: phasor solution reproduces P,Q")
    m, _ = make_model()
    P, Q = 0.85, 0.3
    op = m.operating_point(P, Q, 1.0)
    P_chk = m.active_power(op["Ef_pu"], op["delta_rad"], 1.0)
    Q_chk = m.reactive_power(op["Ef_pu"], op["delta_rad"], 1.0)
    assert_true(abs(P_chk - P) < 1e-6, f"Recovered P={P_chk:.4f} (target {P})")
    assert_true(abs(Q_chk - Q) < 1e-6, f"Recovered Q={Q_chk:.4f} (target {Q})")
    assert_true(op["over_excited"], "pf=0.85 lagging -> over-excited (Ef>Vt)")


def test_efficiency_bounds():
    print("\n[Test 6] Efficiency strictly in (0,1) and P_mech > P_elec")
    m, _ = make_model()
    for P in [0.2, 0.5, 0.85, 1.0]:
        op = m.operating_point(P, 0.2, 1.0)
        assert_true(0.0 < op["efficiency"] < 1.0, f"P={P}: eta={op['efficiency']:.4f} in (0,1)")
        assert_true(op["P_mech_pu"] > op["P_elec_pu"], "P_mech > P_elec (losses positive)")


def test_swing_energy_conservation():
    print("\n[Test 7] Swing ODE: undamped steady-state stays put; damped returns to sync")
    m, _ = make_model()
    Ef, Pm = 1.7, 0.85
    # Start exactly at equilibrium, no damping -> delta and omega should not drift
    m_nd = m
    D_save = m_nd.D
    m_nd.D = 0.0
    r = m_nd.simulate_swing(Pm=Pm, Ef=Ef, duration_s=3.0, dt=0.005)
    m_nd.D = D_save
    drift = np.max(np.abs(r["delta_deg"] - r["delta_deg"][0]))
    omega_drift = np.max(np.abs(r["omega_pu"] - 1.0))
    assert_true(drift < 0.5, f"Equilibrium start: delta drift {drift:.4f} deg ~ 0")
    assert_true(omega_drift < 1e-3, f"omega stays at 1.0 pu (drift {omega_drift:.2e})")


def test_swing_step_recovery():
    print("\n[Test 8] Swing ODE: damped recovery to new equilibrium after load step")
    m, _ = make_model()
    Ef, Pm = 1.7, 0.7
    r = m.simulate_swing(Pm=Pm, Ef=Ef, duration_s=8.0, dt=0.005,
                         P_step=0.1, t_step=0.5)
    # New equilibrium: sin(delta) = (Pm+0.1)*Xs/(Ef)
    d_new = np.degrees(np.arcsin((Pm + 0.1) * m.Xs / Ef))
    final = np.mean(r["delta_deg"][-100:])
    assert_true(r["success"], "Integration succeeded")
    assert_true(abs(final - d_new) < 1.0, f"Settles near new eq {d_new:.2f}deg (got {final:.2f})")
    assert_true(abs(r["omega_pu"][-1] - 1.0) < 1e-3, "Speed returns to synchronous")


def test_stability_limit():
    print("\n[Test 9] Loss of synchronism when Pm > Pmax")
    m, _ = make_model()
    Ef = 1.2
    Pmax = m.pmax(Ef, 1.0)
    # Drive mechanical power above pull-out -> delta runs away
    r = m.simulate_swing(Pm=Pmax * 1.3, Ef=Ef, delta0=np.radians(30),
                         duration_s=3.0, dt=0.002)
    assert_true(r["delta_deg"][-1] > 90.0, f"delta exceeds 90deg (={r['delta_deg'][-1]:.1f}) -> unstable")
    # And a stable case stays below 90
    r2 = m.simulate_swing(Pm=Pmax * 0.5, Ef=Ef, duration_s=3.0, dt=0.002)
    assert_true(np.all(r2["delta_deg"] < 90.0), "Pm<Pmax stays synchronised (delta<90)")


def test_capability_curve():
    print("\n[Test 10] Capability curve: armature & field limit circles")
    m, _ = make_model()
    c = m.capability_curve(1.0)
    r_arm = np.hypot(c["P_armature"], c["Q_armature"])
    assert_true(np.allclose(r_arm, 1.0, atol=1e-9), "Armature limit is unit circle (S=1 pu)")
    assert_true(c["field_radius"] > 0 and c["field_center_Q"] < 0,
                f"Field circle r={c['field_radius']:.3f}, center_Q={c['field_center_Q']:.3f}")


def test_avr_field_control():
    print("\n[Test 11] AVR field control runs and keeps machine synchronised")
    m, cm = make_model()
    r = cm.predict({"P_pu": 0.8, "Q_pu": 0.25, "simulate": True,
                    "avr": True, "Vref_pu": 1.0, "duration_s": 3.0})
    tr = r["transient"]
    assert_true(tr["success"], "AVR transient integrates successfully")
    assert_true(tr["stable"], "Machine stays stable under AVR")
    assert_true(np.all(np.isfinite(tr["Ef_pu"])), "Ef trajectory finite under AVR")


def test_predict_interface():
    print("\n[Test 12] ComponentModel predict() interface")
    _, cm = make_model()
    r = cm.predict({"P_pu": 0.85, "Q_pu": 0.3})
    for key in ["Ef_pu", "delta_deg", "If_A", "Q_pu", "efficiency",
                "Pmax_pu", "stable", "over_excited"]:
        assert_true(key in r, f"Key '{key}' in output")
    info = cm.get_info()
    assert_true(info["component_id"] == "EC179", "component_id == EC179")
    assert_true("Kundur" in info["source"], "Kundur cited in source")


def test_benchmark():
    print("\n[Test 13] Benchmark: 5s swing transient")
    m, _ = make_model()
    t0 = time.perf_counter()
    m.simulate_swing(Pm=0.85, Ef=1.7, duration_s=5.0, dt=0.005, P_step=0.1, t_step=1.0)
    elapsed = time.perf_counter() - t0
    print(f"  5s swing simulation in {elapsed*1000:.1f} ms")
    assert_true(elapsed < 5.0, "Completes in < 5 s")


if __name__ == "__main__":
    tests = [
        test_power_angle_law,
        test_power_monotone_and_pmax,
        test_excitation_controls_Q,
        test_emf_from_field,
        test_operating_point_consistency,
        test_efficiency_bounds,
        test_swing_energy_conservation,
        test_swing_step_recovery,
        test_stability_limit,
        test_capability_curve,
        test_avr_field_control,
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
    print(f"EC179 WRSG F2a dq-frame -- Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    sys.exit(0 if failed == 0 else 1)
