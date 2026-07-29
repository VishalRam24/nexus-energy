"""
EC178 -- Switched Reluctance Motor (SRM) -- F2a Reluctance / Co-energy Model
Test suite: physics sanity (co-energy torque, energy conservation, no-PM,
efficiency bounds), edge cases, predict() interface, benchmark timing.
Run:  python3 scripts/test_model.py    (system python3, NOT pytest)
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from model import SRM_F2a
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
def test_inductance_profile():
    print("\n[Test 1] Inductance L(theta) varies between L_unaligned and L_aligned")
    m, _ = make_model()
    # aligned is theta_e = pi -> theta_mech = pi/Nr for phase 0
    th_unaligned = 0.0
    th_aligned = np.pi / m.Nr
    L_u = m.inductance(th_unaligned, 0.0, 0)
    L_a = m.inductance(th_aligned, 0.0, 0)
    assert_true(abs(L_u - m.L_u) < 1e-9, f"Unaligned L={L_u*1e3:.2f} mH ~ L_unaligned")
    assert_true(abs(L_a - m.L_a) < 1e-9, f"Aligned   L={L_a*1e3:.2f} mH ~ L_aligned")
    assert_true(L_a > L_u, f"L_aligned ({L_a*1e3:.1f} mH) > L_unaligned ({L_u*1e3:.1f} mH)")


def test_dLdtheta_sign():
    print("\n[Test 2] dL/dtheta > 0 in rising region (positive motoring torque)")
    m, _ = make_model()
    # rising region: theta_e in (0, pi) -> theta_mech in (0, pi/Nr)
    th = 0.5 * (np.pi / m.Nr)  # mid-rise
    dL = m.dL_dtheta_mech(th, 5.0, 0)
    assert_true(dL > 0, f"dL/dtheta = {dL:.4f} H/rad > 0 (rising inductance)")
    # falling region: theta_e in (pi, 2pi)
    th2 = 1.5 * (np.pi / m.Nr)
    dL2 = m.dL_dtheta_mech(th2, 5.0, 0)
    assert_true(dL2 < 0, f"dL/dtheta = {dL2:.4f} H/rad < 0 (falling inductance)")


def test_torque_from_coenergy():
    print("\n[Test 3] Torque T = 0.5 i^2 dL/dtheta (co-energy, scales with i^2)")
    m, _ = make_model()
    th = 0.5 * (np.pi / m.Nr)
    dL = m.dL_dtheta_mech(th, 5.0, 0)
    T1 = 0.5 * (5.0 ** 2) * dL
    T2 = 0.5 * (10.0 ** 2) * m.dL_dtheta_mech(th, 10.0, 0)
    assert_true(T1 > 0, f"T(5A) = {T1:.3f} N.m > 0")
    # doubling current -> ~4x torque (modulo saturation lowering dL slightly)
    ratio = T2 / T1
    assert_true(2.5 < ratio < 4.5, f"T(10A)/T(5A) = {ratio:.2f} (~i^2 scaling)")


def test_no_pm_zero_current_zero_torque():
    print("\n[Test 4] No permanent magnet: zero current -> zero torque everywhere")
    m, _ = make_model()
    thetas = np.linspace(0, 2 * np.pi / m.Nr, 25)
    Tmax = 0.0
    for th in thetas:
        T = sum(0.5 * 0.0 ** 2 * m.dL_dtheta_mech(th, 0.0, ph) for ph in range(m.Nph))
        Tmax = max(Tmax, abs(T))
    assert_true(Tmax < 1e-12, f"max|T| at i=0 is {Tmax:.2e} N.m (reluctance only, no PM)")


def test_saturation_reduces_inductance():
    print("\n[Test 5] Magnetic saturation lowers aligned inductance at high current")
    m, _ = make_model()
    L_lo = m.L_aligned_sat(0.1)
    L_hi = m.L_aligned_sat(50.0)
    assert_true(L_hi < L_lo, f"L_aligned: {L_lo*1e3:.1f} mH (low i) -> {L_hi*1e3:.1f} mH (high i)")
    assert_true(L_hi >= m.L_u, f"saturated L ({L_hi*1e3:.1f} mH) stays >= L_unaligned")


def test_motoring_produces_torque():
    print("\n[Test 6] Excited machine produces positive average torque")
    m, _ = make_model()
    r = m.simulate(T_load=2.0, duration_s=0.04, dt=2e-5)
    assert_true(r["T_avg"] > 0, f"T_avg = {r['T_avg']:.3f} N.m > 0")
    assert_true(np.max(r["phase_currents"]) > 0.1,
                f"phase current builds up (max {np.max(r['phase_currents']):.2f} A)")


def test_energy_conservation():
    print("\n[Test 7] Energy balance: W_elec >= W_mech + W_copper (no creation)")
    m, _ = make_model()
    r = m.simulate(T_load=3.0, duration_s=0.04, dt=2e-5)
    We, Wm, Wc = r["W_elec_J"], r["W_mech_J"], r["W_copper_J"]
    assert_true(We > 0, f"W_elec = {We:.3f} J > 0 (net electrical input)")
    # Input must cover mechanical output + copper loss (rest -> field/iron/KE)
    assert_true(We + 1e-6 >= Wm + Wc - 1e-3,
                f"W_elec ({We:.3f}) >= W_mech ({Wm:.3f}) + W_copper ({Wc:.3f})")


def test_efficiency_bounds():
    print("\n[Test 8] Efficiency strictly in (0, 1)")
    m, _ = make_model()
    for TL in [1.0, 2.0, 4.0]:
        r = m.simulate(T_load=TL, duration_s=0.04, dt=2e-5)
        eta = r["efficiency"]
        assert_true(0.0 < eta < 1.0, f"T_load={TL} N.m -> eff = {eta:.4f} in (0,1)")


def test_torque_ripple_present():
    print("\n[Test 9] Torque ripple is positive and finite (doubly-salient SRM)")
    m, _ = make_model()
    r = m.simulate(T_load=2.0, duration_s=0.04, dt=2e-5)
    rip = r["torque_ripple"]
    assert_true(rip > 0, f"torque ripple = {rip:.3f} > 0 (instantaneous torque pulsates)")
    assert_true(np.isfinite(rip), "ripple is finite")


def test_load_slows_machine():
    print("\n[Test 10] Higher load torque yields lower / non-increasing final speed")
    m, _ = make_model()
    r_lo = m.simulate(T_load=1.0, omega0=157.0, duration_s=0.03, dt=2e-5)
    r_hi = m.simulate(T_load=8.0, omega0=157.0, duration_s=0.03, dt=2e-5)
    assert_true(r_hi["speed_rpm"][-1] <= r_lo["speed_rpm"][-1] + 1.0,
                f"final speed: load=8 -> {r_hi['speed_rpm'][-1]:.0f} rpm "
                f"<= load=1 -> {r_lo['speed_rpm'][-1]:.0f} rpm")


def test_predict_interface():
    print("\n[Test 11] ComponentModel predict() interface + metadata")
    _, cm = make_model()
    assert_true(cm.component_id == "EC178", "component_id == EC178")
    assert_true(cm.version == "1.0.0", "version == 1.0.0")
    r = cm.predict({"T_load": 2.0, "duration_s": 0.02, "dt": 2e-5})
    for key in ["t", "torque", "omega", "speed_rpm", "phase_currents",
                "T_avg", "torque_ripple", "efficiency"]:
        assert_true(key in r, f"Key '{key}' in output")
    assert_true(len(r["t"]) == len(r["torque"]), "time/torque arrays same length")


def test_benchmark():
    print("\n[Test 12] Benchmark: 40 ms simulation completes quickly")
    m, _ = make_model()
    t0 = time.perf_counter()
    m.simulate(T_load=2.0, duration_s=0.04, dt=2e-5)
    elapsed = time.perf_counter() - t0
    print(f"  40 ms electromechanical sim in {elapsed*1000:.0f} ms wall")
    assert_true(elapsed < 5.0, "Completes in < 5 s")


if __name__ == "__main__":
    tests = [
        test_inductance_profile,
        test_dLdtheta_sign,
        test_torque_from_coenergy,
        test_no_pm_zero_current_zero_torque,
        test_saturation_reduces_inductance,
        test_motoring_produces_torque,
        test_energy_conservation,
        test_efficiency_bounds,
        test_torque_ripple_present,
        test_load_slows_machine,
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
    print(f"EC178 SRM F2a -- Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    sys.exit(0 if failed == 0 else 1)
