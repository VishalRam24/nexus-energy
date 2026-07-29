"""
EC137 -- Attenuator WEC -- F2a Coupled Oscillators
Test suite: physics sanity (energy conservation, capture-width bound,
power from relative joint motion, optimal PTO damping), edge cases,
predict() interface, benchmark timing.
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from model import AttenuatorWEC_F2a
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
def test_power_nonnegative():
    print("\n[Test 1] PTO power is always non-negative (resistive extraction)")
    m, _ = make_model()
    r = m.simulate(2.5, 8.0, dt=0.1, duration_s=60.0)
    assert_true(np.all(r["power_pto"] >= -1e-9), "All per-joint PTO power >= 0")
    assert_true(np.all(r["power_total_elec"] >= -1e-9), "Total electrical power >= 0")
    assert_true(r["mean_power_elec_W"] > 0, f"Mean elec power > 0: {r['mean_power_elec_W']/1e3:.1f} kW")


def test_power_from_relative_motion():
    print("\n[Test 2] Power comes from relative joint motion (theta_dot)")
    m, _ = make_model()
    r = m.simulate(3.0, 9.0, B_pto=1.3e8, dt=0.1, duration_s=60.0)
    # Reconstruct power from theta_dot and B_pto -> must match reported.
    recon = (1.3e8 * r["theta_dot"] ** 2).sum(axis=0)
    err = np.max(np.abs(recon - r["power_total_mech"])) / (np.max(r["power_total_mech"]) + 1e-9)
    assert_true(err < 1e-9, f"P = B_pto*theta_dot^2 holds (rel err {err:.2e})")
    # Zero relative motion -> zero power: stiff joints (huge B_pto) lock motion.
    r_lock = m.simulate(3.0, 9.0, B_pto=1e12, dt=0.1, duration_s=40.0)
    assert_true(r_lock["mean_power_mech_W"] < r["mean_power_mech_W"],
                "Over-damped (locked) joints extract less than tuned PTO")


def test_energy_conservation():
    print("\n[Test 3] Energy balance: excitation work = PTO + radiation dissipation")
    m, _ = make_model()
    for Hs, Te in [(2.0, 8.0), (3.0, 9.0), (4.0, 11.0)]:
        r = m.simulate(Hs, Te, B_pto=1.3e8, dt=0.05, duration_s=100.0)
        assert_true(abs(r["energy_residual"]) < 0.05,
                    f"Hs={Hs},Te={Te}: |residual|={abs(r['energy_residual'])*100:.2f}% < 5%")


def test_capture_width_bounded():
    print("\n[Test 4] Capture width bounded by radiation limit (Falnes theorem)")
    m, _ = make_model()
    for Hs, Te in [(1.0, 7.0), (3.0, 9.0), (5.0, 12.0)]:
        r = m.simulate(Hs, Te, dt=0.1, duration_s=60.0)
        wavelength = 2.0 * np.pi / m.dispersion_k(2.0 * np.pi / Te)
        # Total point-absorber bound = n_joint * lambda/2pi.
        cw_bound = m.n_joint * wavelength / (2.0 * np.pi)
        assert_true(r["capture_width_m"] <= cw_bound + 1e-6,
                    f"Hs={Hs}: CW={r['capture_width_m']:.2f} <= bound {cw_bound:.1f} m")
        assert_true(r["capture_width_m"] >= 0, "CW non-negative")


def test_cwr_realistic_range():
    print("\n[Test 5] Tuned capture-width ratio in realistic attenuator range")
    m, _ = make_model()
    o = m.optimal_B_pto(3.0, 9.0, dt=0.1, duration_s=80.0)
    r = m.simulate(3.0, 9.0, B_pto=o["B_opt"], dt=0.1, duration_s=80.0)
    # Attenuators: CWR (vs narrow width) typically 0.2-8; absolute CW a few-10 m.
    assert_true(0.5 < r["capture_width_ratio"] < 8.0,
                f"CWR={r['capture_width_ratio']:.2f} in plausible band")
    assert_true(2.0 < r["capture_width_m"] < 30.0,
                f"CW={r['capture_width_m']:.2f} m physically reasonable")


def test_optimal_pto_damping():
    print("\n[Test 6] Optimal PTO damping maximises power (Falnes impedance match)")
    m, _ = make_model()
    o = m.optimal_B_pto(3.0, 9.0, dt=0.1, duration_s=80.0, n_scan=13)
    P = o["P_grid"]
    i_best = int(np.argmax(P))
    assert_true(0 < i_best < len(P) - 1,
                f"Optimum is interior (idx {i_best}/{len(P)-1}) -> true max, not edge")
    # Power must drop on both sides of the optimum (concave / single peak).
    assert_true(P[i_best] > P[0] and P[i_best] > P[-1],
                "P(opt) exceeds both extremes of the B_pto scan")
    assert_true(o["P_max_elec_W"] > 0, f"P_max={o['P_max_elec_W']/1e3:.1f} kW")


def test_power_increases_with_Hs():
    print("\n[Test 7] Mean power increases with wave height (P ~ H_s^2)")
    m, _ = make_model()
    P = [m.simulate(Hs, 9.0, B_pto=1.3e8, dt=0.1, duration_s=60.0)["mean_power_elec_W"]
         for Hs in [1.0, 2.0, 3.0, 4.0]]
    for a, b in zip(P[:-1], P[1:]):
        assert_true(b > a, f"power {b/1e3:.1f} kW > {a/1e3:.1f} kW")
    # Roughly quadratic: P(2Hs)/P(Hs) ~ 4 for linear hydrodynamics.
    ratio = P[3] / P[1]   # Hs 4 vs 2
    assert_true(3.0 < ratio < 5.0, f"P(4m)/P(2m)={ratio:.2f} ~ 4 (quadratic scaling)")


def test_joint_phasing():
    print("\n[Test 8] Joints respond with travelling-wave phase lag (attenuator)")
    m, _ = make_model()
    r = m.simulate(3.0, 9.0, dt=0.05, duration_s=80.0)
    # Different joints should NOT be perfectly in phase (else not an attenuator).
    th = r["theta"][:, len(r["t"]) // 2:]
    # Peak-to-peak amplitudes nonzero; cross-correlation lag between j0 and j1.
    j0 = th[0] - th[0].mean()
    j1 = th[1] - th[1].mean()
    corr = np.corrcoef(j0, j1)[0, 1]
    assert_true(abs(corr) < 0.999, f"Joints 0,1 not identical (corr={corr:.3f})")
    assert_true(np.ptp(th[0]) > 1e-4, "Joint motion is excited (nonzero amplitude)")


def test_zero_wave_zero_power():
    print("\n[Test 9] Edge: zero wave height -> zero motion and power")
    m, _ = make_model()
    r = m.simulate(0.0, 9.0, dt=0.2, duration_s=30.0)
    assert_true(r["mean_power_elec_W"] < 1e-6, "No waves -> ~zero power")
    assert_true(np.max(np.abs(r["theta"])) < 1e-6, "No waves -> joints stay still")


def test_efficiency_bounds():
    print("\n[Test 10] Wave-to-wire efficiency factor in (0,1); elec < mech")
    m, _ = make_model()
    r = m.simulate(3.0, 9.0, dt=0.1, duration_s=60.0)
    assert_true(0.0 < r["efficiency"] < 1.0, f"eta_pto*eta_gen={r['efficiency']:.3f} in (0,1)")
    assert_true(r["mean_power_elec_W"] < r["mean_power_mech_W"] + 1e-6,
                "Electrical power < mechanical (PTO losses)")


def test_predict_interface():
    print("\n[Test 11] ComponentModel predict() interface")
    _, cm = make_model()
    r = cm.predict({"H_s": 2.5, "T_e": 8.5, "dt": 0.2, "duration_s": 30.0})
    for key in ["t", "theta", "theta_dot", "power_total_elec",
                "mean_power_elec_W", "capture_width_m", "capture_width_ratio",
                "wave_power_per_m_W", "energy_residual"]:
        assert_true(key in r, f"Key '{key}' in output")
    assert_true(r["theta"].shape[0] == cm._model.n_joint, "theta has n_joint rows")
    assert_true(len(r["t"]) == r["theta"].shape[1], "time and theta aligned")
    info = cm.get_info()
    assert_true(info["component_id"] == "EC137", "get_info reports EC137")


def test_benchmark():
    print("\n[Test 12] Benchmark: 120 s simulation at dt=0.1")
    m, _ = make_model()
    t0 = time.perf_counter()
    m.simulate(3.0, 9.0, dt=0.1, duration_s=120.0)
    elapsed = time.perf_counter() - t0
    print(f"  120 s simulation in {elapsed*1000:.1f} ms")
    assert_true(elapsed < 5.0, "Completes in < 5 s")


if __name__ == "__main__":
    tests = [
        test_power_nonnegative,
        test_power_from_relative_motion,
        test_energy_conservation,
        test_capture_width_bounded,
        test_cwr_realistic_range,
        test_optimal_pto_damping,
        test_power_increases_with_Hs,
        test_joint_phasing,
        test_zero_wave_zero_power,
        test_efficiency_bounds,
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
    print(f"EC137 Attenuator WEC F2a -- Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    sys.exit(0 if failed == 0 else 1)
