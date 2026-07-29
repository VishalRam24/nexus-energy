"""
EC063 -- Vertical Axis Wind Turbine (VAWT) -- F2a DMST + Rotor Dynamics
Test suite: aerodynamic physics sanity, ODE behaviour, edge cases, benchmark.
Run with system python3:  python3 scripts/test_model.py   (NO pytest)
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from model import VAWT_F2a
from predict import ComponentModel

PASS = "✓"
FAIL = "✗"

BETZ = 16.0 / 27.0  # 0.5926


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
def test_cp_below_betz():
    print("\n[Test 1] Cp respects the Betz limit at all TSR")
    m, _ = make_model()
    for lam in np.linspace(0.5, 11.0, 60):
        cp = m.cp(lam)
        assert_true(cp <= BETZ + 1e-6,
                    f"Cp({lam:.2f})={cp:.3f} <= Betz {BETZ:.3f}")
    print("  All 60 TSR points below Betz.")


def test_cp_peak_reasonable():
    print("\n[Test 2] Peak Cp and optimal TSR are VAWT-realistic")
    m, _ = make_model()
    cpmax, lam_opt = m.cp_max()
    assert_true(0.30 < cpmax < BETZ,
                f"Peak Cp={cpmax:.3f} in (0.30, Betz) for a good Darrieus")
    assert_true(2.5 < lam_opt < 5.5,
                f"Optimal TSR={lam_opt:.2f} in typical Darrieus band [2.5,5.5]")


def test_cp_single_peak_shape():
    print("\n[Test 3] Cp(lambda) rises then falls (single peak)")
    m, _ = make_model()
    _, lam_opt = m.cp_max()
    # rising before the peak
    assert_true(m.cp(lam_opt) > m.cp(lam_opt * 0.5),
                "Cp larger at peak than at half the optimal TSR")
    # falling after the peak
    assert_true(m.cp(lam_opt) > m.cp(lam_opt + 2.5),
                "Cp smaller well past the optimal TSR")


def test_cp_zero_at_zero_tsr():
    print("\n[Test 4] No power extraction at TSR -> 0 (Darrieus low-start)")
    m, _ = make_model()
    assert_true(abs(m.cp(0.0)) < 0.02, f"Cp(0)={m.cp(0.0):.4f} ~ 0")
    assert_true(m.cp(0.5) < 0.05, f"Cp(0.5)={m.cp(0.5):.4f} small at low TSR")


def test_power_scales_cube():
    print("\n[Test 5] Aero power scales ~ U^3 at fixed TSR")
    m, _ = make_model()
    lam = 4.0
    U1, U2 = 6.0, 12.0
    w1 = lam * U1 / m.R
    w2 = lam * U2 / m.R
    P1 = m.aero_power(U1, w1)
    P2 = m.aero_power(U2, w2)
    ratio = P2 / P1
    assert_true(abs(ratio - 8.0) < 0.05,
                f"P(2U)/P(U)={ratio:.3f} ~ 8 (cube law)")


def test_torque_positive_in_band():
    print("\n[Test 6] Aero torque positive across the operating TSR band")
    m, _ = make_model()
    U = 10.0
    for lam in [2.0, 3.0, 4.0, 5.0, 6.0]:
        w = lam * U / m.R
        T = m.aero_torque(U, w)
        assert_true(T > 0, f"T_aero(TSR={lam})={T:.1f} N.m > 0")


def test_spinup_runaway_bounded():
    print("\n[Test 7] No-load spin-up runaway is bounded (drag braking)")
    m, _ = make_model()
    r = m.simulate(10.0, T_load=0.0, omega0=6.0, dt=0.5, duration_s=300.0)
    tsr = r["tip_speed_ratio"][-1]
    assert_true(tsr < 11.0, f"Free-run TSR={tsr:.2f} stays bounded < 11")
    assert_true(r["cp"][-1] < 0.05,
                f"Cp at runaway={r['cp'][-1]:.3f} ~ 0 (no net torque)")


def test_load_moves_operating_point():
    print("\n[Test 8] Higher generator load -> lower TSR, higher Cp")
    m, _ = make_model()
    r_lo = m.simulate(10.0, T_load=100.0, omega0=8.0, dt=0.5, duration_s=300.0)
    r_hi = m.simulate(10.0, T_load=300.0, omega0=8.0, dt=0.5, duration_s=300.0)
    tsr_lo, tsr_hi = r_lo["tip_speed_ratio"][-1], r_hi["tip_speed_ratio"][-1]
    assert_true(tsr_hi < tsr_lo,
                f"TSR drops with load: {tsr_hi:.2f} < {tsr_lo:.2f}")
    assert_true(r_hi["cp"][-1] > r_lo["cp"][-1],
                f"Cp rises toward peak: {r_hi['cp'][-1]:.3f} > {r_lo['cp'][-1]:.3f}")


def test_steady_state_reached():
    print("\n[Test 9] Rotor ODE reaches near steady state")
    m, _ = make_model()
    r = m.simulate(10.0, T_load=250.0, omega0=8.0, dt=0.5, duration_s=400.0)
    dw = abs(r["omega"][-1] - r["omega"][-2])
    assert_true(dw < 0.01, f"Near SS: d(omega)={dw:.5f} rad/s between last steps")


def test_wind_step_response():
    print("\n[Test 10] Rotor accelerates when wind steps up")
    m, _ = make_model()

    def wind(t):
        return 8.0 if t < 60.0 else 13.0

    r = m.simulate(wind, T_load=200.0, omega0=8.0, dt=0.5, duration_s=140.0)
    i_before = np.argmin(np.abs(r["t"] - 55.0))
    assert_true(r["omega"][-1] > r["omega"][i_before],
                "Rotor speed rises after wind step up")
    assert_true(r["power_elec"][-1] > r["power_elec"][i_before],
                "Electrical power rises after wind step up")


def test_energy_consistency():
    print("\n[Test 11] Electrical power <= aero power (loss + gen efficiency)")
    m, _ = make_model()
    r = m.simulate(11.0, T_load=300.0, omega0=8.0, dt=0.5, duration_s=200.0)
    for i in range(len(r["t"])):
        Pa, Pe = r["power_aero"][i], r["power_elec"][i]
        if Pa > 1.0:
            assert_true(Pe <= Pa + 1e-6,
                        f"P_elec={Pe:.0f} <= P_aero={Pa:.0f}")
    print("  All samples: electrical <= aerodynamic power.")


def test_predict_interface():
    print("\n[Test 12] ComponentModel predict() interface")
    _, cm = make_model()
    r = cm.predict({"wind_speed": 10.0, "T_load_Nm": 200.0,
                    "dt": 1.0, "duration_s": 20.0})
    for key in ["t", "omega", "rpm", "tip_speed_ratio", "cp",
                "power_aero", "power_elec", "torque_aero"]:
        assert_true(key in r, f"Key '{key}' in output")
    assert_true(len(r["t"]) == len(r["omega"]), "Arrays same length")
    info = cm.get_info()
    assert_true(info["component_id"] == "EC063", "get_info reports EC063")


def test_benchmark():
    print("\n[Test 13] Benchmark: 120 s simulation at dt=0.5")
    m, _ = make_model()
    t0 = time.perf_counter()
    m.simulate(10.0, T_load=200.0, omega0=8.0, dt=0.5, duration_s=120.0)
    elapsed = time.perf_counter() - t0
    print(f"  120 s simulation in {elapsed*1000:.1f} ms")
    assert_true(elapsed < 5.0, "Completes in < 5 s")


if __name__ == "__main__":
    tests = [
        test_cp_below_betz,
        test_cp_peak_reasonable,
        test_cp_single_peak_shape,
        test_cp_zero_at_zero_tsr,
        test_power_scales_cube,
        test_torque_positive_in_band,
        test_spinup_runaway_bounded,
        test_load_moves_operating_point,
        test_steady_state_reached,
        test_wind_step_response,
        test_energy_consistency,
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
    print(f"EC063 VAWT F2a -- Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    sys.exit(0 if failed == 0 else 1)
