"""
EC056 -- Linear Fresnel CSP -- F2a Physics-Lumped Receiver Thermal ODE
Test suite: optics/IAM sanity, energy conservation, T^4 radiation,
P=0 at DNI=0, ODE convergence, edge cases, predict() interface, benchmark.

Run:  python3 scripts/test_model.py   (system python3, NOT pytest)
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from model import LinearFresnelF2a
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
def test_iam_peak_unity():
    print("\n[Test 1] IAM = 1 at normal incidence, optics in (0,1)")
    m, _ = make_model()
    assert_true(abs(m.iam_longitudinal(0.0) - 1.0) < 1e-9, "IAM_L(0)=1")
    assert_true(abs(m.iam_transversal(0.0) - 1.0) < 1e-9, "IAM_T(0)=1")
    assert_true(abs(m.end_loss_factor(0.0) - 1.0) < 1e-9, "f_end(0)=1")
    eta0 = m.optical_efficiency(0.0, 0.0)
    assert_true(0.0 < eta0 < 1.0, f"eta_opt(0,0)={eta0:.4f} in (0,1)")


def test_iam_monotone_decreasing():
    print("\n[Test 2] IAM decreases with incidence angle")
    m, _ = make_model()
    L = [m.iam_longitudinal(a) for a in [0, 15, 30, 45, 60]]
    T = [m.iam_transversal(a) for a in [0, 15, 30, 45, 55]]
    assert_true(all(L[i] >= L[i+1] for i in range(len(L)-1)), "IAM_L monotone down")
    assert_true(all(T[i] >= T[i+1] for i in range(len(T)-1)), "IAM_T monotone down")
    assert_true(m.optical_efficiency(50, 50) < m.optical_efficiency(0, 0),
                "eta_opt drops at high angles")


def test_end_loss():
    print("\n[Test 3] End loss reduces optics at off-normal longitudinal angle")
    m, _ = make_model()
    assert_true(m.end_loss_factor(40.0) < 1.0, "f_end(40)<1")
    assert_true(m.end_loss_factor(40.0) <= m.end_loss_factor(10.0) + 1e-12,
                "end loss grows with theta_L")


def test_power_zero_at_no_sun():
    print("\n[Test 4] No absorbed power at DNI=0; P->0 at steady state (no stored heat)")
    m, _ = make_model()
    assert_true(m.absorbed_power_per_m(0.0, 0.0, 0.0) == 0.0, "q_abs=0 at DNI=0")
    # Start wall at the HTF inlet temp so there is no stored heat to deliver:
    # with no sun and no thermal store, the receiver delivers ~0 useful power.
    r = m.simulate(0.0, 0.0, 0.0, 298.15, 473.15, T_w0_K=473.15,
                   dt=10.0, duration_s=1200.0)
    assert_true(r["P_electric_W"][-1] < 1.0, "P_electric->0 at DNI=0 steady state")
    assert_true(r["Q_to_fluid_W"][-1] <= 1e-3 * abs(r["Q_to_fluid_W"][0] + 1.0) + 1.0,
                "no net useful heat to fluid at DNI=0 steady state")
    # A hot wall (stored heat) cools when the sun is off:
    rc = m.simulate(0.0, 0.0, 0.0, 298.15, 473.15, T_w0_K=520.0,
                    dt=10.0, duration_s=600.0)
    assert_true(rc["T_wall_K"][-1] < rc["T_wall_K"][0], "hot wall cools when DNI=0")


def test_energy_conservation_steady():
    print("\n[Test 5] Steady-state energy balance: q_abs = q_conv+q_rad+q_htf")
    m, _ = make_model()
    Tw, d = m.steady_wall_temp(850.0, 10.0, 15.0, 298.15, 473.15)
    residual = d["q_abs_per_m"] - (d["q_conv_per_m"] + d["q_rad_per_m"] + d["q_htf_per_m"])
    scale = max(1.0, abs(d["q_abs_per_m"]))
    assert_true(abs(residual) / scale < 1e-3,
                f"balance residual {residual:.3f} W/m << q_abs {d['q_abs_per_m']:.1f}")


def test_radiation_T4():
    print("\n[Test 6] Radiative loss scales ~ T^4")
    m, _ = make_model()
    T_amb = 298.15
    # isolate radiation: ratio should follow (T1^4 - Tsky^4)/(T2^4 - Tsky^4)
    T1, T2 = 600.0, 700.0
    r1 = m.q_rad_per_m(T1, T_amb)
    r2 = m.q_rad_per_m(T2, T_amb)
    Tsky = T_amb - m.T_sky_off
    expected = (T2**4 - Tsky**4) / (T1**4 - Tsky**4)
    got = r2 / r1
    assert_true(abs(got - expected) / expected < 1e-9, f"q_rad ratio matches T^4 law")
    assert_true(r2 > r1, "radiation increases with wall T")


def test_htf_outlet_rises():
    print("\n[Test 7] HTF outlet hotter than inlet under sun; rises with DNI")
    m, _ = make_model()
    _, d_lo = m.steady_wall_temp(400.0, 0.0, 0.0, 298.15, 473.15)
    _, d_hi = m.steady_wall_temp(900.0, 0.0, 0.0, 298.15, 473.15)
    assert_true(d_hi["T_htf_out_K"] > 473.15, "T_out > T_in under strong sun")
    assert_true(d_hi["T_htf_out_K"] > d_lo["T_htf_out_K"], "T_out rises with DNI")
    assert_true(d_hi["Q_to_fluid_W"] > d_lo["Q_to_fluid_W"], "delivered heat rises with DNI")


def test_thermal_efficiency_range():
    print("\n[Test 8] Thermal efficiency physical (0 < eta_th < eta_opt)")
    m, _ = make_model()
    _, d = m.steady_wall_temp(850.0, 0.0, 0.0, 298.15, 473.15)
    assert_true(0.0 < d["eta_thermal"] < 1.0, f"eta_th={d['eta_thermal']:.3f} in (0,1)")
    assert_true(d["eta_thermal"] <= d["eta_optical"] + 1e-6,
                f"eta_th {d['eta_thermal']:.3f} <= eta_opt {d['eta_optical']:.3f}")


def test_ode_reaches_steady_state():
    print("\n[Test 9] Wall ODE converges to steady state")
    m, _ = make_model()
    r = m.simulate(800.0, 0.0, 0.0, 298.15, 473.15, T_w0_K=300.0,
                   dt=5.0, duration_s=3000.0)
    dT = abs(r["T_wall_K"][-1] - r["T_wall_K"][-2])
    assert_true(dT < 0.05, f"near steady: dT={dT:.4f} K between last steps")
    assert_true(r["T_wall_K"][-1] > 473.15, "hot wall above HTF inlet at steady state")


def test_cold_start_heats_up():
    print("\n[Test 10] Cold-start wall heats toward operating temperature")
    m, _ = make_model()
    r = m.simulate(900.0, 0.0, 0.0, 298.15, 473.15, T_w0_K=300.0,
                   dt=5.0, duration_s=1500.0)
    assert_true(r["T_wall_K"][-1] > r["T_wall_K"][0], "wall heats up from cold start")
    # higher DNI -> hotter steady wall
    r2 = m.simulate(500.0, 0.0, 0.0, 298.15, 473.15, T_w0_K=300.0,
                    dt=5.0, duration_s=3000.0)
    rH = m.simulate(1000.0, 0.0, 0.0, 298.15, 473.15, T_w0_K=300.0,
                    dt=5.0, duration_s=3000.0)
    assert_true(rH["T_wall_K"][-1] > r2["T_wall_K"][-1], "hotter wall at higher DNI")


def test_predict_interface():
    print("\n[Test 11] ComponentModel predict() interface")
    _, cm = make_model()
    r = cm.predict({"dni": 800.0, "theta_L_deg": 10.0, "theta_T_deg": 15.0,
                    "T_htf_in_C": 200.0, "duration_s": 600.0, "dt": 20.0})
    for key in ["t", "T_wall_K", "T_htf_out_K", "q_abs_per_m", "q_conv_per_m",
                "q_rad_per_m", "q_htf_per_m", "Q_to_fluid_W", "eta_thermal",
                "eta_optical", "P_electric_W", "T_wall_C", "T_htf_out_C"]:
        assert_true(key in r, f"Key '{key}' in output")
    assert_true(len(r["t"]) == len(r["T_wall_K"]) == len(r["T_htf_out_K"]),
                "arrays same length")
    info = cm.get_info()
    assert_true(info["component_id"] == "EC056", "get_info id=EC056")


def test_benchmark():
    print("\n[Test 12] Benchmark: 30-min sim at dt=5s")
    m, _ = make_model()
    t0 = time.perf_counter()
    m.simulate(800.0, 0.0, 0.0, 298.15, 473.15, T_w0_K=300.0, dt=5.0, duration_s=1800.0)
    elapsed = time.perf_counter() - t0
    print(f"  1800 s simulation in {elapsed*1000:.1f} ms")
    assert_true(elapsed < 5.0, "completes in < 5 s")


if __name__ == "__main__":
    tests = [
        test_iam_peak_unity,
        test_iam_monotone_decreasing,
        test_end_loss,
        test_power_zero_at_no_sun,
        test_energy_conservation_steady,
        test_radiation_T4,
        test_htf_outlet_rises,
        test_thermal_efficiency_range,
        test_ode_reaches_steady_state,
        test_cold_start_heats_up,
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
    print(f"EC056 Linear Fresnel F2a -- Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    sys.exit(0 if failed == 0 else 1)
