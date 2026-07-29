"""
EC173 -- Distribution Transformer -- F2a Equivalent-Circuit + Thermal ODE
Test suite: equivalent-circuit consistency, efficiency physics, energy
conservation, thermal ODE transient, edge cases, predict() interface, benchmark.
Run:  python3 scripts/test_model.py   (system python3, NOT pytest)
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from model import DistributionTransformerF2a
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
def test_equivalent_circuit_consistency():
    print("\n[Test 1] Equivalent-circuit params recover rated test losses")
    m, _ = make_model()
    ec = m.equivalent_circuit()
    # R_eq_pu * S_rated should reproduce P_k; G_c_pu * S_rated -> P_0
    assert_true(abs(ec["R_eq_pu"] * m.S_rated - m.P_k) < 1.0,
                f"R_eq_pu*S = {ec['R_eq_pu']*m.S_rated:.1f} W ~ P_k={m.P_k} W")
    assert_true(abs(m.G_c_pu * m.S_rated - m.P_0) < 1.0,
                f"G_c_pu*S = {m.G_c_pu*m.S_rated:.1f} W ~ P_0={m.P_0} W")
    # |Z_eq| = u_k  ->  R^2 + X^2 = u_k^2
    z = np.hypot(ec["R_eq_pu"], ec["X_eq_pu"])
    assert_true(abs(z - m.u_k) < 1e-9, f"|Z_eq_pu|={z:.5f} == u_k={m.u_k}")
    assert_true(ec["X_eq_pu"] > ec["R_eq_pu"], "X_eq > R_eq (expected for power xfmr)")


def test_efficiency_range():
    print("\n[Test 2] Efficiency in (0.97, 0.995) at sensible loads")
    m, _ = make_model()
    for K in [0.2, 0.4, 0.6, 0.8, 1.0]:
        eta = float(m.efficiency(K, 1.0, 75.0, 0.9))
        assert_true(0.97 < eta < 0.995, f"eta(K={K})={eta*100:.3f}%")


def test_peak_efficiency_partial_load():
    print("\n[Test 3] Peak efficiency at PARTIAL load (PLR_opt = sqrt(P0/Pk))")
    m, _ = make_model()
    K = np.linspace(0.01, 1.2, 400)
    eta = m.efficiency(K, 1.0, 75.0, 1.0)
    K_peak = K[int(np.argmax(eta))]
    K_opt = m.optimal_load_fraction()
    assert_true(0.2 < K_opt < 0.6, f"PLR_opt={K_opt:.3f} is partial load")
    assert_true(abs(K_peak - K_opt) < 0.05,
                f"argmax eta at K={K_peak:.3f} ~ PLR_opt={K_opt:.3f}")
    # Efficiency at PLR_opt exceeds efficiency at full load
    assert_true(float(m.efficiency(K_opt, 1.0, 75.0, 1.0)) >
                float(m.efficiency(1.0, 1.0, 75.0, 1.0)),
                "eta(PLR_opt) > eta(full load)")


def test_loss_balance_at_optimum():
    print("\n[Test 4] At PLR_opt: copper loss == core loss")
    m, _ = make_model()
    K_opt = m.optimal_load_fraction()
    p_core = float(m.core_loss(1.0))
    p_cu = float(m.copper_loss(K_opt, 75.0))  # at reference temp
    assert_true(abs(p_core - p_cu) / p_core < 0.01,
                f"P_core={p_core:.1f} W ~ P_cu={p_cu:.1f} W at PLR_opt")


def test_energy_conservation():
    print("\n[Test 5] Energy conservation: P_in = P_out + losses")
    m, _ = make_model()
    for K in [0.3, 0.7, 1.0]:
        P_out = float(m.output_power(K, 0.9))
        P_loss = float(m.total_loss(K, 1.0, 75.0))
        eta = float(m.efficiency(K, 1.0, 75.0, 0.9))
        P_in = P_out + P_loss
        assert_true(abs(eta - P_out / P_in) < 1e-9,
                    f"K={K}: eta matches P_out/(P_out+loss) ({eta*100:.3f}%)")


def test_core_loss_voltage_dependence():
    print("\n[Test 6] Core loss follows Steinmetz V^n_B, copper indep. of V")
    m, _ = make_model()
    assert_true(m.core_loss(1.1) > m.core_loss(1.0) > m.core_loss(0.9),
                "core loss monotone in voltage")
    ratio = float(m.core_loss(1.1) / m.core_loss(1.0))
    assert_true(abs(ratio - 1.1 ** m.n_B) < 1e-6,
                f"core loss ratio {ratio:.4f} == 1.1^n_B")


def test_voltage_regulation():
    print("\n[Test 7] Voltage regulation positive (lagging pf), grows with load")
    m, _ = make_model()
    vr_half = m.voltage_regulation(0.5, 0.9)["vr_exact_pu"]
    vr_full = m.voltage_regulation(1.0, 0.9)["vr_exact_pu"]
    assert_true(vr_full > vr_half > 0, f"VR: half={vr_half*100:.2f}% full={vr_full*100:.2f}%")
    assert_true(vr_full < 0.06, f"VR full={vr_full*100:.2f}% < 6% (sane for u_k=4%)")
    # Leading pf reduces (can reverse) regulation
    vr_lead = m.voltage_regulation(1.0, 0.9, leading=True)["vr_exact_pu"]
    assert_true(vr_lead < vr_full, "leading pf lowers VR")


def test_thermal_ode_rises_with_load():
    print("\n[Test 8] Thermal ODE: hot-spot rises and increases with load")
    m, _ = make_model()
    r_lo = m.simulate_thermal(0.3, 20.0, dt=120.0, duration=28800.0)
    r_hi = m.simulate_thermal(1.0, 20.0, dt=120.0, duration=28800.0)
    assert_true(r_hi["T_hot_spot"][-1] > r_lo["T_hot_spot"][-1],
                f"T_hs(K=1.0)={r_hi['T_hot_spot'][-1]:.1f} > "
                f"T_hs(K=0.3)={r_lo['T_hot_spot'][-1]:.1f} degC")
    assert_true(r_hi["T_hot_spot"][-1] > 20.0, "hot-spot above ambient")
    # At rated load steady hot-spot ~ ambient + 55 + 23 = ~98 degC
    assert_true(90.0 < r_hi["T_hot_spot"][-1] < 105.0,
                f"rated steady T_hs={r_hi['T_hot_spot'][-1]:.1f} degC ~ 98 degC")


def test_thermal_transient_lag():
    print("\n[Test 9] Cold-start transient: T rises monotonically, oil lags winding")
    m, _ = make_model()
    r = m.simulate_thermal(1.0, 20.0, dt=60.0, duration=43200.0,
                           theta_oil0=0.0, theta_hs0=0.0)
    Ths = r["T_hot_spot"]
    assert_true(Ths[-1] > Ths[0], "hot-spot warms from cold start")
    assert_true(np.all(np.diff(Ths) > -1e-6), "monotone non-decreasing under step-up")
    # converges to steady form within the run
    dT = abs(Ths[-1] - Ths[-2])
    assert_true(dT < 0.1, f"near steady state at end (dT={dT:.4f} K/step)")


def test_thermal_matches_steady_closed_form():
    print("\n[Test 10] ODE steady state matches closed-form hot_spot_steady")
    m, _ = make_model()
    r = m.simulate_thermal(0.8, 25.0, dt=120.0, duration=86400.0)
    closed = float(m.hot_spot_steady(0.8, 25.0))
    assert_true(abs(r["T_hot_spot"][-1] - closed) < 0.5,
                f"ODE end {r['T_hot_spot'][-1]:.2f} ~ closed-form {closed:.2f} degC")


def test_daily_profile_low_load_factor():
    print("\n[Test 11] Daily profile has low average load factor (<0.6)")
    m, _ = make_model()
    prof, hourly = m.residential_daily_profile()
    lf = float(hourly.mean())
    assert_true(0.3 < lf < 0.6, f"daily load factor={lf:.3f} (low, partial-load device)")
    # 24h ODE run stays in safe thermal band
    r = m.simulate_thermal(prof, 25.0, dt=300.0, duration=86400.0)
    assert_true(r["T_hot_spot"].max() < 120.0,
                f"peak daily T_hs={r['T_hot_spot'].max():.1f} degC < 120")


def test_predict_interface_and_benchmark():
    print("\n[Test 12] predict() interface + benchmark timing")
    _, cm = make_model()
    t0 = time.perf_counter()
    r = cm.predict({"load_fraction": 0.5, "power_factor": 0.9, "duration_s": 14400.0})
    elapsed = time.perf_counter() - t0
    for key in ["operating_point", "equivalent_circuit", "t", "T_hot_spot",
                "T_top_oil", "load", "p_total", "T_hot_spot_final"]:
        assert_true(key in r, f"key '{key}' in predict() output")
    op = r["operating_point"]
    assert_true(0 < op["efficiency"] < 1, f"eta={op['efficiency']*100:.2f}% in (0,1)")
    assert_true(len(r["t"]) == len(r["T_hot_spot"]), "arrays same length")
    print(f"  predict() in {elapsed*1000:.1f} ms")
    assert_true(elapsed < 5.0, "predict() completes < 5 s")


if __name__ == "__main__":
    tests = [
        test_equivalent_circuit_consistency,
        test_efficiency_range,
        test_peak_efficiency_partial_load,
        test_loss_balance_at_optimum,
        test_energy_conservation,
        test_core_loss_voltage_dependence,
        test_voltage_regulation,
        test_thermal_ode_rises_with_load,
        test_thermal_transient_lag,
        test_thermal_matches_steady_closed_form,
        test_daily_profile_low_load_factor,
        test_predict_interface_and_benchmark,
    ]
    passed = failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"  ERROR: {e}")

    print(f"\n{'='*60}")
    print(f"EC173 Distribution Transformer F2a -- Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    sys.exit(0 if failed == 0 else 1)
