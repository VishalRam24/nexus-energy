"""
EC161 -- Dual Active Bridge (DAB) -- F2a Phase-Shift Averaged
Test suite: SPS power-transfer formula, bidirectionality, P_max at phi=pi/2,
energy conservation, efficiency bounds, ZVS region, averaged ODE, interface.
Run:  python3 scripts/test_model.py   (system python3, NOT pytest)
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from model import DAB_F2a
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
def test_sps_formula():
    print("\n[Test 1] SPS power-transfer formula matches De Doncker (1991) Eq.6")
    m, _ = make_model()
    v1, v2, phi = 400.0, 200.0, 0.6
    expected = (m.n * v1 * v2) / (2 * np.pi * m.f_sw * m.L_s) * phi * (1 - abs(phi) / np.pi)
    got = float(m.power_transfer(v1, v2, phi))
    assert_true(abs(got - expected) < 1e-6, f"P={got:.2f} W matches formula {expected:.2f} W")


def test_pmax_at_pi_over_2():
    print("\n[Test 2] |P| maximised at phi = pi/2")
    m, _ = make_model()
    phis = np.linspace(0.01, np.pi - 0.01, 400)
    P = np.array([float(m.power_transfer(400.0, 200.0, p)) for p in phis])
    idx = np.argmax(P)
    assert_true(abs(phis[idx] - np.pi / 2) < 0.02, f"argmax phi={phis[idx]:.4f} ~ pi/2={np.pi/2:.4f}")
    pmax = float(m.power_max(400.0, 200.0))
    assert_true(P[idx] <= pmax + 1e-6 and abs(P[idx] - pmax) < 1.0,
                f"grid peak {P[idx]:.2f} W bounded by power_max() {pmax:.2f} W")
    assert_true(float(m.power_transfer(400.0, 200.0, np.pi / 2)) == pmax, "power_max() = P(pi/2)")


def test_bidirectional():
    print("\n[Test 3] Bidirectional: sign(phi) flips power direction, odd symmetry")
    m, _ = make_model()
    p_pos = float(m.power_transfer(400.0, 200.0, 0.7))
    p_neg = float(m.power_transfer(400.0, 200.0, -0.7))
    assert_true(p_pos > 0 and p_neg < 0, f"P(+0.7)={p_pos:.1f}>0, P(-0.7)={p_neg:.1f}<0")
    assert_true(abs(p_pos + p_neg) < 1e-6, "odd symmetry P(phi) = -P(-phi)")


def test_zero_phi_zero_power():
    print("\n[Test 4] phi=0 transfers zero power")
    m, _ = make_model()
    assert_true(abs(float(m.power_transfer(400.0, 200.0, 0.0))) < 1e-9, "P(0)=0")


def test_phase_inversion():
    print("\n[Test 5] phase_for_power inverts power_transfer (both signs)")
    m, _ = make_model()
    for p_tgt in [3000.0, 7000.0, -5000.0]:
        phi = float(m.phase_for_power(400.0, 200.0, p_tgt))
        p_back = float(m.power_transfer(400.0, 200.0, phi))
        assert_true(abs(p_back - p_tgt) < 1.0, f"target {p_tgt:.0f} W -> phi {phi:.4f} -> {p_back:.1f} W")


def test_efficiency_bounds():
    print("\n[Test 6] Efficiency strictly in (0, 1) across loads")
    m, _ = make_model()
    for p in [500.0, 2000.0, 5000.0, 9000.0]:
        phi = float(m.phase_for_power(400.0, 200.0, p))
        eta = float(m.efficiency(400.0, 200.0, phi))
        assert_true(0.0 < eta < 1.0, f"P={p:.0f} W -> eta={eta:.4f} in (0,1)")


def test_energy_conservation():
    print("\n[Test 7] Energy conservation: P_in = P_out + P_loss")
    m, _ = make_model()
    phi = 0.6
    p_out = abs(float(m.power_transfer(400.0, 200.0, phi)))
    p_loss = float(m.losses(400.0, 200.0, phi))
    eta = float(m.efficiency(400.0, 200.0, phi))
    p_in = p_out + p_loss
    assert_true(p_loss >= 0.0, f"P_loss={p_loss:.2f} W >= 0")
    assert_true(abs(eta - p_out / p_in) < 1e-9, f"eta={eta:.4f} == P_out/(P_out+P_loss)")


def test_losses_nonneg_and_rms_monotone():
    print("\n[Test 8] Losses >= 0 and RMS current grows with |phi|")
    m, _ = make_model()
    i_prev = -1.0
    for phi in np.linspace(0.0, 1.4, 30):
        loss = float(m.losses(400.0, 200.0, phi))
        irms = float(m.inductor_rms_current(400.0, 200.0, phi))
        assert_true(loss >= -1e-9, f"loss(phi={phi:.2f})={loss:.3f} >= 0")
        assert_true(irms >= i_prev - 1e-6, f"i_rms monotone up at phi={phi:.2f}: {irms:.2f} A")
        i_prev = irms


def test_zvs_region():
    print("\n[Test 9] ZVS region: matched voltage (d=1) gives full ZVS for phi>0")
    m, _ = make_model()
    # choose v2 so that n*v2 = v1 -> d=1
    v1 = 400.0
    v2 = v1 / m.n
    z = m.zvs_region(v1, v2, 0.5)
    assert_true(bool(z["full_zvs"]), f"d={float(z['d']):.3f} -> full ZVS at phi=0.5")
    # strong mismatch at tiny phi loses ZVS on one bridge
    z2 = m.zvs_region(800.0, 50.0, 0.02)
    assert_true(not bool(z2["full_zvs"]), "large voltage mismatch at small phi loses full ZVS")


def test_ode_output_regulates():
    print("\n[Test 10] Averaged ODE drives V_out toward power balance (solve_ivp)")
    m, _ = make_model()
    # Start below nominal; positive phi should raise the bus toward steady state.
    r = m.simulate(0.6, v1=400.0, v2_0=150.0, r_load=4.0, dt=2e-5, duration_s=4e-3)
    assert_true(r["v_out"][-1] > r["v_out"][0], f"V_out rises {r['v_out'][0]:.1f}->{r['v_out'][-1]:.1f} V")
    dv = abs(r["v_out"][-1] - r["v_out"][-2])
    assert_true(dv < 1.0, f"approaching steady state, last dV={dv:.4f} V")


def test_predict_interface():
    print("\n[Test 11] ComponentModel predict() interface")
    _, cm = make_model()
    r = cm.predict({"p_target": 5000.0, "dt": 2e-5, "duration_s": 1e-3})
    for key in ["t", "v_out", "phi", "power_transfer", "power_loss",
                "efficiency", "i_rms", "full_zvs"]:
        assert_true(key in r, f"Key '{key}' in output")
    assert_true(len(r["t"]) == len(r["v_out"]), "Arrays same length")


def test_benchmark():
    print("\n[Test 12] Benchmark: 5 ms averaged sim")
    m, _ = make_model()
    t0 = time.perf_counter()
    m.simulate(0.6, v1=400.0, v2_0=200.0, r_load=4.0, dt=2e-5, duration_s=5e-3)
    elapsed = time.perf_counter() - t0
    print(f"  5 ms simulation in {elapsed*1000:.1f} ms")
    assert_true(elapsed < 5.0, "Completes in < 5 s")


if __name__ == "__main__":
    tests = [
        test_sps_formula,
        test_pmax_at_pi_over_2,
        test_bidirectional,
        test_zero_phi_zero_power,
        test_phase_inversion,
        test_efficiency_bounds,
        test_energy_conservation,
        test_losses_nonneg_and_rms_monotone,
        test_zvs_region,
        test_ode_output_regulates,
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
    print(f"EC161 DAB F2a -- Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    sys.exit(0 if failed == 0 else 1)
