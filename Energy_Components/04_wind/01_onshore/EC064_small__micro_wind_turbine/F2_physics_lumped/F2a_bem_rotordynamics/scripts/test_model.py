"""
EC064 -- Small / Micro Wind Turbine -- F2a BEM Rotor-Dynamics
Test suite: physics sanity (Betz, P~U^3, energy balance), ODE transient,
edge cases, predict() interface, benchmark timing.
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from model import SmallWindTurbineF2a, BETZ_LIMIT
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
def test_cp_below_betz():
    print("\n[Test 1] Cp <= Betz limit (16/27) for all lambda, beta")
    m, _ = make_model()
    lam = np.linspace(0.0, 20.0, 500)
    for beta in [0.0, 2.0, 5.0, 10.0]:
        cp = m.Cp(lam, beta)
        assert_true(np.all(cp <= BETZ_LIMIT + 1e-9),
                    f"max Cp(beta={beta})={np.max(cp):.4f} <= {BETZ_LIMIT:.4f}")
    assert_true(np.all(cp >= 0.0), "Cp non-negative everywhere")


def test_cp_peak_at_lambda_opt():
    print("\n[Test 2] Cp peaks near lambda_opt and equals Cp_max")
    m, _ = make_model()
    lam = np.linspace(1.0, 14.0, 1400)
    cp = m.Cp(lam, 0.0)
    lam_peak = lam[np.argmax(cp)]
    assert_true(abs(np.max(cp) - m.Cp_max) < 1e-3,
                f"peak Cp={np.max(cp):.4f} == Cp_max={m.Cp_max:.4f}")
    assert_true(abs(lam_peak - m.lambda_opt) < 2.0,
                f"peak at lambda={lam_peak:.2f} near lambda_opt={m.lambda_opt:.1f}")


def test_power_cubic_in_wind():
    print("\n[Test 3] Aero power scales as U^3 at fixed TSR")
    m, _ = make_model()
    # Hold lambda fixed: omega = lambda*U/R, so Cp identical, P ~ U^3.
    lam = m.lambda_opt
    U1, U2 = 6.0, 12.0  # factor 2 in wind -> factor 8 in power
    w1 = lam * U1 / m.R
    w2 = lam * U2 / m.R
    P1 = m.aero_power(w1, U1)
    P2 = m.aero_power(w2, U2)
    ratio = P2 / P1
    assert_true(abs(ratio - 8.0) < 0.05, f"P(2U)/P(U)={ratio:.3f} ~ 8 (U^3 law)")


def test_betz_power_bound():
    print("\n[Test 4] Extracted power below Betz-limited wind power")
    m, _ = make_model()
    U = 10.0
    P_wind = 0.5 * m.rho * m.A * U ** 3
    w = m.lambda_opt * U / m.R
    P = m.aero_power(w, U)
    assert_true(P <= BETZ_LIMIT * P_wind + 1e-6,
                f"P={P:.1f} W <= Betz*P_wind={BETZ_LIMIT*P_wind:.1f} W")


def test_spin_up_transient():
    print("\n[Test 5] Rotor spins up to the optimal-TSR operating point")
    m, _ = make_model()
    # Large inertia + the low-Cp self-start region make settling take ~120 s.
    r = m.simulate(8.0, omega0=1.0, dt=0.2, duration_s=160.0)
    assert_true(r["omega"][-1] > r["omega"][0],
                f"omega rose {r['omega'][0]:.2f} -> {r['omega'][-1]:.2f} rad/s")
    dw = abs(r["omega"][-1] - r["omega"][-2])
    assert_true(dw < 0.05, f"near steady state: |domega|={dw:.4f} rad/s/step")
    assert_true(r["omega"][-1] < 40.0, f"omega_final={r['omega'][-1]:.2f} bounded")
    # Optimal-torque controller settles the rotor near lambda_opt (peak Cp).
    assert_true(abs(r["tsr"][-1] - m.lambda_opt) < 1.0,
                f"settled TSR={r['tsr'][-1]:.2f} near lambda_opt={m.lambda_opt:.1f}")
    assert_true(r["Cp"][-1] > 0.30, f"operating Cp={r['Cp'][-1]:.3f} near peak")


def test_steady_state_torque_balance():
    print("\n[Test 6] Settled omega satisfies torque balance")
    m, _ = make_model()
    r = m.simulate(8.0, omega0=1.0, dt=0.2, duration_s=160.0)
    w = r["omega"][-1]
    net = m.aero_torque(w, 8.0) - m.gen_torque(w) - m.loss_torque(w)
    assert_true(abs(net) < 5.0, f"net torque at settle ={net:.3f} N.m ~ 0")
    w_ss = m.steady_state(8.0)
    assert_true(abs(w - w_ss) < 1.0,
                f"sim omega={w:.2f} matches root-find omega={w_ss:.2f}")


def test_energy_balance():
    print("\n[Test 7] Energy balance: dE_kin = (P_aero - P_gen - P_loss) dt")
    m, _ = make_model()
    r = m.simulate(9.0, omega0=2.0, dt=0.02, duration_s=20.0)
    t, w = r["t"], r["omega"]
    dE_kin = 0.5 * m.J * (w[-1] ** 2 - w[0] ** 2)
    P_gen_mech = r["T_gen"] * w
    P_loss = r["T_loss"] * w
    P_net = r["P_aero"] - P_gen_mech - P_loss
    int_net = np.trapezoid(P_net, t) if hasattr(np, "trapezoid") else np.trapz(P_net, t)
    rel = abs(dE_kin - int_net) / (abs(int_net) + 1.0)
    assert_true(rel < 0.05,
                f"dE_kin={dE_kin:.1f} J vs integ(P_net)={int_net:.1f} J (rel {rel:.3%})")


def test_furling_reduces_power():
    print("\n[Test 8] Furling sheds area above v_furl; parked above cut-out")
    m, _ = make_model()
    assert_true(abs(m.furl_factor(8.0) - 1.0) < 1e-9, "no furl below v_furl")
    f_mid = m.furl_factor(0.5 * (m.v_furl + m.v_cut_out))
    assert_true(0.0 < f_mid < 1.0, f"partial furl factor={f_mid:.3f} in (0,1)")
    assert_true(m.furl_factor(m.v_cut_out + 1.0) == 0.0, "fully furled above cut-out")
    assert_true(m.aero_power(20.0, m.v_cut_out + 2.0) == 0.0, "no power above cut-out")


def test_below_cut_in_and_gen_load():
    print("\n[Test 9] No aero torque below cut-in; gen torque grows with omega")
    m, _ = make_model()
    assert_true(m.aero_torque(10.0, m.v_cut_in - 0.5) == 0.0, "no torque below cut-in")
    assert_true(m.gen_torque(10.0) > m.gen_torque(5.0), "T_gen increases with omega")
    # power-limiting above rated speed
    T_hi = m.gen_torque(2.0 * m.omega_rated)
    P_hi = m.gen_power_elec(2.0 * m.omega_rated)
    assert_true(P_hi <= m.P_rated * 1.05, f"P_elec capped ~rated: {P_hi:.1f} W")


def test_higher_wind_higher_power():
    print("\n[Test 10] Settled electrical power monotone in wind, capped at rated")
    m, _ = make_model()
    # Use the torque-balance operating point directly (steady state).
    P_prev = -1.0
    for U in [6.0, 8.0, 10.0, 11.0]:
        w = m.steady_state(U)
        P = m.gen_power_elec(w)
        assert_true(P > P_prev, f"U={U}: P_elec={P:.1f} W > prev {P_prev:.1f} W")
        P_prev = P
    # Above rated: power-limited near P_rated, not runaway.
    w_hi = m.steady_state(12.0)
    P_hi = m.gen_power_elec(w_hi)
    assert_true(P_hi <= m.P_rated * 1.05,
                f"above-rated P_elec={P_hi:.1f} W capped near {m.P_rated:.0f} W")


def test_predict_interface():
    print("\n[Test 11] ComponentModel predict() interface + turbulence input")
    _, cm = make_model()
    r = cm.predict({"wind_speed_ms": 8.0, "dt": 0.1, "duration_s": 10.0})
    for key in ["t", "omega", "rpm", "tsr", "Cp", "P_aero", "P_elec",
                "T_aero", "T_gen", "T_loss", "efficiency", "wind_speed"]:
        assert_true(key in r, f"Key '{key}' in output")
    assert_true(len(r["t"]) == len(r["omega"]), "Arrays same length")
    rt = cm.predict({"wind_speed_ms": 9.0, "TI": 0.18, "dt": 0.1, "duration_s": 10.0})
    assert_true(np.std(rt["wind_speed"]) > 0.0, "turbulent wind series varies")


def test_benchmark():
    print("\n[Test 12] Benchmark: 60 s transient at dt=0.05")
    m, _ = make_model()
    t0 = time.perf_counter()
    m.simulate(8.0, omega0=1.0, dt=0.05, duration_s=60.0)
    elapsed = time.perf_counter() - t0
    print(f"  60 s simulation in {elapsed*1000:.1f} ms")
    assert_true(elapsed < 5.0, "Completes in < 5 s")


if __name__ == "__main__":
    tests = [
        test_cp_below_betz,
        test_cp_peak_at_lambda_opt,
        test_power_cubic_in_wind,
        test_betz_power_bound,
        test_spin_up_transient,
        test_steady_state_torque_balance,
        test_energy_balance,
        test_furling_reduces_power,
        test_below_cut_in_and_gen_load,
        test_higher_wind_higher_power,
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
    print(f"EC064 Small Wind F2a -- Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    sys.exit(0 if failed == 0 else 1)
