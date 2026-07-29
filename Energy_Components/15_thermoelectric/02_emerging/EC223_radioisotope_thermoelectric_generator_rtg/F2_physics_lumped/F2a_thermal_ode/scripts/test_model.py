"""
EC223 -- Radioisotope Thermoelectric Generator (RTG) -- F2a Physics-Lumped
Test suite: physics sanity (decay law, conservation, efficiency bounds,
power decline, radiator T^4), edge cases, predict() interface, benchmark.
Run:  python3 scripts/test_model.py   (system python3, NOT pytest)
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from model import RTG_F2a, SIGMA, LN2
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
def test_decay_exponential():
    print("\n[Test 1] Decay heat follows exponential Q0*exp(-ln2*t/t_half)")
    m, _ = make_model()
    # Heat must halve after exactly one half-life.
    q0 = m.decay_heat(0.0)
    q_half = m.decay_heat(m.t_half)
    assert_true(abs(q0 - m.Q0) < 1e-6, f"Q(0)={q0:.1f} == Q0={m.Q0:.1f}")
    assert_true(abs(q_half / q0 - 0.5) < 1e-9,
                f"Q(t_half)/Q0={q_half/q0:.6f} == 0.5")
    # Strictly decreasing
    ts = np.linspace(0, 200, 50)
    q = m.decay_heat(ts)
    assert_true(np.all(np.diff(q) < 0), "Decay heat strictly decreasing")


def test_energy_conservation_couple():
    print("\n[Test 2] Couple energy balance: Q_h - Q_c == P_e (exact)")
    m, _ = make_model()
    for Th in [700.0, 925.0, 1100.0]:
        cs = m.couple_state(Th, m.T_cold)
        bal = cs["Q_h_couple"] - cs["Q_c_couple"] - cs["P_e_couple"]
        assert_true(abs(bal) < 1e-9, f"Th={Th}: Qh-Qc-Pe={bal:.2e} ~ 0")


def test_efficiency_below_carnot():
    print("\n[Test 3] Module efficiency < Carnot and <= ZT-limit")
    m, _ = make_model()
    for Th in [650.0, 925.0, 1200.0]:
        cs = m.couple_state(Th, m.T_cold)
        etac = m.eta_carnot(Th, m.T_cold)
        etazt = m.eta_zt_max(Th, m.T_cold)
        assert_true(0 < cs["eta_module"] < etac,
                    f"Th={Th}: eta={cs['eta_module']*100:.2f}% < Carnot {etac*100:.1f}%")
        assert_true(cs["eta_module"] <= etazt + 1e-9,
                    f"Th={Th}: eta <= eta_zt_max {etazt*100:.2f}%")


def test_efficiency_in_unit_interval():
    print("\n[Test 4] All efficiencies in (0,1) over mission")
    m, _ = make_model()
    r = m.simulate(60.0, 60)
    for eta in r["eta_module"]:
        assert_true(0 < eta < 1.0, f"eta_module={eta:.4f} in (0,1)")
    for eta in r["eta_carnot"]:
        assert_true(0 < eta < 1.0, f"eta_carnot={eta:.4f} in (0,1)")


def test_power_declines():
    print("\n[Test 5] Electrical power declines monotonically with isotope decay")
    m, _ = make_model()
    r = m.simulate(50.0, 100)
    P = r["P_electric_W"]
    assert_true(np.all(np.diff(P) <= 1e-6), "P_electric monotonically non-increasing")
    assert_true(P[-1] < P[0], f"P(50yr)={P[-1]:.1f} < P(0)={P[0]:.1f} W")
    assert_true(P[0] > 0, f"BOL power positive: {P[0]:.1f} W")


def test_hot_temp_declines():
    print("\n[Test 6] Hot-side temperature falls as decay heat drops")
    m, _ = make_model()
    r = m.simulate(50.0, 100)
    T = r["T_hot_K"]
    assert_true(T[-1] < T[0], f"T_hot(50yr)={T[-1]:.0f} < T_hot(0)={T[0]:.0f} K")
    assert_true(np.all(T > m.T_cold), "T_hot stays above cold side throughout")


def test_radiator_t4():
    print("\n[Test 7] Radiator rejection scales as T^4 (Stefan-Boltzmann)")
    m, _ = make_model()
    q1 = m.Q_radiator(573.0)
    q2 = m.Q_radiator(2.0 * 573.0)
    # Doubling T (>> T_space) should multiply rejection by ~16
    ratio = q2 / q1
    assert_true(abs(ratio - 16.0) < 0.5, f"Q_rad(2T)/Q_rad(T)={ratio:.2f} ~ 16")
    # Sanity: matches eps*sigma*A formula
    expect = m.eps * SIGMA * m.A_rad * (573.0**4 - m.T_space**4)
    assert_true(abs(q1 - expect) < 1e-6, "Q_radiator matches eps*sigma*A*(T^4-Ts^4)")


def test_cold_side_balance():
    print("\n[Test 8] Cold-side energy balance: radiator sized to reject Q_c at T_cold")
    m, _ = make_model()
    Th0 = m.T_hot_0
    Q_c, Q_rad, res = m.radiator_balance(Th0)
    rel = abs(res) / Q_c
    assert_true(rel < 0.05, f"|Q_c-Q_rad|/Q_c={rel*100:.2f}% < 5% (Q_c={Q_c:.0f}, Q_rad={Q_rad:.0f})")


def test_matched_load_maximises_power():
    print("\n[Test 9] Matched load (R_L=R) maximises electrical power")
    m, _ = make_model()
    Th = 925.0
    P_matched = m.couple_state(Th, m.T_cold, R_load=m.R)["P_e_couple"]
    for rl in [0.5 * m.R, 2.0 * m.R, 5.0 * m.R]:
        P = m.couple_state(Th, m.T_cold, R_load=rl)["P_e_couple"]
        assert_true(P <= P_matched + 1e-12,
                    f"P(R_L={rl:.4f})={P:.4f} <= P_matched={P_matched:.4f}")


def test_global_energy_balance():
    print("\n[Test 10] Global balance at BOL steady state: Q_decay ~ Q_te + Q_par")
    m, _ = make_model()
    Th = m.steady_T_hot(0.0)
    cs = m.couple_state(Th, m.T_cold)
    Q_in = m.decay_heat(0.0)
    Q_out = cs["Q_h_total"] + m.K_hp * (Th - m.T_cold)
    rel = abs(Q_in - Q_out) / Q_in
    assert_true(rel < 1e-3, f"Steady balance residual {rel*100:.3f}% (Qin={Q_in:.0f}, Qout={Q_out:.0f})")


def test_predict_interface():
    print("\n[Test 11] ComponentModel predict() interface")
    _, cm = make_model()
    r = cm.predict({"mission_years": 30.0, "n_points": 25})
    for key in ["t_years", "T_hot_K", "Q_decay_W", "P_electric_W",
                "eta_module", "eta_carnot", "eta_zt_max", "current_A",
                "Q_radiator_W", "power_fraction"]:
        assert_true(key in r, f"Key '{key}' in output")
    assert_true(len(r["t_years"]) == len(r["P_electric_W"]), "Arrays same length")
    assert_true(abs(r["power_fraction"][0] - 1.0) < 1e-9, "power_fraction[0] == 1.0")
    info = cm.get_info()
    assert_true(info["component_id"] == "EC223" and info["version"] == "1.0.0",
                "get_info id/version correct")


def test_benchmark():
    print("\n[Test 12] Benchmark: 50yr / 200-point ODE simulation")
    m, _ = make_model()
    t0 = time.perf_counter()
    m.simulate(50.0, 200)
    elapsed = time.perf_counter() - t0
    print(f"  50yr simulation in {elapsed*1000:.1f} ms")
    assert_true(elapsed < 5.0, "Completes in < 5 s")


if __name__ == "__main__":
    tests = [
        test_decay_exponential,
        test_energy_conservation_couple,
        test_efficiency_below_carnot,
        test_efficiency_in_unit_interval,
        test_power_declines,
        test_hot_temp_declines,
        test_radiator_t4,
        test_cold_side_balance,
        test_matched_load_maximises_power,
        test_global_energy_balance,
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
    print(f"EC223 RTG F2a -- Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    sys.exit(0 if failed == 0 else 1)
