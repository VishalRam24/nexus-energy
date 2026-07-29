"""
EC216 -- Thermoelectric Generator (TEG) -- F2a Coupled Thermal-Electrical -- Test suite.
"""
import sys, os, time
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from model import TEG_CoupledF2a
from predict import ComponentModel

PASS, FAIL = "\u2713", "\u2717"


def assert_true(cond, msg):
    if cond:
        print(f"  {PASS}  {msg}")
    else:
        print(f"  {FAIL}  FAILED: {msg}")
        raise AssertionError(msg)


def make():
    cm = ComponentModel()
    return cm._model, cm


def test_efficiency_below_carnot():
    """Efficiency must be strictly below Carnot limit."""
    print("\n[Test 1] Efficiency <= Carnot")
    m, _ = make()
    for T_hot in [350.0, 400.0, 473.0, 550.0]:
        T_cold = 300.0
        R_L = m.matched_load_resistance(T_hot, T_cold)
        res = m.solve_steady_state(T_hot, T_cold, R_L)
        eta_carnot = 1.0 - T_cold / T_hot
        assert_true(
            res["efficiency"] < eta_carnot,
            f"T_hot={T_hot}K: eta={res['efficiency']:.4f} < Carnot={eta_carnot:.4f}"
        )
        assert_true(res["efficiency"] > 0, f"  eta > 0 for dT={T_hot - T_cold}K")


def test_max_power_matched_load():
    """Power at matched load should approximate V_oc^2 / (4*R_int)."""
    print("\n[Test 2] Power at matched load ~ V_oc^2 / (4*R_int)")
    m, _ = make()
    T_hot, T_cold = 473.0, 300.0
    R_L = m.matched_load_resistance(T_hot, T_cold)
    res = m.solve_steady_state(T_hot, T_cold, R_L)
    P_est = res["V_oc"]**2 / (4.0 * res["R_int"])
    ratio = res["P"] / P_est if P_est > 0 else 0
    assert_true(
        0.7 < ratio < 1.3,
        f"P={res['P']:.3f} W, P_est={P_est:.3f} W, ratio={ratio:.3f}"
    )


def test_zero_power_equal_temps():
    """Zero power when T_hot = T_cold."""
    print("\n[Test 3] Zero power when T_hot = T_cold")
    m, _ = make()
    res = m.solve_steady_state(300.0, 300.0, 1.0)
    assert_true(abs(res["P"]) < 1e-10, f"P={res['P']:.2e} ~ 0")
    assert_true(abs(res["V"]) < 1e-10, f"V={res['V']:.2e} ~ 0")
    assert_true(abs(res["I"]) < 1e-10, f"I={res['I']:.2e} ~ 0")


def test_monotonic_power_vs_delta_t():
    """Higher delta-T must yield higher power (monotonic increase)."""
    print("\n[Test 4] Monotonic: higher dT -> higher power")
    m, _ = make()
    T_cold = 300.0
    dTs = [20, 50, 100, 150, 200]
    powers = []
    for dT in dTs:
        T_hot = T_cold + dT
        R_L = m.matched_load_resistance(T_hot, T_cold)
        res = m.solve_steady_state(T_hot, T_cold, R_L)
        powers.append(res["P"])

    for i in range(1, len(powers)):
        assert_true(
            powers[i] > powers[i - 1],
            f"P(dT={dTs[i]})={powers[i]:.3f} > P(dT={dTs[i-1]})={powers[i-1]:.3f}"
        )


def test_energy_conservation():
    """Electrical power must equal Q_h - Q_c (energy conservation)."""
    print("\n[Test 5] Energy conservation: P_elec = Q_h - Q_c")
    m, _ = make()
    for T_hot in [373.0, 473.0, 573.0]:
        T_cold = 300.0
        R_L = m.matched_load_resistance(T_hot, T_cold)
        res = m.solve_steady_state(T_hot, T_cold, R_L)
        P_balance = res["Q_h"] - res["Q_c"]
        err = abs(res["P"] - P_balance)
        assert_true(
            err < 0.1 * max(res["P"], 1e-6) + 1e-6,
            f"T_hot={T_hot}K: P={res['P']:.4f}, Q_h-Q_c={P_balance:.4f}, err={err:.2e}"
        )


def test_junction_temps_between_source_sink():
    """Junction temps must be between source and sink temperatures."""
    print("\n[Test 6] Junction temps between source and sink")
    m, _ = make()
    T_hot, T_cold = 500.0, 290.0
    R_L = m.matched_load_resistance(T_hot, T_cold)
    res = m.solve_steady_state(T_hot, T_cold, R_L)
    T_hj = res["T_h_junction"]
    T_cj = res["T_c_junction"]
    assert_true(
        T_cj <= T_hj,
        f"T_c_junction={T_cj:.2f} <= T_h_junction={T_hj:.2f}"
    )
    assert_true(
        T_hj <= T_hot,
        f"T_h_junction={T_hj:.2f} <= T_source={T_hot:.2f}"
    )
    assert_true(
        T_cj >= T_cold,
        f"T_c_junction={T_cj:.2f} >= T_sink={T_cold:.2f}"
    )


def test_predict_interface():
    """ComponentModel predict() returns expected keys."""
    print("\n[Test 7] ComponentModel predict() interface")
    _, cm = make()
    r = cm.predict({"T_hot_K": 450.0, "T_cold_K": 300.0})
    expected_keys = ["P", "V", "I", "efficiency", "Q_h", "Q_c", "V_oc",
                     "R_int", "T_h_junction", "T_c_junction", "eta_carnot",
                     "converged", "iterations"]
    for k in expected_keys:
        assert_true(k in r, f"Key '{k}' present")

    info = cm.get_info()
    assert_true(info["component_id"] == "EC216", "component_id = EC216")
    assert_true("F2a" in info["fidelity"], "fidelity contains F2a")


def test_benchmark():
    """Benchmark: single solve_steady_state call."""
    print("\n[Test 8] Benchmark")
    m, _ = make()
    R_L = m.matched_load_resistance(473.0, 300.0)
    # Warm up
    m.solve_steady_state(473.0, 300.0, R_L)

    n_runs = 100
    t0 = time.perf_counter()
    for _ in range(n_runs):
        m.solve_steady_state(473.0, 300.0, R_L)
    elapsed = (time.perf_counter() - t0) / n_runs
    print(f"  Single solve in {elapsed * 1000:.2f} ms")
    assert_true(elapsed < 1.0, "< 1 s per solve")


def test_load_sweep():
    """Power peaks near matched load resistance."""
    print("\n[Test 9] Power peaks near matched load")
    m, _ = make()
    T_hot, T_cold = 473.0, 300.0
    R_match = m.matched_load_resistance(T_hot, T_cold)
    iv = m.iv_curve(T_hot, T_cold, N_points=50)
    idx_max = np.argmax(iv["P"])
    R_at_max = iv["R_load"][idx_max]
    ratio = R_at_max / R_match
    assert_true(
        0.3 < ratio < 3.0,
        f"R_at_Pmax={R_at_max:.3f}, R_match={R_match:.3f}, ratio={ratio:.2f}"
    )


def test_convergence():
    """Solver should converge for typical operating conditions."""
    print("\n[Test 10] Solver convergence")
    m, _ = make()
    for T_hot in [320.0, 400.0, 500.0, 580.0]:
        R_L = m.matched_load_resistance(T_hot, 300.0)
        res = m.solve_steady_state(T_hot, 300.0, R_L)
        assert_true(
            res["converged"],
            f"Converged at T_hot={T_hot}K in {res['iterations']} iterations"
        )


if __name__ == "__main__":
    tests = [
        test_efficiency_below_carnot,
        test_max_power_matched_load,
        test_zero_power_equal_temps,
        test_monotonic_power_vs_delta_t,
        test_energy_conservation,
        test_junction_temps_between_source_sink,
        test_predict_interface,
        test_benchmark,
        test_load_sweep,
        test_convergence,
    ]
    p = f = 0
    for t in tests:
        try:
            t()
            p += 1
        except Exception as e:
            f += 1
            print(f"  ERROR: {e}")
    print(f"\n{'=' * 60}")
    print(f"EC216 TEG F2a -- {p} passed, {f} failed")
    print(f"{'=' * 60}")
    sys.exit(0 if f == 0 else 1)
