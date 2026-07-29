"""
EC058 -- Flat Plate Solar Collector -- F2a Dynamic Thermal -- Test suite.
"""
import sys, os, time
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from model import FlatPlateCollectorF2a
from predict import ComponentModel

PASS, FAIL = "\u2713", "\u2717"


def assert_true(c, m):
    if c:
        print(f"  {PASS}  {m}")
    else:
        print(f"  {FAIL}  FAILED: {m}")
        raise AssertionError(m)


def make():
    cm = ComponentModel()
    return cm._model, cm


# ---- Test 1: Steady-state matches Hottel-Whillier ----
def test_steady_state_hwb():
    print("\n[Test 1] Steady-state matches Hottel-Whillier formula")
    m, _ = make()
    G = 800.0
    T_in = 40.0
    T_amb = 25.0

    r = m.steady_state(G, T_in, T_amb, theta_deg=0.0)

    # Manual HWB: Q_u = A * F_R * [tau_alpha * G - U_L * (T_in - T_amb)]
    # At steady state with small a2, U_L ~ a1
    Q_expected_approx = m.A_c * m.F_R * (m.tau_alpha_n * G - m.a1 * (T_in - T_amb))
    Q_expected_approx = max(0.0, Q_expected_approx)

    # Should be close (not exact due to temperature-dependent U_L iteration)
    rel_err = abs(r["Q_useful_W"] - Q_expected_approx) / max(Q_expected_approx, 1.0)
    assert_true(rel_err < 0.05,
                f"Q_useful={r['Q_useful_W']:.1f} W ~ HWB approx {Q_expected_approx:.1f} W "
                f"(rel err {rel_err:.3f})")
    assert_true(0.3 < r["efficiency"] < 0.8,
                f"eta={r['efficiency']:.3f} in reasonable range")


# ---- Test 2: Dynamic cold start heats up ----
def test_dynamic_cold_start():
    print("\n[Test 2] Dynamic: collector heats up from cold start")
    m, _ = make()
    t_end = 3600.0  # 1 hour
    t_eval = np.linspace(0, t_end, 200)

    G_func = lambda t: 800.0
    T_in_func = lambda t: 20.0
    T_amb_func = lambda t: 20.0

    res = m.simulate(
        t_span=(0, t_end), t_eval=t_eval, T_m0=20.0,
        G_func=G_func, T_in_func=T_in_func, T_amb_func=T_amb_func,
    )

    assert_true(res["solver_success"], "ODE solver converged")
    assert_true(res["T_mean_C"][-1] > res["T_mean_C"][0],
                f"T_mean rose from {res['T_mean_C'][0]:.1f} to {res['T_mean_C'][-1]:.1f} C")
    assert_true(res["T_outlet_C"][-1] > 20.0,
                f"T_out={res['T_outlet_C'][-1]:.1f} C > T_in=20 C")


# ---- Test 3: Zero irradiance cooldown ----
def test_zero_irradiance_cooldown():
    print("\n[Test 3] Zero irradiance: thermal inertia then cooldown")
    m, _ = make()
    t_end = 7200.0  # 2 hours
    t_eval = np.linspace(0, t_end, 300)

    T_amb = 10.0
    T_m0 = 60.0  # Start hot

    res = m.simulate(
        t_span=(0, t_end), t_eval=t_eval, T_m0=T_m0,
        G_func=lambda t: 0.0,
        T_in_func=lambda t: T_amb,
        T_amb_func=lambda t: T_amb,
    )

    assert_true(res["solver_success"], "ODE solver converged")
    # Should cool down but still be above ambient initially
    assert_true(res["T_mean_C"][0] > T_amb,
                f"T_mean starts at {res['T_mean_C'][0]:.1f} > T_amb={T_amb}")
    assert_true(res["T_mean_C"][-1] < T_m0,
                f"T_mean dropped from {T_m0:.1f} to {res['T_mean_C'][-1]:.1f} C")
    # After 2 hours with no sun, should approach ambient
    assert_true(res["T_mean_C"][-1] < T_m0 * 0.5,
                f"Significant cooldown occurred")


# ---- Test 4: Higher flow rate -> lower T_out but higher Q_useful ----
def test_flow_rate_effect():
    print("\n[Test 4] Higher flow rate -> lower T_out but higher Q_useful")
    m, _ = make()
    G = 800.0
    T_in = 40.0
    T_amb = 25.0

    r_low = m.steady_state(G, T_in, T_amb)

    # Create a model with higher flow rate
    import json
    params_path = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")
    with open(params_path) as f:
        params_hi = json.load(f)
    params_hi["unit"]["m_dot_spec"]["value"] = 0.04  # double flow rate
    m_hi = FlatPlateCollectorF2a(params_hi)

    r_hi = m_hi.steady_state(G, T_in, T_amb)

    assert_true(r_hi["T_outlet_C"] < r_low["T_outlet_C"],
                f"T_out(hi_flow)={r_hi['T_outlet_C']:.1f} < T_out(lo_flow)={r_low['T_outlet_C']:.1f}")
    assert_true(r_hi["Q_useful_W"] >= r_low["Q_useful_W"] * 0.99,
                f"Q(hi_flow)={r_hi['Q_useful_W']:.1f} >= Q(lo_flow)={r_low['Q_useful_W']:.1f}")


# ---- Test 5: Efficiency decreases with higher (T_m - T_amb) ----
def test_efficiency_vs_temperature():
    print("\n[Test 5] Efficiency decreases with higher (T_m - T_amb)")
    m, _ = make()
    G = 800.0
    T_amb = 20.0

    etas = []
    T_ins = [25.0, 50.0, 75.0, 95.0]
    for T_in in T_ins:
        r = m.steady_state(G, T_in, T_amb)
        etas.append(r["efficiency"])

    for i in range(len(etas) - 1):
        assert_true(etas[i] >= etas[i + 1],
                    f"eta(T_in={T_ins[i]})={etas[i]:.3f} >= eta(T_in={T_ins[i+1]})={etas[i+1]:.3f}")


# ---- Test 6: Energy conservation in steady state ----
def test_energy_conservation_steady():
    print("\n[Test 6] Energy conservation in steady state (Q_solar = Q_loss + Q_useful)")
    m, _ = make()
    G = 800.0
    T_in = 50.0
    T_amb = 20.0

    r = m.steady_state(G, T_in, T_amb)
    T_m = r["T_mean_C"]
    dT = T_m - T_amb

    Q_solar = m.A_c * m.F_R * m.tau_alpha_n * G
    Q_loss = m.A_c * (m.a1 * dT + m.a2 * dT ** 2)
    Q_useful = r["Q_useful_W"]

    # In steady state: Q_solar = Q_loss + Q_useful (approximately, since our model
    # separates F_R into optical and loss terms)
    balance = Q_solar - Q_loss - Q_useful
    # The balance should be close to zero relative to Q_solar
    # (not exact due to F_R being in the solar term but losses using T_m)
    rel_balance = abs(balance) / Q_solar if Q_solar > 1.0 else abs(balance)
    assert_true(rel_balance < 0.15,
                f"Energy balance residual: {balance:.1f} W "
                f"(rel={rel_balance:.3f}, Q_solar={Q_solar:.1f}, "
                f"Q_loss={Q_loss:.1f}, Q_u={Q_useful:.1f})")


# ---- Test 7: Predict interface ----
def test_predict_interface():
    print("\n[Test 7] ComponentModel predict() interface")
    _, cm = make()
    r = cm.predict({
        "irradiance_W_m2": 800.0,
        "T_inlet_C": 40.0,
        "T_ambient_C": 25.0,
        "incidence_angle_deg": 30.0,
    })
    for k in ["Q_useful_W", "T_outlet_C", "T_mean_C", "efficiency", "iam_factor"]:
        assert_true(k in r, f"Key '{k}' present")
    assert_true(r["iam_factor"] < 1.0, f"IAM at 30 deg = {r['iam_factor']:.3f} < 1.0")
    assert_true(r["Q_useful_W"] > 0, f"Q_useful={r['Q_useful_W']:.1f} > 0")


# ---- Test 8: Benchmark ----
def test_benchmark():
    print("\n[Test 8] Benchmark: dynamic 1-hour simulation")
    m, _ = make()
    t_eval = np.linspace(0, 3600, 100)

    t0 = time.perf_counter()
    for _ in range(10):
        m.simulate(
            t_span=(0, 3600), t_eval=t_eval, T_m0=20.0,
            G_func=lambda t: 800.0,
            T_in_func=lambda t: 30.0,
            T_amb_func=lambda t: 25.0,
        )
    elapsed = (time.perf_counter() - t0) / 10
    print(f"  1-hour simulation in {elapsed * 1000:.1f} ms")
    assert_true(elapsed < 10.0, "< 10 s per simulation")


if __name__ == "__main__":
    tests = [
        test_steady_state_hwb,
        test_dynamic_cold_start,
        test_zero_irradiance_cooldown,
        test_flow_rate_effect,
        test_efficiency_vs_temperature,
        test_energy_conservation_steady,
        test_predict_interface,
        test_benchmark,
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
    print(f"EC058 Flat Plate Collector F2a -- {p} passed, {f} failed")
    print(f"{'=' * 60}")
    sys.exit(0 if f == 0 else 1)
