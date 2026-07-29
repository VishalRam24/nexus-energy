"""
EC154 -- Enhanced Geothermal System (EGS) -- F2a Physics-Lumped
Test suite: energy conservation, thermal drawdown monotonicity, Carnot bound,
NTU effectiveness, edge cases, predict() interface, benchmark timing.
Run with system python3 (NOT pytest):  python3 scripts/test_model.py
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from model import EGS_F2a
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
def test_effectiveness_range():
    print("\n[Test 1] Fracture HX effectiveness in (0, 1)")
    m, _ = make_model()
    eps = m.effectiveness()
    assert_true(0.0 < eps < 1.0, f"eps={eps:.4f} in (0,1)")
    # Lower flow -> longer residence -> higher effectiveness
    eps_low = m.effectiveness(10.0)
    eps_high = m.effectiveness(150.0)
    assert_true(eps_low > eps_high,
                f"eps(10 kg/s)={eps_low:.4f} > eps(150 kg/s)={eps_high:.4f}")


def test_reservoir_cools_monotonically():
    print("\n[Test 2] Reservoir temperature declines monotonically (thermal drawdown)")
    m, _ = make_model()
    r = m.simulate(years=30.0, n_points=200)
    T = r["T_rock_degC"]
    diffs = np.diff(T)
    assert_true(np.all(diffs <= 1e-9), "T_rock non-increasing at every step")
    assert_true(T[-1] < T[0] - 1.0,
                f"Net cooling: T_rock {T[0]:.1f}C -> {T[-1]:.1f}C over 30 yr")


def test_drawdown_toward_injection():
    print("\n[Test 3] Rock asymptotes toward injection temperature, never below")
    m, _ = make_model()
    r = m.simulate(years=200.0, n_points=100)
    T = r["T_rock_degC"]
    assert_true(T[-1] > m.T_inject - 1e-6,
                f"T_rock_end={T[-1]:.2f}C >= T_inject={m.T_inject:.1f}C")
    assert_true(T[-1] < r["T_rock_degC"][0],
                "Long-horizon temperature well below initial")


def test_energy_conservation():
    print("\n[Test 4] Energy conservation: rock dU == integral of Q_extract")
    m, _ = make_model()
    r = m.simulate(years=40.0, n_points=400)
    err = r["energy_balance_err"]
    assert_true(err < 1e-3, f"relative energy-balance residual={err:.2e} < 1e-3")


def test_efficiency_below_carnot():
    print("\n[Test 5] Cycle efficiency strictly below Carnot bound, in (0,1)")
    m, _ = make_model()
    r = m.simulate(years=30.0, n_points=200)
    ec, eta = r["eta_carnot"], r["eta_cycle"]
    assert_true(np.all(eta < ec + 1e-12), "eta_cycle <= eta_carnot everywhere")
    assert_true(np.all(eta < ec * 0.999 + 1e-9),
                "eta_cycle strictly below Carnot (eta_util < 1)")
    assert_true(np.all((eta > 0.0) & (eta < 1.0)), "0 < eta_cycle < 1")
    assert_true(np.all((ec > 0.0) & (ec < 1.0)), "0 < eta_carnot < 1")


def test_produced_temp_bounded():
    print("\n[Test 6] Produced fluid temperature within [T_inj, T_rock]")
    m, _ = make_model()
    r = m.simulate(years=30.0, n_points=200)
    Tp, Tr = r["T_prod_degC"], r["T_rock_degC"]
    assert_true(np.all(Tp >= m.T_inject - 1e-9), "T_prod >= T_inject")
    assert_true(np.all(Tp <= Tr + 1e-9), "T_prod <= T_rock")


def test_power_declines():
    print("\n[Test 7] Net power declines as reservoir cools")
    m, _ = make_model()
    r = m.simulate(years=30.0, n_points=200)
    P = r["P_net_kW"]
    assert_true(P[0] > 0.0, f"P_net(0)={P[0]:.0f} kW > 0")
    assert_true(P[-1] < P[0],
                f"P_net declines: {P[0]:.0f} kW -> {P[-1]:.0f} kW")
    assert_true(np.all(np.diff(P) <= 1e-6), "P_net non-increasing over life")


def test_net_below_gross():
    print("\n[Test 8] Net power below gross (pump parasitic positive)")
    m, _ = make_model()
    r = m.simulate(years=10.0, n_points=100)
    Pg, Pn, Pp = r["P_gross_kW"], r["P_net_kW"], r["P_pump_kW"]
    assert_true(np.all(Pn <= Pg + 1e-9), "P_net <= P_gross")
    assert_true(np.all(Pp >= -1e-9), "P_pump >= 0")
    nz = Pg > 1.0
    assert_true(np.all(Pn[nz] < Pg[nz]), "Net strictly below gross when producing")


def test_flow_sensitivity():
    print("\n[Test 9] Higher flow -> faster drawdown (shorter tau_res)")
    m, _ = make_model()
    tau_low = m.reservoir_time_constant_yr(20.0)
    tau_high = m.reservoir_time_constant_yr(120.0)
    assert_true(tau_low > tau_high,
                f"tau(20 kg/s)={tau_low:.1f} yr > tau(120 kg/s)={tau_high:.1f} yr")
    assert_true(tau_low > 0.0, "tau_res positive")


def test_hotter_rock_more_power():
    print("\n[Test 10] Hotter initial reservoir -> more initial net power")
    m, _ = make_model()
    r_cool = m.simulate(years=1.0, n_points=10, T_geo_init_degC=160.0)
    r_hot = m.simulate(years=1.0, n_points=10, T_geo_init_degC=280.0)
    assert_true(r_hot["P_net_kW"][0] > r_cool["P_net_kW"][0],
                f"P_net(280C)={r_hot['P_net_kW'][0]:.0f} > "
                f"P_net(160C)={r_cool['P_net_kW'][0]:.0f} kW")


def test_predict_interface():
    print("\n[Test 11] ComponentModel predict() interface")
    _, cm = make_model()
    r = cm.predict({"years": 5.0, "n_points": 50})
    for key in ["t_years", "T_rock_degC", "T_prod_degC", "P_net_kW",
                "eta_cycle", "eta_carnot", "tau_res_yr", "energy_balance_err"]:
        assert_true(key in r, f"Key '{key}' in output")
    assert_true(len(r["t_years"]) == len(r["P_net_kW"]) == 50,
                "Arrays consistent length")
    info = cm.get_info()
    assert_true(info["component_id"] == "EC154" and info["version"] == "1.0.0",
                "get_info metadata correct")


def test_benchmark():
    print("\n[Test 12] Benchmark: 30-yr sim, 200 points")
    m, _ = make_model()
    t0 = time.perf_counter()
    m.simulate(years=30.0, n_points=200)
    elapsed = time.perf_counter() - t0
    print(f"  30-yr simulation in {elapsed*1000:.1f} ms")
    assert_true(elapsed < 5.0, "Completes in < 5 s")


if __name__ == "__main__":
    tests = [
        test_effectiveness_range,
        test_reservoir_cools_monotonically,
        test_drawdown_toward_injection,
        test_energy_conservation,
        test_efficiency_below_carnot,
        test_produced_temp_bounded,
        test_power_declines,
        test_net_below_gross,
        test_flow_sensitivity,
        test_hotter_rock_more_power,
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
    print(f"EC154 EGS F2a -- Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    sys.exit(0 if failed == 0 else 1)
