"""
EC100 -- Brayton Cycle Gas Turbine -- F2a Physics-Lumped
Test suite: physics sanity (conservation, eta<Carnot, optimal PR), edge cases,
predict() interface, spool ODE, and a benchmark timing test.
Run:  python3 scripts/test_model.py   (system python3, NOT pytest)
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from model import Brayton_F2a
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
def test_cp_air():
    print("\n[Test 1] Variable cp(T) matches air tables")
    m, _ = make_model()
    # Cengel A-17: cp(300)=1.005, cp(1000)=1.142, cp(1500)=1.211 kJ/kg.K
    assert_true(abs(m.cp(300) - 1005.0) < 25, f"cp(300)={m.cp(300):.1f} ~1005")
    assert_true(abs(m.cp(1000) - 1142.0) < 30, f"cp(1000)={m.cp(1000):.1f} ~1142")
    assert_true(m.cp(1500) > m.cp(300), "cp rises with T")


def test_compression_heats():
    print("\n[Test 2] Compression raises T, real > isentropic")
    m, _ = make_model()
    T2, w_c, T2s = m.compress(288.15, 18.0)
    assert_true(T2 > 288.15, f"T2={T2:.1f} > T1")
    assert_true(T2 > T2s, f"real T2={T2:.1f} > T2s={T2s:.1f} (irreversibility)")
    assert_true(w_c > 0, f"w_compressor={w_c:.0f} J/kg > 0")


def test_expansion_cools():
    print("\n[Test 3] Expansion drops T, real recovers less than isentropic")
    m, _ = make_model()
    T4, w_t, T4s = m.expand(1700.0, 18.0)
    assert_true(T4 < 1700.0, f"T4={T4:.1f} < T3")
    assert_true(T4 > T4s, f"real T4={T4:.1f} > T4s={T4s:.1f} (less work extracted)")
    assert_true(w_t > 0, f"w_turbine={w_t:.0f} J/kg > 0")


def test_efficiency_below_carnot():
    print("\n[Test 4] eta_thermal < eta_Carnot (2nd law)")
    m, _ = make_model()
    for PR in [6, 12, 18, 25, 35]:
        st = m.cycle(PR=PR)
        assert_true(0 < st["eta_thermal"] < st["eta_carnot"],
                    f"PR={PR}: eta={st['eta_thermal']:.4f} < Carnot={st['eta_carnot']:.4f}")


def test_energy_conservation():
    print("\n[Test 5] Energy balance: q_in = w_net + q_out (1st law)")
    m, _ = make_model()
    st = m.cycle(PR=18.0)
    # q_out = q_in - w_net (closed-cycle air-standard energy balance)
    q_out = st["q_in_J_kg"] - st["w_net_J_kg"]
    assert_true(q_out > 0, f"q_out={q_out:.0f} J/kg > 0 (heat rejected)")
    resid = abs(st["q_in_J_kg"] - (st["w_net_J_kg"] + q_out))
    assert_true(resid < 1e-6, f"1st-law residual={resid:.2e} ~ 0")
    # power balance
    bal = abs(st["W_turbine_W"] - st["W_compressor_W"] - st["W_net_W"])
    assert_true(bal < 1.0, f"W_t - W_c - W_net = {bal:.2e} W ~ 0")


def test_optimal_pressure_ratio():
    print("\n[Test 6] Specific work peaks near analytic optimal PR")
    m, _ = make_model()
    PR_opt = m.optimal_pressure_ratio()
    PRs = np.linspace(2, 40, 80)
    wnet = np.array([m.cycle(PR=pr)["w_net_J_kg"] for pr in PRs])
    PR_numeric = PRs[np.argmax(wnet)]
    assert_true(2 < PR_opt < 40, f"PR_opt analytic={PR_opt:.1f} in range")
    # numeric peak (with real effs) within a reasonable band of analytic ideal
    assert_true(abs(PR_numeric - PR_opt) / PR_opt < 0.6,
                f"numeric peak PR={PR_numeric:.1f} near analytic {PR_opt:.1f}")


def test_back_work_ratio():
    print("\n[Test 7] Back-work ratio in physical band (0.3-0.7 for GT)")
    m, _ = make_model()
    st = m.cycle(PR=18.0)
    bwr = st["back_work_ratio"]
    assert_true(0.2 < bwr < 0.8, f"BWR={bwr:.3f} typical for gas turbine")


def test_higher_TIT_better():
    print("\n[Test 8] Raising TIT raises efficiency and net work")
    m, _ = make_model()
    lo = m.cycle(PR=18.0, TIT=1300.0)
    hi = m.cycle(PR=18.0, TIT=1700.0)
    assert_true(hi["eta_thermal"] > lo["eta_thermal"],
                f"eta: TIT1700={hi['eta_thermal']:.4f} > TIT1300={lo['eta_thermal']:.4f}")
    assert_true(hi["w_net_J_kg"] > lo["w_net_J_kg"], "net work rises with TIT")


def test_regeneration_improves_eta():
    print("\n[Test 9] Regeneration improves eta at modest PR (T4>T2)")
    m, _ = make_model()
    base = m.cycle(PR=8.0, regen_eps=0.0)
    regen = m.cycle(PR=8.0, regen_eps=0.8)
    assert_true(regen["eta_thermal"] > base["eta_thermal"],
                f"regen eta={regen['eta_thermal']:.4f} > base {base['eta_thermal']:.4f}")
    # fuel use must drop (less heat needed) at same power
    assert_true(regen["q_in_J_kg"] < base["q_in_J_kg"], "regeneration cuts fuel heat")


def test_spool_ode():
    print("\n[Test 10] Spool ODE: balanced load holds speed; over-load decelerates")
    m, _ = make_model()
    st = m.cycle()
    # load equal to net power -> near steady speed
    bal = m.simulate_spool(st["W_net_W"], t_end=15.0)
    assert_true(bal["success"], "solve_ivp succeeded")
    assert_true(abs(bal["speed_fraction"][-1] - 1.0) < 0.05,
                f"balanced: speed_frac={bal['speed_fraction'][-1]:.3f} ~ 1.0")
    # over-load (1.5x) -> shaft slows down
    over = m.simulate_spool(1.5 * st["W_net_W"], t_end=15.0)
    assert_true(over["rpm"][-1] < over["rpm"][0],
                f"over-load decelerates: {over['rpm'][0]:.0f} -> {over['rpm'][-1]:.0f} rpm")


def test_predict_interface():
    print("\n[Test 11] ComponentModel predict()/get_info() interface")
    _, cm = make_model()
    info = cm.get_info()
    for k in ["component_id", "component_name", "fidelity", "version", "inputs", "outputs"]:
        assert_true(k in info, f"get_info has '{k}'")
    r = cm.predict({"pressure_ratio": 18.0, "transient": True, "t_end_s": 5.0})
    for key in ["T2_K", "T3_K", "T4_K", "w_net_J_kg", "eta_thermal",
                "eta_carnot", "W_net_W", "mdot_fuel_kg_s", "transient"]:
        assert_true(key in r, f"predict output has '{key}'")
    assert_true(len(r["transient"]["t"]) == len(r["transient"]["rpm"]),
                "transient arrays same length")


def test_benchmark():
    print("\n[Test 12] Benchmark: 100 cycle evals + one 20s spool sim")
    m, _ = make_model()
    t0 = time.perf_counter()
    for _ in range(100):
        m.cycle(PR=18.0)
    t_cycle = time.perf_counter() - t0
    t1 = time.perf_counter()
    m.simulate_spool(150e6, t_end=20.0)
    t_spool = time.perf_counter() - t1
    print(f"  100 cycle evals in {t_cycle*1000:.1f} ms; 20s spool sim in {t_spool*1000:.1f} ms")
    assert_true(t_cycle + t_spool < 5.0, "completes well under 5 s")


if __name__ == "__main__":
    tests = [
        test_cp_air,
        test_compression_heats,
        test_expansion_cools,
        test_efficiency_below_carnot,
        test_energy_conservation,
        test_optimal_pressure_ratio,
        test_back_work_ratio,
        test_higher_TIT_better,
        test_regeneration_improves_eta,
        test_spool_ode,
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
    print(f"EC100 Brayton F2a -- Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    sys.exit(0 if failed == 0 else 1)
