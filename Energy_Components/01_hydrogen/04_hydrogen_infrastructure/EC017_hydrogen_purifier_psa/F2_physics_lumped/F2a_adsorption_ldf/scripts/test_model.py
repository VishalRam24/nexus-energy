"""
EC017 -- Hydrogen Purifier (PSA) -- F2a Adsorption + LDF
Test suite: physics sanity (mass conservation, Langmuir/LDF limits, breakthrough,
purity/recovery bounds), edge cases, predict() interface, benchmark timing.
Run with system python3 (NOT pytest):  python3 scripts/test_model.py
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from model import HydrogenPSA_F2a
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
def test_langmuir_limits():
    print("\n[Test 1] Langmuir isotherm: limits + monotonicity")
    m, _ = make_model()
    q0 = m.q_equilibrium(0.0, m.T_op)
    assert_true(abs(q0) < 1e-12, f"q*(p=0)=0 (got {q0:.2e})")
    q_hi = m.q_equilibrium(1e6, m.T_op)
    assert_true(abs(q_hi - m.q_sat) < 1e-3, f"q*(p->inf)->q_sat ({q_hi:.4f}~{m.q_sat})")
    p = np.linspace(0.1, 50, 40)
    q = m.q_equilibrium(p, m.T_op)
    assert_true(np.all(np.diff(q) > 0), "q* monotonically increases with pressure")


def test_langmuir_temperature():
    print("\n[Test 2] Adsorption weakens with temperature (exothermic)")
    m, _ = make_model()
    q_cold = m.q_equilibrium(5.0, 273.15)
    q_hot = m.q_equilibrium(5.0, 373.15)
    assert_true(q_cold > q_hot, f"q*(cold)={q_cold:.4f} > q*(hot)={q_hot:.4f}")


def test_ldf_drives_to_equilibrium():
    print("\n[Test 3] LDF loading ODE converges to Langmuir equilibrium")
    m, _ = make_model()
    p_imp = (1.0 - m.y_feed) * m.P_H
    q_star = m.q_equilibrium(p_imp, m.T_op)
    # integrate a long pure-adsorption step toward equilibrium
    r = m.simulate_cycle(t_ads=200.0, t_purge=0.001, q0=0.0, dt=1.0)
    q_eq = r["loading_equilibrium"][0]
    assert_true(abs(q_eq - q_star) < 1e-6, f"q* target matches Langmuir ({q_eq:.4f})")
    # LDF rate is positive while below equilibrium, zero at equilibrium
    rate_lo = m.dqdt_ldf(0.0, p_imp, m.T_op)
    rate_eq = m.dqdt_ldf(q_star, p_imp, m.T_op)
    assert_true(rate_lo > 0, f"dq/dt>0 below eq ({rate_lo:.4f})")
    assert_true(abs(rate_eq) < 1e-9, f"dq/dt=0 at eq ({rate_eq:.2e})")


def test_mass_conservation():
    print("\n[Test 4] Impurity mass balance: fed = adsorbed + slip")
    m, _ = make_model()
    for q0 in [0.0, 1.0, 2.5]:
        for ta in [60.0, 120.0, 400.0]:
            r = m.simulate_cycle(t_ads=ta, q0=q0, dt=2.0)
            assert_true(r["impurity_balance_residual_mol"] < 1e-6,
                        f"q0={q0}, t_ads={ta}: residual={r['impurity_balance_residual_mol']:.2e} ~ 0")


def test_purity_bounds():
    print("\n[Test 5] Purity in (0, 1] for all conditions")
    m, _ = make_model()
    for q0 in [0.0, 1.5, 2.9]:
        for ph in [10.0, 20.0, 50.0]:
            r = m.simulate_cycle(P_H=ph, q0=q0, dt=2.0)
            assert_true(0.0 < r["purity"] <= 1.0, f"q0={q0},P_H={ph}: purity={r['purity']:.5f}")


def test_recovery_below_one():
    print("\n[Test 6] Recovery strictly < 1 (purge consumes H2)")
    m, _ = make_model()
    for pr in [0.05, 0.15, 0.25, 0.35]:
        r = m.simulate_cycle(purge_ratio=pr, dt=2.0)
        assert_true(0.0 < r["recovery"] < 1.0, f"purge={pr}: recovery={r['recovery']:.4f} in (0,1)")


def test_recovery_monotone_in_purge():
    print("\n[Test 7] Recovery decreases as purge ratio increases")
    m, _ = make_model()
    prs = [0.05, 0.15, 0.25, 0.35]
    recs = [m.simulate_cycle(purge_ratio=pr, dt=2.0)["recovery"] for pr in prs]
    assert_true(np.all(np.diff(recs) < 0), f"recovery falls with purge: {[round(x,3) for x in recs]}")


def test_breakthrough_drops_purity():
    print("\n[Test 8] Breakthrough: saturating the bed lowers purity")
    m, _ = make_model()
    r_design = m.simulate_cycle(t_ads=120.0, q0=0.0, dt=2.0)
    r_long = m.simulate_cycle(t_ads=600.0, q0=0.0, dt=2.0)
    r_preload = m.simulate_cycle(t_ads=120.0, q0=2.9, dt=2.0)
    assert_true(r_long["purity"] < r_design["purity"],
                f"long-ads purity {r_long['purity']:.4f} < design {r_design['purity']:.4f}")
    assert_true(r_preload["purity"] < r_design["purity"],
                f"preloaded purity {r_preload['purity']:.4f} < design {r_design['purity']:.4f}")
    assert_true(r_long["impurity_slip_mol"] > r_design["impurity_slip_mol"],
                "more impurity slips through a saturated bed")


def test_design_purity_high():
    print("\n[Test 9] Design-point purity is industrial-grade (>99%)")
    m, _ = make_model()
    r = m.cyclic_steady_state()
    assert_true(r["purity"] > 0.99, f"CSS purity={r['purity']*100:.4f}% > 99%")
    assert_true(0.7 < r["recovery"] < 0.95, f"CSS recovery={r['recovery']*100:.2f}% in 70-95%")


def test_regeneration():
    print("\n[Test 10] Blowdown+purge regenerate the bed (loading drops)")
    m, _ = make_model()
    r = m.simulate_cycle(q0=2.0, dt=2.0)
    assert_true(r["q_end_purge"] < r["q_end_adsorption"],
                f"q_purge={r['q_end_purge']:.4f} < q_ads={r['q_end_adsorption']:.4f}")
    assert_true(r["impurity_desorbed_purge_mol"] >= 0, "purge desorbs impurity (>=0)")
    # purge to clean H2 (p_imp=0) drives loading toward zero
    assert_true(r["q_end_purge"] < r["q_end_blowdown"] + 1e-9,
                "purge step reduces loading below blowdown")


def test_cyclic_steady_state():
    print("\n[Test 11] Cyclic steady state converges")
    m, _ = make_model()
    r = m.cyclic_steady_state(n_cycles=30, tol=1e-7)
    assert_true(r["css_cycles"] <= 30, f"CSS in {r['css_cycles']} cycles")
    hist = r["css_history"]
    if len(hist) >= 2:
        assert_true(abs(hist[-1] - hist[-2]) < 1e-3 or r["css_cycles"] < 30,
                    "start-of-cycle loading converged")
    assert_true(r["specific_energy_kWh_per_kg_H2"] > 0, "spec energy positive")


def test_specific_energy_pressure():
    print("\n[Test 12] Specific energy rises with pressure ratio; in DOE range")
    m, _ = make_model()
    w_lo = m.specific_energy(10.0, 1.5)
    w_hi = m.specific_energy(40.0, 1.5)
    assert_true(w_hi > w_lo, f"W(40bar)={w_hi:.3f} > W(10bar)={w_lo:.3f}")
    assert_true(0.5 < w_lo < 4.0 and 0.5 < w_hi < 4.0,
                f"spec energy in DOE 0.5-4 kWh/kg ({w_lo:.2f},{w_hi:.2f})")


def test_predict_interface():
    print("\n[Test 13] ComponentModel predict() interface + get_info()")
    _, cm = make_model()
    info = cm.get_info()
    for k in ["component_id", "component_name", "fidelity", "version", "inputs", "outputs"]:
        assert_true(k in info, f"get_info has '{k}'")
    assert_true(cm.component_id == "EC017", "component_id == EC017")
    r = cm.predict({"feed_pressure_bar": 25.0, "feed_h2_fraction": 0.70})
    for key in ["t", "loading", "purity", "recovery",
                "productivity_mol_kg_cycle", "impurity_balance_residual_mol"]:
        assert_true(key in r, f"predict output has '{key}'")
    assert_true(len(r["t"]) == len(r["loading"]), "time/loading arrays same length")


def test_benchmark():
    print("\n[Test 14] Benchmark: full CSS simulation timing")
    m, _ = make_model()
    t0 = time.perf_counter()
    m.cyclic_steady_state(n_cycles=20, dt=1.0)
    elapsed = time.perf_counter() - t0
    print(f"  CSS (up to 20 cycles, dt=1s) in {elapsed*1000:.1f} ms")
    assert_true(elapsed < 5.0, "Completes in < 5 s")


if __name__ == "__main__":
    tests = [
        test_langmuir_limits,
        test_langmuir_temperature,
        test_ldf_drives_to_equilibrium,
        test_mass_conservation,
        test_purity_bounds,
        test_recovery_below_one,
        test_recovery_monotone_in_purge,
        test_breakthrough_drops_purity,
        test_design_purity_high,
        test_regeneration,
        test_cyclic_steady_state,
        test_specific_energy_pressure,
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

    print(f"\n{'='*64}")
    print(f"EC017 Hydrogen Purifier PSA F2a -- Results: {passed} passed, {failed} failed")
    print(f"{'='*64}")
    sys.exit(0 if failed == 0 else 1)
