"""
EC156 -- Geothermal Heat Pump (GHP) -- F2a Vapor-Cycle + Borehole ODE
Test suite: thermodynamic sanity, energy balance, ground-source stability,
ODE convergence, edge cases, predict() interface, benchmark timing.
Run: python3 scripts/test_model.py   (system python3, NOT pytest)
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from model import GHP_F2a
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
def test_r410a_saturation():
    print("\n[Test 1] R-410A saturation pressure matches NIST")
    m, _ = make_model()
    # NIST REFPROP bubble-point: 0 degC ~ 798 kPa, 40 degC ~ 2419 kPa
    p0 = m.p_sat(0.0) / 1e3
    p40 = m.p_sat(40.0) / 1e3
    assert_true(abs(p0 - 798.0) / 798.0 < 0.03, f"Psat(0C)={p0:.0f} kPa ~ 798 (NIST)")
    assert_true(abs(p40 - 2419.0) / 2419.0 < 0.03, f"Psat(40C)={p40:.0f} kPa ~ 2419 (NIST)")
    assert_true(m.p_sat(50.0) > m.p_sat(10.0), "Psat monotonically increases with T")


def test_cop_above_one():
    print("\n[Test 2] Heating COP > 1 across operating envelope")
    m, _ = make_model()
    for Tl in [0, 5, 8, 12, 18]:
        for Ts in [30, 40, 45, 55]:
            op = m.cycle(Tl, Ts)
            assert_true(op["COP"] > 1.0, f"COP({Tl}C loop,{Ts}C supply)={op['COP']:.2f} > 1")


def test_cop_below_carnot():
    print("\n[Test 3] COP < Carnot ceiling (2nd law)")
    m, _ = make_model()
    for Tl in [0, 5, 8, 12, 18]:
        for Ts in [30, 45, 55]:
            op = m.cycle(Tl, Ts)
            assert_true(op["COP"] < op["COP_carnot"],
                        f"COP={op['COP']:.2f} < Carnot={op['COP_carnot']:.2f} ({Tl},{Ts})")


def test_energy_balance():
    print("\n[Test 4] Cycle energy balance: Q_cond = Q_evap + W_comp")
    m, _ = make_model()
    for Tl, Ts in [(8, 45), (2, 50), (15, 35)]:
        op = m.cycle(Tl, Ts)
        resid = abs(op["Q_cond"] - (op["Q_evap"] + op["W_comp"]))
        assert_true(resid < 1e-6 * op["Q_cond"],
                    f"|Qc-(Qe+Wc)|={resid:.2e} W ~ 0 ({Tl},{Ts})")


def test_rated_capacity():
    print("\n[Test 5] Rated condenser duty near nameplate ~10 kW")
    m, _ = make_model()
    op = m.cycle(8.0, 45.0)
    q = op["Q_cond"] / 1e3
    assert_true(8.0 < q < 12.0, f"Q_cond(rated)={q:.2f} kW in [8,12]")


def test_cop_improves_with_warmer_ground():
    print("\n[Test 6] COP increases with warmer source / lower lift")
    m, _ = make_model()
    cop_cold = m.cycle(2.0, 45.0)["COP"]
    cop_warm = m.cycle(15.0, 45.0)["COP"]
    assert_true(cop_warm > cop_cold,
                f"COP(15C ground)={cop_warm:.2f} > COP(2C)={cop_cold:.2f}")
    cop_lowsink = m.cycle(8.0, 30.0)["COP"]
    cop_hisink = m.cycle(8.0, 55.0)["COP"]
    assert_true(cop_lowsink > cop_hisink,
                f"COP(30C supply)={cop_lowsink:.2f} > COP(55C)={cop_hisink:.2f}")


def test_ground_source_depletion():
    print("\n[Test 7] Transient: loop cools as ground source is drawn down")
    m, _ = make_model()
    r = m.simulate(T_supply_c=45, Q_demand_kW=8, dt=600, duration_s=5 * 86400)
    assert_true(r["success"], "solve_ivp converged")
    assert_true(r["T_loop"][-1] < r["T_loop"][0],
                f"T_loop {r['T_loop'][0]:.2f}->{r['T_loop'][-1]:.2f} (depletion)")
    assert_true(r["COP"][-1] < r["COP"][0],
                f"COP {r['COP'][0]:.2f}->{r['COP'][-1]:.2f} (drops as source cools)")


def test_ground_stays_stable():
    print("\n[Test 8] Ground source remains stable & physical (no runaway)")
    m, _ = make_model()
    r = m.simulate(T_supply_c=45, Q_demand_kW=8, dt=600, duration_s=10 * 86400)
    assert_true(np.all(r["T_ground"] > -5.0), "Ground node stays above freezing-ish")
    assert_true(np.all(r["T_ground"] <= m.T_ground0 + 1e-6),
                "Ground never exceeds undisturbed T in heating draw")
    assert_true(np.all(r["T_loop"] > -10.0) and np.all(r["T_loop"] < m.T_ground0 + 1.0),
                "Loop temperature bounded below undisturbed ground")


def test_cop_all_bounds_transient():
    print("\n[Test 9] Transient COP obeys 1 < COP < Carnot at every step")
    m, _ = make_model()
    r = m.simulate(T_supply_c=50, Q_demand_kW=9, dt=600, duration_s=7 * 86400)
    assert_true(np.all(r["COP"] > 1.0), "All transient COP > 1")
    assert_true(np.all(r["COP"] < r["COP_carnot"]), "All transient COP < Carnot")


def test_volumetric_efficiency():
    print("\n[Test 10] Volumetric efficiency falls with pressure ratio")
    m, _ = make_model()
    op_low = m.cycle(15.0, 30.0)   # low lift -> low PR
    op_high = m.cycle(0.0, 55.0)   # high lift -> high PR
    assert_true(op_high["PR"] > op_low["PR"], f"PR {op_high['PR']:.2f} > {op_low['PR']:.2f}")
    assert_true(op_high["eta_vol"] < op_low["eta_vol"],
                f"eta_vol {op_high['eta_vol']:.3f} < {op_low['eta_vol']:.3f} at high PR")
    assert_true(0.3 <= op_high["eta_vol"] <= 1.0, "eta_vol bounded in [0.3,1]")


def test_predict_interface():
    print("\n[Test 11] ComponentModel predict() / get_info() interface")
    _, cm = make_model()
    info = cm.get_info()
    for k in ["component_id", "component_name", "fidelity", "version", "inputs", "outputs"]:
        assert_true(k in info, f"get_info has '{k}'")
    assert_true(info["component_id"] == "EC156", "component_id == EC156")
    r = cm.predict({"T_supply_c": 45, "Q_demand_kW": 8, "dt": 600, "duration_s": 86400})
    for key in ["t", "T_loop", "T_ground", "COP", "COP_carnot",
                "Q_cond_kW", "Q_evap_kW", "W_elec_kW"]:
        assert_true(key in r, f"Output key '{key}' present")
    assert_true(len(r["t"]) == len(r["COP"]), "Output arrays same length")


def test_benchmark():
    print("\n[Test 12] Benchmark: 30-day sim at 10-min step")
    m, _ = make_model()
    t0 = time.perf_counter()
    m.simulate(T_supply_c=45, Q_demand_kW=8, dt=600, duration_s=30 * 86400)
    elapsed = time.perf_counter() - t0
    print(f"  30-day simulation in {elapsed*1000:.1f} ms")
    assert_true(elapsed < 5.0, "Completes in < 5 s")


if __name__ == "__main__":
    tests = [
        test_r410a_saturation,
        test_cop_above_one,
        test_cop_below_carnot,
        test_energy_balance,
        test_rated_capacity,
        test_cop_improves_with_warmer_ground,
        test_ground_source_depletion,
        test_ground_stays_stable,
        test_cop_all_bounds_transient,
        test_volumetric_efficiency,
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
    print(f"EC156 GHP F2a (vapor-cycle + borehole) -- Results: "
          f"{passed} passed, {failed} failed")
    print(f"{'='*64}")
    sys.exit(0 if failed == 0 else 1)
