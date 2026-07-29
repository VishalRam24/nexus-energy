"""
EC097 -- Rankine Cycle (Steam Turbine) -- F2a Physics-Lumped Thermo Cycle
Test suite: thermodynamic sanity, energy/entropy bounds, ODE convergence.
Custom harness (no pytest). Run: python3 scripts/test_model.py
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from model import RankineCycleF2a
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
def test_efficiency_below_carnot():
    print("\n[Test 1] Second law: eta_thermal < eta_carnot")
    m, _ = make_model()
    for kw in [{}, {"reheat": True}, {"regeneration": True}]:
        r = m.solve_cycle(**kw)
        assert_true(r["eta_thermal"] < r["eta_carnot"],
                    f"{kw or 'base'}: eta={r['eta_thermal']:.4f} < "
                    f"Carnot={r['eta_carnot']:.4f}")


def test_energy_conservation():
    print("\n[Test 2] First law: q_boiler = w_net + q_cond")
    m, _ = make_model()
    # Plain and reheat cycles close exactly (single through-flow).
    for kw in [{}, {"reheat": True}]:
        r = m.solve_cycle(**kw)
        lhs = r["q_boiler"]
        rhs = r["w_net"] + r["q_cond"]
        resid = abs(lhs - rhs)
        assert_true(resid < 1e-6 * max(1.0, abs(lhs)),
                    f"{kw or 'base'}: |q_boiler - (w_net+q_cond)| = {resid:.2e} kJ/kg")


def test_efficiency_realistic_band():
    print("\n[Test 3] eta_thermal in realistic Rankine band 0.30-0.45")
    m, _ = make_model()
    for kw in [{}, {"reheat": True}, {"regeneration": True}]:
        r = m.solve_cycle(**kw)
        assert_true(0.30 <= r["eta_thermal"] <= 0.45,
                    f"{kw or 'base'}: eta={r['eta_thermal']:.4f} in [0.30, 0.45]")


def test_turbine_gt_pump_work():
    print("\n[Test 4] Turbine work >> pump work (back-work ratio small)")
    m, _ = make_model()
    r = m.solve_cycle()
    assert_true(r["w_turbine"] > r["w_pump"],
                f"w_turbine={r['w_turbine']:.1f} > w_pump={r['w_pump']:.2f} kJ/kg")
    bwr = r["w_pump"] / r["w_turbine"]
    assert_true(bwr < 0.05,
                f"back-work ratio={bwr:.4f} < 0.05 (steam plant, not gas turbine)")


def test_net_work_positive():
    print("\n[Test 5] Net work and powers positive")
    m, _ = make_model()
    r = m.solve_cycle()
    assert_true(r["w_net"] > 0, f"w_net={r['w_net']:.1f} kJ/kg > 0")
    assert_true(r["P_elec_W"] > 0 and r["P_elec_W"] < r["Q_in_W"],
                f"0 < P_elec={r['P_elec_W']/1e6:.1f} MW < Q_in={r['Q_in_W']/1e6:.1f} MW")


def test_carnot_increases_with_superheat():
    print("\n[Test 6] Higher superheat T raises Carnot bound and eta")
    m, _ = make_model()
    r_lo = m.solve_cycle(T_superheat=400.0)
    r_hi = m.solve_cycle(T_superheat=600.0)
    assert_true(r_hi["eta_carnot"] > r_lo["eta_carnot"],
                f"Carnot: {r_hi['eta_carnot']:.4f} > {r_lo['eta_carnot']:.4f}")
    assert_true(r_hi["eta_thermal"] > r_lo["eta_thermal"],
                f"eta_th: {r_hi['eta_thermal']:.4f} > {r_lo['eta_thermal']:.4f}")


def test_lower_condenser_raises_eta():
    print("\n[Test 7] Lower condenser pressure raises efficiency")
    m, _ = make_model()
    r_hi_p = m.solve_cycle(P_condenser=0.30)
    r_lo_p = m.solve_cycle(P_condenser=0.06)
    assert_true(r_lo_p["eta_thermal"] > r_hi_p["eta_thermal"],
                f"eta(0.06bar)={r_lo_p['eta_thermal']:.4f} > "
                f"eta(0.30bar)={r_hi_p['eta_thermal']:.4f}")


def test_turbine_exit_quality():
    print("\n[Test 8] Turbine exit quality physical (0 < x <= 1)")
    m, _ = make_model()
    for kw in [{}, {"reheat": True}]:
        r = m.solve_cycle(**kw)
        x = r["x_turbine_exit"]
        assert_true(0.0 < x <= 1.0, f"{kw or 'base'}: x_exit={x:.3f} in (0,1]")
    # Reheat should raise exit quality (drier steam) vs plain cycle.
    assert_true(m.solve_cycle(reheat=True)["x_turbine_exit"]
                > m.solve_cycle()["x_turbine_exit"],
                "reheat raises turbine exit quality")


def test_transient_ode_convergence():
    print("\n[Test 9] Boiler-drum ODE converges to steady state")
    m, _ = make_model()
    r = m.simulate(m.Q_fuel_design, dt=10.0, duration_s=6000.0)
    assert_true(len(r["t"]) > 1, "transient produced a time series")
    dT = abs(r["T_drum"][-1] - r["T_drum"][-2])
    assert_true(dT < 0.5, f"near steady state: dT={dT:.4f} K between last steps")
    # Drum approaches Q_fuel/UA above saturation.
    T_ss_expect = r["T_sat_boiler_K"] + m.Q_fuel_design / m.UA_boiler
    err = abs(r["T_drum"][-1] - T_ss_expect)
    assert_true(err < 2.0,
                f"final T_drum {r['T_drum'][-1]:.1f} K ~ analytic SS "
                f"{T_ss_expect:.1f} K (err {err:.2f})")


def test_transient_step_response():
    print("\n[Test 10] Fuel step up raises drum temperature and eta")
    m, _ = make_model()

    def step_fuel(t):
        return 0.6 * m.Q_fuel_design if t < 2000 else m.Q_fuel_design

    r = m.simulate(step_fuel, dt=20.0, duration_s=6000.0)
    i_before = np.argmin(np.abs(r["t"] - 1900.0))
    i_after = np.argmin(np.abs(r["t"] - 5900.0))
    assert_true(r["T_drum"][i_after] > r["T_drum"][i_before],
                "drum temperature rises after fuel step up")
    assert_true(r["eta_thermal"][i_after] >= r["eta_thermal"][i_before] - 1e-6,
                "efficiency does not fall after fuel step up")


def test_predict_interface():
    print("\n[Test 11] ComponentModel predict()/get_info() interface")
    _, cm = make_model()
    info = cm.get_info()
    assert_true(info["component_id"] == "EC097", "component_id == EC097")
    assert_true(cm.version == "1.0.0", "version == 1.0.0")
    r = cm.predict({"reheat": True})
    for key in ["eta_thermal", "eta_carnot", "w_net", "P_elec_W", "q_boiler"]:
        assert_true(key in r, f"steady key '{key}' in output")
    rt = cm.predict({"mode": "transient", "duration_s": 1000.0, "dt": 50.0})
    assert_true("transient" in rt, "transient mode returns 'transient' block")
    tr = rt["transient"]
    assert_true(len(tr["t"]) == len(tr["eta_thermal"]),
                "transient arrays same length")


def test_benchmark():
    print("\n[Test 12] Benchmark: 100 steady solves + 3000 s transient")
    m, _ = make_model()
    t0 = time.perf_counter()
    for _ in range(100):
        m.solve_cycle()
    t_steady = time.perf_counter() - t0
    t0 = time.perf_counter()
    m.simulate(m.Q_fuel_design, dt=10.0, duration_s=3000.0)
    t_tr = time.perf_counter() - t0
    print(f"  100 steady solves: {t_steady*1000:.1f} ms; "
          f"3000 s transient: {t_tr*1000:.1f} ms")
    assert_true(t_steady < 10.0 and t_tr < 60.0, "completes within time budget")


if __name__ == "__main__":
    tests = [
        test_efficiency_below_carnot,
        test_energy_conservation,
        test_efficiency_realistic_band,
        test_turbine_gt_pump_work,
        test_net_work_positive,
        test_carnot_increases_with_superheat,
        test_lower_condenser_raises_eta,
        test_turbine_exit_quality,
        test_transient_ode_convergence,
        test_transient_step_response,
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
    print(f"EC097 Rankine F2a -- Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    sys.exit(0 if failed == 0 else 1)
