"""
EC138 -- Ocean Thermal Energy Conversion (OTEC) -- F2a Physics-Lumped Thermal Cycle
Test suite: physics sanity (Carnot bound, energy conservation, parasitic), edge
cases (low dT -> no net power), predict() interface, ODE convergence, benchmark.
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from model import OTEC_F2a
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
def test_carnot_small():
    print("\n[Test 1] Carnot efficiency is small for OTEC dT (~20 K)")
    m, _ = make_model()
    eta_c = float(m.carnot_efficiency(26.0, 5.0))
    assert_true(0.0 < eta_c < 0.10, f"eta_Carnot={eta_c*100:.2f}% in (0, 10%)")
    assert_true(0.03 < eta_c < 0.08, f"eta_Carnot={eta_c*100:.2f}% in typical 3-7% band")


def test_cycle_below_carnot():
    print("\n[Test 2] Real cycle efficiency strictly below reservoir Carnot")
    m, _ = make_model()
    eta_c = float(m.carnot_efficiency(26.0, 5.0))
    # WF sat temps are inside the gap (pinch) -> even smaller
    eta_cyc = float(m.cycle_efficiency(26.0 - 2.5, 5.0 + 2.5))
    assert_true(0.0 < eta_cyc < eta_c,
                f"eta_cycle={eta_cyc*100:.2f}% < eta_Carnot={eta_c*100:.2f}%")


def test_net_below_cycle():
    print("\n[Test 3] Net efficiency < cycle efficiency (parasitic penalty)")
    m, _ = make_model()
    ss = m.steady_state(26.0, 5.0)
    assert_true(ss["eta_net"] < ss["eta_cycle"],
                f"eta_net={ss['eta_net']*100:.2f}% < eta_cycle={ss['eta_cycle']*100:.2f}%")
    assert_true(0.0 < ss["eta_net"] < 0.05,
                f"eta_net={ss['eta_net']*100:.2f}% in realistic 0-5% OTEC band")


def test_energy_conservation():
    print("\n[Test 4] Energy conservation: Q_evap = P_gross + Q_cond")
    m, _ = make_model()
    ss = m.steady_state(26.0, 5.0)
    resid = ss["Q_evap_kw"] - (ss["P_gross_kw"] + ss["Q_cond_kw"])
    assert_true(abs(resid) < 1e-6 * max(ss["Q_evap_kw"], 1.0),
                f"|Q_evap - (P_gross + Q_cond)| = {abs(resid):.3e} kW ~ 0")
    assert_true(ss["Q_evap_kw"] > ss["Q_cond_kw"],
                f"Q_evap={ss['Q_evap_kw']:.0f} > Q_cond={ss['Q_cond_kw']:.0f} kW")


def test_parasitic_dominant():
    print("\n[Test 5] Seawater pumping is the dominant parasitic load")
    m, _ = make_model()
    ss = m.steady_state(26.0, 5.0)
    sw = ss["P_warm_pump_kw"] + ss["P_cold_pump_kw"]
    assert_true(sw > ss["P_wf_pump_kw"],
                f"seawater pumps {sw:.1f} kW > wf pump {ss['P_wf_pump_kw']:.1f} kW")
    frac = ss["P_parasitic_kw"] / ss["P_gross_kw"]
    assert_true(0.10 < frac < 0.60,
                f"parasitic = {frac*100:.1f}% of gross (typical 20-40%)")


def test_no_net_power_low_dT():
    print("\n[Test 6] Insufficient dT -> net power non-positive")
    m, _ = make_model()
    # Warm 9 C, cold 5 C: dT only 4 K -> after pinch the cycle barely works,
    # parasitic pumping must exceed gross power.
    ss = m.steady_state(9.0, 5.0)
    assert_true(ss["P_net_kw"] <= 0.0,
                f"P_net={ss['P_net_kw']:.1f} kW <= 0 at dT=4 K")
    # And full design dT gives positive net power
    ss2 = m.steady_state(26.0, 5.0)
    assert_true(ss2["P_net_kw"] > 0.0,
                f"P_net={ss2['P_net_kw']:.1f} kW > 0 at dT=21 K")


def test_net_power_increases_with_dT():
    print("\n[Test 7] Net power increases with warm-water temperature")
    m, _ = make_model()
    p_prev = -1e9
    for Tw in [16.0, 20.0, 24.0, 28.0]:
        p = m.steady_state(Tw, 5.0)["P_net_kw"]
        assert_true(p >= p_prev - 1e-6, f"P_net({Tw}C)={p:.1f} kW >= prev {p_prev:.1f}")
        p_prev = p


def test_ode_steady_state():
    print("\n[Test 8] Lumped HX ODE settles to steady state")
    m, _ = make_model()
    r = m.simulate(26.0, 5.0, dt=30.0, duration_s=7200.0)
    assert_true(r["success"], "solve_ivp reported success")
    dTe = abs(r["T_evap_c"][-1] - r["T_evap_c"][-2])
    dTc = abs(r["T_cond_c"][-1] - r["T_cond_c"][-2])
    assert_true(dTe < 0.05 and dTc < 0.05,
                f"near SS: dT_evap={dTe:.4f}, dT_cond={dTc:.4f} K per step")
    # WF sat temps must lie inside the reservoir gap
    assert_true(5.0 < r["T_cond_c"][-1] < r["T_evap_c"][-1] < 26.0,
                f"T_cold < T_cond({r['T_cond_c'][-1]:.2f}) < "
                f"T_evap({r['T_evap_c'][-1]:.2f}) < T_warm")


def test_ode_transient_step():
    print("\n[Test 9] Step up in warm-water T raises net power after lag")
    m, _ = make_model()
    def Tw_step(t):
        return 22.0 if t < 1800.0 else 28.0
    r = m.simulate(Tw_step, 5.0, dt=30.0, duration_s=3600.0)
    i_before = np.argmin(np.abs(r["t"] - 1700.0))
    i_after = np.argmin(np.abs(r["t"] - 3500.0))
    assert_true(r["P_net_kw"][i_after] > r["P_net_kw"][i_before],
                f"P_net rose {r['P_net_kw'][i_before]:.0f} -> "
                f"{r['P_net_kw'][i_after]:.0f} kW after warm-T step")


def test_efficiency_bounds_timeseries():
    print("\n[Test 10] All time-series efficiencies stay within physical bounds")
    m, _ = make_model()
    r = m.simulate(26.0, 5.0, dt=60.0, duration_s=3600.0)
    assert_true(np.all(r["eta_carnot"] > 0) and np.all(r["eta_carnot"] < 0.10),
                "eta_carnot in (0, 0.10) for all t")
    assert_true(np.all(r["eta_cycle"] >= 0) and np.all(r["eta_cycle"] < r["eta_carnot"] + 1e-9),
                "eta_cycle in [0, eta_carnot) for all t")


def test_predict_interface():
    print("\n[Test 11] ComponentModel predict() interface")
    _, cm = make_model()
    r = cm.predict({"T_warm_in_c": 26.0, "T_cold_in_c": 5.0,
                    "dt": 60.0, "duration_s": 600.0})
    for key in ["t", "T_evap_c", "T_cond_c", "eta_carnot", "eta_cycle",
                "eta_net", "P_gross_kw", "P_net_kw", "P_parasitic_kw",
                "steady_state"]:
        assert_true(key in r, f"Key '{key}' in output")
    assert_true(len(r["t"]) == len(r["P_net_kw"]), "Arrays same length")
    info = cm.get_info()
    assert_true(info["component_id"] == "EC138", "get_info id = EC138")


def test_benchmark():
    print("\n[Test 12] Benchmark: 1 h sim at dt=10 s")
    m, _ = make_model()
    t0 = time.perf_counter()
    m.simulate(26.0, 5.0, dt=10.0, duration_s=3600.0)
    elapsed = time.perf_counter() - t0
    print(f"  3600 s simulation in {elapsed*1000:.1f} ms")
    assert_true(elapsed < 5.0, "Completes in < 5 s")


if __name__ == "__main__":
    tests = [
        test_carnot_small,
        test_cycle_below_carnot,
        test_net_below_cycle,
        test_energy_conservation,
        test_parasitic_dominant,
        test_no_net_power_low_dT,
        test_net_power_increases_with_dT,
        test_ode_steady_state,
        test_ode_transient_step,
        test_efficiency_bounds_timeseries,
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
    print(f"EC138 OTEC F2a -- Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    sys.exit(0 if failed == 0 else 1)
