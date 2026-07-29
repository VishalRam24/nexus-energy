"""
EC214 -- Mechanical Vapor Compression (MVC) -- F2a Physics-Lumped
Test suite: physics sanity (conservation, heat-pump, BPE), edge cases,
predict() interface, benchmark timing. Custom harness (NO pytest).
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from model import MVC_F2a
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
def test_compressor_lifts_above_brine():
    print("\n[Test 1] Compressor lifts vapor sat. temp above boiling brine")
    m, _ = make_model()
    Tb, Tv, Ts = m.temperatures()
    assert_true(Tv < Tb, f"T_vapor={Tv:.3f} < T_brine={Tb:.3f} (BPE depresses vapor sat)")
    assert_true(Ts > Tb, f"T_steam={Ts:.3f} > T_brine={Tb:.3f} (compressor lifts it ABOVE brine)")
    pr = m.compressor_work()[2]
    assert_true(pr > 1.0, f"pressure ratio={pr:.4f} > 1 (compression)")


def test_bpe_physical():
    print("\n[Test 2] Boiling-point elevation positive and salinity-monotone")
    m, _ = make_model()
    bpe = m.bpe()
    assert_true(0.2 < bpe < 1.5, f"BPE={bpe:.3f} K in (0.2,1.5) for seawater")
    m.sal_ppm = 70000.0
    bpe_high = m.bpe()
    m.sal_ppm = 35000.0
    bpe_low = m.bpe()
    assert_true(bpe_high > bpe_low, f"BPE rises with salinity: {bpe_high:.3f} > {bpe_low:.3f}")


def test_sec_realistic():
    print("\n[Test 3] Specific electric energy in realistic MVC band 7-12 kWh/m3")
    m, _ = make_model()
    sec = m.specific_energy()
    assert_true(7.0 <= sec <= 12.0, f"SEC={sec:.2f} kWh/m3 in [7,12] (Veza 1995; El-Dessouky 2002)")


def test_heat_pump_amplification():
    print("\n[Test 4] Heat-pump effect: latent heat reused >> compressor work")
    m, _ = make_model()
    w_act = m.compressor_work()[0]
    hfg = m.hfg(m.T_brine_C)
    gor = m.gor()
    assert_true(hfg > w_act, f"hfg={hfg:.0f} kJ/kg >> w={w_act:.1f} kJ/kg")
    assert_true(gor > 1.0, f"GOR_equiv={gor:.1f} > 1 (genuine heat pump, no external steam)")


def test_sec_rises_with_lift():
    print("\n[Test 5] SEC increases with compressor temperature lift")
    m, _ = make_model()
    sec_low = m.specific_energy(dT_lift=2.5)
    sec_high = m.specific_energy(dT_lift=5.0)
    assert_true(sec_high > sec_low, f"SEC(5K)={sec_high:.2f} > SEC(2.5K)={sec_low:.2f}")


def test_design_point_consistency():
    print("\n[Test 6] Design point: Q = m_dist*hfg and P = m_dist*w (energy balance)")
    m, _ = make_model()
    d = m.design_point()
    Q_check = d["m_dist_kg_s"] * m.hfg(m.T_brine_C)
    assert_true(abs(Q_check - d["Q_evap_kW"]) / d["Q_evap_kW"] < 1e-6,
                f"Q_evap={d['Q_evap_kW']:.1f} = m_dist*hfg={Q_check:.1f}")
    w = m.compressor_work()[0]
    P_check = d["m_dist_kg_s"] * w
    assert_true(abs(P_check - d["P_elec_kW"]) / d["P_elec_kW"] < 1e-6,
                f"P_elec={d['P_elec_kW']:.2f} = m_dist*w={P_check:.2f}")


def test_transient_heats_to_boiling():
    print("\n[Test 7] Transient ODE: cold brine heats up toward boiling setpoint")
    m, _ = make_model()
    r = m.simulate(T0_brine_C=45.0, duration_s=3000.0, dt=10.0)
    assert_true(r["success"], "solve_ivp succeeded")
    assert_true(r["T_brine_C"][-1] > r["T_brine_C"][0] + 5.0,
                f"T_brine rose {r['T_brine_C'][0]:.1f}->{r['T_brine_C'][-1]:.1f} C")
    assert_true(r["T_brine_C"][-1] <= m.T_brine_C + 0.5,
                f"T_brine clamps at boiling pt {r['T_brine_C'][-1]:.2f} <= {m.T_brine_C+0.5:.1f} C")


def test_transient_reaches_design_production():
    print("\n[Test 8] Steady distillate approaches design capacity")
    m, _ = make_model()
    r = m.simulate(T0_brine_C=55.0, duration_s=4000.0, dt=10.0)
    dist_final = r["distillate_m3_day"][-1]
    assert_true(abs(dist_final - m.cap_m3_day) / m.cap_m3_day < 0.05,
                f"distillate {dist_final:.1f} ~ design {m.cap_m3_day:.0f} m3/day (within 5%)")


def test_no_boiling_below_saturation():
    print("\n[Test 9] No distillate produced while brine is sub-cooled")
    m, _ = make_model()
    r = m.simulate(T0_brine_C=40.0, duration_s=100.0, dt=5.0)
    # very start: brine well below boiling -> distillate must be ~0
    assert_true(r["distillate_m3_day"][0] < 1e-6,
                f"distillate at t=0 ={r['distillate_m3_day'][0]:.3e} ~ 0 (sub-cooled)")


def test_level_mass_balance():
    print("\n[Test 10] Sump level stays bounded (mass conservation)")
    m, _ = make_model()
    r = m.simulate(T0_brine_C=55.0, level0_m=1.0, duration_s=3000.0, dt=10.0)
    L = r["level_m"]
    assert_true(np.all(np.isfinite(L)), "level finite")
    assert_true(np.all(L >= -1e-6), "level non-negative")
    assert_true(abs(L[-1] - L[0]) < 0.5, f"level near-steady: {L[0]:.3f}->{L[-1]:.3f} m")


def test_predict_interface():
    print("\n[Test 11] ComponentModel predict()/get_info() interface")
    _, cm = make_model()
    info = cm.get_info()
    for k in ["component_id", "component_name", "fidelity", "version", "inputs", "outputs"]:
        assert_true(k in info, f"get_info has '{k}'")
    assert_true(info["component_id"] == "EC214", "component_id == EC214")
    r = cm.predict({"duration_s": 1000.0, "dt": 20.0})
    for key in ["SEC_kWh_m3", "GOR_equiv", "distillate_m3_day_final", "t", "T_brine_C_series"]:
        assert_true(key in r, f"predict output has '{key}'")
    assert_true(len(r["t"]) == len(r["T_brine_C_series"]), "series arrays same length")
    assert_true(7.0 <= r["SEC_kWh_m3"] <= 12.0, f"predict SEC={r['SEC_kWh_m3']:.2f} realistic")


def test_benchmark():
    print("\n[Test 12] Benchmark: 3000 s transient sim timing")
    m, _ = make_model()
    t0 = time.perf_counter()
    m.simulate(T0_brine_C=50.0, duration_s=3000.0, dt=10.0)
    elapsed = time.perf_counter() - t0
    print(f"  3000 s simulation in {elapsed*1000:.1f} ms")
    assert_true(elapsed < 5.0, "Completes in < 5 s")


if __name__ == "__main__":
    tests = [
        test_compressor_lifts_above_brine,
        test_bpe_physical,
        test_sec_realistic,
        test_heat_pump_amplification,
        test_sec_rises_with_lift,
        test_design_point_consistency,
        test_transient_heats_to_boiling,
        test_transient_reaches_design_production,
        test_no_boiling_below_saturation,
        test_level_mass_balance,
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
    print(f"EC214 MVC F2a -- Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    sys.exit(0 if failed == 0 else 1)
