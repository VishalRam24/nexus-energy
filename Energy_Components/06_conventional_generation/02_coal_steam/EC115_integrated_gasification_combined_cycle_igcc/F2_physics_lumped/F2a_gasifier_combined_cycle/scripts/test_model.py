"""
EC115 -- IGCC -- F2a Physics-Lumped
Test suite: conservation, thermodynamic bounds, ODE convergence, edge cases.
Run:  python3 scripts/test_model.py   (system python3, NOT pytest)
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from model import IGCC_F2a
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


def design_coal(m):
    return m.Q_coal_design / m.LHV_coal


# ---------------------------------------------------------------------------
def test_cold_gas_efficiency():
    print("\n[Test 1] Cold-gas efficiency < 1 (Higman & van der Burgt 2008)")
    m, _ = make_model()
    assert_true(0.0 < m.cge < 1.0, f"CGE={m.cge:.3f} in (0,1)")
    # syngas chemical power must be strictly less than coal power
    q_coal = m.coal_power_mw(20.0)
    q_syn = m.syngas_power_mw(20.0)
    assert_true(q_syn < q_coal, f"Q_syngas={q_syn:.1f} < Q_coal={q_coal:.1f} MW")


def test_combined_cycle_beats_single():
    print("\n[Test 2] Combined-cycle eff > either single cycle")
    m, _ = make_model()
    eta_cc = m.combined_cycle_efficiency()
    assert_true(eta_cc > m.eta_B, f"eta_CC={eta_cc:.3f} > eta_Brayton={m.eta_B:.3f}")
    assert_true(eta_cc > m.eta_R, f"eta_CC={eta_cc:.3f} > eta_Rankine={m.eta_R:.3f}")
    assert_true(eta_cc < 1.0, f"eta_CC={eta_cc:.3f} < 1")


def test_net_efficiency_range():
    print("\n[Test 3] Net plant efficiency in 0.38-0.45 (IGCC LHV basis)")
    m, _ = make_model()
    eta = m.net_efficiency()
    assert_true(0.38 <= eta <= 0.45, f"eta_net={eta:.4f} in [0.38, 0.45]")


def test_efficiency_below_carnot():
    print("\n[Test 4] Every efficiency < Carnot bound")
    m, _ = make_model()
    eta_c = m.carnot_efficiency()
    assert_true(0 < eta_c < 1, f"Carnot={eta_c:.3f} in (0,1)")
    assert_true(m.net_efficiency() < eta_c, f"eta_net < Carnot ({eta_c:.3f})")
    assert_true(m.combined_cycle_efficiency() < eta_c, f"eta_CC < Carnot")
    assert_true(m.eta_B < eta_c and m.eta_R < eta_c, "Brayton & Rankine < Carnot")


def test_energy_conservation():
    print("\n[Test 5] Energy balance: gross = Brayton + Rankine, all <= syngas power")
    m, _ = make_model()
    mc = design_coal(m)
    q_syn = m.syngas_power_mw(mc)
    w_b = m.brayton_work_mw(mc)
    w_r = m.rankine_work_mw(mc)
    w_g = m.gross_power_mw(mc)
    assert_true(abs(w_g - (w_b + w_r)) < 1e-6, "gross = Brayton + Rankine")
    assert_true(w_g <= q_syn + 1e-6, f"gross work {w_g:.1f} <= syngas power {q_syn:.1f}")
    # full chain: net work <= coal power, no energy created
    assert_true(m.net_power_mw(mc) <= m.coal_power_mw(mc), "net <= coal power")


def test_mass_conservation_monotone():
    print("\n[Test 6] Mass/energy: outputs scale monotonically with coal feed")
    m, _ = make_model()
    coals = np.linspace(10.0, 40.0, 8)
    p_prev = -1.0
    s_prev = -1.0
    for c in coals:
        p = float(m.net_power_mw(c))
        s = float(m.syngas_rate_nm3s(c))
        assert_true(p > p_prev, f"net power rises with coal: {p:.1f} MW")
        assert_true(s > s_prev, f"syngas flow rises with coal: {s:.1f} Nm3/s")
        p_prev, s_prev = p, s


def test_design_point_power():
    print("\n[Test 7] Design coal feed delivers ~rated net power (400 MW)")
    m, _ = make_model()
    p = float(m.net_power_mw(design_coal(m)))
    assert_true(abs(p - m.P_rated) < 1.0, f"net power {p:.1f} ~ rated {m.P_rated} MW")


def test_thermal_ode_lag_and_settle():
    print("\n[Test 8] Combustor metal ODE lags gas, settles toward T_gas")
    m, _ = make_model()
    r = m.simulate(design_coal(m), T_metal_0=m.T_comp, dt=5.0, duration_s=3000.0)
    # metal starts below gas, monotonically rises, approaches T_gas
    assert_true(r["T_metal"][0] < r["T_gas"][0], "metal starts below gas temp")
    assert_true(r["T_metal"][-1] > r["T_metal"][0], "metal heats up")
    gap = abs(r["T_metal"][-1] - r["T_gas"][-1])
    assert_true(gap < 5.0, f"metal -> gas at long time (gap={gap:.2f} K)")
    # never exceeds the driving gas temperature (passive lumped capacitance)
    assert_true(np.all(r["T_metal"] <= r["T_gas"] + 1e-6), "metal never exceeds gas T")


def test_thermal_time_constant():
    print("\n[Test 9] ODE matches analytic tau (1st-order lumped capacitance)")
    m, _ = make_model()
    tau = m.time_constant()
    r = m.simulate(design_coal(m), T_metal_0=m.T_comp, dt=2.0, duration_s=tau)
    T0, Tinf = r["T_metal"][0], r["T_gas"][-1]
    # at t = tau, response should reach ~63.2% of (Tinf - T0)
    frac = (r["T_metal"][-1] - T0) / (Tinf - T0)
    assert_true(0.60 < frac < 0.66, f"response at tau = {frac*100:.1f}% (~63.2%)")


def test_co2_intensity():
    print("\n[Test 10] CO2 intensity in IGCC range (no CCS, ~650-850 g/kWh)")
    m, _ = make_model()
    ci = float(m.co2_intensity_g_per_kwh(design_coal(m)))
    assert_true(600.0 < ci < 900.0, f"CO2 intensity {ci:.0f} g/kWh (no-CCS IGCC)")


def test_predict_interface():
    print("\n[Test 11] ComponentModel predict()/get_info() interface")
    _, cm = make_model()
    info = cm.get_info()
    assert_true(info["component_id"] == "EC115", "component_id == EC115")
    r = cm.predict({"dt": 10.0, "duration_s": 200.0})
    for key in ["t", "T_metal", "T_gas", "net_power_mw", "net_efficiency",
                "combined_cycle_efficiency", "carnot_efficiency", "syngas_rate_nm3s"]:
        assert_true(key in r, f"Key '{key}' in output")
    assert_true(len(r["t"]) == len(r["T_metal"]), "Arrays same length")


def test_benchmark():
    print("\n[Test 12] Benchmark: 600 s sim at dt=2")
    m, _ = make_model()
    t0 = time.perf_counter()
    m.simulate(design_coal(m), dt=2.0, duration_s=600.0)
    elapsed = time.perf_counter() - t0
    print(f"  600 s simulation in {elapsed*1000:.1f} ms")
    assert_true(elapsed < 5.0, "Completes in < 5 s")


if __name__ == "__main__":
    tests = [
        test_cold_gas_efficiency,
        test_combined_cycle_beats_single,
        test_net_efficiency_range,
        test_efficiency_below_carnot,
        test_energy_conservation,
        test_mass_conservation_monotone,
        test_design_point_power,
        test_thermal_ode_lag_and_settle,
        test_thermal_time_constant,
        test_co2_intensity,
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
    print(f"EC115 IGCC F2a -- Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    sys.exit(0 if failed == 0 else 1)
