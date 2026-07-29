"""
EC190 -- LNG Regasification Terminal -- F2a Physics-Lumped Vaporizer Thermal
Test suite: energy conservation, vaporization heat balance, send-out tracking,
cold-energy accounting, ODE behaviour, edge cases, predict() interface, timing.
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from model import LNGRegasF2a
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
def test_regas_heat_components():
    print("\n[Test 1] Regas specific heat = latent + sensible, dominated by latent")
    m, _ = make_model()
    q = m.regas_specific_heat()
    # Methane: latent ~511 kJ/kg, total regas to 5C ~ 0.9-1.1 MJ/kg (Mokhatab 2014)
    assert_true(7.0e5 < q < 1.3e6, f"q_regas={q/1e3:.1f} kJ/kg in [700,1300]")
    assert_true(q > m.h_fg, f"total {q/1e3:.1f} > latent {m.h_fg/1e3:.1f} kJ/kg (sensible added)")


def test_process_demand_scales():
    print("\n[Test 2] Process heat demand scales linearly with send-out rate")
    m, _ = make_model()
    q1 = m.process_heat_demand_W(250.0)
    q2 = m.process_heat_demand_W(500.0)
    assert_true(abs(q2 - 2.0 * q1) < 1e-3 * q2, f"linear: {q2/1e6:.2f} ~ 2x {q1/1e6:.2f} MW")
    # 500 t/h ~ 139 kg/s * ~1 MJ/kg ~ 130-150 MW
    assert_true(80e6 < q2 < 200e6, f"500 t/h demand {q2/1e6:.1f} MW in [80,200]")


def test_energy_conservation():
    print("\n[Test 3] Energy conservation: E_source = E_process + E_stored")
    _, cm = make_model()
    r = cm.predict({"sendout_rate_ton_per_h": 500.0, "duration_s": 7200.0, "dt": 30.0})
    eb = r["energy_balance"]
    rel = abs(eb["residual_J"]) / max(abs(eb["E_source_J"]), 1.0)
    assert_true(rel < 1e-3, f"residual {eb['residual_J']:.3e} J, rel={rel:.2e} < 1e-3")


def test_thermal_transient_relaxes():
    print("\n[Test 4] Lumped ODE relaxes metal T toward steady state")
    m, cm = make_model()
    # cold start well below steady
    r = cm.predict({"sendout_rate_ton_per_h": 500.0, "T_metal0_K": 150.0,
                    "duration_s": 7200.0, "dt": 20.0})
    T = r["T_metal"]
    assert_true(T[-1] > T[0], f"metal warms: {T[-1]:.1f} > {T[0]:.1f} K")
    dT = abs(T[-1] - T[-2])
    assert_true(dT < 0.05, f"near steady: |dT|={dT:.4f} K between last steps")


def test_steady_metal_T_matches_analytic():
    print("\n[Test 5] ODE steady metal T matches analytic UA-network solution")
    m, cm = make_model()
    # Use a low flow so process is capacity-limited (not flow-limited)
    r = cm.predict({"sendout_rate_ton_per_h": 5000.0, "T_metal0_K": 250.0,
                    "duration_s": 7200.0, "dt": 20.0})
    T_ss_ode = r["T_metal"][-1]
    T_ss_ana = m.steady_metal_T(5000.0)
    assert_true(abs(T_ss_ode - T_ss_ana) < 0.5,
                f"ODE {T_ss_ode:.2f} ~ analytic {T_ss_ana:.2f} K")
    # metal sits between cold stream and seawater source
    assert_true(m._process_stream_mean_T() < T_ss_ode < m.T_heat_source,
                f"{m._process_stream_mean_T():.1f} < {T_ss_ode:.1f} < {m.T_heat_source:.1f} K")


def test_sendout_tracks_demand():
    print("\n[Test 6] Send-out gas tracks demand when heat is sufficient")
    m, cm = make_model()
    r = cm.predict({"sendout_rate_ton_per_h": 300.0, "duration_s": 5400.0, "dt": 30.0})
    # at steady state Q_process should meet Q_demand -> sendout equals requested
    m_requested = 300.0 * 1000.0 / 3600.0
    assert_true(abs(r["sendout_kg_s"][-1] - m_requested) < 0.02 * m_requested,
                f"sendout {r['sendout_kg_s'][-1]:.2f} ~ requested {m_requested:.2f} kg/s")
    assert_true(r["Q_process_W"][-1] <= r["Q_demand_W"][-1] + 1e-3,
                "Q_process never exceeds demand")


def test_sendout_demand_limited():
    print("\n[Test 7] Cold heat source starves send-out (capacity-limited)")
    m, cm = make_model()
    # Very high flow demand + near-freezing seawater -> capacity limited
    r = cm.predict({"sendout_rate_ton_per_h": 5000.0, "T_heat_source_K": 276.0,
                    "duration_s": 5400.0, "dt": 30.0})
    m_requested = 5000.0 * 1000.0 / 3600.0
    assert_true(r["sendout_kg_s"][-1] < m_requested,
                f"starved: delivered {r['sendout_kg_s'][-1]:.1f} < requested {m_requested:.1f} kg/s")
    assert_true(r["Q_process_W"][-1] < r["Q_demand_W"][-1],
                "Q_process below demand when heat-limited")


def test_pump_work_positive_small():
    print("\n[Test 8] Pump work positive and small vs vaporization heat")
    m, _ = make_model()
    Wp = m.pump_work_W(500.0)
    Qv = m.process_heat_demand_W(500.0)
    assert_true(Wp > 0, f"pump work {Wp/1e6:.3f} MW > 0")
    assert_true(Wp < 0.1 * Qv, f"pump {Wp/1e6:.2f} MW << vaporization {Qv/1e6:.1f} MW")


def test_cold_exergy_accounting():
    print("\n[Test 9] Cold exergy positive, below total regas heat, rises with ambient")
    m, _ = make_model()
    Ex_lo = m.cold_exergy_W(500.0, T_ambient_K=283.15)
    Ex_hi = m.cold_exergy_W(500.0, T_ambient_K=303.15)
    Q = m.process_heat_demand_W(500.0)
    assert_true(Ex_lo > 0, f"cold exergy {Ex_lo/1e6:.2f} MW > 0")
    assert_true(Ex_lo < Q, f"exergy {Ex_lo/1e6:.1f} < heat {Q/1e6:.1f} MW (2nd law)")
    assert_true(Ex_hi > Ex_lo, f"warmer ambient raises cold exergy: {Ex_hi/1e6:.1f} > {Ex_lo/1e6:.1f} MW")


def test_zero_flow_edge():
    print("\n[Test 10] Zero send-out: no process heat, metal warms to source")
    m, cm = make_model()
    r = cm.predict({"sendout_rate_ton_per_h": 0.0, "T_metal0_K": 200.0,
                    "duration_s": 7200.0, "dt": 30.0})
    assert_true(np.all(r["Q_process_W"] < 1e-6), "no process heat drawn at zero flow")
    assert_true(abs(r["T_metal"][-1] - m.T_heat_source) < 1.0,
                f"metal -> source {r['T_metal'][-1]:.1f} ~ {m.T_heat_source:.1f} K")


def test_predict_interface():
    print("\n[Test 11] ComponentModel predict()/get_info() interface")
    _, cm = make_model()
    info = cm.get_info()
    for k in ["component_id", "fidelity", "version", "inputs", "outputs"]:
        assert_true(k in info, f"get_info has '{k}'")
    assert_true(info["component_id"] == "EC190", "component_id == EC190")
    r = cm.predict({"sendout_rate_ton_per_h": 400.0, "duration_s": 600.0, "dt": 30.0})
    for key in ["t", "T_metal", "Q_source_W", "Q_process_W", "sendout_kg_s",
                "pump_W", "cold_exergy_W", "energy_balance"]:
        assert_true(key in r, f"output has '{key}'")
    assert_true(len(r["t"]) == len(r["T_metal"]) == len(r["sendout_kg_s"]),
                "time-series arrays same length")


def test_benchmark():
    print("\n[Test 12] Benchmark: 1 h transient at dt=10 s")
    _, cm = make_model()
    t0 = time.perf_counter()
    cm.predict({"sendout_rate_ton_per_h": 500.0, "duration_s": 3600.0, "dt": 10.0})
    elapsed = time.perf_counter() - t0
    print(f"  1 h simulation in {elapsed*1000:.1f} ms")
    assert_true(elapsed < 5.0, "completes in < 5 s")


if __name__ == "__main__":
    tests = [
        test_regas_heat_components,
        test_process_demand_scales,
        test_energy_conservation,
        test_thermal_transient_relaxes,
        test_steady_metal_T_matches_analytic,
        test_sendout_tracks_demand,
        test_sendout_demand_limited,
        test_pump_work_positive_small,
        test_cold_exergy_accounting,
        test_zero_flow_edge,
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
    print(f"EC190 LNG Regas F2a -- Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    sys.exit(0 if failed == 0 else 1)
