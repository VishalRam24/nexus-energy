"""
EC207 -- CO2 Compression & Pipeline -- F2a Physics-Lumped
Test suite: real-gas Z, energy conservation, supercritical phase, stage
heating, SEC realism, pipeline pressure drop, transient ODE, predict()
interface, and a benchmark timing test.  (NO pytest -- custom harness.)
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from model import CO2CompressionPipelineF2a
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
def test_z_factor_realgas():
    print("\n[Test 1] Real-gas Z: dilute ~1, dense-phase << 1, strong variation")
    m, _ = make_model()
    Z_low = m.z_factor(308.15, 1.5)     # near ideal
    Z_dense = m.z_factor(308.15, 150.0)  # dense / supercritical
    assert_true(0.95 < Z_low <= 1.02, f"Z(1.5 bar)={Z_low:.3f} ~ 1 (near ideal)")
    assert_true(Z_dense < 0.6, f"Z(150 bar)={Z_dense:.3f} << 1 (strong real-gas)")
    assert_true(Z_low - Z_dense > 0.3, f"Z varies strongly: {Z_low:.3f} -> {Z_dense:.3f}")


def test_supercritical_dense_phase():
    print("\n[Test 2] Supercritical / dense phase achieved at discharge")
    m, _ = make_model()
    res = m.compress()
    assert_true(res["P_discharge_bar"] > m.P_crit / 1e5,
                f"P_disch={res['P_discharge_bar']:.1f} > P_crit={m.P_crit/1e5:.1f} bar")
    sc = m.is_supercritical(m.T_intercool, res["P_discharge_bar"])
    assert_true(sc, "T>T_crit AND P>P_crit -> dense/supercritical phase")
    rho = m.density_real(m.T_intercool, res["P_discharge_bar"])
    assert_true(rho > 600.0, f"Dense-phase density rho={rho:.0f} kg/m3 > 600")


def test_stage_discharge_hotter():
    print("\n[Test 3] T_discharge > T_in for every stage (polytropic heating)")
    m, _ = make_model()
    res = m.compress()
    for s in range(m.N):
        Td = res["stage_T_discharge"][s]
        Ti = res["stage_T_in"][s]
        assert_true(Td > Ti, f"stage {s+1}: T_disch={Td:.1f} > T_in={Ti:.1f} K")
    # intercooling resets inlet of later stages back to T_intercool
    assert_true(abs(res["stage_T_in"][1] - m.T_intercool) < 1e-6,
                "Intercooling returns stage-2 inlet to T_intercool")


def test_sec_realistic():
    print("\n[Test 4] Specific energy ~90-120 kWh/tCO2 (McCollum & Ogden 2006)")
    m, _ = make_model()
    res = m.compress()
    sec = res["SEC_kWh_per_tCO2"]
    assert_true(70.0 < sec < 140.0, f"SEC={sec:.1f} kWh/tCO2 in literature band")


def test_energy_conservation():
    print("\n[Test 5] Energy conservation: shaft power = m_dot * w_specific")
    m, _ = make_model()
    res = m.compress()
    m_dot = 100.0
    P_shaft = m.shaft_power_kw(m_dot)
    expect = m_dot * res["w_specific_J_per_kg"] / 1000.0
    assert_true(abs(P_shaft - expect) < 1e-6, f"P_shaft={P_shaft:.2f} kW == m_dot*w")
    # sum of stage works (pre mech-loss) reconciles with total
    w_sum = res["stage_work"].sum() / m.eta_m
    assert_true(abs(w_sum - res["w_specific_J_per_kg"]) < 1e-6,
                "Sum of stage works == total specific work")


def test_work_increases_with_pressure():
    print("\n[Test 6] Compression work monotonically increases with P_out")
    m, _ = make_model()
    sec_prev = -1.0
    for P_out in [110.0, 130.0, 150.0, 180.0]:
        sec = m.compress(P_out=P_out)["SEC_kWh_per_tCO2"]
        assert_true(sec > sec_prev, f"SEC(P_out={P_out})={sec:.1f} > prev={sec_prev:.1f}")
        sec_prev = sec


def test_pipeline_pressure_drop():
    print("\n[Test 7] Dense-phase pipeline dP positive & grows with length")
    m, _ = make_model()
    dP_100 = m.pipeline_pressure_drop_bar(100.0, length_km=100.0, diameter_m=0.508)
    dP_200 = m.pipeline_pressure_drop_bar(100.0, length_km=200.0, diameter_m=0.508)
    assert_true(dP_100 > 0.0, f"dP(100km)={dP_100:.2f} bar > 0")
    assert_true(dP_200 > dP_100 * 1.9, f"dP scales ~linearly with L: {dP_100:.2f}->{dP_200:.2f}")
    # higher flow -> larger drop (quadratic-ish in v)
    dP_hi = m.pipeline_pressure_drop_bar(200.0, length_km=100.0, diameter_m=0.508)
    assert_true(dP_hi > dP_100, f"Higher flow raises dP: {dP_hi:.2f} > {dP_100:.2f}")


def test_transient_ode_converges():
    print("\n[Test 8] Lumped pressure-transient ODE reaches steady state")
    m, _ = make_model()
    tr = m.simulate_pressure_transient(m_in_kg_s=100.0, P0_bar=80.0,
                                       P_delivery_bar=80.0, duration_s=600.0)
    P_end = tr["P_discharge_bar"][-1]
    P_ss = tr["P_steady_state_bar"]
    assert_true(P_end > tr["P_discharge_bar"][0],
                f"Pressure builds up: {tr['P_discharge_bar'][0]:.1f} -> {P_end:.1f} bar")
    assert_true(abs(P_end - P_ss) / P_ss < 0.05,
                f"Converges to SS: P_end={P_end:.2f} ~ P_ss={P_ss:.2f} bar")


def test_transient_massbalance():
    print("\n[Test 9] At steady state pipeline outflow == compressor inflow")
    m, _ = make_model()
    tr = m.simulate_pressure_transient(m_in_kg_s=80.0, P_delivery_bar=80.0,
                                       duration_s=800.0)
    m_out_final = tr["m_out_kg_s"][-1]
    assert_true(abs(m_out_final - tr["m_in_kg_s"]) / tr["m_in_kg_s"] < 0.05,
                f"m_out={m_out_final:.2f} ~ m_in={tr['m_in_kg_s']:.2f} kg/s")


def test_predict_interface():
    print("\n[Test 10] ComponentModel predict() interface")
    _, cm = make_model()
    r = cm.predict({"mass_flow_kg_s": 100.0, "duration_s": 100.0})
    for key in ["SEC_kWh_per_tCO2", "specific_work_J_per_kg", "shaft_power_kW",
                "P_discharge_bar", "Z_inlet", "Z_discharge", "pipeline_dP_bar",
                "supercritical", "transient"]:
        assert_true(key in r, f"Key '{key}' in output")
    assert_true(len(r["transient"]["t"]) == len(r["transient"]["P_discharge_bar"]),
                "Transient arrays same length")
    info = cm.get_info()
    assert_true(info["component_id"] == "EC207", "get_info component_id == EC207")


def test_benchmark():
    print("\n[Test 11] Benchmark: full predict() with 300s transient")
    _, cm = make_model()
    t0 = time.perf_counter()
    cm.predict({"mass_flow_kg_s": 100.0, "duration_s": 300.0})
    elapsed = time.perf_counter() - t0
    print(f"  full predict (compression + pipeline + ODE) in {elapsed*1000:.1f} ms")
    assert_true(elapsed < 5.0, "Completes in < 5 s")


if __name__ == "__main__":
    tests = [
        test_z_factor_realgas,
        test_supercritical_dense_phase,
        test_stage_discharge_hotter,
        test_sec_realistic,
        test_energy_conservation,
        test_work_increases_with_pressure,
        test_pipeline_pressure_drop,
        test_transient_ode_converges,
        test_transient_massbalance,
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
    print(f"EC207 CO2 Compression & Pipeline F2a -- Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    sys.exit(0 if failed == 0 else 1)
