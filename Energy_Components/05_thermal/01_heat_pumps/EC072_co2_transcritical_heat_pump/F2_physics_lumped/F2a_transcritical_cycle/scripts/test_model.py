"""
EC072 -- CO2 Transcritical Heat Pump (R744) -- F2a Transcritical Cycle
Test suite: cycle thermodynamics, gas-cooler glide, optimum pressure,
energy balance, lumped ODE behavior, edge cases, predict() interface, timing.
NO pytest -- custom assert harness, run as `python3 scripts/test_model.py`.
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from model import CO2TranscriticalHPF2a
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
def test_transcritical_regime():
    print("\n[Test 1] High side is supercritical (transcritical cycle)")
    m, _ = make_model()
    st = m.cycle_states(0.0, 15.0, 90.0)
    assert_true(st["P_high"] > m.P_crit,
                f"P_high={st['P_high']:.1f} > P_crit={m.P_crit:.2f} bar")
    assert_true(st["P_low"] < m.P_crit,
                f"P_low={st['P_low']:.1f} bar subcritical (evaporator)")
    assert_true(st["T2"] - 273.15 > 60.0,
                f"Discharge T2={st['T2']-273.15:.1f} C is high (hot gas)")


def test_cop_greater_than_one():
    print("\n[Test 2] Heating COP > 1 across operating envelope")
    m, _ = make_model()
    for Ts in [-10.0, 0.0, 10.0, 20.0]:
        for Twin in [10.0, 20.0, 30.0]:
            cop = m.cop(Ts, Twin)   # optimum P_high
            assert_true(cop > 1.0, f"COP(Ts={Ts},Twin={Twin})={cop:.3f} > 1")


def test_gas_cooler_glide():
    print("\n[Test 3] Gas cooler shows temperature GLIDE (CO2 signature)")
    m, _ = make_model()
    prof = m.glide(0.0, 15.0, 90.0, n=20)
    glide_span = prof[0] - prof[-1]
    assert_true(glide_span > 20.0,
                f"Glide span={glide_span:.1f} K (non-isothermal heat rejection)")
    # Monotonic cooling along the gas cooler.
    assert_true(np.all(np.diff(prof) <= 1e-9),
                "CO2 temperature decreases monotonically through gas cooler")
    assert_true(abs(prof[-1] - (15.0 + m.pinch_gc)) < 1e-6,
                f"Outlet glides to T_water_in+pinch = {15.0+m.pinch_gc:.1f} C")


def test_cp_peaks_at_pseudocritical():
    print("\n[Test 4] Supercritical cp peaks at pseudocritical T (Span-Wagner)")
    m, _ = make_model()
    T_pc = m.T_pseudocritical(90.0)
    cp_at_pc = m.cp_super(T_pc, 90.0)
    cp_far = m.cp_super(T_pc + 60.0, 90.0)
    assert_true(cp_at_pc > cp_far * 2.0,
                f"cp(T_pc)={cp_at_pc:.2f} >> cp(far)={cp_far:.2f} kJ/(kg.K)")
    # Pseudocritical T rises with pressure.
    assert_true(m.T_pseudocritical(110.0) > m.T_pseudocritical(80.0),
                "Pseudocritical temperature rises with high-side pressure")


def test_optimum_pressure_exists():
    print("\n[Test 5] Optimum high-side pressure exists and is transcritical")
    m, _ = make_model()
    P_opt, cop_opt = m.optimum_high_pressure_search(0.0, 15.0)
    assert_true(P_opt > m.P_crit, f"P_opt={P_opt:.1f} > P_crit (transcritical)")
    # COP at optimum beats COP at low and high extremes.
    cop_lo = m.cop(0.0, 15.0, 76.0)
    cop_hi = m.cop(0.0, 15.0, 128.0)
    assert_true(cop_opt >= cop_lo and cop_opt >= cop_hi,
                f"COP_opt={cop_opt:.3f} >= COP(76)={cop_lo:.3f}, "
                f"COP(128)={cop_hi:.3f}")
    # Liao correlation lands in transcritical regime too.
    P_liao = float(m.optimum_high_pressure_liao(35.0))
    assert_true(P_liao > m.P_crit, f"Liao P_opt(35C)={P_liao:.1f} > P_crit")


def test_energy_balance():
    print("\n[Test 6] Energy balance: q_gc = w_elec * COP within rounding")
    m, _ = make_model()
    st = m.cycle_states(0.0, 15.0, 95.0)
    lhs = st["q_gc"]
    rhs = st["w_elec"] * st["cop"]
    assert_true(abs(lhs - rhs) / lhs < 1e-9,
                f"q_gc={lhs:.2f} == w_elec*COP={rhs:.2f} kJ/kg")
    # First law: heat out exceeds work in (heat pump amplification).
    assert_true(st["q_gc"] > st["w_elec"],
                f"q_gc={st['q_gc']:.1f} > w_elec={st['w_elec']:.1f} kJ/kg")


def test_compressor_work_sign():
    print("\n[Test 7] Compression raises temperature; actual work >= isentropic")
    m, _ = make_model()
    st = m.cycle_states(0.0, 15.0, 90.0)
    assert_true(st["T2"] > st["T1"], f"T2={st['T2']:.1f} > T1={st['T1']:.1f} K")
    assert_true(st["T2"] >= st["T2s"] - 1e-6,
                f"Actual discharge T2={st['T2']:.1f} >= isentropic "
                f"T2s={st['T2s']:.1f} K")
    assert_true(st["w_act"] >= st["w_is"] - 1e-9,
                f"w_act={st['w_act']:.2f} >= w_is={st['w_is']:.2f}")


def test_cop_drops_with_higher_water():
    print("\n[Test 8] COP falls as required water-inlet temperature rises")
    m, _ = make_model()
    cop_cold = m.cop(0.0, 15.0)
    cop_warm = m.cop(0.0, 40.0)
    assert_true(cop_cold > cop_warm,
                f"COP(Twin=15)={cop_cold:.3f} > COP(Twin=40)={cop_warm:.3f}")
    # COP rises with warmer source.
    cop_lowsrc = m.cop(-15.0, 15.0)
    cop_hisrc = m.cop(15.0, 15.0)
    assert_true(cop_hisrc > cop_lowsrc,
                f"COP(src=15)={cop_hisrc:.3f} > COP(src=-15)={cop_lowsrc:.3f}")


def test_lumped_ode_heats_water():
    print("\n[Test 9] Lumped ODE charges water monotonically toward target")
    m, _ = make_model()
    r = m.simulate(5.0, 15.0, 65.0, duration_s=1800.0)
    assert_true(r["T_water"][-1] > r["T_water"][0],
                f"Water warms {r['T_water'][0]:.1f} -> {r['T_water'][-1]:.1f} C")
    assert_true(np.all(np.diff(r["T_water"]) >= -1e-6),
                "Water temperature is non-decreasing (monotone charge)")
    assert_true(r["T_water"][-1] <= 65.0 + 0.05,
                f"Self-limits at target: T_final={r['T_water'][-1]:.3f} <= ~65 C")
    assert_true(np.all(r["cop"] > 1.0), "COP > 1 throughout transient")


def test_latent_heat_declines_to_crit():
    print("\n[Test 10] Evaporator latent heat declines toward critical point")
    m, _ = make_model()
    h_cold = m.latent_heat_evap(263.15)   # -10 C
    h_warm = m.latent_heat_evap(298.15)   # 25 C, near critical
    assert_true(h_cold > h_warm > 0.0,
                f"h_fg(-10C)={h_cold:.1f} > h_fg(25C)={h_warm:.1f} > 0 kJ/kg")


def test_predict_interface():
    print("\n[Test 11] ComponentModel predict() interface + get_info()")
    _, cm = make_model()
    info = cm.get_info()
    for k in ["component_id", "fidelity", "version", "inputs", "outputs"]:
        assert_true(k in info, f"get_info has '{k}'")
    assert_true(info["component_id"] == "EC072", "component_id == EC072")
    r = cm.predict({"T_source_c": 5.0, "T_water_in_c": 15.0,
                    "T_water_target_c": 60.0, "duration_s": 600.0})
    for key in ["t", "T_water", "cop", "Q_heat_kW", "P_elec_kW",
                "cop_design", "P_high_bar"]:
        assert_true(key in r, f"Key '{key}' in predict output")
    assert_true(len(r["t"]) == len(r["T_water"]) == len(r["cop"]),
                "Output arrays share length")
    assert_true(r["cop_design"] > 1.0, f"cop_design={r['cop_design']:.3f} > 1")


def test_benchmark():
    print("\n[Test 12] Benchmark: 1800 s transient + optimum-pressure search")
    m, _ = make_model()
    t0 = time.perf_counter()
    m.simulate(0.0, 15.0, 65.0, duration_s=1800.0)
    elapsed = time.perf_counter() - t0
    print(f"  1800 s simulation in {elapsed*1000:.1f} ms")
    assert_true(elapsed < 5.0, "Completes in < 5 s")


if __name__ == "__main__":
    tests = [
        test_transcritical_regime,
        test_cop_greater_than_one,
        test_gas_cooler_glide,
        test_cp_peaks_at_pseudocritical,
        test_optimum_pressure_exists,
        test_energy_balance,
        test_compressor_work_sign,
        test_cop_drops_with_higher_water,
        test_lumped_ode_heats_water,
        test_latent_heat_declines_to_crit,
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
    print(f"EC072 CO2 Transcritical HP F2a -- Results: "
          f"{passed} passed, {failed} failed")
    print(f"{'='*60}")
    sys.exit(0 if failed == 0 else 1)
