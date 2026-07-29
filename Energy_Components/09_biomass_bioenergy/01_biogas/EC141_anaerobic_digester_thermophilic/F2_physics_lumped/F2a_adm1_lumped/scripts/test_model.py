"""
EC141 -- Anaerobic Digester (Thermophilic) -- F2a Simplified ADM1 + Thermal
Test suite: kinetics sanity, mass conservation, thermophilic thermal, edge cases.
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from model import AnaerobicDigesterThermophilicF2a
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
def test_monod_kinetics():
    print("\n[Test 1] Monod kinetics: saturating, monotone in (0,1)")
    m, _ = make_model()
    f_lo = m.monod(0.1, m.Ks_met)
    f_hi = m.monod(100.0, m.Ks_met)
    assert_true(0.0 < f_lo < f_hi < 1.0, f"monod 0<{f_lo:.3f}<{f_hi:.3f}<1")
    f_half = m.monod(m.Ks_met, m.Ks_met)
    assert_true(abs(f_half - 0.5) < 1e-9, f"S=Ks gives 0.5 ({f_half:.4f})")


def test_vfa_inhibition():
    print("\n[Test 2] VFA inhibition factor decreases with VFA")
    m, _ = make_model()
    I0 = m.inhibition_vfa(0.0)
    I_hi = m.inhibition_vfa(50.0)
    assert_true(abs(I0 - 1.0) < 1e-9, f"no inhibition at Sa=0 (I={I0:.3f})")
    assert_true(0.0 < I_hi < I0, f"inhibition grows: I(50)={I_hi:.3f} < 1")


def test_steady_state_biogas_positive():
    print("\n[Test 3] Steady-state produces positive realistic biogas")
    m, _ = make_model()
    r = m.simulate(S_in_COD=50.0, Q_in=6.667, T0_degC=55.0,
                   duration_days=120.0, dt_days=1.0)
    ch4 = r["Q_CH4_m3_day"][-1]
    biogas = r["Q_biogas_m3_day"][-1]
    assert_true(ch4 > 0, f"CH4>0 ({ch4:.1f} m3/day)")
    assert_true(biogas > ch4, f"biogas>CH4 ({biogas:.1f}>{ch4:.1f})")


def test_methane_yield_realistic():
    print("\n[Test 4] CH4 yield per kgCOD_fed is realistic (< 0.35 m3/kgCOD)")
    m, _ = make_model()
    r = m.simulate(S_in_COD=50.0, Q_in=6.667, T0_degC=55.0,
                   duration_days=200.0, dt_days=1.0)
    cod_fed = 6.667 * 50.0                       # kgCOD/day fed
    ch4 = r["Q_CH4_m3_day"][-1]                  # m3/day
    yield_per_cod = ch4 / cod_fed                # m3CH4/kgCOD
    # theoretical max 0.35 m3/kgCOD; real digesters 0.20-0.34
    assert_true(0.05 < yield_per_cod <= 0.35,
                f"CH4 yield {yield_per_cod:.3f} m3/kgCOD in (0.05, 0.35]")


def test_cod_mass_conservation():
    print("\n[Test 5] COD mass balance closes at steady state")
    m, _ = make_model()
    S_in, Q_in = 50.0, 6.667
    r = m.simulate(S_in_COD=S_in, Q_in=Q_in, T0_degC=55.0,
                   duration_days=300.0, dt_days=2.0)
    # total COD in liquid (substrate + biomass), last step
    cod_liq = (r["Xc"][-1] + r["Ss"][-1] + r["Sa_VFA"][-1]
               + r["Xaci"][-1] + r["Xmet"][-1])
    # COD balance: in = out(liquid) + gas (in COD), per day, at SS
    cod_in = Q_in * S_in                                   # kgCOD/day
    cod_out_liq = Q_in * cod_liq                           # kgCOD/day leaving
    # CH4 carries 0.35 m3/kgCOD -> COD_to_gas = Q_ch4 / 0.35
    cod_gas = r["Q_CH4_m3_day"][-1] / m.f_ch4_cod          # kgCOD/day
    residual = cod_in - cod_out_liq - cod_gas
    rel = abs(residual) / cod_in
    assert_true(rel < 0.05, f"COD balance closes: residual {rel*100:.2f}% < 5%")


def test_thermophilic_temperature():
    print("\n[Test 6] Heater holds thermophilic setpoint (~55 C)")
    m, _ = make_model()
    r = m.simulate(S_in_COD=50.0, Q_in=6.667, T0_degC=20.0,
                   duration_days=10.0, dt_days=0.1)
    T_final = r["temperature_degC"][-1]
    assert_true(50.0 < T_final < 60.0,
                f"T reaches thermophilic band: {T_final:.2f} C in (50,60)")
    assert_true(r["temperature_degC"][0] < T_final,
                f"digester heated from cold start ({r['temperature_degC'][0]:.1f}->{T_final:.1f})")


def test_heating_demand_positive():
    print("\n[Test 7] Heating demand positive (thermophilic loses heat to ambient)")
    m, _ = make_model()
    q = m.heating_demand_W(m.T_set, 6.667)
    assert_true(q > 0, f"heating demand {q/1000:.1f} kW > 0")
    # higher feed -> more sensible load
    q2 = m.heating_demand_W(m.T_set, 20.0)
    assert_true(q2 > q, f"more feed raises demand ({q2/1000:.1f}>{q/1000:.1f} kW)")


def test_washout_at_high_dilution():
    print("\n[Test 8] Washout: very high dilution collapses methanogens/biogas")
    m, _ = make_model()
    # D > mu_max_met -> methanogens cannot sustain -> washout
    r_ok = m.simulate(50.0, 6.667, 55.0, duration_days=150.0, dt_days=2.0)
    r_wash = m.simulate(50.0, 90.0, 55.0, duration_days=150.0, dt_days=2.0)
    assert_true(r_wash["Q_CH4_m3_day"][-1] < r_ok["Q_CH4_m3_day"][-1],
                f"high D lowers CH4 ({r_wash['Q_CH4_m3_day'][-1]:.2f} < {r_ok['Q_CH4_m3_day'][-1]:.2f})")


def test_no_feed_no_biogas():
    print("\n[Test 9] Edge: no substrate feed -> biogas decays toward zero")
    m, _ = make_model()
    r = m.simulate(S_in_COD=0.0, Q_in=6.667, T0_degC=55.0,
                   duration_days=200.0, dt_days=2.0)
    assert_true(r["Q_CH4_m3_day"][-1] < 1.0,
                f"CH4 ~0 with no feed ({r['Q_CH4_m3_day'][-1]:.3f} m3/day)")


def test_concentrations_nonnegative():
    print("\n[Test 10] All state concentrations stay >= 0")
    m, _ = make_model()
    r = m.simulate(80.0, 10.0, 55.0, duration_days=100.0, dt_days=1.0)
    for key in ["Xc", "Ss", "Sa_VFA", "Xaci", "Xmet"]:
        assert_true(np.all(r[key] >= -1e-6), f"{key} >= 0 (min {r[key].min():.4f})")


def test_predict_interface():
    print("\n[Test 11] ComponentModel predict() / get_info() interface")
    _, cm = make_model()
    info = cm.get_info()
    assert_true(info["component_id"] == "EC141", "component_id EC141")
    assert_true("Batstone" in info["source"], "ADM1 (Batstone) cited in source")
    r = cm.predict({"S_in_COD": 50.0, "Q_in": 6.667,
                    "duration_days": 20.0, "dt_days": 1.0})
    for key in ["t", "temperature", "Q_CH4_m3_day", "Q_biogas_m3_day",
                "energy_kWh_day", "heating_demand_W", "HRT_days"]:
        assert_true(key in r, f"Key '{key}' in output")
    assert_true(len(r["t"]) == len(r["Q_CH4_m3_day"]), "Arrays same length")
    assert_true(abs(r["HRT_days"] - 15.0) < 0.5, f"HRT ~15 d ({r['HRT_days']:.1f})")


def test_benchmark():
    print("\n[Test 12] Benchmark: 60-day sim at dt=0.5 d")
    m, _ = make_model()
    t0 = time.perf_counter()
    m.simulate(50.0, 6.667, 55.0, duration_days=60.0, dt_days=0.5)
    elapsed = time.perf_counter() - t0
    print(f"  60-day simulation in {elapsed*1000:.1f} ms")
    assert_true(elapsed < 5.0, "Completes in < 5 s")


if __name__ == "__main__":
    tests = [
        test_monod_kinetics,
        test_vfa_inhibition,
        test_steady_state_biogas_positive,
        test_methane_yield_realistic,
        test_cod_mass_conservation,
        test_thermophilic_temperature,
        test_heating_demand_positive,
        test_washout_at_high_dilution,
        test_no_feed_no_biogas,
        test_concentrations_nonnegative,
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
    print(f"EC141 AD Thermophilic F2a -- Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    sys.exit(0 if failed == 0 else 1)
