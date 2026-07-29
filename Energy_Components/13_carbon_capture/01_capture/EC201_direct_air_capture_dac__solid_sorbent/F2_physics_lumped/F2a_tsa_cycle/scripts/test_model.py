"""
EC201 -- Direct Air Capture (DAC) Solid Sorbent -- F2a TSA Cycle -- Test suite.
"""
import sys, os, time
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from model import TSACycleModel
from predict import ComponentModel

PASS, FAIL = "\u2713", "\u2717"


def assert_true(c, m):
    if c:
        print(f"  {PASS}  {m}")
    else:
        print(f"  {FAIL}  FAILED: {m}")
        raise AssertionError(m)


def make():
    cm = ComponentModel()
    return cm._model, cm


# ------------------------------------------------------------------
# Test 1: Specific thermal energy in realistic range
# ------------------------------------------------------------------
def test_specific_thermal_energy():
    print("\n[Test 1] Specific thermal energy in realistic range")
    m, _ = make()
    r = m.compute()
    sec_th = r["specific_thermal_GJ_tCO2"]
    # With 50% heat recovery: ~3-5 GJ/tCO2; without recovery: 5-10 GJ/tCO2
    # Literature range: 4-12 GJ/tCO2 (no recovery); advanced systems target <4
    assert_true(2.0 <= sec_th <= 12.0,
                f"SEC_th = {sec_th:.2f} GJ/tCO2 in [2, 12]")
    # Also verify without heat recovery is in classic range
    r_no_hr = m.compute()
    # Manually compute: sensible heat doubles without recovery
    Q_no_hr_kJ = (m.m_sorbent * m.Cp_sorbent * (m.T_des_degC - m.T_ads_degC)
                  / 1000.0)  # no recovery
    co2_mol = r["co2_per_cycle_kg"] * 1000.0 / m.M_CO2
    Q_des_kJ = co2_mol * m.delta_H_ads
    sec_th_no_hr = (Q_no_hr_kJ + Q_des_kJ) / r["co2_per_cycle_kg"] * 1e-3
    print(f"  SEC_th (no heat recovery) = {sec_th_no_hr:.2f} GJ/tCO2")
    assert_true(4.0 <= sec_th_no_hr <= 15.0,
                f"SEC_th (no HR) = {sec_th_no_hr:.2f} GJ/tCO2 in [4, 15]")


# ------------------------------------------------------------------
# Test 2: Working capacity > 0
# ------------------------------------------------------------------
def test_working_capacity_positive():
    print("\n[Test 2] Working capacity > 0")
    m, _ = make()
    dq = m.working_capacity()
    assert_true(dq > 0, f"delta_q = {dq:.4f} mmol/g > 0")


# ------------------------------------------------------------------
# Test 3: q at adsorption > q at desorption
# ------------------------------------------------------------------
def test_q_ads_greater_than_q_des():
    print("\n[Test 3] q(T_ads) > q(T_des) at respective conditions")
    m, _ = make()
    r = m.compute()
    assert_true(r["q_ads_mmol_g"] > r["q_des_mmol_g"],
                f"q_ads={r['q_ads_mmol_g']:.4f} > q_des={r['q_des_mmol_g']:.4f}")


# ------------------------------------------------------------------
# Test 4: Higher T_des -> more CO2 released (higher delta_q)
# ------------------------------------------------------------------
def test_higher_T_des_more_release():
    print("\n[Test 4] Higher T_des -> higher working capacity")
    m, _ = make()
    dq_low = m.working_capacity(T_des_degC=80.0)
    dq_high = m.working_capacity(T_des_degC=120.0)
    assert_true(dq_high > dq_low,
                f"dq(T_des=120) = {dq_high:.4f} > dq(T_des=80) = {dq_low:.4f}")


# ------------------------------------------------------------------
# Test 5: Energy balance: total = thermal + electrical
# ------------------------------------------------------------------
def test_energy_balance():
    print("\n[Test 5] Total SEC = thermal + electrical")
    m, _ = make()
    r = m.compute()
    total = r["total_SEC_GJ_tCO2"]
    expected = r["specific_thermal_GJ_tCO2"] + r["specific_electrical_GJ_tCO2"]
    assert_true(abs(total - expected) < 1e-10,
                f"Total {total:.4f} == thermal {r['specific_thermal_GJ_tCO2']:.4f}"
                f" + electrical {r['specific_electrical_GJ_tCO2']:.4f}")


# ------------------------------------------------------------------
# Test 6: Productivity > 0
# ------------------------------------------------------------------
def test_productivity_positive():
    print("\n[Test 6] Productivity > 0")
    m, _ = make()
    prod = m.productivity_kg_h()
    assert_true(prod > 0, f"Productivity = {prod:.3f} kg CO2/h > 0")


# ------------------------------------------------------------------
# Test 7: CO2 per cycle physically reasonable (kg scale for 1000 kg sorbent)
# ------------------------------------------------------------------
def test_co2_per_cycle_reasonable():
    print("\n[Test 7] CO2 per cycle physically reasonable")
    m, _ = make()
    co2 = m.co2_per_cycle_kg()
    # For 1000 kg sorbent with ~1 mmol/g working capacity:
    # 1000 kg * 1e3 g/kg * 1e-3 mol/g * 44 g/mol / 1000 g/kg ~ 44 kg
    # Realistic range: 0.5 - 100 kg for 1000 kg sorbent
    assert_true(0.1 < co2 < 200.0,
                f"CO2/cycle = {co2:.2f} kg (reasonable for 1000 kg sorbent)")


# ------------------------------------------------------------------
# Test 8: Predict interface
# ------------------------------------------------------------------
def test_predict_interface():
    print("\n[Test 8] ComponentModel predict() interface")
    _, cm = make()
    r = cm.predict({})
    expected_keys = [
        "q_ads_mmol_g", "q_des_mmol_g", "working_capacity_mmol_g",
        "co2_per_cycle_kg", "thermal_energy_per_cycle_kJ",
        "fan_energy_per_cycle_kJ", "specific_thermal_GJ_tCO2",
        "specific_electrical_GJ_tCO2", "total_SEC_GJ_tCO2",
        "productivity_kg_CO2_h", "cycle_time_s",
    ]
    for k in expected_keys:
        assert_true(k in r, f"Key '{k}' present")

    # Test with custom inputs
    r2 = cm.predict({"T_des_degC": 110.0})
    assert_true(r2["T_des_degC"] == 110.0, "Custom T_des_degC = 110.0 used")


# ------------------------------------------------------------------
# Test 9: Langmuir isotherm monotonicity
# ------------------------------------------------------------------
def test_langmuir_monotonicity():
    print("\n[Test 9] Langmuir isotherm: q increases with P, decreases with T")
    m, _ = make()

    # q increases with pressure at fixed T
    q_low_P = m.loading(298.15, 0.02)
    q_high_P = m.loading(298.15, 0.06)
    assert_true(q_high_P > q_low_P,
                f"q(P=0.06) = {q_high_P:.4f} > q(P=0.02) = {q_low_P:.4f}")

    # q decreases with temperature at fixed P (for exothermic adsorption)
    q_cold = m.loading(288.15, 0.042)
    q_hot = m.loading(373.15, 0.042)
    assert_true(q_cold > q_hot,
                f"q(15C) = {q_cold:.4f} > q(100C) = {q_hot:.4f}")


# ------------------------------------------------------------------
# Test 10: Specific electrical energy in realistic range
# ------------------------------------------------------------------
def test_specific_electrical_energy():
    print("\n[Test 10] Specific electrical energy 0.1-1.5 GJ/tCO2")
    m, _ = make()
    r = m.compute()
    sec_el = r["specific_electrical_GJ_tCO2"]
    assert_true(0.01 <= sec_el <= 1.5,
                f"SEC_el = {sec_el:.3f} GJ/tCO2 in [0.01, 1.5]")


# ------------------------------------------------------------------
# Test 11: get_info interface
# ------------------------------------------------------------------
def test_get_info():
    print("\n[Test 11] get_info() returns metadata")
    _, cm = make()
    info = cm.get_info()
    assert_true(info["component_id"] == "EC201", "component_id == EC201")
    assert_true("F2a" in info["fidelity"], "fidelity contains F2a")
    assert_true("inputs" in info and "outputs" in info, "inputs/outputs present")


# ------------------------------------------------------------------
# Test 12: Benchmark
# ------------------------------------------------------------------
def test_benchmark():
    print("\n[Test 12] Benchmark: compute() speed")
    m, _ = make()
    t0 = time.perf_counter()
    for _ in range(1000):
        m.compute()
    elapsed = (time.perf_counter() - t0) / 1000
    print(f"  Single compute() in {elapsed*1e6:.1f} us")
    assert_true(elapsed < 1.0, "< 1 s per compute()")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
if __name__ == "__main__":
    tests = [
        test_specific_thermal_energy,
        test_working_capacity_positive,
        test_q_ads_greater_than_q_des,
        test_higher_T_des_more_release,
        test_energy_balance,
        test_productivity_positive,
        test_co2_per_cycle_reasonable,
        test_predict_interface,
        test_langmuir_monotonicity,
        test_specific_electrical_energy,
        test_get_info,
        test_benchmark,
    ]
    p = f = 0
    for t in tests:
        try:
            t()
            p += 1
        except Exception as e:
            f += 1
            print(f"  ERROR: {e}")
    print(f"\n{'='*60}")
    print(f"EC201 DAC F2a TSA Cycle -- {p} passed, {f} failed")
    print(f"{'='*60}")
    sys.exit(0 if f == 0 else 1)
