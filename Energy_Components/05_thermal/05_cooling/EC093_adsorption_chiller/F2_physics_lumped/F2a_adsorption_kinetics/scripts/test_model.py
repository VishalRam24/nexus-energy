"""
EC093 — Adsorption Chiller — F2a Physics-Lumped (Adsorption Kinetics)
Test suite: physics sanity, conservation, isotherm/kinetics limits,
cyclic behaviour, COP band, predict() interface, benchmark timing.
No pytest — run as `python3 scripts/test_model.py`.
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from model import AdsorptionChillerF2a, p_sat_water
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
def test_psat_monotone():
    print("\n[Test 1] Water saturation pressure increasing & sane")
    p10 = p_sat_water(283.15)   # 10 C ~ 1228 Pa
    p100 = p_sat_water(373.15)  # 100 C ~ 101325 Pa
    assert_true(p100 > p10, f"p_sat(100C)={p100:.0f} > p_sat(10C)={p10:.0f}")
    assert_true(1000 < p10 < 1500, f"p_sat(10C)={p10:.0f} Pa near 1228")
    assert_true(9.5e4 < p100 < 1.06e5, f"p_sat(100C)={p100:.0f} Pa near 101325")


def test_isotherm_monotone_T():
    print("\n[Test 2] D-A isotherm: uptake falls as bed heats (desorption)")
    m, _ = make_model()
    T_src = 287.15  # evaporator 14 C
    w_prev = float(m.w_equilibrium(298.15, T_src))
    for Tb in np.linspace(300, 360, 20):
        w = float(m.w_equilibrium(Tb, T_src))
        assert_true(w <= w_prev + 1e-9, f"w*({Tb-273.15:.0f}C)={w:.4f} <= prev {w_prev:.4f}")
        w_prev = w
    print("  Monotone decrease over 20 bed temperatures.")


def test_isotherm_bounds():
    print("\n[Test 3] Equilibrium uptake bounded in [0, W0]")
    m, _ = make_model()
    for Tb in [290, 310, 340, 370]:
        for Ts in [280, 290, 300]:
            w = float(m.w_equilibrium(Tb, Ts))
            assert_true(0.0 <= w <= m.W0 + 1e-12, f"0<=w*({Tb},{Ts})={w:.4f}<=W0={m.W0}")


def test_ldf_arrhenius():
    print("\n[Test 4] LDF rate constant rises with temperature (Arrhenius)")
    m, _ = make_model()
    k_cold = float(m.k_ldf(303.15))
    k_hot = float(m.k_ldf(358.15))
    assert_true(k_hot > k_cold, f"k(85C)={k_hot:.4e} > k(30C)={k_cold:.4e}")
    tau = 1.0 / k_hot
    assert_true(5 < tau < 600, f"hot-bed time constant {tau:.0f}s physically reasonable")


def test_cop_in_band():
    print("\n[Test 5] Thermal COP in physical band for silica-gel/water")
    _, cm = make_model()
    r = cm.predict({"T_hot": 85, "T_cool": 30, "T_chilled": 14,
                    "t_half_cycle": 400, "n_cycles": 8})
    cop = r["thermal_COP"]
    # Single-stage silica-gel/water ideal thermal COP: ~0.3-0.75 (Wang & Oliveira 2006)
    assert_true(0.3 <= cop <= 0.75, f"COP={cop:.3f} in [0.3, 0.75]")
    assert_true(cop < 1.0, f"COP={cop:.3f} < 1 (heat-driven, sub-unity)")


def test_cooling_power_positive():
    print("\n[Test 6] Positive cooling power & SCP at design point")
    _, cm = make_model()
    r = cm.predict({"n_cycles": 8})
    assert_true(r["cooling_power_kW"] > 0, f"Q_cool={r['cooling_power_kW']:.2f} kW > 0")
    assert_true(100 < r["SCP_W_per_kg"] < 1000,
                f"SCP={r['SCP_W_per_kg']:.0f} W/kg in plausible range")


def test_mass_conservation():
    print("\n[Test 7] Mass conservation: water adsorbed == desorbed per cycle")
    _, cm = make_model()
    r = cm.predict({"n_cycles": 10})
    dw_a, dw_d = r["dw_adsorbed"], r["dw_desorbed"]
    assert_true(abs(dw_a - dw_d) < 0.02 * max(abs(dw_a), 1e-6),
                f"dw_ads={dw_a:.4f} ~= dw_des={dw_d:.4f} (cyclic steady state)")
    assert_true(dw_a > 0, f"net uptake swing dw={dw_a:.4f} > 0")


def test_energy_balance_cop():
    print("\n[Test 8] Energy balance: driving heat > cooling delivered")
    _, cm = make_model()
    r = cm.predict({"n_cycles": 8})
    assert_true(r["E_des_J"] > r["E_evap_J"],
                f"Q_drive={r['E_des_J']:.1e} J > Q_cool={r['E_evap_J']:.1e} J (COP<1)")
    cop_check = r["E_evap_J"] / r["E_des_J"]
    assert_true(abs(cop_check - r["thermal_COP"]) < 1e-6, "COP consistent with energies")


def test_cyclic_bed_temperature():
    print("\n[Test 9] Bed swings between cool and hot temperatures")
    _, cm = make_model()
    r = cm.predict({"T_hot": 85, "T_cool": 30, "n_cycles": 8})
    T_des_max = r["T_bed_des"].max() - 273.15
    T_ads_min = r["T_bed_ads"].min() - 273.15
    assert_true(T_des_max > 70, f"desorbing bed reaches {T_des_max:.1f}C (driven toward 85C)")
    assert_true(T_ads_min < 45, f"adsorbing bed cools to {T_ads_min:.1f}C (toward 30C)")
    assert_true(T_des_max > T_ads_min, "bed temperature genuinely cycles")


def test_hot_temp_raises_capacity():
    print("\n[Test 10] Higher driving temperature increases cooling capacity")
    _, cm = make_model()
    q_lo = cm.predict({"T_hot": 70, "n_cycles": 6})["cooling_power_kW"]
    q_hi = cm.predict({"T_hot": 90, "n_cycles": 6})["cooling_power_kW"]
    assert_true(q_hi > q_lo, f"Q_cool(90C)={q_hi:.2f} > Q_cool(70C)={q_lo:.2f} kW")


def test_predict_interface():
    print("\n[Test 11] ComponentModel predict()/get_info() interface")
    _, cm = make_model()
    info = cm.get_info()
    for k in ["component_id", "component_name", "fidelity", "version", "inputs", "outputs"]:
        assert_true(k in info, f"get_info has '{k}'")
    assert_true(info["component_id"] == "EC093", "component_id == EC093")
    r = cm.predict({"n_cycles": 4})
    for key in ["thermal_COP", "cooling_power_kW", "SCP_W_per_kg",
                "T_bed_ads", "w_ads", "Q_evap"]:
        assert_true(key in r, f"predict output has '{key}'")
    assert_true(len(r["t_ads"]) == len(r["w_ads"]), "ads arrays same length")


def test_benchmark():
    print("\n[Test 12] Benchmark: 8-cycle simulation timing")
    _, cm = make_model()
    t0 = time.perf_counter()
    cm.predict({"n_cycles": 8})
    elapsed = time.perf_counter() - t0
    print(f"  8-cycle simulation in {elapsed*1000:.1f} ms")
    assert_true(elapsed < 5.0, "Completes in < 5 s")


if __name__ == "__main__":
    tests = [
        test_psat_monotone,
        test_isotherm_monotone_T,
        test_isotherm_bounds,
        test_ldf_arrhenius,
        test_cop_in_band,
        test_cooling_power_positive,
        test_mass_conservation,
        test_energy_balance_cop,
        test_cyclic_bed_temperature,
        test_hot_temp_raises_capacity,
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
    print(f"EC093 Adsorption Chiller F2a — Results: {passed} passed, {failed} failed")
    print(f"{'='*64}")
    sys.exit(0 if failed == 0 else 1)
