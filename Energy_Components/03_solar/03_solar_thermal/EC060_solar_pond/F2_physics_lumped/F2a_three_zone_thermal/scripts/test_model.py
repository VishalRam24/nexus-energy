"""
EC060 -- Solar Pond (Salinity-Gradient) -- F2a Three-Zone Lumped
Test suite: Beer-Lambert optics, energy conservation, multi-day charging,
loss terms, night behaviour, edge cases, predict() interface, benchmark.
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from model import SolarPondF2a
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
def test_beer_lambert_attenuation():
    print("\n[Test 1] Beer-Lambert: radiation attenuates with depth")
    m, _ = make_model()
    f_surface = m.transmitted_fraction(0.0)
    f_lcz = m.solar_fraction_to_lcz()
    assert_true(0.0 < f_lcz < f_surface <= 1.0,
                f"0 < f_LCZ={f_lcz:.4f} < f_surface={f_surface:.4f} <= 1")
    # Monotonic decay with depth
    depths = np.linspace(0.0, 3.0, 20)
    fr = m.transmitted_fraction(depths)
    assert_true(np.all(np.diff(fr) < 0), "Transmitted fraction strictly decreases with depth")
    # Exponential form check at the LCZ top depth
    expected = m.tau_surface * np.exp(-m.mu * (m.h_ucz + m.h_ncz))
    assert_true(abs(f_lcz - expected) < 1e-12, "f_LCZ matches exp(-mu*depth) law")


def test_thicker_ncz_blocks_more_light():
    print("\n[Test 2] Thicker NCZ -> less light reaches LCZ (more insulation)")
    import copy
    _, cm = make_model()
    base = cm._model.solar_fraction_to_lcz()
    unit = copy.deepcopy(cm._raw["unit"])
    unit["h_ncz"]["value"] = 2.0
    m2 = SolarPondF2a({"unit": unit})
    thick = m2.solar_fraction_to_lcz()
    assert_true(thick < base, f"f_LCZ(NCZ=2.0)={thick:.4f} < f_LCZ(NCZ=1.2)={base:.4f}")


def test_lcz_heats_over_days():
    print("\n[Test 3] LCZ storage heats up over multiple days")
    _, cm = make_model()
    r = cm.predict({"G": 250.0, "T_lcz_init": 20.0, "T_amb": 20.0,
                    "duration_days": 30.0, "dt_hours": 6.0})
    assert_true(r["T_lcz"][-1] > r["T_lcz"][0] + 10.0,
                f"LCZ warmed {r['T_lcz'][0]:.1f}->{r['T_lcz'][-1]:.1f} C over 30 d")
    assert_true(np.all(np.diff(r["T_lcz"]) > -1e-6),
                "LCZ temperature monotonically rising while charging (no extraction)")


def test_lcz_reaches_storage_band():
    print("\n[Test 4] LCZ reaches realistic 70-90 C storage band at steady state")
    _, cm = make_model()
    # 200 W/m2 daily-mean global insolation -> documented operating band
    r = cm.predict({"G": 200.0, "T_lcz_init": 20.0, "T_amb": 20.0,
                    "duration_days": 400.0, "dt_hours": 12.0})
    T_ss = float(r["T_lcz"][-1])
    assert_true(70.0 <= T_ss <= 95.0, f"steady LCZ={T_ss:.1f} C in [70, 95] (Tabor 1981)")
    assert_true(r["T_lcz"][-1] > r["T_ucz"][-1] + 30.0,
                "LCZ much hotter than UCZ surface zone")


def test_energy_conservation():
    print("\n[Test 5] Energy conservation: integ(net flux) == stored energy change")
    m, cm = make_model()
    r = cm.predict({"G": 250.0, "T_lcz_init": 20.0, "T_amb": 20.0,
                    "duration_days": 20.0, "dt_hours": 1.0})
    net = r["Q_solar_W"] - r["Q_ncz_W"] - r["Q_ground_W"] - r["Q_extract_W"]
    dE_int = np.trapezoid(net, r["t"])
    dE_state = m.C_lcz * (r["T_lcz"][-1] - r["T_lcz"][0])
    rel = abs(dE_int - dE_state) / abs(dE_state)
    assert_true(rel < 1e-3, f"LCZ energy balance closes, rel_err={rel:.2e}")


def test_solar_zero_at_night():
    print("\n[Test 6] Q_solar = 0 at night (diurnal profile)")
    m, cm = make_model()
    r = cm.predict({"G": 700.0, "T_lcz_init": 75.0, "T_amb": 20.0,
                    "duration_days": 3.0, "dt_hours": 0.5, "diurnal": True})
    assert_true(np.isclose(r["Q_solar_W"].min(), 0.0),
                f"min Q_solar={r['Q_solar_W'].min():.3e} W (=0 at night)")
    assert_true(r["Q_solar_W"].max() > 0.0, "Daytime Q_solar > 0")
    # At least one sample with exactly zero solar input
    assert_true(np.any(r["G"] == 0.0), "Night samples have G=0")


def test_loss_terms_signs():
    print("\n[Test 7] Loss terms positive when LCZ hot, increase with temperature")
    m, _ = make_model()
    q_ncz_warm = m.Q_ncz_path(80.0, 20.0)
    q_ncz_hot = m.Q_ncz_path(90.0, 20.0)
    q_ground = m.Q_ground(80.0)
    assert_true(q_ncz_warm > 0 and q_ground > 0, "NCZ and ground losses positive when hot")
    assert_true(q_ncz_hot > q_ncz_warm, "NCZ loss increases with LCZ temperature")
    assert_true(m.Q_top(40.0, 20.0) > 0, "Top loss positive when UCZ above ambient")


def test_heat_extraction_cools():
    print("\n[Test 8] Heat extraction lowers steady-state LCZ temperature")
    _, cm = make_model()
    r0 = cm.predict({"G": 250.0, "T_lcz_init": 90.0, "T_amb": 20.0,
                     "duration_days": 60.0, "dt_hours": 12.0, "Q_extract_W": 0.0})
    rE = cm.predict({"G": 250.0, "T_lcz_init": 90.0, "T_amb": 20.0,
                     "duration_days": 60.0, "dt_hours": 12.0, "Q_extract_W": 4.0e5})
    assert_true(rE["T_lcz"][-1] < r0["T_lcz"][-1],
                f"extraction cools: {rE['T_lcz'][-1]:.1f} < {r0['T_lcz'][-1]:.1f} C")


def test_night_cooldown():
    print("\n[Test 9] With no sun, a hot pond cools (loses heat)")
    m, cm = make_model()
    r = cm.predict({"G": 0.0, "T_lcz_init": 85.0, "T_amb": 15.0,
                    "duration_days": 5.0, "dt_hours": 2.0})
    assert_true(r["T_lcz"][-1] < r["T_lcz"][0],
                f"LCZ cools in dark: {r['T_lcz'][0]:.1f}->{r['T_lcz'][-1]:.1f} C")
    assert_true(r["T_lcz"][-1] > m.T_ground,
                "LCZ stays above ground sink temperature")


def test_predict_interface():
    print("\n[Test 10] ComponentModel predict() interface")
    _, cm = make_model()
    r = cm.predict({"G": 200.0, "duration_days": 2.0, "dt_hours": 3.0})
    for key in ["t", "t_days", "T_lcz", "T_ucz", "Q_solar_W", "Q_ncz_W",
                "Q_ground_W", "Q_top_W", "Q_extract_W", "f_lcz"]:
        assert_true(key in r, f"Key '{key}' in output")
    assert_true(len(r["t"]) == len(r["T_lcz"]) == len(r["Q_solar_W"]),
                "All time arrays same length")
    info = cm.get_info()
    assert_true(info["component_id"] == "EC060" and info["version"] == "1.0.0",
                "get_info() metadata correct")


def test_benchmark():
    print("\n[Test 11] Benchmark: 30-day simulation")
    _, cm = make_model()
    t0 = time.perf_counter()
    cm.predict({"G": 250.0, "T_lcz_init": 20.0, "T_amb": 20.0,
                "duration_days": 30.0, "dt_hours": 1.0})
    elapsed = time.perf_counter() - t0
    print(f"  30-day sim in {elapsed*1000:.1f} ms")
    assert_true(elapsed < 5.0, "Completes in < 5 s")


def _raw_unit(cm):
    return {k: v for k, v in cm._raw["unit"].items()}


if __name__ == "__main__":
    tests = [
        test_beer_lambert_attenuation,
        test_thicker_ncz_blocks_more_light,
        test_lcz_heats_over_days,
        test_lcz_reaches_storage_band,
        test_energy_conservation,
        test_solar_zero_at_night,
        test_loss_terms_signs,
        test_heat_extraction_cools,
        test_night_cooldown,
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
    print(f"EC060 Solar Pond F2a -- Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    sys.exit(0 if failed == 0 else 1)
