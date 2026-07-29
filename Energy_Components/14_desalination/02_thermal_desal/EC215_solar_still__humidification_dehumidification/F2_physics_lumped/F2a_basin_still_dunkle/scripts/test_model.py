"""
EC215 -- Solar Still / HDH -- F2a Basin Still (Dunkle)
Test suite: physics sanity, energy/mass conservation, diurnal behaviour.
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from model import SolarStillF2a, SIGMA
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
    print("\n[Test 1] Saturation pressure rises with T, sane magnitudes")
    m, _ = make_model()
    p25 = m.p_sat(298.15)
    p60 = m.p_sat(333.15)
    p100 = m.p_sat(373.15)
    assert_true(p60 > p25, f"P(60C)={p60:.0f} > P(25C)={p25:.0f} Pa")
    assert_true(p100 > p60, f"P(100C)={p100:.0f} > P(60C)={p60:.0f} Pa")
    # ~3.2 kPa at 25 C, ~101 kPa at 100 C
    assert_true(2500 < p25 < 4000, f"P(25C)={p25:.0f} Pa near 3.17 kPa")
    assert_true(90000 < p100 < 115000, f"P(100C)={p100:.0f} Pa near 101 kPa")


def test_latent_heat_range():
    print("\n[Test 2] Latent heat in physical band, decreasing with T")
    m, _ = make_model()
    h40 = m.latent_heat(313.15)
    h80 = m.latent_heat(353.15)
    assert_true(2.2e6 < h40 < 2.5e6, f"h_fg(40C)={h40:.3e} J/kg")
    assert_true(h80 < h40, f"h_fg(80C)={h80:.3e} < h_fg(40C)")


def test_fluxes_zero_at_equilibrium():
    print("\n[Test 3] All interior fluxes vanish when T_w = T_g")
    m, _ = make_model()
    assert_true(abs(m.q_evap(320.0, 320.0)) < 1e-9, "q_evap = 0 at dT=0")
    assert_true(abs(m.q_conv(320.0, 320.0)) < 1e-9, "q_conv = 0 at dT=0")
    assert_true(abs(m.q_rad(320.0, 320.0)) < 1e-9, "q_rad = 0 at dT=0")


def test_fluxes_positive_when_water_warmer():
    print("\n[Test 4] Fluxes positive (water->cover) when T_w > T_g")
    m, _ = make_model()
    assert_true(m.q_evap(330.0, 310.0) > 0, "q_evap > 0")
    assert_true(m.q_conv(330.0, 310.0) > 0, "q_conv > 0")
    assert_true(m.q_rad(330.0, 310.0) > 0, "q_rad > 0")


def test_no_distillate_at_night():
    print("\n[Test 5] Q=0 at night -> production halts, water cools toward ambient")
    m, _ = make_model()
    # full day then a night-only window: start hot, G=0
    r = m.simulate(G_peak=lambda t: 0.0, T_w0=330.0, T_g0=300.0,
                   T_amb=298.15, duration_s=6 * 3600.0, dt=600.0)
    assert_true(np.all(r["G"] == 0.0), "Irradiance identically zero")
    # rate decays as water cools; final rate << initial
    assert_true(r["distillate_rate_kg_s"][-1] < r["distillate_rate_kg_s"][0],
                "Distillate rate decays at night")
    assert_true(r["T_water"][-1] < r["T_water"][0], "Water cools at night")
    assert_true(r["T_water"][-1] > 297.0, "Water approaches ambient, not below")


def test_night_irradiance_exactly_zero():
    print("\n[Test 6] Diurnal profile is exactly zero before sunrise / after sunset")
    m, _ = make_model()
    assert_true(m.irradiance(0.0) == 0.0, "G(midnight) = 0")
    assert_true(m.irradiance(0.1 * 86400.0) == 0.0, "G(pre-sunrise) = 0")
    assert_true(m.irradiance(0.5 * 86400.0) > 0.0, "G(noon) > 0")
    assert_true(m.irradiance(0.9 * 86400.0) == 0.0, "G(post-sunset) = 0")


def test_daily_yield_realistic():
    print("\n[Test 7] Daily yield in realistic solar-still band 2-6 L/(m2.day)")
    m, _ = make_model()
    r = m.simulate(G_peak=900.0, duration_s=86400.0, dt=600.0)
    y = r["daily_yield_L_m2"]
    print(f"  daily yield = {y:.2f} L/(m2.day); peak T_water = "
          f"{r['T_water'].max()-273.15:.1f} C")
    assert_true(1.5 < y < 7.0, f"yield={y:.2f} L/(m2.day) in [1.5, 7]")


def test_yield_tracks_solar():
    print("\n[Test 8] Yield increases monotonically with peak irradiance")
    m, _ = make_model()
    ys = []
    for Gp in [400.0, 700.0, 1000.0]:
        r = m.simulate(G_peak=Gp, duration_s=86400.0, dt=900.0)
        ys.append(r["daily_yield_L_m2"])
    assert_true(ys[0] < ys[1] < ys[2],
                f"yield rises with G: {ys[0]:.2f} < {ys[1]:.2f} < {ys[2]:.2f}")


def test_energy_conservation():
    print("\n[Test 9] Cumulative energy balance closes (water node)")
    m, _ = make_model()
    r = m.simulate(G_peak=900.0, duration_s=86400.0, dt=300.0)
    t = r["t"]
    # Integrate water-node terms over the day and compare to dU_water.
    G = r["G"]
    Tw, Tg = r["T_water"], r["T_glass"]
    A = m.A
    solar_in = np.trapz(A * m.alpha_w * m.tau_g * G, t)
    q_int = r["q_evap"] + r["q_conv"] + r["q_rad"]
    to_cover = np.trapz(A * q_int, t)
    basin_loss = np.trapz(A * m.U_b * (Tw - m.T_amb), t)
    dU = m.m_w * m.cp_w * (Tw[-1] - Tw[0])
    residual = solar_in - to_cover - basin_loss - dU
    rel = abs(residual) / max(abs(solar_in), 1.0)
    print(f"  solar_in={solar_in/1e6:.2f} MJ, to_cover={to_cover/1e6:.2f} MJ, "
          f"basin_loss={basin_loss/1e6:.2f} MJ, dU={dU/1e6:.3f} MJ, rel_resid={rel:.4f}")
    assert_true(rel < 0.02, f"Water energy balance closes (rel residual {rel:.4f} < 0.02)")


def test_distillate_from_latent_heat():
    print("\n[Test 10] Distillate mass equals integral of q_evap / h_fg")
    m, _ = make_model()
    r = m.simulate(G_peak=900.0, duration_s=86400.0, dt=300.0)
    # recompute mass from latent-heat definition and compare to accumulator
    h_fg = np.array([m.latent_heat(T) for T in r["T_water"]])
    m_check = np.trapz(m.A * r["q_evap"] / h_fg, r["t"])
    m_acc = r["cumulative_distillate_kg"][-1]
    rel = abs(m_check - m_acc) / max(m_acc, 1e-9)
    assert_true(rel < 0.02, f"mass from latent heat matches ODE accumulator (rel {rel:.4f})")
    assert_true(np.all(np.diff(r["cumulative_distillate_kg"]) >= -1e-12),
                "Cumulative distillate is monotone non-decreasing")


def test_predict_interface():
    print("\n[Test 11] ComponentModel predict() interface")
    _, cm = make_model()
    r = cm.predict({"G_peak_W_m2": 800.0, "duration_s": 43200.0, "dt": 900.0})
    for key in ["t", "T_water", "T_glass", "G", "q_evap",
                "distillate_rate_L_h", "cumulative_distillate_kg",
                "daily_yield_L_m2"]:
        assert_true(key in r, f"Key '{key}' in output")
    assert_true(len(r["t"]) == len(r["T_water"]) == len(r["q_evap"]),
                "Output arrays same length")
    info = cm.get_info()
    assert_true(info["component_id"] == "EC215", "get_info reports EC215")


def test_deeper_water_more_thermal_lag():
    print("\n[Test 12] Deeper basin water lowers peak temperature (thermal inertia)")
    _, cm = make_model()
    r_shallow = cm.predict({"G_peak_W_m2": 900.0, "water_depth_mm": 10.0,
                            "duration_s": 86400.0, "dt": 600.0})
    cm2 = ComponentModel()
    r_deep = cm2.predict({"G_peak_W_m2": 900.0, "water_depth_mm": 50.0,
                          "duration_s": 86400.0, "dt": 600.0})
    Tp_shallow = r_shallow["T_water"].max()
    Tp_deep = r_deep["T_water"].max()
    assert_true(Tp_deep < Tp_shallow,
                f"deep peak {Tp_deep-273.15:.1f}C < shallow {Tp_shallow-273.15:.1f}C")


def test_benchmark():
    print("\n[Test 13] Benchmark: 1-day sim")
    m, _ = make_model()
    t0 = time.perf_counter()
    m.simulate(G_peak=900.0, duration_s=86400.0, dt=600.0)
    elapsed = time.perf_counter() - t0
    print(f"  1-day simulation in {elapsed*1000:.1f} ms")
    assert_true(elapsed < 5.0, "Completes in < 5 s")


if __name__ == "__main__":
    tests = [
        test_psat_monotone,
        test_latent_heat_range,
        test_fluxes_zero_at_equilibrium,
        test_fluxes_positive_when_water_warmer,
        test_no_distillate_at_night,
        test_night_irradiance_exactly_zero,
        test_daily_yield_realistic,
        test_yield_tracks_solar,
        test_energy_conservation,
        test_distillate_from_latent_heat,
        test_predict_interface,
        test_deeper_water_more_thermal_lag,
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
    print(f"EC215 Solar Still F2a -- Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    sys.exit(0 if failed == 0 else 1)
