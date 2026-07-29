"""
EC066 -- Offshore Floating Wind Turbine -- F2a Aero-Hydro Coupled Model
Test suite: BEM/Betz sanity, ODE conservation & bounds, aero-hydro coupling,
edge cases, predict() interface, benchmark timing. NO pytest.
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from model import FloatingWindF2a, BETZ
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
def test_cp_below_betz():
    print("\n[Test 1] Cp(lambda,beta) never exceeds Betz limit 16/27")
    m, _ = make_model()
    lam = np.linspace(0.5, 18.0, 400)
    for beta in [0.0, 2.0, 5.0, 10.0, 20.0]:
        cp = m.power_coefficient(lam, beta)
        assert_true(np.all(cp <= BETZ + 1e-12),
                    f"max Cp(beta={beta})={cp.max():.4f} <= Betz {BETZ:.4f}")


def test_cp_peak_realistic():
    print("\n[Test 2] Peak Cp matches IEA-15MW (~0.45-0.50), positive somewhere")
    m, _ = make_model()
    lam = np.linspace(0.5, 16.0, 600)
    cp = m.power_coefficient(lam, 0.0)
    cp_peak = cp.max()
    lam_peak = lam[np.argmax(cp)]
    assert_true(0.44 < cp_peak < 0.52, f"peak Cp={cp_peak:.4f} in (0.44,0.52)")
    assert_true(6.0 < lam_peak < 12.0,
                f"optimal TSR={lam_peak:.2f} in (6,12) for large rotor")


def test_cp_pitch_feathering():
    print("\n[Test 3] Blade pitch (feathering) reduces Cp at optimal TSR")
    m, _ = make_model()
    cp0 = m.power_coefficient(m.tsr_opt, 0.0)
    cp15 = m.power_coefficient(m.tsr_opt, 15.0)
    assert_true(cp15 < cp0, f"Cp(beta=15)={cp15:.4f} < Cp(beta=0)={cp0:.4f}")


def test_energy_conservation():
    print("\n[Test 4] Aero power <= kinetic wind flux (Cp<=1 energy bound)")
    m, _ = make_model()
    for V in [5.0, 9.0, 12.0, 20.0]:
        for Om in [0.3, 0.6, 0.79]:
            P = m.aero_power(V, Om, 0.0)
            P_avail = 0.5 * m.rho * m.A * V ** 3
            assert_true(P <= P_avail + 1e-6,
                        f"P_aero({V},{Om})={P/1e6:.2f}MW <= flux {P_avail/1e6:.2f}MW")


def test_thrust_bounded():
    print("\n[Test 5] Thrust coefficient bounded by momentum limit (Ct<=1)")
    m, _ = make_model()
    lam = np.linspace(0.5, 16.0, 200)
    for beta in [0.0, 5.0, 15.0]:
        ct = m.thrust_coefficient(lam, beta)
        assert_true(np.all((ct >= 0) & (ct <= 1.0)),
                    f"Ct(beta={beta}) in [0,1], max={ct.max():.3f}")


def test_rotor_spins_up():
    print("\n[Test 6] Rotor ODE spins up from rest under wind")
    m, _ = make_model()
    r = m.simulate(10.0, dt=0.1, duration_s=120.0, Omega0=0.2)
    assert_true(r["rotor_speed"][-1] > r["rotor_speed"][0],
                f"Omega: {r['rotor_speed'][0]:.3f} -> {r['rotor_speed'][-1]:.3f} rad/s")
    assert_true(r["rotor_speed"][-1] <= m.Omega_rated * 1.25,
                f"rotor speed {r['rotor_speed'][-1]:.3f} <= ~rated {m.Omega_rated:.3f}")


def test_power_bounded_rated():
    print("\n[Test 7] Electrical power capped near rated above rated wind")
    m, cm = make_model()
    r = m.simulate(18.0, dt=0.1, duration_s=120.0)
    P_max = r["power_elec"].max()
    assert_true(P_max <= m.P_rated * 1.02,
                f"P_elec max={P_max/1e6:.2f}MW <= rated {m.P_rated/1e6:.1f}MW")
    assert_true(P_max > 0.5 * m.P_rated,
                f"P_elec max={P_max/1e6:.2f}MW > half rated above rated wind")


def test_platform_motion_bounded():
    print("\n[Test 8] Platform surge & pitch stay bounded (stable spring-damper)")
    m, _ = make_model()
    r = m.simulate(11.0, dt=0.05, duration_s=200.0, H_wave=4.0, T_wave=10.0)
    surge_pk = abs(r["surge"]).max()
    pitch_pk = abs(r["pitch_deg"]).max()
    assert_true(surge_pk < 50.0, f"peak surge={surge_pk:.2f} m bounded (<50)")
    assert_true(pitch_pk < 15.0, f"peak pitch={pitch_pk:.2f} deg bounded (<15)")
    assert_true(np.all(np.isfinite(r["surge"])) and np.all(np.isfinite(r["pitch"])),
                "trajectory finite (no blow-up)")


def test_aero_hydro_coupling():
    print("\n[Test 9] Platform motion modulates relative wind (aero-hydro link)")
    m, _ = make_model()
    calm = m.simulate(11.0, dt=0.05, duration_s=150.0, H_wave=0.0)
    rough = m.simulate(11.0, dt=0.05, duration_s=150.0, H_wave=5.0, T_wave=9.0)
    # waves drive platform motion -> V_rel fluctuates more than in calm sea
    var_calm = np.var(calm["V_rel"][-1000:])
    var_rough = np.var(rough["V_rel"][-1000:])
    assert_true(var_rough > var_calm,
                f"V_rel variance: rough {var_rough:.4f} > calm {var_calm:.4f}")
    # V_rel must differ from free wind because of platform velocity coupling
    dev = np.max(np.abs(rough["V_rel"] - rough["V_wind"]))
    assert_true(dev > 1e-3, f"max |V_rel - V_wind|={dev:.4f} m/s > 0 (coupled)")


def test_cut_in_out():
    print("\n[Test 10] No power below cut-in / above cut-out")
    m, _ = make_model()
    r_low = m.simulate(2.0, dt=0.2, duration_s=30.0)
    r_high = m.simulate(28.0, dt=0.2, duration_s=30.0)
    assert_true(np.all(r_low["power_elec"] == 0.0), "P=0 below cut-in (2 m/s)")
    assert_true(np.all(r_high["power_elec"] == 0.0), "P=0 above cut-out (28 m/s)")


def test_predict_interface():
    print("\n[Test 11] ComponentModel predict() interface & get_info()")
    _, cm = make_model()
    info = cm.get_info()
    assert_true(info["component_id"] == "EC066", "component_id EC066")
    assert_true(abs(info["betz_limit"] - 16.0 / 27.0) < 1e-9, "Betz limit reported")
    r = cm.predict({"wind_speed_ms": 9.0, "dt": 0.2, "duration_s": 20.0,
                    "wave_height_m": 2.0})
    for key in ["t", "rotor_speed", "cp", "V_rel", "power_elec",
                "thrust", "surge", "pitch_deg", "power_elec_mean_MW"]:
        assert_true(key in r, f"Key '{key}' in output")
    assert_true(len(r["t"]) == len(r["power_elec"]), "Arrays same length")
    assert_true(r["cp_max"] <= BETZ + 1e-9, f"reported cp_max {r['cp_max']:.4f} <= Betz")


def test_benchmark():
    print("\n[Test 12] Benchmark: 120 s coupled sim at dt=0.05")
    m, _ = make_model()
    t0 = time.perf_counter()
    m.simulate(10.59, dt=0.05, duration_s=120.0, H_wave=3.0, T_wave=10.0)
    elapsed = time.perf_counter() - t0
    print(f"  120 s aero-hydro simulation in {elapsed*1000:.1f} ms")
    assert_true(elapsed < 5.0, "Completes in < 5 s")


if __name__ == "__main__":
    tests = [
        test_cp_below_betz,
        test_cp_peak_realistic,
        test_cp_pitch_feathering,
        test_energy_conservation,
        test_thrust_bounded,
        test_rotor_spins_up,
        test_power_bounded_rated,
        test_platform_motion_bounded,
        test_aero_hydro_coupling,
        test_cut_in_out,
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
    print(f"EC066 Floating Wind F2a -- Results: {passed} passed, {failed} failed")
    print(f"{'='*64}")
    sys.exit(0 if failed == 0 else 1)
