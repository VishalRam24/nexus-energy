"""
EC132 -- Tidal Stream Turbine -- F2a Physics-Lumped Rotor Dynamics
Test suite: physics sanity (Betz, v^3, energy conservation, rated limiting),
ODE behaviour, edge cases, predict() interface, benchmark timing.
Run: python3 scripts/test_model.py   (NO pytest)
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from model import TidalStreamTurbineF2a, BETZ_LIMIT
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
    print("\n[Test 1] Cp(lambda) <= Betz limit everywhere")
    m, _ = make_model()
    lam = np.linspace(0.0, 12.0, 500)
    cp = m.cp(lam)
    assert_true(np.all(cp <= BETZ_LIMIT + 1e-12),
                f"max Cp={cp.max():.4f} <= Betz={BETZ_LIMIT:.4f}")
    assert_true(np.all(cp >= -1e-12), "Cp >= 0 everywhere")
    assert_true(m.Cp_max <= BETZ_LIMIT, f"design Cp_max={m.Cp_max} <= Betz")


def test_cp_peak_at_lambda_opt():
    print("\n[Test 2] Cp peaks near lambda_opt at Cp_max")
    m, _ = make_model()
    lam = np.linspace(0.1, m.lambda_zero, 2000)
    cp = m.cp(lam)
    lam_peak = lam[np.argmax(cp)]
    assert_true(abs(lam_peak - m.lambda_opt) < 0.3,
                f"peak at lambda={lam_peak:.2f} ~ lambda_opt={m.lambda_opt}")
    assert_true(abs(cp.max() - m.Cp_max) < 1e-3,
                f"peak Cp={cp.max():.4f} ~ Cp_max={m.Cp_max}")
    assert_true(m.cp(m.lambda_zero) < 1e-9, "Cp -> 0 at lambda_zero")
    assert_true(m.cp(0.0) < 1e-9, "Cp = 0 at lambda=0")


def test_power_cubic_in_v():
    print("\n[Test 3] Available power scales as v^3")
    m, _ = make_model()
    p1 = m.power_available(1.0)
    p2 = m.power_available(2.0)
    p3 = m.power_available(3.0)
    assert_true(abs(p2 / p1 - 8.0) < 1e-6, f"P(2)/P(1)={p2/p1:.4f} ~ 8")
    assert_true(abs(p3 / p1 - 27.0) < 1e-6, f"P(3)/P(1)={p3/p1:.4f} ~ 27")


def test_seawater_density():
    print("\n[Test 4] Seawater density hardcoded ~1025 kg/m3 (>> air)")
    m, _ = make_model()
    assert_true(abs(m.rho - 1025.0) < 1.0, f"rho={m.rho} kg/m3")
    assert_true(m.rho / 1.225 > 800.0, f"rho is {m.rho/1.225:.0f}x denser than air")


def test_rated_power_limiting():
    print("\n[Test 5] Electrical power never exceeds rated, even at high v")
    m, _ = make_model()
    omega = m.omega_rated
    for v in [2.5, 3.0, 3.5, 3.9]:
        p = m.electrical_power_w(omega, v)
        assert_true(p <= m.P_rated_w + 1e-6,
                    f"P_elec(v={v})={p/1e3:.1f} kW <= {m.P_rated_w/1e3:.0f} kW")


def test_cut_in_cut_out_gating():
    print("\n[Test 6] Cut-in / cut-out gating zeroes power")
    m, _ = make_model()
    omega = m.omega_rated
    assert_true(m.electrical_power_w(omega, m.v_cut_in - 0.1) == 0.0,
                "P=0 below cut-in")
    assert_true(m.electrical_power_w(omega, m.v_cut_out + 0.1) == 0.0,
                "P=0 above cut-out")
    assert_true(m.electrical_power_w(omega, 2.0) > 0.0,
                "P>0 in operating band")


def test_ode_spins_up():
    print("\n[Test 7] Rotor ODE: from rest the rotor spins up under flow")
    m, _ = make_model()
    r = m.simulate(v_mean=2.5, v_amp=0.0, duration_s=2000.0, dt=20.0, omega0=0.05)
    assert_true(r["omega"][-1] > r["omega"][0],
                f"omega rises {r['omega'][0]:.3f}->{r['omega'][-1]:.3f} rad/s")
    assert_true(r["omega"][-1] < 5.0, "omega stays bounded (no runaway)")


def test_ode_torque_balance_steady():
    print("\n[Test 8] Steady flow -> rotor approaches torque balance")
    m, _ = make_model()
    r = m.simulate(v_mean=2.0, v_amp=0.0, duration_s=8000.0, dt=20.0)
    domega = abs(r["omega"][-1] - r["omega"][-2])
    assert_true(domega < 1e-3, f"near steady: |domega|={domega:.2e} rad/s/step")
    # at steady, T_hydro ~ T_gen
    w = r["omega"][-1]
    Th = m.hydro_torque(w, 2.0)
    Tg = m.gen_torque(w)
    assert_true(abs(Th - Tg) / max(Th, 1.0) < 0.05,
                f"T_hydro={Th:.3e} ~ T_gen={Tg:.3e}")


def test_energy_conservation():
    print("\n[Test 9] Energy conservation: E_elec <= E_available (over a tide)")
    m, _ = make_model()
    r = m.simulate(v_mean=2.0, v_amp=1.0, duration_s=12000.0, dt=60.0)
    e_elec = r["energy_electrical_wh"]
    e_avail = r["energy_available_wh"]
    assert_true(e_elec <= e_avail + 1e-6,
                f"E_elec={e_elec/1e3:.1f} kWh <= E_avail={e_avail/1e3:.1f} kWh")
    eff_overall = e_elec / e_avail
    assert_true(eff_overall < BETZ_LIMIT,
                f"overall extraction {eff_overall:.3f} < Betz {BETZ_LIMIT:.3f}")
    assert_true(e_elec > 0.0, "non-trivial energy extracted")


def test_bidirectional_generation():
    print("\n[Test 10] Bidirectional (ebb+flood): negative flow still generates")
    m, _ = make_model()
    # mean 0, amplitude 2.5 -> flow swings negative (ebb) and positive (flood)
    r = m.simulate(v_mean=0.0, v_amp=2.5, tidal_period_s=20000.0,
                   duration_s=20000.0, dt=60.0)
    assert_true(r["v"].min() >= 0.0, "speed magnitude is non-negative (|v| used)")
    assert_true(r["power_electrical_w"].max() > 0.0,
                "generates during ebb and flood half-cycles")
    # there should be two generation lobes per period -> non-trivial energy
    assert_true(r["energy_electrical_wh"] > 0.0, "net energy over full tide > 0")


def test_predict_interface():
    print("\n[Test 11] ComponentModel predict()/get_info() interface")
    _, cm = make_model()
    info = cm.get_info()
    assert_true(info["component_id"] == "EC132", "component_id EC132")
    assert_true(cm.version == "1.0.0", "version 1.0.0")
    r = cm.predict({"v_mean": 2.0, "v_amp": 0.5, "duration_s": 3000.0, "dt": 60.0})
    for key in ["t", "v", "omega", "cp", "power_electrical_w",
                "capacity_factor", "energy_electrical_wh"]:
        assert_true(key in r, f"key '{key}' in output")
    assert_true(len(r["t"]) == len(r["power_electrical_w"]),
                "time-series arrays same length")
    assert_true(0.0 <= r["capacity_factor"] <= 1.0,
                f"CF={r['capacity_factor']:.3f} in [0,1]")


def test_benchmark():
    print("\n[Test 12] Benchmark: full M2 tidal cycle (~12.4 h) sim")
    m, _ = make_model()
    t0 = time.perf_counter()
    r = m.simulate(v_mean=2.0, v_amp=1.0, tidal_period_s=44712.0,
                   duration_s=44712.0, dt=60.0)
    elapsed = time.perf_counter() - t0
    print(f"  full tidal cycle in {elapsed*1000:.1f} ms, "
          f"CF={r['capacity_factor']:.3f}, "
          f"E={r['energy_electrical_wh']/1000:.0f} kWh")
    assert_true(elapsed < 5.0, "completes in < 5 s")


if __name__ == "__main__":
    tests = [
        test_cp_below_betz,
        test_cp_peak_at_lambda_opt,
        test_power_cubic_in_v,
        test_seawater_density,
        test_rated_power_limiting,
        test_cut_in_cut_out_gating,
        test_ode_spins_up,
        test_ode_torque_balance_steady,
        test_energy_conservation,
        test_bidirectional_generation,
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
    print(f"EC132 Tidal Stream Turbine F2a -- Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    sys.exit(0 if failed == 0 else 1)
