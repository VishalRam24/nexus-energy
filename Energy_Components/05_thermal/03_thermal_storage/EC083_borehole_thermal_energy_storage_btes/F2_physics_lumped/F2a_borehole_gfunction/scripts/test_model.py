"""
EC083 -- Borehole Thermal Energy Storage (BTES) -- F2a Physics-Lumped
Test suite: energy conservation, ground cooling on extraction, seasonal
storage behaviour, g-function limits, edge cases, predict() interface.

Run:  python3 scripts/test_model.py     (system python3, NOT pytest)
"""

import sys
import os
import time
import numpy as np
from scipy.integrate import trapezoid

sys.path.insert(0, os.path.dirname(__file__))
from model import BTES_F2a
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


DAY = 86400.0


# ---------------------------------------------------------------------------
def test_charge_warms_store():
    print("\n[Test 1] Charging (heat injection) warms the ground store")
    m, _ = make_model()
    r = m.simulate(500000.0, T_store0=10.0, T_amb=8.0, duration_s=60 * DAY)
    assert_true(r["success"], "solve_ivp succeeded")
    assert_true(r["T_store"][-1] > 10.0,
                f"T_store rose 10 -> {r['T_store'][-1]:.2f} C")
    # monotone non-decreasing under steady charge
    assert_true(np.all(np.diff(r["T_store"]) > -1e-6),
                "T_store monotonically increasing under steady charge")


def test_discharge_cools_ground():
    print("\n[Test 2] Heat extraction COOLS the ground store")
    m, _ = make_model()
    r = m.simulate(-300000.0, T_store0=40.0, T_amb=8.0, duration_s=60 * DAY)
    assert_true(r["T_store"][-1] < 40.0,
                f"T_store cooled 40 -> {r['T_store'][-1]:.2f} C")
    assert_true(np.all(np.diff(r["T_store"]) < 1e-6),
                "T_store monotonically decreasing under steady discharge")


def test_energy_conservation():
    print("\n[Test 3] Energy conservation: dE_store = (Q_in - Q_loss) dt")
    m, _ = make_model()
    Q = 400000.0
    dur = 90 * DAY
    r = m.simulate(Q, T_store0=10.0, T_amb=8.0, duration_s=dur, n_out=2000)
    E_injected = Q * dur
    E_loss = trapezoid(r["Q_loss"], r["t"])
    dE_store = r["E_stored_J"][-1] - r["E_stored_J"][0]
    residual = E_injected - E_loss - dE_store
    rel = abs(residual) / E_injected
    print(f"  E_in={E_injected/3.6e9:.1f} MWh, loss={E_loss/3.6e9:.1f} MWh, "
          f"dE_store={dE_store/3.6e9:.1f} MWh, rel.resid={rel:.2e}")
    assert_true(rel < 1e-2, f"Energy balance closes to {rel:.2e} (<1%)")


def test_fluid_outlet_charge_vs_discharge():
    print("\n[Test 4] Fluid outlet hotter than store on charge, colder on discharge")
    m, _ = make_model()
    rc = m.simulate(500000.0, T_store0=20.0, T_amb=8.0, duration_s=30 * DAY)
    assert_true(rc["T_out"][-1] > rc["T_store"][-1],
                f"charge: T_out={rc['T_out'][-1]:.2f} > T_store={rc['T_store'][-1]:.2f}")
    rd = m.simulate(-300000.0, T_store0=40.0, T_amb=8.0, duration_s=30 * DAY)
    assert_true(rd["T_out"][-1] < rd["T_store"][-1],
                f"discharge: T_out={rd['T_out'][-1]:.2f} < T_store={rd['T_store'][-1]:.2f}")


def test_Rb_temperature_drop():
    print("\n[Test 5] Borehole resistance R_b sets fluid<->wall temperature drop")
    m, _ = make_model()
    Q = 600000.0
    Tw = 25.0
    Tmean, Tin, Tout = m.fluid_temperatures(Q, Tw)
    q_line = Q / m.L_tot
    expected_drop = q_line * m.R_b
    assert_true(abs((Tmean - Tw) - expected_drop) < 1e-9,
                f"T_fluid_mean - T_wall = q*R_b = {expected_drop:.3f} K")
    # outlet warmer than inlet on charge
    assert_true(Tout > Tin, f"charge: T_out={Tout:.2f} > T_in={Tin:.2f}")


def test_gfunction_monotone_and_bounded():
    print("\n[Test 6] g-function increases with time, saturates at Eskilson plateau")
    m, _ = make_model()
    ts = np.array([1e3, 1e5, 1e7, 1e8, 1e9, 1e10])
    g = m.g_function(ts)
    assert_true(np.all(np.diff(g) >= -1e-9), "g(t) non-decreasing in time")
    g_plateau = np.log(m.H / (2.0 * m.r_b))
    assert_true(g[-1] <= g_plateau + 1e-9,
                f"g saturates at ln(H/2r_b)={g_plateau:.3f} (g_max={g[-1]:.3f})")
    assert_true(g[0] < g[-1], "g grows over the simulated range")


def test_ground_resistance_positive():
    print("\n[Test 7] Ground thermal resistance R_g(t) > 0 and = g/(2 pi k)")
    m, _ = make_model()
    t = 30 * DAY
    Rg = m.ground_resistance(t)
    assert_true(Rg > 0.0, f"R_g={Rg:.4f} (m.K)/W > 0")
    assert_true(abs(Rg - m.g_function(t) / (2 * np.pi * m.k_g)) < 1e-12,
                "R_g = g/(2 pi k_g)")


def test_seasonal_cycle_recovery():
    print("\n[Test 8] Seasonal charge then discharge: store returns toward start")
    m, _ = make_model()
    # 180 d charge, then 180 d discharge of similar magnitude
    def Q(t):
        return 500000.0 if t < 180 * DAY else -500000.0
    r = m.simulate(Q, T_store0=10.0, T_amb=8.0, duration_s=360 * DAY, n_out=720)
    i_peak = np.argmax(r["T_store"])
    T_peak = r["T_store"][i_peak]
    T_end = r["T_store"][-1]
    print(f"  T_start=10.0, T_peak={T_peak:.2f}, T_end={T_end:.2f}")
    assert_true(T_peak > 10.0, "store charged up in summer half")
    assert_true(T_end < T_peak, "store discharged down in winter half")
    assert_true(T_end < T_peak - 2.0, "meaningful seasonal swing recovered")


def test_idle_relaxes_to_ambient():
    print("\n[Test 9] Idle warm store loses heat toward surroundings")
    m, _ = make_model()
    r = m.simulate(0.0, T_store0=40.0, T_amb=8.0, duration_s=365 * DAY)
    assert_true(r["T_store"][-1] < 40.0,
                f"idle store cools 40 -> {r['T_store'][-1]:.2f} C")
    assert_true(r["T_store"][-1] > 8.0,
                f"but stays above ambient (T={r['T_store'][-1]:.2f} > 8 C)")
    assert_true(np.all(r["Q_loss"] >= -1e-6), "Q_loss >= 0 for warm store")


def test_predict_interface():
    print("\n[Test 10] ComponentModel predict() interface keys & shapes")
    _, cm = make_model()
    r = cm.predict({"Q_fluid_W": 500000.0, "duration_s": 30 * DAY, "n_out": 100})
    for key in ["t", "t_days", "T_store", "T_wall", "T_fluid_mean",
                "T_in", "T_out", "Q_fluid", "Q_loss", "E_stored_MWh"]:
        assert_true(key in r, f"key '{key}' present")
    assert_true(len(r["t"]) == len(r["T_store"]) == len(r["T_out"]),
                "all arrays same length")
    info = cm.get_info()
    assert_true(info["component_id"] == "EC083" and info["version"] == "1.0.0",
                "get_info id/version correct")


def test_zero_load_no_drift():
    print("\n[Test 11] Zero load at undisturbed temp: no spurious drift")
    m, _ = make_model()
    r = m.simulate(0.0, T_store0=10.0, T_amb=10.0, duration_s=180 * DAY)
    drift = abs(r["T_store"][-1] - 10.0)
    assert_true(drift < 1e-3, f"no drift (|dT|={drift:.2e} C) when balanced")
    assert_true(abs(r["E_stored_MWh"][-1]) < 1e-3, "E_stored ~ 0")


def test_benchmark():
    print("\n[Test 12] Benchmark: 1-year simulation runtime")
    m, _ = make_model()
    t0 = time.perf_counter()
    m.simulate(500000.0, T_store0=10.0, T_amb=8.0, duration_s=365 * DAY)
    elapsed = time.perf_counter() - t0
    print(f"  1-year sim in {elapsed*1000:.1f} ms")
    assert_true(elapsed < 5.0, "completes in < 5 s")


if __name__ == "__main__":
    tests = [
        test_charge_warms_store,
        test_discharge_cools_ground,
        test_energy_conservation,
        test_fluid_outlet_charge_vs_discharge,
        test_Rb_temperature_drop,
        test_gfunction_monotone_and_bounded,
        test_ground_resistance_positive,
        test_seasonal_cycle_recovery,
        test_idle_relaxes_to_ambient,
        test_predict_interface,
        test_zero_load_no_drift,
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
    print(f"EC083 BTES F2a -- Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    sys.exit(0 if failed == 0 else 1)
