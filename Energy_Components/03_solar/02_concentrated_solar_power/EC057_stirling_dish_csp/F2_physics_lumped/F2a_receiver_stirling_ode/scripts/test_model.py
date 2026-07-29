"""
EC057 -- Stirling Dish CSP -- F2a Physics-Lumped
Test suite: receiver-ODE physics sanity, Carnot bound, energy conservation,
P=0 at DNI=0, T^4 radiation, edge cases, predict() interface, benchmark timing.
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from model import StirlingDishF2a
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
def test_radiation_T4():
    print("\n[Test 1] Radiative loss strictly proportional to T_rec^4")
    m, _ = make_model()
    T_amb = 25.0
    T_sky_K = (T_amb + 273.15) - m.T_sky_offset
    sky_emit = m.eps_rec * m.SIGMA * m.A_rec * T_sky_K**4  # constant sky term
    # Emission term alone = Q_rad + sky_emit = eps*sigma*A*T_rec^4 -> must scale T^4.
    T1, T2 = 200.0, (473.15 * 2) - 273.15  # double absolute temperature
    emit1 = m.Q_rad(T1, T_amb) + sky_emit
    emit2 = m.Q_rad(T2, T_amb) + sky_emit
    ratio = emit2 / emit1
    assert_true(abs(ratio - 16.0) < 1e-6, f"Doubling abs T -> emission x{ratio:.4f} (exactly 16 = 2^4)")
    # And the coefficient matches eps*sigma*A exactly.
    coeff = emit1 / (473.15**4)
    assert_true(abs(coeff - m.eps_rec * m.SIGMA * m.A_rec) < 1e-12,
                "Emission coefficient == eps*sigma*A_rec")


def test_efficiency_below_carnot():
    print("\n[Test 2] eta_stirling strictly below Carnot at all temperatures")
    m, _ = make_model()
    for T_rec in [400.0, 600.0, 720.0, 800.0]:
        eta_c = float(m.eta_carnot(T_rec, 25.0))
        eta_s = float(m.eta_stirling(T_rec, 25.0))
        assert_true(eta_s < eta_c, f"T={T_rec}C: eta_stir={eta_s:.3f} < Carnot={eta_c:.3f}")
        assert_true(0.0 < eta_s < 1.0, f"eta_stir={eta_s:.3f} in (0,1)")


def test_zero_dni_zero_power():
    print("\n[Test 3] P_elec = 0 when DNI = 0 (no sun -> no power)")
    m, _ = make_model()
    # Even starting hot, with no sun the engine draw goes to zero (Q_avail<=0).
    P = float(m.power_output_w(720.0, 25.0, 0.0, 0.0))
    assert_true(P == 0.0, f"P_elec={P:.1f} W at DNI=0 (hot start)")
    # Full transient from cold with DNI=0 -> stays ~ambient, zero power throughout.
    r = m.simulate(0.0, 0.0, 25.0, 25.0, 30.0, 1800.0)
    assert_true(np.all(r["P_elec_kw"] == 0.0), "P_elec=0 for entire DNI=0 run")
    # No sun: receiver drifts between ambient (25C) and the radiative sky sink
    # (T_amb - 20K = 5C); it must stay cold (below ambient) and never run.
    assert_true(5.0 <= r["T_rec_c"][-1] <= 25.0,
                f"Receiver cools to sky/ambient band ({r['T_rec_c'][-1]:.1f}C in [5,25])")


def test_receiver_heats_up():
    print("\n[Test 4] Receiver heats from cold start under DNI")
    m, _ = make_model()
    r = m.simulate(900.0, 0.0, 25.0, 25.0, 10.0, 1800.0)
    assert_true(r["T_rec_c"][-1] > 500.0, f"T_rec rises to {r['T_rec_c'][-1]:.1f} C (>500)")
    assert_true(r["T_rec_c"][-1] < 850.0, f"T_rec={r['T_rec_c'][-1]:.1f} C stays < 850 (bounded)")
    assert_true(r["T_rec_c"][-1] > r["T_rec_c"][0], "Monotone net heating from cold")


def test_steady_state_reached():
    print("\n[Test 5] Receiver ODE reaches approximate steady state")
    m, _ = make_model()
    r = m.simulate(900.0, 0.0, 25.0, 25.0, 10.0, 3600.0)
    dT = abs(r["T_rec_c"][-1] - r["T_rec_c"][-2])
    assert_true(dT < 0.05, f"Near SS: dT={dT:.4f} K between last two steps")


def test_energy_conservation():
    print("\n[Test 6] Instantaneous energy balance closes (Q_abs = losses + Q_eng + storage)")
    m, _ = make_model()
    T_rec, T_amb, dni = 600.0, 25.0, 900.0
    Q_abs = float(m.Q_absorbed(dni, 0.0))
    Q_loss = float(m.Q_loss(T_rec, T_amb))
    Q_eng = float(m.Q_engine(T_rec, T_amb, dni, 0.0))
    storage = m.C_rec * float(m.dTdt(T_rec, T_amb, dni, 0.0))  # m*cp*dT/dt
    residual = Q_abs - (Q_loss + Q_eng + storage)
    assert_true(abs(residual) < 1e-6 * max(Q_abs, 1.0),
                f"Balance residual={residual:.3e} W (<< Q_abs={Q_abs:.0f} W)")


def test_power_in_rated_range():
    print("\n[Test 7] Steady-state net power near rated 25 kWe class")
    m, _ = make_model()
    r = m.simulate(900.0, 0.0, 600.0, 25.0, 10.0, 3600.0)
    P = r["P_elec_kw"][-1]
    assert_true(8.0 < P < 30.0, f"Steady P_elec={P:.1f} kW in plausible dish range")
    eta = r["eta_system"][-1]
    assert_true(0.10 < eta < 0.35, f"System efficiency={eta*100:.1f}% (10-35%, dish class)")


def test_higher_dni_more_power():
    print("\n[Test 8] Monotone: higher DNI -> higher steady power")
    m, _ = make_model()
    P_low = m.simulate(500.0, 0.0, 600.0, 25.0, 20.0, 3600.0)["P_elec_kw"][-1]
    P_high = m.simulate(1000.0, 0.0, 600.0, 25.0, 20.0, 3600.0)["P_elec_kw"][-1]
    assert_true(P_high > P_low, f"P(1000)={P_high:.1f} > P(500)={P_low:.1f} kW")


def test_losses_increase_with_temperature():
    print("\n[Test 9] Receiver losses increase with receiver temperature")
    m, _ = make_model()
    q_lo = float(m.Q_loss(400.0, 25.0))
    q_hi = float(m.Q_loss(800.0, 25.0))
    assert_true(q_hi > q_lo, f"Q_loss(800C)={q_hi:.0f} > Q_loss(400C)={q_lo:.0f} W")
    # Radiation should dominate at high T.
    assert_true(m.Q_rad(800.0, 25.0) > m.Q_conv(800.0, 25.0),
                "Radiative loss dominates convective at 800 C")


def test_predict_interface():
    print("\n[Test 10] ComponentModel predict()/get_info() interface")
    _, cm = make_model()
    info = cm.get_info()
    for k in ["component_id", "component_name", "fidelity", "version", "inputs", "outputs"]:
        assert_true(k in info, f"get_info has '{k}'")
    assert_true(info["component_id"] == "EC057", "component_id == EC057")
    r = cm.predict({"DNI": 900.0, "dt": 20.0, "duration_s": 600.0})
    for key in ["t", "T_rec_c", "P_elec_kw", "Q_absorbed_kw", "Q_loss_kw",
                "eta_carnot", "eta_stirling", "eta_system"]:
        assert_true(key in r, f"Output key '{key}' present")
    assert_true(len(r["t"]) == len(r["P_elec_kw"]), "Output arrays same length")


def test_benchmark():
    print("\n[Test 11] Benchmark: 1 h transient at dt=5 s")
    m, _ = make_model()
    t0 = time.perf_counter()
    m.simulate(900.0, 0.0, 25.0, 25.0, 5.0, 3600.0)
    elapsed = time.perf_counter() - t0
    print(f"  3600 s simulation in {elapsed*1000:.1f} ms")
    assert_true(elapsed < 5.0, "Completes in < 5 s")


if __name__ == "__main__":
    tests = [
        test_radiation_T4,
        test_efficiency_below_carnot,
        test_zero_dni_zero_power,
        test_receiver_heats_up,
        test_steady_state_reached,
        test_energy_conservation,
        test_power_in_rated_range,
        test_higher_dni_more_power,
        test_losses_increase_with_temperature,
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
    print(f"EC057 Stirling Dish CSP F2a -- Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    sys.exit(0 if failed == 0 else 1)
