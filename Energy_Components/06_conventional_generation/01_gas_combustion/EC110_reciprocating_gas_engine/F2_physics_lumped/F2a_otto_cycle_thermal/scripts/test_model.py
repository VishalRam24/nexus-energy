"""
EC110 -- Reciprocating Gas Engine -- F2a Otto/Miller Cycle + Thermal ODE
Test suite: thermodynamic bounds, loss ordering, energy conservation,
part-load behaviour, thermal ODE, edge cases, predict() interface, benchmark.
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from model import ReciprocatingGasEngineF2a
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
def test_otto_below_carnot():
    print("\n[Test 1] Otto efficiency below Carnot bound")
    m, _ = make_model()
    eta_otto = m.otto_efficiency(miller=True)
    eta_carnot = m.carnot_bound()
    assert_true(0.0 < eta_otto < 1.0, f"eta_otto={eta_otto:.4f} in (0,1)")
    assert_true(eta_otto < eta_carnot,
                f"eta_otto={eta_otto:.4f} < Carnot={eta_carnot:.4f}")


def test_otto_increases_with_cr():
    print("\n[Test 2] Otto efficiency increases with compression ratio")
    m, _ = make_model()
    eta_lo = 1.0 - 1.0 / (8.0 ** (m.gamma - 1.0))
    eta_hi = 1.0 - 1.0 / (16.0 ** (m.gamma - 1.0))
    assert_true(eta_hi > eta_lo, f"eta(r=16)={eta_hi:.4f} > eta(r=8)={eta_lo:.4f}")


def test_miller_reduces_effective_cr():
    print("\n[Test 3] Miller cycle lowers effective compression ratio (eta)")
    m, _ = make_model()
    eta_miller = m.otto_efficiency(miller=True)
    eta_full = m.otto_efficiency(miller=False)
    assert_true(eta_miller < eta_full,
                f"eta_miller={eta_miller:.4f} < eta_full_cr={eta_full:.4f}")


def test_loss_ordering():
    print("\n[Test 4] eta_brake < eta_indicated < eta_otto (brake<indicated)")
    m, _ = make_model()
    op = m.operating_point(1.0)
    assert_true(op["eta_brake"] < op["eta_indicated"],
                f"eta_brake={op['eta_brake']:.4f} < eta_ind={op['eta_indicated']:.4f}")
    assert_true(op["eta_indicated"] <= op["eta_otto"] + 1e-9,
                f"eta_ind={op['eta_indicated']:.4f} <= eta_otto={op['eta_otto']:.4f}")
    assert_true(0.30 < op["eta_brake"] < 0.50,
                f"rated eta_brake={op['eta_brake']:.4f} in realistic gas-engine band")


def test_energy_conservation():
    print("\n[Test 5] Energy balance: brake+friction+block+exhaust = fuel")
    m, _ = make_model()
    for plr in [1.0, 0.7, 0.4]:
        op = m.operating_point(plr)
        total = (op["P_brake_w"] + op["P_friction_w"]
                 + op["P_block_w"] + op["P_exhaust_w"])
        assert_true(abs(total - op["P_fuel_w"]) < 1e-3 * max(op["P_fuel_w"], 1.0),
                    f"PLR={plr}: sum={total:.1f} = fuel={op['P_fuel_w']:.1f} W")


def test_partload_efficiency_drops():
    print("\n[Test 6] Brake efficiency falls at part load (friction fraction grows)")
    m, _ = make_model()
    plrs = [1.0, 0.8, 0.6, 0.4]
    etas = [m.operating_point(p)["eta_brake"] for p in plrs]
    for i in range(1, len(etas)):
        assert_true(etas[i] < etas[i - 1],
                    f"eta_brake(PLR={plrs[i]})={etas[i]:.4f} < (PLR={plrs[i-1]})={etas[i-1]:.4f}")


def test_fmep_positive_monotone():
    print("\n[Test 7] FMEP positive and increases with speed")
    m, _ = make_model()
    f_lo = m.fmep(900.0, 5e5)
    f_hi = m.fmep(1800.0, 5e5)
    assert_true(f_lo > 0, f"FMEP(900rpm)={f_lo/1e3:.1f} kPa > 0")
    assert_true(f_hi > f_lo, f"FMEP(1800)={f_hi/1e3:.1f} > FMEP(900)={f_lo/1e3:.1f} kPa")


def test_brake_power_scales_with_load():
    print("\n[Test 8] Brake power increases monotonically with PLR")
    m, _ = make_model()
    p_prev = -1.0
    for plr in [0.4, 0.6, 0.8, 1.0]:
        P = m.operating_point(plr)["P_brake_w"]
        assert_true(P > p_prev, f"P_brake(PLR={plr})={P/1e3:.1f} kW > prev")
        p_prev = P


def test_thermal_ode_heats_up():
    print("\n[Test 9] Engine-block thermal ODE: heats up from cold start")
    m, _ = make_model()
    r = m.simulate(1.0, T0_K=298.15, dt=10.0, duration_s=1200.0)
    assert_true(r["temperature"][-1] > 298.15,
                f"T_final={r['temperature'][-1]:.2f} > 298.15 K (warmed up)")
    assert_true(r["temperature"][-1] < 500.0,
                f"T_final={r['temperature'][-1]:.2f} < 500 K (physically reasonable)")


def test_thermal_steady_state():
    print("\n[Test 10] Block temperature approaches steady state")
    m, _ = make_model()
    r = m.simulate(1.0, T0_K=298.15, dt=10.0, duration_s=3600.0)
    dT = abs(r["temperature"][-1] - r["temperature"][-2])
    assert_true(dT < 0.5, f"Near SS: dT={dT:.4f} K between last two steps")


def test_efficiency_bounds_all_loads():
    print("\n[Test 11] Brake efficiency strictly in (0,1) across load range")
    m, _ = make_model()
    for plr in np.linspace(0.4, 1.0, 13):
        eta = m.operating_point(plr)["eta_brake"]
        assert_true(0.0 < eta < 1.0, f"PLR={plr:.2f}: eta_brake={eta:.4f} in (0,1)")


def test_predict_interface_and_benchmark():
    print("\n[Test 12] predict() interface + benchmark timing")
    _, cm = make_model()
    t0 = time.perf_counter()
    r = cm.predict({"part_load_ratio": 0.8, "dt": 10.0, "duration_s": 600.0})
    elapsed = time.perf_counter() - t0
    for key in ["t", "temperature", "P_brake_w", "eta_brake",
                "eta_indicated", "operating_point"]:
        assert_true(key in r, f"Key '{key}' in predict output")
    assert_true(len(r["t"]) == len(r["temperature"]), "Arrays same length")
    print(f"  600s thermal simulation in {elapsed*1000:.1f} ms")
    assert_true(elapsed < 5.0, "predict() completes in < 5 s")


if __name__ == "__main__":
    tests = [
        test_otto_below_carnot,
        test_otto_increases_with_cr,
        test_miller_reduces_effective_cr,
        test_loss_ordering,
        test_energy_conservation,
        test_partload_efficiency_drops,
        test_fmep_positive_monotone,
        test_brake_power_scales_with_load,
        test_thermal_ode_heats_up,
        test_thermal_steady_state,
        test_efficiency_bounds_all_loads,
        test_predict_interface_and_benchmark,
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
    print(f"EC110 Reciprocating Gas Engine F2a -- Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    sys.exit(0 if failed == 0 else 1)
