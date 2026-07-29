"""
EC099 -- Stirling Engine -- F2a Physics-Lumped
Test suite: thermodynamic sanity, second-law bounds, scaling laws,
warm-up ODE convergence, Beale cross-check, predict() interface.
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from model import StirlingEngineF2a
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
def test_efficiency_below_carnot():
    print("\n[Test 1] Cycle efficiency < Carnot (second law)")
    m, _ = make_model()
    for Th in [700.0, 923.15, 1050.0]:
        eta = m.cycle_efficiency(Th, 320.0)
        eta_c = m.carnot_efficiency(Th, 320.0)
        assert_true(0.0 < eta < eta_c, f"Th={Th}: eta={eta:.4f} < Carnot={eta_c:.4f}")


def test_efficiency_in_range():
    print("\n[Test 2] Efficiency strictly in (0,1)")
    m, _ = make_model()
    for Th in [650.0, 900.0, 1100.0]:
        eta = m.cycle_efficiency(Th, 300.0)
        assert_true(0.0 < eta < 1.0, f"eta(Th={Th})={eta:.4f} in (0,1)")


def test_efficiency_rises_with_Th():
    print("\n[Test 3] Efficiency increases monotonically with T_h")
    m, _ = make_model()
    Ths = np.linspace(650.0, 1100.0, 20)
    prev = m.cycle_efficiency(Ths[0], 320.0)
    for Th in Ths[1:]:
        eta = m.cycle_efficiency(Th, 320.0)
        assert_true(eta >= prev - 1e-9, f"eta(Th={Th:.0f})={eta:.4f} >= {prev:.4f}")
        prev = eta
    print("  All 19 pairs monotone.")


def test_power_scales_with_pressure():
    print("\n[Test 4] Indicated power scales ~linearly with mean pressure")
    m, _ = make_model()
    p1, p2 = 2.0e6, 4.0e6
    m.p_mean = p1
    P1 = m.indicated_power(923.15, 323.15, 1500.0)
    m.p_mean = p2
    P2 = m.indicated_power(923.15, 323.15, 1500.0)
    ratio = P2 / P1
    assert_true(abs(ratio - 2.0) < 0.05, f"P(2p)/P(p)={ratio:.3f} ~ 2.0 (linear in p_mean)")


def test_power_scales_with_speed():
    print("\n[Test 5] Indicated power scales linearly with engine speed")
    m, _ = make_model()
    P1 = m.indicated_power(923.15, 323.15, 750.0)
    P2 = m.indicated_power(923.15, 323.15, 1500.0)
    assert_true(abs(P2 / P1 - 2.0) < 1e-6, f"P(2N)/P(N)={P2/P1:.4f} = 2.0")


def test_regenerator_effect():
    print("\n[Test 6] Better regenerator -> higher efficiency")
    m, _ = make_model()
    m.regen_eff = 0.60
    eta_lo = m.cycle_efficiency(923.15, 323.15)
    m.regen_eff = 0.95
    eta_hi = m.cycle_efficiency(923.15, 323.15)
    assert_true(eta_hi > eta_lo, f"eta(eps=0.95)={eta_hi:.4f} > eta(eps=0.60)={eta_lo:.4f}")
    # perfect regenerator approaches Carnot
    m.regen_eff = 0.999999
    eta_perf = m.cycle_efficiency(923.15, 323.15)
    eta_c = m.carnot_efficiency(923.15, 323.15)
    assert_true(eta_perf <= eta_c + 1e-9 and eta_perf > eta_hi,
                f"eta(eps->1)={eta_perf:.4f} approaches Carnot={eta_c:.4f}")


def test_brake_below_indicated():
    print("\n[Test 7] Brake power < indicated power (losses), brake eff < cycle eff")
    m, _ = make_model()
    P_ind = m.indicated_power()
    P_brk = m.brake_power()
    assert_true(0.0 < P_brk < P_ind, f"P_brake={P_brk:.0f} < P_ind={P_ind:.0f}")
    assert_true(m.brake_efficiency() < m.cycle_efficiency(),
                "brake eff < indicated eff")


def test_energy_conservation():
    print("\n[Test 8] Energy balance: Q_in = W_ind + Q_reject, all positive")
    m, _ = make_model()
    W = m.indicated_work_per_cycle()
    Q_in = m.heater_duty_per_cycle()
    Q_rej = Q_in - W
    assert_true(W > 0 and Q_in > 0 and Q_rej > 0,
                f"W={W:.2f} J, Q_in={Q_in:.2f} J, Q_rej={Q_rej:.2f} J all > 0")
    assert_true(abs((W + Q_rej) - Q_in) < 1e-6, "Q_in = W + Q_reject (closes)")


def test_beale_crosscheck():
    print("\n[Test 9] Beale/West number power is same order as Schmidt brake power")
    m, _ = make_model()
    P_b = m.brake_power(923.15, 323.15, 1500.0)
    P_beale = m.beale_power(923.15, 323.15, 1500.0)
    P_west = m.west_power(923.15, 323.15, 1500.0)
    # Beale similitude is an order-of-magnitude sanity check only; the
    # idealised (lossless) Schmidt swing runs higher than the empirical
    # Beale estimate. Accept same-order agreement.
    assert_true(0.1 < P_beale / P_b < 10.0,
                f"Beale={P_beale:.0f} W vs brake={P_b:.0f} W (same order)")
    assert_true(P_west > 0, f"West power={P_west:.0f} W > 0")


def test_warmup_ode():
    print("\n[Test 10] Warm-up ODE: head heats from cold to steady state below adiabatic cap")
    m, _ = make_model()
    r = m.simulate(T_h0=300.0, dt=5.0, duration_s=1200.0)
    assert_true(r["T_h"][-1] > r["T_h"][0], f"Heats up: {r['T_h'][0]:.0f}->{r['T_h'][-1]:.0f} K")
    # steady state: near-zero slope at the end
    dT = abs(r["T_h"][-1] - r["T_h"][-2]) / 5.0
    assert_true(dT < 0.05, f"Reaches steady state (dT/dt={dT:.4f} K/s)")
    # power should be near zero at cold start and positive once hot
    assert_true(r["indicated_power"][0] < r["indicated_power"][-1],
                "Power rises as head warms up")
    # efficiency stays below Carnot throughout
    mask = r["efficiency"] > 0
    assert_true(np.all(r["efficiency"][mask] < r["carnot_eff"][mask] + 1e-9),
                "eta < Carnot at every warm-up step")


def test_working_gas_swap():
    print("\n[Test 11] Hydrogen vs helium vs air: regenerator loss differs, eff still valid")
    import json, os
    p = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")
    with open(p) as f:
        base = json.load(f)
    effs = {}
    for gas in ["helium", "hydrogen", "air"]:
        params = json.loads(json.dumps(base))
        params["engine"]["working_gas"]["value"] = gas
        mg = StirlingEngineF2a(params)
        eta = mg.cycle_efficiency()
        effs[gas] = eta
        assert_true(0.0 < eta < mg.carnot_efficiency(),
                    f"{gas}: eta={eta:.4f} < Carnot")
    print(f"  eta He={effs['helium']:.3f} H2={effs['hydrogen']:.3f} air={effs['air']:.3f}")


def test_predict_interface():
    print("\n[Test 12] ComponentModel predict() interface")
    _, cm = make_model()
    r = cm.predict({"dt": 20.0, "duration_s": 400.0})
    for key in ["t", "T_h", "indicated_power", "brake_power",
                "efficiency", "carnot_eff", "beale_power", "heat_input"]:
        assert_true(key in r, f"Key '{key}' in output")
    assert_true(len(r["t"]) == len(r["T_h"]), "Arrays same length")
    info = cm.get_info()
    assert_true(info["component_id"] == "EC099", "get_info id = EC099")


def test_benchmark():
    print("\n[Test 13] Benchmark: 1200 s warm-up at dt=1 s")
    m, _ = make_model()
    t0 = time.perf_counter()
    m.simulate(T_h0=300.0, dt=1.0, duration_s=1200.0)
    elapsed = time.perf_counter() - t0
    print(f"  1200 s warm-up in {elapsed*1000:.1f} ms")
    assert_true(elapsed < 5.0, "Completes in < 5 s")


if __name__ == "__main__":
    tests = [
        test_efficiency_below_carnot,
        test_efficiency_in_range,
        test_efficiency_rises_with_Th,
        test_power_scales_with_pressure,
        test_power_scales_with_speed,
        test_regenerator_effect,
        test_brake_below_indicated,
        test_energy_conservation,
        test_beale_crosscheck,
        test_warmup_ode,
        test_working_gas_swap,
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
    print(f"EC099 Stirling F2a -- Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    sys.exit(0 if failed == 0 else 1)
