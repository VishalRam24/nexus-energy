"""
EC107 -- Micro-CHP (Stirling-based) -- F2a Physics-Lumped
Test suite: physics sanity (Carnot bound, energy conservation, eff ranges),
warm-up ODE behaviour, edge cases, predict() interface, benchmark timing.
Run with: python3 scripts/test_model.py   (system python3, NOT pytest)
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from model import StirlingCHP_F2a
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
def test_carnot_bound():
    print("\n[Test 1] Electrical eff strictly below Carnot")
    m, _ = make_model()
    eta_C = m.carnot_efficiency()
    ss = m.steady_state(1.0)
    assert_true(0.0 < eta_C < 1.0, f"eta_Carnot={eta_C:.4f} in (0,1)")
    assert_true(ss["eta_elec"] < eta_C, f"eta_elec={ss['eta_elec']:.4f} < eta_C={eta_C:.4f}")
    assert_true(m.indicated_efficiency() < eta_C,
                f"eta_ind={m.indicated_efficiency():.4f} < eta_C={eta_C:.4f}")


def test_efficiency_targets():
    print("\n[Test 2] eta_e ~10-15%, eta_th ~75-85%, eta_total LHV ~0.85-0.98")
    m, _ = make_model()
    ss = m.steady_state(1.0)
    assert_true(0.08 < ss["eta_elec"] < 0.18, f"eta_elec={ss['eta_elec']:.4f} (~0.10-0.15)")
    assert_true(0.70 < ss["eta_th"] < 0.88, f"eta_th={ss['eta_th']:.4f} (~0.80)")
    # Condensing Stirling micro-CHP on an LHV basis recovers flue latent/sensible
    # heat, so total (CHP) efficiency legitimately sits high (~0.90-0.97 LHV);
    # the only floor on losses is residual stack+casing loss, so total stays <1.
    assert_true(0.85 < ss["eta_total"] < 0.98 and ss["eta_total"] < 1.0,
                f"eta_total={ss['eta_total']:.4f} (LHV condensing, in [0.85,0.98) and <1)")


def test_total_lt_one_and_gt_elec():
    print("\n[Test 3] total_eff < 1 and total_eff > electrical_eff")
    m, _ = make_model()
    for load in [0.3, 0.6, 1.0]:
        ss = m.steady_state(load)
        assert_true(ss["eta_total"] < 1.0, f"load={load}: eta_total={ss['eta_total']:.4f} < 1")
        assert_true(ss["eta_total"] > ss["eta_elec"],
                    f"load={load}: total={ss['eta_total']:.4f} > elec={ss['eta_elec']:.4f}")


def test_energy_conservation():
    print("\n[Test 4] Energy balance: P_elec + Q_th + losses = Q_fuel")
    m, _ = make_model()
    ss = m.steady_state(1.0)
    out = ss["P_elec_W"] + ss["Q_th_W"]
    losses = ss["Q_fuel_W"] - out
    assert_true(losses >= -1e-6, f"non-negative losses={losses:.2f} W")
    assert_true(out <= ss["Q_fuel_W"] + 1e-6,
                f"P_elec+Q_th={out:.1f} <= Q_fuel={ss['Q_fuel_W']:.1f}")
    # Internal split must also sum: Q_th = Q_reject + Q_flue_rec
    split = ss["Q_reject_W"] + ss["Q_flue_rec_W"]
    assert_true(abs(split - ss["Q_th_W"]) < 1e-6,
                f"Q_reject+Q_flue_rec={split:.1f} == Q_th={ss['Q_th_W']:.1f}")


def test_power_to_heat_ratio():
    print("\n[Test 5] Power-to-heat ratio low (heat-led CHP)")
    m, _ = make_model()
    ss = m.steady_state(1.0)
    assert_true(0.0 < ss["power_to_heat"] < 0.3,
                f"P:H={ss['power_to_heat']:.4f} (heat-led, <0.3)")


def test_beale_power_order():
    print("\n[Test 6] Beale-number indicated power ~ kW order")
    m, _ = make_model()
    P_ind = m.beale_power()
    assert_true(300.0 < P_ind < 5000.0, f"Beale P_ind={P_ind:.0f} W (sub-kW..few-kW)")
    # Electrical = eta_mech * P_ind should be near the ~1 kWe class
    assert_true(200.0 < m.eta_mech * P_ind < 3000.0,
                f"P_elec(Beale)={m.eta_mech*P_ind:.0f} W ~ 1 kWe class")


def test_warmup_heats_up():
    print("\n[Test 7] Warm-up ODE: cold start heads toward T_hot")
    m, _ = make_model()
    r = m.simulate(1.0, m.T_amb, 10.0, 3600.0)
    T0, Tf = r["temperature"][0], r["temperature"][-1]
    assert_true(Tf > T0 + 100.0, f"T rose from {T0:.1f} to {Tf:.1f} K")
    assert_true(Tf < m.T_hot + 1.0, f"T_final={Tf:.1f} K stays below/at T_hot={m.T_hot:.1f}")
    assert_true(np.all(np.diff(r["temperature"]) >= -1e-6), "Temperature monotonically rising")


def test_warmup_gates_electricity():
    print("\n[Test 8] Electrical output zero when cold, rises with warm-up")
    m, _ = make_model()
    r = m.simulate(1.0, m.T_cold, 10.0, 3600.0)
    assert_true(r["P_elec_W"][0] < 1e-6, f"cold start P_elec[0]={r['P_elec_W'][0]:.4f} ~ 0")
    assert_true(r["P_elec_W"][-1] > r["P_elec_W"][0], "P_elec grows as head heats")
    assert_true(np.all(np.diff(r["warmup_factor"]) >= -1e-9), "warmup_factor non-decreasing")


def test_conservation_during_warmup():
    print("\n[Test 9] eta_total near-flat & bounded during warm-up (energy conserved)")
    m, _ = make_model()
    r = m.simulate(1.0, m.T_amb, 10.0, 1800.0)
    et = r["eta_total"][r["eta_total"] > 0]
    assert_true(np.all(et < 1.0 + 1e-9), f"all eta_total < 1 (max={et.max():.4f})")
    # Cold work-not-extracted is rejected as heat, so eta_total is nearly
    # constant -- but NOT exactly: while cold the engine does no mechanical
    # work, so the mechanical/alternator loss fraction (1-eta_mech) of the
    # would-be indicated work is delivered as useful heat rather than lost.
    # This makes eta_total slightly HIGHER when cold, by at most
    # eta_elec*(1/eta_mech - 1) ~= 0.106*(1/0.85 - 1) ~= 0.019. Bound the
    # spread by that physical ceiling (plus a small margin).
    ceiling = m.steady_state(1.0)["eta_elec"] * (1.0 / m.eta_mech - 1.0) + 1e-3
    assert_true(et.max() - et.min() < ceiling,
                f"eta_total spread={et.max()-et.min():.2e} < mech-loss ceiling={ceiling:.2e}")


def test_zero_load():
    print("\n[Test 10] Zero burner load -> zero output, no spurious heat")
    m, _ = make_model()
    ss = m.steady_state(0.0)
    assert_true(ss["P_elec_W"] == 0.0 and ss["Q_th_W"] == 0.0, "all powers zero at load=0")
    r = m.simulate(0.0, m.T_amb, 30.0, 600.0)
    assert_true(np.all(r["P_elec_W"] < 1e-9), "no electricity with burner off")
    assert_true(r["temperature"][-1] <= m.T_amb + 1e-6, "cold unit does not self-heat")


def test_predict_interface():
    print("\n[Test 11] ComponentModel predict() interface")
    _, cm = make_model()
    r = cm.predict({"load_fraction": 1.0, "dt": 30.0, "duration_s": 600.0})
    for key in ["t", "temperature", "P_elec_W", "Q_th_W", "eta_elec",
                "eta_th", "eta_total", "warmup_factor", "steady_state"]:
        assert_true(key in r, f"Key '{key}' in output")
    assert_true(len(r["t"]) == len(r["temperature"]) == len(r["P_elec_W"]),
                "Time-series arrays same length")
    info = cm.get_info()
    assert_true(info["component_id"] == "EC107", "get_info reports EC107")


def test_benchmark():
    print("\n[Test 12] Benchmark: 1 h warm-up sim at dt=5 s")
    m, _ = make_model()
    t0 = time.perf_counter()
    m.simulate(1.0, m.T_amb, 5.0, 3600.0)
    elapsed = time.perf_counter() - t0
    print(f"  3600 s warm-up simulation in {elapsed*1000:.1f} ms")
    assert_true(elapsed < 5.0, "Completes in < 5 s")


if __name__ == "__main__":
    tests = [
        test_carnot_bound,
        test_efficiency_targets,
        test_total_lt_one_and_gt_elec,
        test_energy_conservation,
        test_power_to_heat_ratio,
        test_beale_power_order,
        test_warmup_heats_up,
        test_warmup_gates_electricity,
        test_conservation_during_warmup,
        test_zero_load,
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
    print(f"EC107 Stirling micro-CHP F2a -- Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    sys.exit(0 if failed == 0 else 1)
