"""
EC067 -- Airborne Wind Energy (AWE) -- F2a Crosswind Pumping-Cycle
Test suite: physics sanity (Loyd bound, energy conservation, positive net
power, monotonicity), edge cases, predict() interface, benchmark timing.
NO pytest -- run as: python3 scripts/test_model.py
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from model import AWE_PumpingCycle_F2a
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
def test_loyd_scaling():
    print("\n[Test 1] Loyd limit scales as v^3 and with (CL/CD)^2")
    m, _ = make_model()
    P5 = m.loyd_power_limit(5.0)
    P10 = m.loyd_power_limit(10.0)
    ratio = P10 / P5
    assert_true(abs(ratio - 8.0) < 1e-6, f"P(10)/P(5)={ratio:.4f} == 2^3=8")
    assert_true(P5 > 0, f"Loyd limit positive: {P5:.1f} W")


def test_power_below_loyd():
    print("\n[Test 2] Reel-out power never exceeds Loyd ideal limit")
    m, _ = make_model()
    for v in [5.0, 8.0, 12.0, 20.0, 28.0]:
        r = m.simulate(v)
        Pmax = np.max(r["P_elec"][r["phase"] > 0]) if np.any(r["phase"] > 0) else 0.0
        assert_true(Pmax <= r["P_loyd_limit"] + 1e-6,
                    f"v={v}: P_peak={Pmax/1000:.1f}kW <= Loyd={r['P_loyd_limit']/1000:.1f}kW")


def test_positive_net_cycle_power():
    print("\n[Test 3] Net cycle power is positive (generates more than it spends)")
    m, _ = make_model()
    for v in [6.0, 10.0, 15.0, 22.0]:
        r = m.simulate(v)
        assert_true(r["P_avg"] > 0, f"v={v}: P_avg={r['P_avg']/1000:.2f} kW > 0")
        assert_true(r["E_net"] > 0, f"v={v}: E_net={r['E_net']/1000:.1f} kJ > 0")


def test_reel_out_exceeds_reel_in():
    print("\n[Test 4] Reel-out energy > reel-in energy (pumping cycle works)")
    m, _ = make_model()
    r = m.simulate(12.0)
    assert_true(r["E_out"] > r["E_in"],
                f"E_out={r['E_out']/1000:.1f}kJ > E_in={r['E_in']/1000:.1f}kJ")


def test_energy_conservation():
    print("\n[Test 5] Energy balance: integral(P dt) == E_net (conservation)")
    m, _ = make_model()
    for v in [8.0, 14.0, 20.0]:
        r = m.simulate(v)
        assert_true(r["energy_residual"] < 1e-6,
                    f"v={v}: residual={r['energy_residual']:.2e} < 1e-6")


def test_duty_cycle_range():
    print("\n[Test 6] Duty cycle in (0,1); reel-out slower => higher duty")
    m, _ = make_model()
    r = m.simulate(10.0)
    assert_true(0.0 < r["duty"] < 1.0, f"duty={r['duty']:.3f} in (0,1)")
    # f_out=0.33 reel-out, f_in=1.5 reel-in => reel-out is slower => duty>0.5
    assert_true(r["duty"] > 0.5, f"reel-out slower => duty={r['duty']:.3f} > 0.5")


def test_tether_ode_endpoints():
    print("\n[Test 7] Tether ODE integrates L_min->L_max->L_min")
    m, _ = make_model()
    r = m.simulate(10.0)
    L = r["L"]
    assert_true(abs(L[0] - m.L_min) < 1e-3, f"L start={L[0]:.2f} == L_min={m.L_min}")
    assert_true(abs(np.max(L) - m.L_max) < 1.0, f"L max={np.max(L):.2f} ~ L_max={m.L_max}")
    assert_true(abs(L[-1] - m.L_min) < 1.0, f"L end={L[-1]:.2f} ~ L_min={m.L_min}")


def test_power_monotone_in_wind():
    print("\n[Test 8] Cycle-average power increases with wind speed (below rated)")
    m, _ = make_model()
    Pavg = [m.simulate(v)["P_avg"] for v in [5.0, 7.0, 9.0, 11.0]]
    for a, b in zip(Pavg[:-1], Pavg[1:]):
        assert_true(b > a, f"P_avg rising: {a/1000:.2f} -> {b/1000:.2f} kW")


def test_tether_drag_penalty():
    print("\n[Test 9] Tether drag lowers effective glide ratio vs bare kite")
    m, _ = make_model()
    G_short = m.effective_glide_ratio(m.L_min)
    G_long = m.effective_glide_ratio(m.L_max)
    G_bare = m.CL / m.CD_kite
    assert_true(G_long < G_short < G_bare,
                f"G(L_max)={G_long:.2f} < G(L_min)={G_short:.2f} < bare={G_bare:.2f}")


def test_cutout_parked():
    print("\n[Test 10] Below cut-in / above cut-out => zero power (parked)")
    m, _ = make_model()
    r_low = m.simulate(2.0)
    r_high = m.simulate(35.0)
    assert_true(r_low["P_avg"] == 0.0, f"v=2: P_avg={r_low['P_avg']} (parked)")
    assert_true(r_high["P_avg"] == 0.0, f"v=35: P_avg={r_high['P_avg']} (parked)")


def test_rated_cap():
    print("\n[Test 11] High wind: instantaneous power capped at rated")
    m, _ = make_model()
    r = m.simulate(28.0)
    Pmax = np.max(r["P_elec"])
    assert_true(Pmax <= m.P_rated + 1e-6,
                f"P_peak={Pmax/1000:.1f}kW <= P_rated={m.P_rated/1000:.0f}kW")


def test_predict_interface():
    print("\n[Test 12] ComponentModel predict()/get_info() interface")
    _, cm = make_model()
    info = cm.get_info()
    for k in ["component_id", "component_name", "fidelity", "version", "inputs", "outputs"]:
        assert_true(k in info, f"get_info has '{k}'")
    assert_true(info["component_id"] == "EC067", "component_id == EC067")
    r = cm.predict({"v_wind": 12.0})
    for key in ["t", "L", "P_elec", "P_avg", "P_loyd_limit", "duty",
                "traction_peak", "E_net", "energy_residual"]:
        assert_true(key in r, f"predict output has '{key}'")
    assert_true(len(r["t"]) == len(r["P_elec"]), "time-series arrays same length")


def test_benchmark():
    print("\n[Test 13] Benchmark: 100 cycle simulations")
    m, _ = make_model()
    t0 = time.perf_counter()
    for _ in range(100):
        m.simulate(10.0, n_eval=100)
    elapsed = time.perf_counter() - t0
    print(f"  100 cycle sims in {elapsed*1000:.1f} ms "
          f"({elapsed*10:.2f} ms each)")
    assert_true(elapsed < 10.0, "100 sims complete in < 10 s")


if __name__ == "__main__":
    tests = [
        test_loyd_scaling,
        test_power_below_loyd,
        test_positive_net_cycle_power,
        test_reel_out_exceeds_reel_in,
        test_energy_conservation,
        test_duty_cycle_range,
        test_tether_ode_endpoints,
        test_power_monotone_in_wind,
        test_tether_drag_penalty,
        test_cutout_parked,
        test_rated_cap,
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
    print(f"EC067 AWE F2a Pumping-Cycle -- Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    sys.exit(0 if failed == 0 else 1)
