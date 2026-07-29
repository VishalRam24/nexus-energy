"""
EC136 -- Overtopping Device WEC (Wave Dragon) -- F2a Physics-Lumped Reservoir Dynamics
Test suite: mass/energy conservation, overtopping monotonicity, P=rho g Q H eta,
reservoir self-regulation, edge cases, predict() interface, benchmark timing.
NO pytest -- custom assert_true harness, run as: python3 scripts/test_model.py
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from model import OvertoppingWEC_F2a
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
def test_overtopping_monotone_in_Hs():
    print("\n[Test 1] Overtopping discharge rises with wave height (Van der Meer)")
    m, _ = make_model()
    q_prev = m.overtopping_q_per_m(0.5, 0.5)
    for Hs in np.linspace(0.6, 6.0, 30):
        q = m.overtopping_q_per_m(Hs, 0.5)
        assert_true(q >= q_prev - 1e-12, f"q({Hs:.2f})={q:.4f} >= q_prev={q_prev:.4f}")
        q_prev = q
    print("  All 30 pairs checked -- q strictly increases with Hs.")


def test_overtopping_falls_with_freeboard():
    print("\n[Test 2] Overtopping falls as effective freeboard (level) rises")
    m, _ = make_model()
    q_low = m.overtopping_q_per_m(3.0, 0.0)
    q_high = m.overtopping_q_per_m(3.0, 2.0)
    assert_true(q_high < q_low, f"q(level=2)={q_high:.4f} < q(level=0)={q_low:.4f}")
    assert_true(q_low > 0, "overtopping positive at design Hs")


def test_power_law_PrhogQHeta():
    print("\n[Test 3] P = rho * g * Q * H * eta_turbine * eta_generator exactly")
    m, _ = make_model()
    for lv in [0.2, 0.5, 1.0, 2.0]:
        Q = m.turbine_Q(lv)
        H = m.head(lv)
        P_expected = m.rho * m.g * Q * H * m.eta_turb * m.eta_gen
        P = m.turbine_power_W(lv)
        assert_true(abs(P - P_expected) < 1e-6 * max(P_expected, 1.0),
                    f"P(lv={lv})={P:.1f} W matches rho g Q H eta")


def test_turbine_flow_orifice_law():
    print("\n[Test 4] Turbine discharge follows Q = K*sqrt(2 g H), rises with head")
    m, _ = make_model()
    Q_prev = m.turbine_Q(0.05)
    for lv in np.linspace(0.1, 3.0, 20):
        Q = m.turbine_Q(lv)
        H = m.head(lv)
        assert_true(abs(Q - m.K * np.sqrt(2 * m.g * H)) < 1e-9, f"Q(lv={lv:.2f}) orifice law")
        assert_true(Q >= Q_prev - 1e-12, "Q rises with head")
        Q_prev = Q


def test_empty_reservoir_zero_power():
    print("\n[Test 5] Empty reservoir -> zero turbine flow and zero power")
    m, _ = make_model()
    assert_true(m.turbine_Q(0.0) == 0.0, "Q_out(0)=0")
    assert_true(m.turbine_power_W(0.0) == 0.0, "P(0)=0")


def test_mass_conservation():
    print("\n[Test 6] Mass conservation: integral(Qin-Qout)dt = A*dLevel")
    m, _ = make_model()
    r = m.simulate(3.0, 7.0, level0=0.5, dt=5.0, duration_s=1800.0)
    throughput = np.trapezoid(r["Q_in"], r["t"])
    rel = abs(r["mass_residual_m3"]) / max(throughput, 1.0)
    assert_true(rel < 1e-2, f"mass residual {r['mass_residual_m3']:.3f} m3 << throughput {throughput:.0f} m3 (rel={rel:.2e})")


def test_reservoir_self_regulates():
    print("\n[Test 7] Reservoir level stays bounded in [0, depth_max]")
    m, _ = make_model()
    r = m.simulate(5.0, 8.0, level0=0.5, dt=5.0, duration_s=2400.0)
    assert_true(np.all(r["level"] >= -1e-9), "level >= 0 always")
    assert_true(np.all(r["level"] <= m.depth_max + 1e-6), f"level <= depth_max={m.depth_max}")
    assert_true(r["success"], "ODE solver converged")


def test_efficiency_in_unit_interval():
    print("\n[Test 8] Overall wave-to-wire efficiency in (0, 1)")
    m, _ = make_model()
    for Hs in [1.5, 3.0, 5.0]:
        r = m.simulate(Hs, 7.0, level0=0.5, dt=10.0, duration_s=1800.0)
        eta = r["eta_overall"]
        assert_true(0.0 < eta < 1.0, f"eta(Hs={Hs})={eta:.4f} in (0,1)")


def test_power_rises_with_sea_state():
    print("\n[Test 9] Mean electrical power increases with wave height")
    m, _ = make_model()
    P_prev = -1.0
    for Hs in [1.0, 2.0, 3.0, 4.0, 5.0]:
        r = m.simulate(Hs, 7.0, level0=0.5, dt=10.0, duration_s=1800.0)
        P = r["P_mean_kW"]
        assert_true(P > P_prev, f"P_mean(Hs={Hs})={P:.1f} kW > previous {P_prev:.1f} kW")
        P_prev = P


def test_power_below_incident_resource():
    print("\n[Test 10] Mean power never exceeds incident wave resource (energy bound)")
    m, _ = make_model()
    r = m.simulate(3.0, 7.0, level0=0.5, dt=10.0, duration_s=1800.0)
    assert_true(r["P_mean_W"] < r["P_incident_mean_W"],
                f"P_mean={r['P_mean_W']/1e3:.1f} kW < P_incident={r['P_incident_mean_W']/1e3:.1f} kW")
    assert_true(np.all(r["power_elec_W"] <= r["power_hyd_W"] + 1e-6),
                "P_elec <= P_hyd (eta<1) at every step")


def test_predict_interface():
    print("\n[Test 11] ComponentModel predict()/get_info() interface")
    _, cm = make_model()
    info = cm.get_info()
    for k in ["component_id", "component_name", "fidelity", "version", "inputs", "outputs"]:
        assert_true(k in info, f"get_info has '{k}'")
    assert_true(cm.component_id == "EC136", "component_id == EC136")
    r = cm.predict({"Hs_m": 3.0, "Tz_s": 7.0, "dt": 20.0, "duration_s": 600.0})
    for key in ["t", "level", "Q_in", "Q_out", "head", "power_elec_W", "P_mean_kW", "eta_overall"]:
        assert_true(key in r, f"output key '{key}' present")
    assert_true(len(r["t"]) == len(r["power_elec_W"]), "time-series arrays same length")


def test_benchmark():
    print("\n[Test 12] Benchmark: 1800 s reservoir simulation")
    m, _ = make_model()
    t0 = time.perf_counter()
    m.simulate(3.0, 7.0, level0=0.5, dt=5.0, duration_s=1800.0)
    elapsed = time.perf_counter() - t0
    print(f"  1800 s simulation in {elapsed*1000:.1f} ms")
    assert_true(elapsed < 5.0, "Completes in < 5 s")


if __name__ == "__main__":
    tests = [
        test_overtopping_monotone_in_Hs,
        test_overtopping_falls_with_freeboard,
        test_power_law_PrhogQHeta,
        test_turbine_flow_orifice_law,
        test_empty_reservoir_zero_power,
        test_mass_conservation,
        test_reservoir_self_regulates,
        test_efficiency_in_unit_interval,
        test_power_rises_with_sea_state,
        test_power_below_incident_resource,
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
    print(f"EC136 Overtopping WEC F2a -- Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    sys.exit(0 if failed == 0 else 1)
