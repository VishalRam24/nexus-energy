"""
EC208 -- CO2 Geological Sequestration -- F2a Physics-Lumped Reservoir / Trapping
Test suite: mass conservation, pressure build-up, fracture constraint,
trapping evolution, physics sanity, edge cases, predict() interface, benchmark.
Run with system python3 (NOT pytest):  python3 scripts/test_model.py
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from model import CO2SequestrationF2a, SEC_PER_YEAR
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
def test_mass_conservation():
    print("\n[Test 1] Mass conservation: sum of trapped == injected")
    m, _ = make_model()
    r = m.simulate(P_wellhead_bar=90.0, injection_years=25.0, sim_years=400.0)
    inj = r["injected_cumulative_t"]
    tot = r["M_total_t"]
    rel = np.abs(tot - inj) / (inj + 1e-9)
    assert_true(np.max(rel[inj > 1e6]) < 1e-3,
                f"max rel mass-balance error = {np.max(rel[inj > 1e6]):.2e} < 1e-3")
    assert_true(r["injected_cumulative_t"][-1] > 0.0, "Some CO2 was injected")


def test_pressure_buildup():
    print("\n[Test 2] Reservoir pressure builds up with injection")
    m, _ = make_model()
    r = m.simulate(P_wellhead_bar=90.0, injection_years=30.0, sim_years=300.0)
    P = r["reservoir_pressure_bar"]
    P0 = m.P0 / 1e5
    assert_true(P[0] >= P0 - 1e-6, f"P starts at P0={P0:.1f} bar")
    idx_inj = np.argmin(np.abs(r["t_years"] - 30.0))
    assert_true(P[idx_inj] > P[0] + 0.5,
                f"P rises during injection: {P[idx_inj]:.2f} > {P[0]:.2f} bar")


def test_below_fracture_pressure():
    print("\n[Test 3] Bottomhole pressure stays below fracture pressure")
    m, _ = make_model()
    P_frac = m.fracture_pressure_pa()
    assert_true(P_frac / 1e5 > 0, f"Fracture pressure = {P_frac/1e5:.1f} bar")
    # nominal-case bottomhole below fracture
    r = m.simulate(P_wellhead_bar=80.0, injection_years=30.0, sim_years=100.0)
    assert_true(r["bottomhole_pressure_bar"] <= r["fracture_pressure_bar"] + 1e-6,
                f"P_bh {r['bottomhole_pressure_bar']:.1f} <= P_frac "
                f"{r['fracture_pressure_bar']:.1f} bar")


def test_injection_stops_after_period():
    print("\n[Test 4] Injection rate is zero after the injection period")
    m, _ = make_model()
    r = m.simulate(P_wellhead_bar=90.0, injection_years=20.0, sim_years=200.0)
    after = r["injection_rate_kg_s"][r["t_years"] > 25.0]
    assert_true(np.all(after == 0.0), "No injection after t > injection_years")
    during = r["injection_rate_kg_s"][r["t_years"] < 15.0]
    assert_true(np.any(during > 0.0), "Injection active during the period")


def test_total_mass_monotone():
    print("\n[Test 5] Total stored mass is non-decreasing (no leakage in F2a)")
    m, _ = make_model()
    r = m.simulate(P_wellhead_bar=90.0, injection_years=30.0, sim_years=400.0)
    dM = np.diff(r["M_total_t"])
    assert_true(np.all(dM > -1e-3 * r["M_total_t"][-1]),
                "M_total non-decreasing within tolerance")
    # after injection ends it should be essentially flat
    post = r["M_total_t"][r["t_years"] > 60.0]
    assert_true(np.ptp(post) / (post.mean() + 1e-9) < 1e-3,
                "M_total flat after injection (conservation post-injection)")


def test_trapping_fractions_sum_to_one():
    print("\n[Test 6] Trapping fractions sum to 1 and are in [0,1]")
    m, _ = make_model()
    r = m.simulate(P_wellhead_bar=90.0, injection_years=30.0, sim_years=400.0)
    tf = r["trapping_fraction"]
    total = tf["structural"] + tf["residual"] + tf["solubility"] + tf["mineral"]
    mask = r["M_total_t"] > 1e6
    assert_true(np.allclose(total[mask], 1.0, atol=1e-6), "Fractions sum to 1")
    for k, v in tf.items():
        assert_true(np.all(v >= -1e-9) and np.all(v <= 1.0 + 1e-9),
                    f"{k} fraction in [0,1]")


def test_trapping_security_evolution():
    print("\n[Test 7] Trapping migrates to more secure mechanisms over time")
    m, _ = make_model()
    r = m.simulate(P_wellhead_bar=90.0, injection_years=30.0, sim_years=800.0)
    tf = r["trapping_fraction"]
    # at end of injection: structural/mobile dominant; at end: it should shrink
    idx_inj = np.argmin(np.abs(r["t_years"] - 30.0))
    assert_true(tf["structural"][idx_inj] > tf["structural"][-1],
                f"Structural fraction decreases: {tf['structural'][idx_inj]:.3f} "
                f"-> {tf['structural'][-1]:.3f}")
    # mineral trapping (most secure) grows monotonically as dissolved CO2 carbonates
    assert_true(tf["mineral"][-1] > tf["mineral"][idx_inj],
                f"Mineral trapping grows: {tf['mineral'][idx_inj]:.4f} "
                f"-> {tf['mineral'][-1]:.4f}")
    # combined secure (residual+solubility+mineral) overtakes the free/mobile plume
    secure_inj = (tf["residual"][idx_inj] + tf["solubility"][idx_inj]
                  + tf["mineral"][idx_inj])
    secure_end = tf["residual"][-1] + tf["solubility"][-1] + tf["mineral"][-1]
    assert_true(secure_end > secure_inj,
                f"Secure (residual+sol+min) fraction grows: {secure_inj:.3f} "
                f"-> {secure_end:.3f}")


def test_mobile_declines_post_injection():
    print("\n[Test 8] Mobile (free) CO2 declines after injection stops")
    m, _ = make_model()
    r = m.simulate(P_wellhead_bar=90.0, injection_years=30.0, sim_years=600.0)
    Mm = r["M_mobile_t"]
    idx_peak = np.argmax(Mm)
    assert_true(Mm[-1] < Mm[idx_peak],
                f"Mobile CO2 decays: peak {Mm[idx_peak]:.2e} -> end {Mm[-1]:.2e} kg")
    assert_true(r["t_years"][idx_peak] <= 35.0,
                "Mobile CO2 peaks at/near end of injection")


def test_plume_and_saturation():
    print("\n[Test 9] Plume radius grows with injection; saturation < 1")
    m, _ = make_model()
    r = m.simulate(P_wellhead_bar=90.0, injection_years=30.0, sim_years=200.0)
    rad = r["plume_radius_m"]
    assert_true(rad[0] < rad[np.argmin(np.abs(r["t_years"] - 30.0))],
                "Plume radius grows during injection")
    assert_true(np.all(r["saturation_avg"] >= -1e-9) and
                np.all(r["saturation_avg"] < 1.0),
                "Average CO2 saturation in [0,1)")


def test_injectivity_increases_with_pressure():
    print("\n[Test 10] Higher wellhead pressure -> higher injectivity (Darcy)")
    m, _ = make_model()
    P_res = m.P0
    q_lo = m.injection_rate_kg_s(60.0 * 1e5, P_res)
    q_hi = m.injection_rate_kg_s(120.0 * 1e5, P_res)
    assert_true(q_hi > q_lo, f"q(120 bar)={q_hi:.2f} > q(60 bar)={q_lo:.2f} kg/s")
    # no injection if driving pressure is below reservoir pressure
    q_none = m.injection_rate_kg_s(0.0, P_res + 1e7)
    assert_true(q_none == 0.0, "No injection when P_bh < P_res")


def test_predict_interface():
    print("\n[Test 11] ComponentModel predict()/get_info() interface")
    _, cm = make_model()
    info = cm.get_info()
    for key in ["component_id", "component_name", "fidelity", "version",
                "inputs", "outputs"]:
        assert_true(key in info, f"get_info has '{key}'")
    assert_true(info["component_id"] == "EC208", "component_id == EC208")
    r = cm.predict({"P_wellhead_bar": 85.0, "injection_years": 20.0,
                    "sim_years": 150.0, "n_points": 120})
    for key in ["t_years", "M_mobile_t", "M_residual_t", "M_dissolved_t",
                "M_mineral_t", "M_total_t", "reservoir_pressure_bar",
                "plume_radius_m", "trapping_fraction"]:
        assert_true(key in r, f"predict output has '{key}'")
    assert_true(len(r["t_years"]) == len(r["M_total_t"]) == 120,
                "Output arrays match n_points")
    assert_true(r["success"], "solve_ivp converged")


def test_benchmark():
    print("\n[Test 12] Benchmark: 1000-year sim")
    m, _ = make_model()
    t0 = time.perf_counter()
    m.simulate(P_wellhead_bar=90.0, injection_years=30.0, sim_years=1000.0)
    elapsed = time.perf_counter() - t0
    print(f"  1000-yr simulation in {elapsed*1000:.1f} ms")
    assert_true(elapsed < 5.0, "Completes in < 5 s")


if __name__ == "__main__":
    tests = [
        test_mass_conservation,
        test_pressure_buildup,
        test_below_fracture_pressure,
        test_injection_stops_after_period,
        test_total_mass_monotone,
        test_trapping_fractions_sum_to_one,
        test_trapping_security_evolution,
        test_mobile_declines_post_injection,
        test_plume_and_saturation,
        test_injectivity_increases_with_pressure,
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
    print(f"EC208 CO2 Sequestration F2a -- Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    sys.exit(0 if failed == 0 else 1)
