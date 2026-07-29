"""
EC203 -- Membrane-Based CO2 Separation -- F2a Solution-Diffusion Cross-Flow
Test suite: mass conservation, purity-recovery tradeoff, selectivity & pressure
ratio effects, ODE behaviour, edge cases, predict() interface, benchmark.
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from model import MembraneF2a
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
    print("\n[Test 1] Mass conservation: feed = retentate + permeate")
    m, _ = make_model()
    F_in = m.F_feed
    for A in [10.0, 50.0, 200.0]:
        r = m.simulate(area_m2=A)
        co2_in = F_in * m.y_feed
        n2_in = F_in * (1.0 - m.y_feed)
        co2_bal = r["F_CO2_retentate"][-1] + r["cum_permeate_CO2"][-1]
        tot_bal = r["retentate_flow_mol_s"] + r["permeate_flow_mol_s"]
        assert_true(abs(co2_bal - co2_in) < 1e-6 * co2_in + 1e-9,
                    f"A={A}: CO2 balance {co2_bal:.6e} == feed {co2_in:.6e}")
        assert_true(abs(tot_bal - F_in) < 1e-6 * F_in + 1e-9,
                    f"A={A}: total mol balance {tot_bal:.6e} == feed {F_in:.6e}")


def test_purity_recovery_tradeoff():
    print("\n[Test 2] Purity-recovery tradeoff (more area -> +recovery, -purity)")
    m, _ = make_model()
    areas = [5.0, 20.0, 50.0, 100.0, 200.0, 500.0]
    recs, purs = [], []
    for A in areas:
        r = m.simulate(area_m2=A)
        recs.append(r["recovery"])
        purs.append(r["purity"])
    # recovery monotonically increases with area
    for i in range(1, len(recs)):
        assert_true(recs[i] >= recs[i - 1] - 1e-6,
                    f"recovery rises: {recs[i]:.3f} >= {recs[i-1]:.3f} (A={areas[i]})")
    # purity monotonically falls as recovery rises
    for i in range(1, len(purs)):
        assert_true(purs[i] <= purs[i - 1] + 1e-6,
                    f"purity falls: {purs[i]:.3f} <= {purs[i-1]:.3f} (A={areas[i]})")
    print(f"  recovery {recs[0]*100:.1f}%->{recs[-1]*100:.1f}%, "
          f"purity {purs[0]*100:.1f}%->{purs[-1]*100:.1f}%")


def test_enrichment():
    print("\n[Test 3] Permeate enriched in CO2 above feed (selective membrane)")
    m, _ = make_model()
    r = m.simulate(area_m2=20.0)
    assert_true(r["purity"] > m.y_feed,
                f"permeate purity {r['purity']:.3f} > feed {m.y_feed:.3f}")
    # retentate is depleted in CO2 below feed
    assert_true(r["retentate_CO2_fraction"] < m.y_feed,
                f"retentate {r['retentate_CO2_fraction']:.3f} < feed {m.y_feed:.3f}")


def test_selectivity_effect():
    print("\n[Test 4] Higher selectivity -> higher permeate purity")
    base, _ = make_model()
    r_lo = base.simulate(area_m2=50.0)
    m_hi, _ = make_model()
    m_hi.alpha = 200.0
    m_hi.Q_N2 = m_hi.Q_CO2 / m_hi.alpha
    r_hi = m_hi.simulate(area_m2=50.0)
    assert_true(r_hi["purity"] > r_lo["purity"],
                f"alpha=200 purity {r_hi['purity']:.3f} > alpha=50 purity {r_lo['purity']:.3f}")
    # selectivity=1 (no separation) -> permeate == feed
    m_one, _ = make_model()
    m_one.alpha = 1.0
    m_one.Q_N2 = m_one.Q_CO2
    r_one = m_one.simulate(area_m2=50.0)
    assert_true(abs(r_one["purity"] - m_one.y_feed) < 0.02,
                f"alpha=1 gives no separation: purity {r_one['purity']:.3f} ~ feed {m_one.y_feed:.3f}")


def test_pressure_ratio_ceiling():
    print("\n[Test 5] Permeate fraction bounded by pressure-ratio ceiling")
    m, _ = make_model()
    phi = m.pressure_ratio
    x = m.y_feed
    y = m.local_permeate_fraction(x)
    assert_true(y <= phi * x + 1e-9,
                f"y={y:.3f} <= phi*x={phi*x:.3f} (enrichment ceiling)")
    # With tiny pressure ratio, purity is strongly limited even at high alpha
    m_lowpr, _ = make_model()
    m_lowpr.p_perm = m_lowpr.p_feed / 1.5   # phi = 1.5
    m_lowpr.alpha = 500.0
    m_lowpr.Q_N2 = m_lowpr.Q_CO2 / m_lowpr.alpha
    y_lim = m_lowpr.local_permeate_fraction(m_lowpr.y_feed)
    assert_true(y_lim <= 1.5 * m_lowpr.y_feed + 1e-9,
                f"low PR caps purity: y={y_lim:.3f} <= {1.5*m_lowpr.y_feed:.3f} despite alpha=500")


def test_local_permeate_consistency():
    print("\n[Test 6] Local permeate fraction satisfies flux-ratio identity")
    m, _ = make_model()
    for x in [0.05, 0.13, 0.4, 0.8]:
        y = m.local_permeate_fraction(x)
        J_co2 = m.Q_CO2 * max(m.p_feed * x - m.p_perm * y, 0.0)
        J_n2 = m.Q_N2 * max(m.p_feed * (1 - x) - m.p_perm * (1 - y), 0.0)
        y_check = J_co2 / (J_co2 + J_n2) if (J_co2 + J_n2) > 0 else 0.0
        # consistent unless clipped by the pressure-ratio ceiling
        ceil_active = abs(y - m.pressure_ratio * x) < 1e-9
        assert_true(ceil_active or abs(y - y_check) < 1e-3,
                    f"x={x}: y={y:.4f} matches flux ratio {y_check:.4f}")


def test_recovery_bounds():
    print("\n[Test 7] Recovery and purity in [0,1]; recovery->1 as area->inf")
    m, _ = make_model()
    r_small = m.simulate(area_m2=1.0)
    r_huge = m.simulate(area_m2=5000.0)
    for r in (r_small, r_huge):
        assert_true(0.0 <= r["recovery"] <= 1.0 + 1e-9, f"recovery {r['recovery']:.3f} in [0,1]")
        assert_true(0.0 <= r["purity"] <= 1.0 + 1e-9, f"purity {r['purity']:.3f} in [0,1]")
    assert_true(r_huge["recovery"] > 0.95,
                f"huge area near-complete recovery: {r_huge['recovery']:.3f}")
    assert_true(r_huge["recovery"] > r_small["recovery"], "recovery grows with area")


def test_area_sizing_roundtrip():
    print("\n[Test 8] area_for_recovery root-find reproduces target recovery")
    m, _ = make_model()
    for target in [0.5, 0.8, 0.9]:
        A = m.area_for_recovery(target)
        r = m.simulate(area_m2=A)
        assert_true(abs(r["recovery"] - target) < 0.02,
                    f"target={target}: A={A:.1f} m2 -> recovery {r['recovery']:.3f}")


def test_two_stage_boosts_purity():
    print("\n[Test 9] Two-stage cascade raises purity above single stage")
    m, _ = make_model()
    s1 = m.simulate(area_m2=100.0)
    ts = m.two_stage(area1_m2=100.0, area2_m2=30.0)
    assert_true(ts["final_purity"] > s1["purity"],
                f"two-stage purity {ts['final_purity']:.3f} > single {s1['purity']:.3f}")
    assert_true(0.0 <= ts["overall_recovery"] <= 1.0 + 1e-9,
                f"overall recovery {ts['overall_recovery']:.3f} in [0,1]")


def test_predict_interface():
    print("\n[Test 10] ComponentModel predict() interface")
    _, cm = make_model()
    r = cm.predict({"area_m2": 50.0, "n_eval": 100})
    for key in ["recovery", "purity", "stage_cut", "area", "permeate_purity",
                "pressure_ratio", "retentate_CO2_fraction"]:
        assert_true(key in r, f"Key '{key}' in output")
    assert_true(len(r["area"]) == len(r["permeate_purity"]), "Profile arrays same length")
    info = cm.get_info()
    assert_true(info["component_id"] == "EC203", "get_info component_id == EC203")


def test_monotone_area_profile():
    print("\n[Test 11] Cumulative permeate CO2 monotonically increases along area")
    m, _ = make_model()
    r = m.simulate(area_m2=200.0, n_eval=100)
    perm = r["cum_permeate_CO2"]
    diffs = np.diff(perm)
    assert_true(np.all(diffs >= -1e-12), "cumulative permeate CO2 non-decreasing")
    ret = r["retentate_x_CO2"]
    assert_true(ret[-1] <= ret[0] + 1e-9,
                f"retentate CO2 fraction depletes: {ret[-1]:.3f} <= {ret[0]:.3f}")


def test_benchmark():
    print("\n[Test 12] Benchmark: single-stage solve_ivp module integration")
    m, _ = make_model()
    t0 = time.perf_counter()
    for _ in range(10):
        m.simulate(area_m2=200.0, n_eval=200)
    elapsed = time.perf_counter() - t0
    print(f"  10 module integrations in {elapsed*1000:.1f} ms "
          f"({elapsed*100:.2f} ms each)")
    assert_true(elapsed < 5.0, "10 integrations in < 5 s")


if __name__ == "__main__":
    tests = [
        test_mass_conservation,
        test_purity_recovery_tradeoff,
        test_enrichment,
        test_selectivity_effect,
        test_pressure_ratio_ceiling,
        test_local_permeate_consistency,
        test_recovery_bounds,
        test_area_sizing_roundtrip,
        test_two_stage_boosts_purity,
        test_predict_interface,
        test_monotone_area_profile,
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
    print(f"EC203 Membrane CO2 F2a -- Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    sys.exit(0 if failed == 0 else 1)
