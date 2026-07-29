"""
EC084 -- Aquifer Thermal Energy Storage (ATES) -- F2a Doublet-Well Physics-Lumped
Test suite: energy conservation, recovery efficiency bounds, monotonicity,
edge cases, predict() interface, benchmark timing.  (NO pytest -- custom harness.)
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from model import ATES_F2a
from predict import ComponentModel

PASS = "✓"
FAIL = "✗"


def assert_true(cond, msg):
    if cond:
        print(f"  {PASS}  {msg}")
    else:
        print(f"  {FAIL}  FAILED: {msg}")
        raise AssertionError(msg)


def make_model(loss_tuning=None):
    cm = ComponentModel()
    if loss_tuning is not None:
        cm._raw["unit"]["loss_tuning"]["value"] = loss_tuning
        cm._model = ATES_F2a(cm._raw)
    return cm._model, cm


# ---------------------------------------------------------------------------
def test_thermal_radius():
    print("\n[Test 1] Doughty thermal radius scales as sqrt(V)")
    m, _ = make_model()
    R1 = m.thermal_radius(50000.0)
    R2 = m.thermal_radius(200000.0)  # 4x volume -> 2x radius
    assert_true(10.0 < R1 < 60.0, f"R_th(50k m3)={R1:.2f} m physically reasonable")
    assert_true(abs(R2 / R1 - 2.0) < 1e-6, f"4x V gives 2x R: {R2:.2f}/{R1:.2f}")


def test_energy_conservation_no_loss():
    print("\n[Test 2] Zero-loss limit: E_extracted == E_injected (conservation)")
    m, _ = make_model(loss_tuning=0.0)
    r = m.simulate(n_cycles=1, n_eval_per_season=60)
    rel = abs(r["E_extracted_J"] - r["E_injected_J"]) / r["E_injected_J"]
    assert_true(rel < 1e-3, f"E_out/E_in mismatch={rel:.2e} (lossless)")
    assert_true(abs(r["recovery_efficiency"] - 1.0) < 1e-3,
                f"eta_recovery={r['recovery_efficiency']:.4f} ~ 1.0 lossless")


def test_recovery_below_one():
    print("\n[Test 3] With losses: 0 < eta_recovery < 1 (2nd-law / loss)")
    m, _ = make_model()
    r = m.simulate(n_cycles=3, n_eval_per_season=40)
    eta = r["recovery_efficiency"]
    assert_true(0.0 < eta < 1.0, f"eta_recovery={eta:.4f} strictly in (0,1)")
    assert_true(r["E_extracted_J"] < r["E_injected_J"],
                "E_extracted < E_injected (conductive losses)")


def test_recovery_in_literature_band():
    print("\n[Test 4] Recovery efficiency in ATES literature band ~0.6-0.95")
    m, _ = make_model()
    r = m.simulate(n_cycles=5, n_eval_per_season=40)
    eta1 = r["seasonal_efficiency"][0]
    etaN = r["seasonal_efficiency"][-1]
    assert_true(0.55 <= eta1 <= 0.85,
                f"1st-cycle eta={eta1:.3f} in [0.55,0.85] (Bloemendal 2014)")
    assert_true(0.6 <= etaN <= 0.97,
                f"steady-cycle eta={etaN:.3f} in [0.6,0.97]")


def test_efficiency_improves_with_cycling():
    print("\n[Test 5] Seasonal efficiency rises over cycles (aquifer pre-warms)")
    m, _ = make_model()
    r = m.simulate(n_cycles=5, n_eval_per_season=40)
    se = r["seasonal_efficiency"]
    assert_true(se[-1] >= se[0] - 1e-9,
                f"eta rises: {se[-1]:.3f} >= {se[0]:.3f}")


def test_bubble_temp_below_injection():
    print("\n[Test 6] Mean bubble temperature never exceeds injection temperature")
    m, _ = make_model()
    r = m.simulate(n_cycles=3, n_eval_per_season=40)
    Tmax = np.max(r["T_storage"])
    assert_true(Tmax <= m.T_inj_warm + 1e-6,
                f"max T_storage={Tmax:.3f} <= T_inj={m.T_inj_warm} (no free heat)")
    assert_true(np.min(r["T_storage"]) >= m.T_ground - 1e-6,
                "T_storage >= T_ground always")


def test_stored_energy_signs():
    print("\n[Test 7] Stored energy: builds during charge, drains during discharge")
    m, _ = make_model()
    r = m.simulate(n_cycles=1, n_eval_per_season=60)
    charge = r["E_stored_kWh"][r["mode"] > 0]
    assert_true(charge[-1] > charge[0], "Energy grows during charge")
    assert_true(np.all(r["E_stored_J"] >= -1.0), "Stored energy stays >= 0")


def test_higher_loss_lowers_recovery():
    print("\n[Test 8] Stronger conductive loss lowers recovery efficiency")
    m_lo, _ = make_model(loss_tuning=0.5)
    m_hi, _ = make_model(loss_tuning=2.0)
    e_lo = m_lo.simulate(n_cycles=3, n_eval_per_season=30)["recovery_efficiency"]
    e_hi = m_hi.simulate(n_cycles=3, n_eval_per_season=30)["recovery_efficiency"]
    assert_true(e_hi < e_lo, f"eta(hi loss)={e_hi:.3f} < eta(lo loss)={e_lo:.3f}")


def test_larger_store_recovers_better():
    print("\n[Test 9] Larger seasonal volume -> better recovery (lower surface/vol)")
    m, _ = make_model()
    e_small = m.simulate(n_cycles=3, V_season=10000.0,
                         n_eval_per_season=30)["recovery_efficiency"]
    e_big = m.simulate(n_cycles=3, V_season=200000.0,
                       n_eval_per_season=30)["recovery_efficiency"]
    assert_true(e_big > e_small,
                f"eta(big)={e_big:.3f} > eta(small)={e_small:.3f}")


def test_predict_interface():
    print("\n[Test 10] ComponentModel predict() interface")
    _, cm = make_model()
    r = cm.predict({"n_cycles": 2})
    for key in ["t", "t_days", "T_storage", "E_stored_kWh", "mode",
                "recovery_efficiency", "thermal_radius_m"]:
        assert_true(key in r, f"Key '{key}' in output")
    assert_true(len(r["t"]) == len(r["T_storage"]), "Time arrays same length")
    info = cm.get_info()
    assert_true(info["component_id"] == "EC084", "component_id == EC084")


def test_benchmark():
    print("\n[Test 11] Benchmark: 5-cycle seasonal simulation timing")
    m, _ = make_model()
    t0 = time.perf_counter()
    m.simulate(n_cycles=5, n_eval_per_season=60)
    elapsed = time.perf_counter() - t0
    print(f"  5-cycle simulation in {elapsed*1000:.1f} ms")
    assert_true(elapsed < 5.0, "Completes in < 5 s")


if __name__ == "__main__":
    tests = [
        test_thermal_radius,
        test_energy_conservation_no_loss,
        test_recovery_below_one,
        test_recovery_in_literature_band,
        test_efficiency_improves_with_cycling,
        test_bubble_temp_below_injection,
        test_stored_energy_signs,
        test_higher_loss_lowers_recovery,
        test_larger_store_recovers_better,
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
    print(f"EC084 ATES F2a -- Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    sys.exit(0 if failed == 0 else 1)
