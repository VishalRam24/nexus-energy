"""
EC210 -- Electrodialysis (ED) -- F2a Ion-Transport Stack Model
Test suite: ion-transport physics sanity, mass conservation, polarization limits,
edge cases, predict() interface, benchmark timing. Custom harness (no pytest).
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from model import ElectrodialysisF2a, F, R
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
def test_current_efficiency_range():
    print("\n[Test 1] Current efficiency in (0, 1)")
    m, _ = make_model()
    xi = m.current_efficiency()
    assert_true(0.0 < xi < 1.0, f"xi={xi:.4f} in (0,1)")
    assert_true(xi <= m.xi_max + 1e-12, f"xi={xi:.4f} <= xi_max={m.xi_max}")


def test_limiting_current_scales_with_conc():
    print("\n[Test 2] Limiting current density proportional to concentration")
    m, _ = make_model()
    i1 = m.limiting_current_density(50.0)
    i2 = m.limiting_current_density(100.0)
    i3 = m.limiting_current_density(200.0)
    assert_true(i2 > i1 and i3 > i2, "i_lim increases with c")
    # linear: doubling c doubles i_lim
    assert_true(abs(i2 / i1 - 2.0) < 1e-6, f"i_lim(100)/i_lim(50)={i2/i1:.4f} ~ 2")


def test_current_capped_below_limiting():
    print("\n[Test 3] Operating current never exceeds limiting current")
    m, _ = make_model()
    for c in [20.0, 50.0, 100.0, 300.0]:
        i_op = float(m.operating_current_density(500.0, c, 0.8))
        i_lim = float(m.limiting_current_density(c))
        assert_true(i_op <= i_lim + 1e-12, f"c={c}: i_op={i_op:.4e} <= i_lim={i_lim:.4e}")


def test_salt_mass_conservation():
    print("\n[Test 4] Salt conserved between diluate and concentrate (rr=1)")
    m, _ = make_model()
    r = m.simulate(150.0, 100.0, 5.0, 100.0, recovery_ratio=1.0)
    dil_lost = r["c_diluate"][0] - r["c_diluate"][-1]
    con_gain = r["c_concentrate"][-1] - r["c_concentrate"][0]
    assert_true(abs(dil_lost - con_gain) < 1e-6,
                f"diluate lost {dil_lost:.4f} == concentrate gained {con_gain:.4f}")


def test_diluate_decreases_concentrate_increases():
    print("\n[Test 5] Diluate desalts, concentrate enriches along path")
    m, _ = make_model()
    r = m.simulate(200.0, 100.0, 5.0, 100.0)
    cd, cc = r["c_diluate"], r["c_concentrate"]
    assert_true(np.all(np.diff(cd) <= 1e-9), "diluate monotonically non-increasing")
    assert_true(np.all(np.diff(cc) >= -1e-9), "concentrate monotonically non-decreasing")
    assert_true(cd[-1] < cd[0], f"product {cd[-1]:.2f} < feed {cd[0]:.2f}")


def test_more_current_more_removal():
    print("\n[Test 6] Higher current density -> more salt removed (until i_lim)")
    m, _ = make_model()
    r_lo = m.simulate(50.0, 100.0, 5.0, 100.0)
    r_hi = m.simulate(120.0, 100.0, 5.0, 100.0)
    assert_true(r_hi["salt_removed_fraction"] > r_lo["salt_removed_fraction"],
                f"removed {r_hi['salt_removed_fraction']:.3f} > {r_lo['salt_removed_fraction']:.3f}")


def test_slower_flow_more_removal():
    print("\n[Test 7] Longer residence (slower flow) -> more removal")
    m, _ = make_model()
    r_fast = m.simulate(200.0, 100.0, 15.0, 100.0)
    r_slow = m.simulate(200.0, 100.0, 3.0, 100.0)
    assert_true(r_slow["salt_removed_fraction"] > r_fast["salt_removed_fraction"],
                f"slow {r_slow['salt_removed_fraction']:.3f} > fast {r_fast['salt_removed_fraction']:.3f}")


def test_membrane_potential_sign():
    print("\n[Test 8] Concentration potential positive when c_con > c_dil")
    m, _ = make_model()
    E = m.membrane_potential(50.0, 200.0)
    assert_true(E > 0, f"E_mem={E:.4f} V > 0 (c_con>c_dil)")
    E0 = m.membrane_potential(100.0, 100.0)
    assert_true(abs(E0) < 1e-9, f"E_mem=0 when c_con==c_dil ({E0:.2e})")


def test_stack_voltage_positive_and_scales():
    print("\n[Test 9] Stack voltage positive and = N * cell-pair voltage")
    m, _ = make_model()
    r = m.simulate(200.0, 100.0, 5.0, 100.0)
    assert_true(np.all(r["stack_voltage"] > 0), "U_stack > 0 everywhere")
    ratio = r["stack_voltage"][0] / r["cell_pair_voltage"][0]
    assert_true(abs(ratio - m.N) < 1e-6, f"U_stack/U_pair={ratio:.1f} == N={m.N}")


def test_sec_physical_range():
    print("\n[Test 10] SEC in physical brackish-ED range (0.05-3 kWh/m3)")
    m, _ = make_model()
    r = m.simulate(150.0, 100.0, 5.0, 100.0)
    sec = r["SEC_kWh_m3"]
    assert_true(0.02 < sec < 5.0, f"SEC={sec:.3f} kWh/m3 in physical range")
    # SEC rises with current density (more energy per m3)
    r2 = m.simulate(50.0, 100.0, 5.0, 100.0)
    assert_true(r["SEC_kWh_m3"] > r2["SEC_kWh_m3"],
                f"SEC(150)={r['SEC_kWh_m3']:.3f} > SEC(50)={r2['SEC_kWh_m3']:.3f}")


def test_zero_current_no_change():
    print("\n[Test 11] Zero current -> no desalination")
    m, _ = make_model()
    r = m.simulate(0.0, 100.0, 5.0, 100.0)
    assert_true(abs(r["salt_removed_fraction"]) < 1e-6,
                f"removed={r['salt_removed_fraction']:.2e} ~ 0 at i=0")


def test_predict_interface():
    print("\n[Test 12] ComponentModel predict() interface")
    _, cm = make_model()
    r = cm.predict({"current_density_A_m2": 200.0, "feed_conc_mol_m3": 100.0})
    for key in ["x", "c_diluate", "c_concentrate", "stack_voltage",
                "SEC_kWh_m3", "current_efficiency", "salt_removed_fraction"]:
        assert_true(key in r, f"Key '{key}' in output")
    assert_true(len(r["x"]) == len(r["c_diluate"]), "Arrays same length")
    info = cm.get_info()
    assert_true(info["component_id"] == "EC210", "component_id == EC210")


def test_benchmark():
    print("\n[Test 13] Benchmark: full stack simulation timing")
    m, _ = make_model()
    t0 = time.perf_counter()
    for _ in range(10):
        m.simulate(200.0, 100.0, 5.0, 100.0)
    elapsed = time.perf_counter() - t0
    print(f"  10 stack simulations in {elapsed*1000:.1f} ms "
          f"({elapsed*100:.1f} ms each)")
    assert_true(elapsed < 5.0, "10 simulations complete in < 5 s")


if __name__ == "__main__":
    tests = [
        test_current_efficiency_range,
        test_limiting_current_scales_with_conc,
        test_current_capped_below_limiting,
        test_salt_mass_conservation,
        test_diluate_decreases_concentrate_increases,
        test_more_current_more_removal,
        test_slower_flow_more_removal,
        test_membrane_potential_sign,
        test_stack_voltage_positive_and_scales,
        test_sec_physical_range,
        test_zero_current_no_change,
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
    print(f"EC210 ED F2a Ion-Transport -- Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    sys.exit(0 if failed == 0 else 1)
