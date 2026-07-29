"""
EC096 -- Magnetic Refrigeration -- F2a AMR (Active Magnetic Regenerator)
Test suite: magnetocaloric physics sanity, AMR cycle conservation, COP bounds,
edge cases, predict() interface, benchmark timing. Custom harness (no pytest).

Run:  python3 scripts/test_model.py
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from model import AMR_F2a, GdMagnetocaloric, MU0, brillouin
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
def test_brillouin_limits():
    print("\n[Test 1] Brillouin function limits")
    J = 3.5
    assert_true(abs(brillouin(J, 1e-9)) < 1e-6, "B_J(0) -> 0")
    assert_true(brillouin(J, 50.0) > 0.99, f"B_J(large)={brillouin(J,50.0):.4f} -> 1 (saturation)")
    assert_true(brillouin(J, 1.0) > 0 and brillouin(J, 1.0) < 1.0, "0 < B_J(1) < 1")


def test_dTad_peaks_at_curie():
    print("\n[Test 2] Delta T_ad peaks at the Curie temperature (Gd, ~293 K)")
    m, _ = make_model()
    mat = m.mat
    H = 1.5 / MU0
    Ts = np.arange(250, 330, 4.0)
    dTads = np.array([mat.delta_T_ad(T, H, 0.0) for T in Ts])
    T_peak = Ts[np.argmax(dTads)]
    assert_true(abs(T_peak - mat.T_C) <= 8.0,
                f"Peak Delta T_ad at T={T_peak:.0f} K near T_C={mat.T_C:.0f} K")
    assert_true(2.0 < dTads.max() < 8.0,
                f"Peak Delta T_ad={dTads.max():.2f} K (Gd ~3-5 K at 1.5 T, Dan'kov 1998)")


def test_dTad_positive_and_field_monotone():
    print("\n[Test 3] Delta T_ad >= 0 and increases with applied field")
    m, _ = make_model()
    mat = m.mat
    d1 = mat.delta_T_ad(mat.T_C, 1.0 / MU0, 0.0)
    d2 = mat.delta_T_ad(mat.T_C, 2.0 / MU0, 0.0)
    assert_true(d1 >= 0 and d2 >= 0, "Delta T_ad >= 0 (magnetisation warms the bed)")
    assert_true(d2 > d1, f"Larger field -> larger MCE: {d2:.3f} > {d1:.3f} K")


def test_magnetisation_orders_below_Tc():
    print("\n[Test 4] Spontaneous magnetisation below T_C, paramagnet above")
    m, _ = make_model()
    mat = m.mat
    M_below = mat.magnetisation(mat.T_C - 40.0, 0.0) / mat.M_sat
    M_above = mat.magnetisation(mat.T_C + 40.0, 0.0) / mat.M_sat
    assert_true(M_below > 0.3, f"Ferromagnetic order below T_C: M/Msat={M_below:.3f}")
    assert_true(M_above < 0.05, f"Near-paramagnetic above T_C: M/Msat={M_above:.3f}")


def test_entropy_decreases_with_field():
    print("\n[Test 5] Magnetic entropy decreases when field applied (ordering)")
    m, _ = make_model()
    mat = m.mat
    H = 1.5 / MU0
    s0 = mat.magnetic_entropy(mat.T_C, 0.0)
    sH = mat.magnetic_entropy(mat.T_C, H)
    assert_true(sH < s0, f"S_mag(H) < S_mag(0): {sH:.3f} < {s0:.3f} J/(kg.K)")


def test_energy_conservation():
    print("\n[Test 6] Cycle energy balance: Q_hot = Q_cold + W_input")
    _, cm = make_model()
    r = cm.predict({"n_cycles": 25})
    resid = abs(r["Q_hot_W"] - r["Q_cold_W"] - r["W_input_W"])
    assert_true(resid < 1e-6 * max(1.0, r["Q_hot_W"]),
                f"1st law residual={resid:.2e} W (Q_hot=Q_cold+W)")


def test_cooling_power_positive():
    print("\n[Test 7] Positive cooling power when reservoir span < AMR no-load span")
    _, cm = make_model()
    r = cm.predict({"T_cold_K": 291.0, "T_hot_K": 296.0, "n_cycles": 25})
    assert_true(r["Q_cold_W"] > 0.0, f"Q_cold={r['Q_cold_W']:.2f} W > 0")
    assert_true(r["W_input_W"] > 0.0, f"W_input={r['W_input_W']:.3f} W > 0")


def test_cop_positive_below_carnot():
    print("\n[Test 8] 0 < COP <= COP_Carnot (2nd law)")
    _, cm = make_model()
    r = cm.predict({"T_cold_K": 291.0, "T_hot_K": 296.0, "n_cycles": 25})
    assert_true(r["COP"] > 0.0, f"COP={r['COP']:.3f} > 0")
    assert_true(r["COP"] <= r["COP_Carnot"] + 1e-9,
                f"COP={r['COP']:.3f} <= Carnot={r['COP_Carnot']:.3f}")


def test_carnot_formula():
    print("\n[Test 9] COP_Carnot = T_cold / (T_hot - T_cold)")
    _, cm = make_model()
    r = cm.predict({"T_cold_K": 290.0, "T_hot_K": 295.0, "n_cycles": 12})
    expected = 290.0 / 5.0
    assert_true(abs(r["COP_Carnot"] - expected) < 1e-6,
                f"COP_Carnot={r['COP_Carnot']:.3f} == {expected:.3f}")


def test_cooling_span_collapses():
    print("\n[Test 10] Cooling power collapses as span exceeds bed capacity")
    _, cm = make_model()
    r_small = cm.predict({"T_cold_K": 291.0, "T_hot_K": 296.0, "n_cycles": 25})
    r_large = cm.predict({"T_cold_K": 283.0, "T_hot_K": 303.0, "n_cycles": 25})
    assert_true(r_large["Q_cold_W"] < r_small["Q_cold_W"],
                f"Q_cold falls with span: {r_large['Q_cold_W']:.2f} < {r_small['Q_cold_W']:.2f} W")


def test_predict_interface():
    print("\n[Test 11] ComponentModel predict() / get_info() interface")
    _, cm = make_model()
    info = cm.get_info()
    for k in ["component_id", "component_name", "fidelity", "version", "inputs", "outputs"]:
        assert_true(k in info, f"get_info has '{k}'")
    assert_true(info["component_id"] == "EC096", "component_id == EC096")
    r = cm.predict({"n_cycles": 12})
    for key in ["COP", "COP_Carnot", "Q_cold_W", "Q_hot_W", "W_input_W",
                "T_span_K", "T_solid_profile", "dTad_profile"]:
        assert_true(key in r, f"predict output has '{key}'")
    assert_true(len(r["T_solid_profile"]) == cm._model.N, "Solid profile has N nodes")


def test_benchmark():
    print("\n[Test 12] Benchmark: 25-cycle AMR simulation timing")
    _, cm = make_model()
    t0 = time.perf_counter()
    cm.predict({"n_cycles": 25})
    elapsed = time.perf_counter() - t0
    print(f"  25-cycle steady-state AMR solve in {elapsed*1000:.0f} ms")
    assert_true(elapsed < 10.0, "Completes in < 10 s")


if __name__ == "__main__":
    tests = [
        test_brillouin_limits,
        test_dTad_peaks_at_curie,
        test_dTad_positive_and_field_monotone,
        test_magnetisation_orders_below_Tc,
        test_entropy_decreases_with_field,
        test_energy_conservation,
        test_cooling_power_positive,
        test_cop_positive_below_carnot,
        test_carnot_formula,
        test_cooling_span_collapses,
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

    print(f"\n{'='*62}")
    print(f"EC096 Magnetic Refrigeration F2a (AMR) -- "
          f"Results: {passed} passed, {failed} failed")
    print(f"{'='*62}")
    sys.exit(0 if failed == 0 else 1)
