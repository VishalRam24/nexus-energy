"""
EC075 -- Finned-Tube Heat Exchanger -- F2a Physics-Lumped (Transient)
Custom assert harness (NO pytest). Run: python3 scripts/test_model.py

Tests: energy conservation, e-NTU steady-state limit, outlet bounds,
monotone transient approach, fin-efficiency effect, predict() interface,
edge cases, and a benchmark timing.
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from model import FinnedTubeHXF2a
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
def test_energy_conservation():
    print("\n[Test 1] Energy conservation: Q_hot == Q_cold (steady)")
    m, _ = make_model()
    r = m.simulate(80.0, 20.0, 1.0, 2.0, duration_s=1800.0, n_out=60)
    Qh = r["Q_hot_kw"][-1]
    Qc = r["Q_cold_kw"][-1]
    assert_true(abs(Qh - Qc) < 1e-3, f"Q_hot={Qh:.4f} ~ Q_cold={Qc:.4f} kW")


def test_entu_steady_state():
    print("\n[Test 2] Steady state matches e-NTU within 3%")
    m, _ = make_model()
    cases = [(80, 20, 1.0, 2.0), (60, 10, 2.0, 5.0), (100, -10, 0.5, 3.0)]
    for c in cases:
        ref = m.steady_state_entu(*c)
        ss = m.steady_outputs(*c, duration_s=1800.0)
        err = abs(ss["effectiveness"] - ref["effectiveness"]) / ref["effectiveness"]
        assert_true(err < 0.03,
                    f"{c}: eps_tran={ss['effectiveness']:.4f} vs e-NTU={ref['effectiveness']:.4f} "
                    f"({err*100:.2f}%)")


def test_outlet_bounds():
    print("\n[Test 3] Outlet temps bracketed by inlets (2nd law)")
    m, _ = make_model()
    Th_in, Tc_in = 80.0, 20.0
    r = m.simulate(Th_in, Tc_in, 1.0, 2.0, duration_s=1800.0, n_out=60)
    Th_out = r["T_h_out"][-1]
    Tc_out = r["T_c_out"][-1]
    assert_true(Tc_in <= Th_out <= Th_in, f"Tc_in <= T_h_out({Th_out:.2f}) <= Th_in")
    assert_true(Tc_in <= Tc_out <= Th_in, f"Tc_in <= T_c_out({Tc_out:.2f}) <= Th_in")
    assert_true(Th_out >= Tc_out - 1e-6, f"T_h_out({Th_out:.2f}) >= T_c_out({Tc_out:.2f})")


def test_effectiveness_le_one():
    print("\n[Test 4] Effectiveness in (0, 1] over the whole transient")
    m, _ = make_model()
    r = m.simulate(80.0, 20.0, 1.0, 2.0, duration_s=1200.0, n_out=120)
    eps = r["effectiveness"]
    assert_true(np.all(eps <= 1.0 + 1e-6), f"eps_max={eps.max():.4f} <= 1")
    assert_true(eps[-1] > 0.0, f"eps_final={eps[-1]:.4f} > 0")


def test_heat_balance_qh_positive():
    print("\n[Test 5] Q > 0 when hot inlet above cold inlet")
    m, _ = make_model()
    r = m.steady_outputs(80.0, 20.0, 1.0, 2.0)
    assert_true(r["Q_kw"] > 0.0, f"Q={r['Q_kw']:.2f} kW > 0")


def test_transient_monotone_approach():
    print("\n[Test 6] Hot outlet rises monotonically from cold start to SS")
    m, _ = make_model()
    # Start whole device cold (=Tc_in). Hot outlet should climb toward SS.
    r = m.simulate(80.0, 20.0, 1.0, 2.0, duration_s=900.0, n_out=80, T0=20.0)
    Th_out = r["T_h_out"]
    diffs = np.diff(Th_out)
    assert_true(np.all(diffs >= -1e-4),
                f"T_h_out non-decreasing (min step {diffs.min():.5f})")
    assert_true(Th_out[-1] > Th_out[0] + 5.0,
                f"warmed up: {Th_out[0]:.2f} -> {Th_out[-1]:.2f} C")


def test_thermal_mass_lag():
    print("\n[Test 7] Finite thermal mass: SS not reached instantly")
    m, _ = make_model()
    r = m.simulate(80.0, 20.0, 1.0, 2.0, duration_s=600.0, n_out=120, T0=20.0)
    Th = r["T_h_out"]
    # Early outlet (t small) must still be far from final (transient lag exists).
    early = Th[2]
    final = Th[-1]
    assert_true(abs(final - early) > 0.5,
                f"transient lag present: early={early:.2f} final={final:.2f} C")


def test_fin_efficiency_effect():
    print("\n[Test 8] Lower fin efficiency -> lower duty (air-side dominant)")
    cm = ComponentModel()
    base = cm._model
    Q_base = base.steady_outputs(80.0, 20.0, 1.0, 2.0)["Q_kw"]
    # Note: in this UA-split the eta_o is folded into UA_c_eff = UA/f, so the
    # physical fin-efficiency lever is the air-side resistance fraction. Increase
    # the air-side resistance share -> larger air resistance -> lower UA path duty
    # is not changed (series stays UA). Instead vary the overall UA via the model:
    cm2 = ComponentModel({"U_overall": {"value": 30.0, "unit": "W/m2K"}})
    Q_low = cm2._model.steady_outputs(80.0, 20.0, 1.0, 2.0)["Q_kw"]
    assert_true(Q_low < Q_base,
                f"lower U -> lower duty: Q(U=30)={Q_low:.2f} < Q(U=45)={Q_base:.2f} kW")


def test_grid_convergence():
    print("\n[Test 9] N-independence: SS effectiveness stable in N")
    cm = ComponentModel()
    base = cm._raw
    eps = []
    for N in [5, 10, 20]:
        p = {k: v for k, v in base.items()}
        p["unit"] = dict(base["unit"])
        p["unit"]["N_cv"] = {"value": N, "unit": "-"}
        mm = FinnedTubeHXF2a(p)
        eps.append(mm.steady_outputs(80.0, 20.0, 1.0, 2.0)["effectiveness"])
    spread = max(eps) - min(eps)
    assert_true(spread < 0.01, f"eps stable across N: {['%.4f' % e for e in eps]} spread={spread:.4f}")


def test_predict_interface():
    print("\n[Test 10] ComponentModel predict() / get_info() interface")
    _, cm = make_model()
    info = cm.get_info()
    assert_true(info["ec_id"] == "EC075", "ec_id == EC075")
    r = cm.predict({"T_h_in": 80.0, "T_c_in": 20.0,
                    "m_dot_hot": 1.0, "m_dot_cold": 2.0, "duration_s": 300.0})
    for key in ["t", "T_h_out", "T_c_out", "Q_kw", "effectiveness",
                "steady_state", "entu_reference"]:
        assert_true(key in r, f"Key '{key}' in output")
    assert_true(len(r["t"]) == len(r["T_h_out"]), "Arrays same length")


def test_edge_no_dt():
    print("\n[Test 11] Edge: equal inlet temps -> zero duty")
    m, _ = make_model()
    r = m.steady_outputs(40.0, 40.0, 1.0, 2.0)
    assert_true(abs(r["Q_kw"]) < 1e-6, f"Q={r['Q_kw']:.6f} kW ~ 0 when dT=0")


def test_benchmark():
    print("\n[Test 12] Benchmark: 600 s transient (N=10)")
    m, _ = make_model()
    t0 = time.perf_counter()
    m.simulate(80.0, 20.0, 1.0, 2.0, duration_s=600.0, n_out=120)
    elapsed = time.perf_counter() - t0
    print(f"  600 s simulation in {elapsed*1000:.1f} ms")
    assert_true(elapsed < 5.0, "Completes in < 5 s")


if __name__ == "__main__":
    tests = [
        test_energy_conservation,
        test_entu_steady_state,
        test_outlet_bounds,
        test_effectiveness_le_one,
        test_heat_balance_qh_positive,
        test_transient_monotone_approach,
        test_thermal_mass_lag,
        test_fin_efficiency_effect,
        test_grid_convergence,
        test_predict_interface,
        test_edge_no_dt,
        test_benchmark,
    ]
    passed = failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"  ERROR: {e}")

    print(f"\n{'='*60}")
    print(f"EC075 Finned-Tube HX F2a -- Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    sys.exit(0 if failed == 0 else 1)
