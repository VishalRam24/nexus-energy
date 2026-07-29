"""
EC174 -- Instrument Transformer (CT / PT) -- F2a Magnetizing-Branch
Test suite: equivalent-circuit accuracy, saturation ODE, energy consistency.
Run with system python3 (NOT pytest):  python3 scripts/test_model.py
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from model import InstrumentTransformerF2a
from predict import ComponentModel

PASS = "✓"
FAIL = "✗"


def assert_true(cond, msg):
    if cond:
        print(f"  {PASS}  {msg}")
    else:
        print(f"  {FAIL}  FAILED: {msg}")
        raise AssertionError(msg)


def make_model(overrides=None):
    cm = ComponentModel(overrides)
    return cm._model, cm


# ---------------------------------------------------------------------------
def test_within_accuracy_class_at_rated():
    print("\n[Test 1] CT meets accuracy class at rated current & rated burden")
    m, _ = make_model()
    r = m.ct_errors(m.I_rated, burden_fraction=1.0)
    re = float(r["ratio_error_pct"])
    ph = float(r["phase_error_min"])
    assert_true(abs(re) <= m.accuracy_class,
                f"|ratio_err|={abs(re):.4f}% <= class {m.accuracy_class}%")
    assert_true(abs(ph) <= m.phase_class_min,
                f"|phase|={abs(ph):.2f} min <= {m.phase_class_min} min")
    assert_true(m.within_accuracy_class(m.I_rated, 1.0), "within_accuracy_class True")


def test_error_grows_with_burden():
    print("\n[Test 2] Ratio AND phase error grow monotonically with burden")
    m, _ = make_model()
    bfs = [0.25, 0.5, 1.0, 2.0, 4.0]
    re = [abs(float(m.ct_errors(m.I_rated, bf)["ratio_error_pct"])) for bf in bfs]
    ph = [abs(float(m.ct_errors(m.I_rated, bf)["phase_error_min"])) for bf in bfs]
    for i in range(1, len(bfs)):
        assert_true(re[i] > re[i - 1], f"ratio_err({bfs[i]})={re[i]:.4f} > {re[i-1]:.4f}")
        assert_true(ph[i] > ph[i - 1], f"phase({bfs[i]})={ph[i]:.3f} > {ph[i-1]:.3f}")


def test_ratio_error_sign():
    print("\n[Test 3] CT secondary under-reads (ratio error negative)")
    m, _ = make_model()
    r = m.ct_errors(m.I_rated, 1.0)
    assert_true(float(r["ratio_error_pct"]) < 0,
                f"ratio_err={float(r['ratio_error_pct']):.4f}% < 0 (excitation steals current)")


def test_excitation_below_ideal():
    print("\n[Test 4] Excitation current is a small fraction of secondary current")
    m, _ = make_model()
    r = m.ct_errors(m.I_rated, 1.0)
    I2_ideal = m.I_rated / m.n
    frac = float(r["excitation_A"]) / I2_ideal
    assert_true(0 < frac < 0.05, f"Ie/I2={frac*100:.3f}% in (0, 5%)")


def test_flux_law_monotone_saturating():
    print("\n[Test 5] Flux law monotone & saturates toward lambda_sat")
    m, _ = make_model()
    ims = np.linspace(0, 50 * m.Im_knee, 200)
    lam = m.flux_linkage(ims)
    assert_true(np.all(np.diff(lam) > 0), "lambda(im) strictly increasing")
    assert_true(lam[-1] < m.lambda_sat, f"lambda saturates below lambda_sat ({lam[-1]:.4f}<{m.lambda_sat:.4f})")
    assert_true(lam[-1] > 0.9 * m.lambda_sat, "approaches lambda_sat at high im")
    # inverse consistency
    im_back = m.magnetizing_current(m.flux_linkage(np.array([m.Im_knee])))
    assert_true(abs(im_back[0] - m.Im_knee) < 1e-6 * m.Im_knee + 1e-9,
                "magnetizing_current is inverse of flux_linkage")


def test_inductance_collapses():
    print("\n[Test 6] Incremental magnetizing inductance collapses in saturation")
    m, _ = make_model()
    L0 = m.magnetizing_inductance(0.0)
    Lhi = m.magnetizing_inductance(20 * m.Im_knee)
    assert_true(Lhi < 0.01 * L0, f"L(20*Imknee)={Lhi:.2e} << L(0)={L0:.2e}")


def test_no_saturation_at_rated():
    print("\n[Test 7] No saturation / negligible distortion at rated current")
    m, _ = make_model()
    s = m.simulate_saturation(m.I_rated * np.sqrt(2.0), burden_fraction=1.0, n_cycles=2)
    assert_true(not s["saturated"], "not saturated at rated")
    assert_true(s["distortion"] < 0.02, f"distortion={s['distortion']:.4f} < 2% at rated")


def test_saturation_at_high_current():
    print("\n[Test 8] CT saturates and distorts at high primary current (fault)")
    m, _ = make_model()
    s_lo = m.simulate_saturation(m.I_rated * np.sqrt(2.0) * 5, 1.0, n_cycles=2)
    s_hi = m.simulate_saturation(m.I_rated * np.sqrt(2.0) * 20, 1.0, n_cycles=2)
    assert_true(s_hi["saturated"], "saturated at 20x rated")
    assert_true(s_hi["distortion"] > 10 * s_lo["distortion"],
                f"distortion grows with current: {s_hi['distortion']:.3f} >> {s_lo['distortion']:.3f}")
    # peak magnetizing current spikes above the knee in saturation
    assert_true(np.max(np.abs(s_hi["i_mag"])) > m.Im_knee,
                "magnetizing current spikes above knee in saturation")


def test_saturation_grows_with_burden():
    print("\n[Test 9] Saturation worsens with higher burden (more EMF needed)")
    m, _ = make_model()
    d_lo = m.simulate_saturation(m.I_rated * np.sqrt(2.0) * 10, 1.0, n_cycles=2)["distortion"]
    d_hi = m.simulate_saturation(m.I_rated * np.sqrt(2.0) * 10, 8.0, n_cycles=2)["distortion"]
    assert_true(d_hi > d_lo, f"distortion(8x burden)={d_hi:.3f} > distortion(1x)={d_lo:.3f}")


def test_energy_consistency():
    print("\n[Test 10] Energy balance: source = dissipation + magnetic storage")
    m, _ = make_model()
    for mult, bf in [(5, 1.0), (20, 1.0), (10, 4.0)]:
        s = m.simulate_saturation(m.I_rated * np.sqrt(2.0) * mult, bf, n_cycles=2)
        assert_true(s["energy_resid"] < 1e-3,
                    f"energy residual={s['energy_resid']:.2e} < 1e-3 (mult={mult},bf={bf})")


def test_pt_mode():
    print("\n[Test 11] PT mode: voltage error within class, grows with burden")
    m, _ = make_model({"type": "PT"})
    r1 = m.pt_errors(m.V_rated, burden_fraction=1.0)
    r4 = m.pt_errors(m.V_rated, burden_fraction=4.0)
    assert_true(abs(float(r1["ratio_error_pct"])) <= m.accuracy_class,
                f"PT |err|={abs(float(r1['ratio_error_pct'])):.4f}% <= class {m.accuracy_class}%")
    assert_true(abs(float(r4["ratio_error_pct"])) > abs(float(r1["ratio_error_pct"])),
                "PT error grows with burden")


def test_predict_interface():
    print("\n[Test 12] ComponentModel predict() interface (both modes)")
    _, cm = make_model()
    a = cm.predict({"mode": "accuracy", "i_primary_A": 200.0, "burden_fraction": 1.0})
    for k in ["ratio_error_pct", "phase_error_min", "within_class", "accuracy_class"]:
        assert_true(k in a, f"accuracy key '{k}' present")
    s = cm.predict({"mode": "saturation", "i_primary_peak_A": 200 * 1.414 * 20, "n_cycles": 2})
    for k in ["t", "i_sec", "i_mag", "flux", "distortion", "saturated", "energy_resid"]:
        assert_true(k in s, f"saturation key '{k}' present")
    assert_true(len(s["t"]) == len(s["i_sec"]), "saturation arrays same length")


def test_benchmark():
    print("\n[Test 13] Benchmark: 2-cycle saturation ODE solve")
    m, _ = make_model()
    t0 = time.perf_counter()
    m.simulate_saturation(m.I_rated * np.sqrt(2.0) * 20, 1.0, n_cycles=2)
    elapsed = time.perf_counter() - t0
    print(f"  2-cycle saturation solve in {elapsed*1000:.1f} ms")
    assert_true(elapsed < 5.0, "Completes in < 5 s")


if __name__ == "__main__":
    tests = [
        test_within_accuracy_class_at_rated,
        test_error_grows_with_burden,
        test_ratio_error_sign,
        test_excitation_below_ideal,
        test_flux_law_monotone_saturating,
        test_inductance_collapses,
        test_no_saturation_at_rated,
        test_saturation_at_high_current,
        test_saturation_grows_with_burden,
        test_energy_consistency,
        test_pt_mode,
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
    print(f"EC174 Instrument Transformer F2a -- Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    sys.exit(0 if failed == 0 else 1)
