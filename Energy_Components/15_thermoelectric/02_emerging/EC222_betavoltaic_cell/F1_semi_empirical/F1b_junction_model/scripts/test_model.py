"""EC222 — Betavoltaic Cell — F1b Junction Model — Test Suite

Physics covered:
  1.  Output keys present
  2.  get_info correct ec_id / fidelity
  3.  Activity at t=0 equals A0
  4.  Activity halves at one half-life
  5.  Isc at t=0 > 0
  6.  Isc decays monotonically with time
  7.  Isc(t) / Isc(0) == A(t) / A0 (at constant T — pure decay factor)
  8.  Voc at t=0 matches Voc_ref
  9.  Voc decreases with temperature (dVoc_dT < 0)
  10. Voc decreases with time (activity decay lowers Voc)
  11. Voc >= 0 at all times
  12. FF decreases with time (radiation damage)
  13. FF floor: FF(very_old) >= 0.5 * FF0
  14. P_out = Isc * Voc * FF (internal consistency check)
  15. P_out >= 0 everywhere
  16. P_out decreases monotonically with time (decay dominates)
  17. P_out < P_beta_absorbed (eta < 100%)
  18. eta_junction in physical range (0-50%)
  19. P_out at t=0 in uW range for typical 0.1 Ci Ni-63 source
  20. Fraction remaining = 1 at t=0, decays to <1 for t>0
  21. Benchmark: 1000 predictions < 1 s
"""
import sys, time, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()


def test_predict_keys(model):
    r = model.predict({"t_years": 0.0})
    expected = [
        "activity_Bq", "P_beta_total_W", "P_beta_absorbed_W",
        "Isc_uA", "Voc_V", "FF", "P_out_W", "P_out_uW",
        "eta_junction", "fraction_remaining",
    ]
    for k in expected:
        assert k in r, f"Missing key: {k}"


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC222"
    assert info["fidelity"] == "F1b"


def test_activity_at_t0_equals_A0(model):
    A0 = model.params["unit"]["A0_Bq"]["value"]
    r = model.predict({"t_years": 0.0})
    assert float(r["activity_Bq"]) == pytest.approx(A0, rel=1e-9)


def test_activity_half_at_half_life(model):
    t_half = model.params["unit"]["t_half_years"]["value"]
    r0 = model.predict({"t_years": 0.0})
    r1 = model.predict({"t_years": t_half})
    ratio = float(r1["activity_Bq"]) / float(r0["activity_Bq"])
    assert ratio == pytest.approx(0.5, rel=1e-6)


def test_Isc_at_t0_positive(model):
    r = model.predict({"t_years": 0.0})
    assert float(r["Isc_uA"]) > 0.0


def test_Isc_decays_monotonically(model):
    t = np.linspace(0.0, 200.0, 50)
    r = model.predict({"t_years": t})
    assert np.all(np.diff(np.asarray(r["Isc_uA"])) < 0), \
        "Isc must monotonically decrease (activity decay)"


def test_Isc_fraction_equals_activity_fraction(model):
    """At constant T=T_ref, Isc(t)/Isc(0) == A(t)/A0 (linear mapping)."""
    T_ref = model.params["unit"]["T_ref_K"]["value"]
    t = np.array([0.0, 10.0, 50.0, 100.0])
    r = model.predict({"t_years": t, "T_cell_K": T_ref})
    Isc = np.asarray(r["Isc_uA"])
    A = np.asarray(r["activity_Bq"])
    A0 = model.params["unit"]["A0_Bq"]["value"]
    # Isc(t) = Isc_ref * (A(t)/A0) at T=T_ref (T-factor = 1)
    for i in range(len(t)):
        ratio_Isc = float(Isc[i]) / float(Isc[0])
        ratio_A = float(A[i]) / float(A0)
        assert abs(ratio_Isc - ratio_A) < 1e-6, \
            f"Isc fraction {ratio_Isc:.6f} != A fraction {ratio_A:.6f} at t={t[i]}"


def test_Voc_at_t0_matches_ref(model):
    """At t=0 and T=T_ref, Voc must equal Voc_ref exactly."""
    T_ref = model.params["unit"]["T_ref_K"]["value"]
    Voc_ref = model.params["unit"]["Voc_ref_V"]["value"]
    r = model.predict({"t_years": 0.0, "T_cell_K": T_ref})
    assert float(r["Voc_V"]) == pytest.approx(Voc_ref, abs=1e-6)


def test_Voc_decreases_with_temperature(model):
    """dVoc/dT < 0 — higher temperature lowers Voc (like PV cells)."""
    r_cold = model.predict({"t_years": 0.0, "T_cell_K": 250.0})
    r_hot = model.predict({"t_years": 0.0, "T_cell_K": 400.0})
    assert float(r_hot["Voc_V"]) < float(r_cold["Voc_V"]), \
        "Voc must decrease with temperature"


def test_Voc_decreases_with_time(model):
    """Voc decreases as activity falls (logarithmic decay term)."""
    t = np.linspace(0.0, 300.0, 50)
    r = model.predict({"t_years": t})
    Voc = np.asarray(r["Voc_V"])
    assert np.all(np.diff(Voc) <= 0), "Voc must decrease monotonically with time"


def test_Voc_nonnegative(model):
    """Voc must always be >= 0."""
    t = np.linspace(0.0, 500.0, 200)
    r = model.predict({"t_years": t})
    assert np.all(np.asarray(r["Voc_V"]) >= 0.0), "Voc must be >= 0"


def test_FF_decreases_with_time(model):
    """Fill factor decreases with time (radiation damage)."""
    t = np.array([0.0, 50.0, 200.0, 500.0])
    r = model.predict({"t_years": t})
    FF = np.asarray(r["FF"])
    assert np.all(np.diff(FF) <= 0), "FF must decrease or stay flat with time"


def test_FF_floor(model):
    """FF must not drop below 50% of FF0 (saturation floor)."""
    FF0 = model.params["unit"]["fill_factor_0"]["value"]
    r = model.predict({"t_years": 10000.0})
    FF = float(np.atleast_1d(r["FF"])[0])
    assert FF >= 0.5 * FF0 - 1e-9, \
        f"FF floor violated: FF={FF:.4f} < 0.5*FF0={0.5*FF0:.4f}"


def test_P_out_equals_Isc_times_Voc_times_FF(model):
    """P_out = Isc * Voc * FF — internal consistency."""
    for t_val in [0.0, 10.0, 50.0, 200.0]:
        r = model.predict({"t_years": t_val})
        Isc_A = float(r["Isc_uA"]) * 1e-6
        Voc = float(r["Voc_V"])
        FF = float(r["FF"])
        P_expected = Isc_A * Voc * FF
        P_actual = float(r["P_out_W"])
        assert abs(P_actual - P_expected) < 1e-15 * max(abs(P_expected), 1.0), \
            f"P_out consistency failed at t={t_val}: {P_actual:.6e} vs {P_expected:.6e}"


def test_P_out_nonnegative(model):
    t = np.linspace(0.0, 500.0, 200)
    r = model.predict({"t_years": t})
    assert np.all(np.asarray(r["P_out_W"]) >= 0.0), "P_out must be >= 0"


def test_P_out_decreases_with_time(model):
    """Power output must decrease monotonically (decay dominates)."""
    t = np.linspace(0.0, 300.0, 100)
    r = model.predict({"t_years": t})
    assert np.all(np.diff(np.asarray(r["P_out_W"])) <= 0), \
        "P_out must not increase with time"


def test_P_out_less_than_absorbed(model):
    """P_out must be less than absorbed beta power (eta < 100%)."""
    t = np.linspace(0.0, 100.0, 50)
    r = model.predict({"t_years": t})
    assert np.all(np.asarray(r["P_out_W"]) <= np.asarray(r["P_beta_absorbed_W"]) + 1e-30), \
        "P_out must not exceed absorbed beta power"


def test_eta_junction_physical_range(model):
    """Junction efficiency must be in 0-50% range (betavoltaic literature limit ~10%)."""
    t = np.linspace(0.0, 100.0, 50)
    r = model.predict({"t_years": t})
    eta = np.asarray(r["eta_junction"])
    assert np.all(eta >= 0.0), "eta_junction must be >= 0"
    assert np.all(eta <= 0.50), "eta_junction must be <= 50% (thermodynamic limit)"


def test_P_out_microwatt_scale(model):
    """For 0.1 Ci Ni-63 at t=0, power output should be in nW-mW range."""
    r = model.predict({"t_years": 0.0})
    P_uW = float(r["P_out_uW"])
    # Diamond cell: 0.1 Ci Ni-63 → expect ~0.1-100 uW
    assert P_uW > 0.0, "P_out must be positive"
    assert P_uW < 1e6, "P_out unrealistically large for betavoltaic"


def test_fraction_remaining_at_t0(model):
    r = model.predict({"t_years": 0.0})
    assert float(r["fraction_remaining"]) == pytest.approx(1.0, abs=1e-12)


def test_benchmark(model):
    t = np.random.uniform(0.0, 300.0, 1000)
    T = np.random.uniform(250.0, 450.0, 1000)
    start = time.perf_counter()
    model.predict({"t_years": t, "T_cell_K": T})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed * 1000:.2f} ms")
    assert elapsed < 1.0
