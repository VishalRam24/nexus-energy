"""EC222 — Betavoltaic Cell — F1a — Test Suite"""
import sys, time, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()


def test_predict_keys(model):
    r = model.predict({"t_years": 0.0})
    for k in ["activity_Bq", "P_beta_W", "P_out_W", "P_out_uW", "fraction_remaining"]:
        assert k in r, f"Missing key: {k}"


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC222"
    assert info["fidelity"] == "F1a"


def test_power_decays_with_time(model):
    """Core physics: betavoltaic power must strictly decrease over time (radioactive decay)."""
    t = np.linspace(0.0, 200.0, 50)
    r = model.predict({"t_years": t})
    assert np.all(np.diff(r["P_out_W"]) < 0), \
        "Power must monotonically decrease (radioactive decay)"


def test_activity_at_t0_equals_A0(model):
    """At t=0, activity must equal A0 exactly."""
    A0 = model.params["unit"]["A0_Bq"]["value"]
    r = model.predict({"t_years": 0.0})
    assert float(r["activity_Bq"]) == pytest.approx(A0, rel=1e-9)


def test_activity_half_at_one_half_life(model):
    """Activity at t=t_half must be exactly A0/2."""
    t_half = model.params["unit"]["t_half_years"]["value"]
    r0 = model.predict({"t_years": 0.0})
    r1 = model.predict({"t_years": t_half})
    ratio = float(r1["activity_Bq"]) / float(r0["activity_Bq"])
    assert ratio == pytest.approx(0.5, rel=1e-6), \
        f"Activity at t_half should be 50% of A0, got {ratio*100:.2f}%"


def test_power_decays_to_half_at_half_life(model):
    """Power at t=t_half must be 50% of initial power."""
    t_half = model.params["unit"]["t_half_years"]["value"]
    r0 = model.predict({"t_years": 0.0})
    r1 = model.predict({"t_years": t_half})
    ratio = float(r1["P_out_W"]) / float(r0["P_out_W"])
    assert ratio == pytest.approx(0.5, rel=1e-6), \
        f"Power at t_half should be 50%, got {ratio*100:.2f}%"


def test_power_scale_microwatt(model):
    """Betavoltaic cells produce uW-scale power at t=0."""
    r = model.predict({"t_years": 0.0})
    P_uW = float(r["P_out_uW"])
    # With 0.1 Ci Ni-63, expect uW range
    assert P_uW > 0.0, "Power must be positive"
    assert P_uW < 1e9, "Power unrealistically large for betavoltaic"


def test_efficiency_range(model):
    """Conversion efficiency must be within 1-8% (betavoltaic literature)."""
    r0 = model.predict({"t_years": 0.0})
    A0 = model.params["unit"]["A0_Bq"]["value"]
    E_MeV = model.params["unit"]["E_beta_MeV"]["value"]
    eta_cap = model.params["unit"]["eta_capture"]["value"]
    eta_conv = model.params["unit"]["eta_conv"]["value"]
    MeV_to_J = 1.602176634e-13
    P_in = A0 * E_MeV * MeV_to_J
    overall_eta = float(r0["P_out_W"]) / P_in
    assert 0.001 < overall_eta < 0.40, \
        f"Overall eta {overall_eta*100:.2f}% outside expected range"


def test_fraction_remaining_monotone_decreasing(model):
    """Fraction remaining must strictly decrease."""
    t = np.linspace(0.0, 300.0, 100)
    r = model.predict({"t_years": t})
    assert np.all(np.diff(r["fraction_remaining"]) < 0), \
        "Fraction remaining must be strictly decreasing"


def test_fraction_at_t0_is_one(model):
    """At t=0, fraction remaining must be 1.0."""
    r = model.predict({"t_years": 0.0})
    assert float(r["fraction_remaining"]) == pytest.approx(1.0, abs=1e-12)


def test_betavoltaic_outlives_design_life(model):
    """After design life (50 years), Ni-63 cell still produces significant power."""
    design_life = model.params["unit"]["design_life_years"]["value"]
    r0 = model.predict({"t_years": 0.0})
    r_end = model.predict({"t_years": design_life})
    fraction = float(r_end["P_out_W"]) / float(r0["P_out_W"])
    # Ni-63: after 50y at t_half=100.2y -> ~71% remaining
    assert fraction > 0.5, \
        f"After {design_life}y design life, Ni-63 cell should retain >50% power, got {fraction*100:.1f}%"


def test_benchmark(model):
    t = np.random.uniform(0.0, 300.0, 1000)
    start = time.perf_counter()
    model.predict({"t_years": t})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed*1000:.2f} ms")
    assert elapsed < 1.0
