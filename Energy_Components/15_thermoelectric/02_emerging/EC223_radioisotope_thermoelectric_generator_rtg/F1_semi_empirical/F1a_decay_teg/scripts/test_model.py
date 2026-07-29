"""EC223 — RTG — F1a — Test Suite"""
import sys, time, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()


def test_predict_keys(model):
    r = model.predict({"t_years": 0.0})
    for k in ["P_thermal_W", "eta_teg", "P_electric_W", "T_hot_K",
              "eta_carnot", "fraction_thermal_remaining", "power_fraction"]:
        assert k in r, f"Missing key: {k}"


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC223"
    assert info["fidelity"] == "F1a"


def test_thermal_power_at_t0(model):
    """At t=0, thermal power must equal P_thermal_0."""
    P0 = model.params["unit"]["P_thermal_0_W"]["value"]
    r = model.predict({"t_years": 0.0})
    assert float(r["P_thermal_W"]) == pytest.approx(P0, rel=1e-9)


def test_thermal_power_decays(model):
    """Thermal power must strictly decrease over time."""
    t = np.linspace(0.0, 100.0, 50)
    r = model.predict({"t_years": t})
    assert np.all(np.diff(r["P_thermal_W"]) < 0), \
        "Thermal power must monotonically decrease"


def test_thermal_half_at_half_life(model):
    """Thermal power at t=t_half must be 50% of P0."""
    t_half = model.params["unit"]["t_half_years"]["value"]
    r0 = model.predict({"t_years": 0.0})
    r1 = model.predict({"t_years": t_half})
    ratio = float(r1["P_thermal_W"]) / float(r0["P_thermal_W"])
    assert ratio == pytest.approx(0.5, rel=1e-6), \
        f"Thermal at t_half should be 50%, got {ratio*100:.2f}%"


def test_electric_power_positive(model):
    """Electric power must be positive at all times in range."""
    t = np.linspace(0.0, 100.0, 100)
    r = model.predict({"t_years": t})
    assert np.all(r["P_electric_W"] > 0.0), "Electric power must be positive"


def test_electric_less_than_thermal(model):
    """Electric power must be less than thermal power (conversion efficiency < 100%)."""
    t = np.linspace(0.0, 100.0, 100)
    r = model.predict({"t_years": t})
    assert np.all(r["P_electric_W"] < r["P_thermal_W"]), \
        "P_electric must always be < P_thermal (eta < 100%)"


def test_eta_teg_in_physical_range(model):
    """TEG efficiency must be in 3-15% range for SiGe at these temperatures."""
    t = np.linspace(0.0, 50.0, 20)
    r = model.predict({"t_years": t})
    assert np.all(r["eta_teg"] > 0.01), "eta_TEG must be > 1%"
    assert np.all(r["eta_teg"] < 0.20), "eta_TEG must be < 20% (physical limit)"


def test_hot_side_temperature_decreases(model):
    """T_hot must decrease as decay power decreases."""
    t = np.linspace(0.0, 100.0, 50)
    r = model.predict({"t_years": t})
    assert np.all(np.diff(r["T_hot_K"]) < 0), \
        "T_hot must decrease monotonically as decay power falls"


def test_rtg_outlives_design_life(model):
    """RTG must still produce meaningful power after design life (50 years for Voyager-class)."""
    design_life = model.params["unit"]["design_life_years"]["value"]
    r0 = model.predict({"t_years": 0.0})
    r_end = model.predict({"t_years": design_life})
    fraction = float(r_end["P_electric_W"]) / float(r0["P_electric_W"])
    # Pu-238 t_half=87.7y: after 50y ~ 67% thermal remaining; with degradation, ~50% electric
    assert fraction > 0.30, \
        f"RTG should retain >30% electric power after {design_life}y, got {fraction*100:.1f}%"


def test_fraction_remaining_is_1_at_t0(model):
    r = model.predict({"t_years": 0.0})
    assert float(r["fraction_thermal_remaining"]) == pytest.approx(1.0, abs=1e-12)


def test_eta_carnot_less_than_1(model):
    """Carnot efficiency must be < 1 for finite temperature difference."""
    t = np.linspace(0.0, 100.0, 20)
    r = model.predict({"t_years": t})
    assert np.all(r["eta_carnot"] < 1.0), "Carnot efficiency must be < 1"
    assert np.all(r["eta_carnot"] > 0.0), "Carnot efficiency must be > 0 (T_hot > T_cold)"


def test_benchmark(model):
    t = np.random.uniform(0.0, 100.0, 1000)
    start = time.perf_counter()
    model.predict({"t_years": t})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed*1000:.2f} ms")
    assert elapsed < 1.0
