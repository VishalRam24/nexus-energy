"""EC135 — Point Absorber WEC — F1b Resonance / PTO — Test Suite"""
import sys, time, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()


def test_predict_keys(model):
    r = model.predict({"H_s": 2.0, "T_e": 10.0})
    for k in ["power_kw", "cwr", "pto_efficiency", "overall_efficiency", "seawater_density_kgm3"]:
        assert k in r


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC135"
    assert info["fidelity"] == "F1b"


def test_power_positive(model):
    r = model.predict({"H_s": 2.0, "T_e": 10.0})
    assert float(r["power_kw"]) > 0.0


def test_storm_cutout(model):
    """Power must be zero above H_s_cutout."""
    m = model._model
    r = model.predict({"H_s": m.H_s_cutout + 0.5, "T_e": 10.0})
    assert float(r["power_kw"]) == 0.0, "Power must be zero above cutout H_s"


def test_rated_power_cap(model):
    """Output must not exceed P_rated."""
    m = model._model
    r = model.predict({"H_s": 5.0, "T_e": m.T_n})
    assert float(r["power_kw"]) <= m.P_rated_kw + 1e-6


def test_resonance_peak(model):
    """At T_e = T_n, CWR should be at peak."""
    m = model._model
    cwr_at_resonance = float(m.capture_width_ratio(m.T_n))
    cwr_off          = float(m.capture_width_ratio(m.T_n + m.sigma * 2))
    assert cwr_at_resonance > cwr_off


def test_pto_efficiency_at_rated(model):
    """PTO efficiency at rated power must equal eta_pto_rated."""
    m = model._model
    eta = float(m.pto_efficiency(m.P_rated_kw))
    assert abs(eta - m.eta_pto_rated) < 1e-9


def test_pto_efficiency_lower_at_part_load(model):
    """PTO efficiency at 10% of rated must be lower than at rated."""
    m = model._model
    eta_rated = float(m.pto_efficiency(m.P_rated_kw))
    eta_low   = float(m.pto_efficiency(0.1 * m.P_rated_kw))
    assert eta_low < eta_rated


def test_salinity_increases_power(model):
    r_low  = model.predict({"H_s": 2.0, "T_e": 10.0, "T_C": 15.0, "S_psu": 30.0})
    r_high = model.predict({"H_s": 2.0, "T_e": 10.0, "T_C": 15.0, "S_psu": 38.0})
    assert float(r_high["power_kw"]) > float(r_low["power_kw"])


def test_colder_water_gives_more_power(model):
    """Colder seawater is denser → more wave power."""
    r_cold = model.predict({"H_s": 2.0, "T_e": 10.0, "T_C": 5.0,  "S_psu": 35.0})
    r_warm = model.predict({"H_s": 2.0, "T_e": 10.0, "T_C": 25.0, "S_psu": 35.0})
    assert float(r_cold["power_kw"]) > float(r_warm["power_kw"])


def test_power_zero_below_cutin(model):
    """Power below H_s cut-in must be zero."""
    m = model._model
    r = model.predict({"H_s": m.H_s_cutin * 0.5, "T_e": 10.0})
    assert float(r["power_kw"]) == 0.0


def test_benchmark(model):
    H_s = np.random.uniform(0.5, 5.9, 1000)
    T_e = np.random.uniform(5.0, 18.0, 1000)
    start = time.perf_counter()
    model.predict({"H_s": H_s, "T_e": T_e})
    elapsed = time.perf_counter() - start
    assert elapsed < 1.0
