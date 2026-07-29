"""EC209 — Reverse Osmosis — F1b Fouling Temperature — Test Suite"""
import sys, time, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()


def test_predict_keys(model):
    r = model.predict({"operating_hours": 0})
    for k in ["permeate_flow_m3_h", "sec_kwh_m3", "rejection_pct", "flux_decline_factor"]:
        assert k in r


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC209"
    assert info["fidelity"] == "F1b"


def test_flux_factor_1_at_start_25C(model):
    """At t=0, T=25C (reference): flux factor should be ~1.0."""
    r = model.predict({"operating_hours": 0, "feed_temperature_degC": 25.0})
    ff = float(np.atleast_1d(r["flux_decline_factor"])[0])
    assert abs(ff - 1.0) < 0.05, f"Flux factor at t=0, T=25C = {ff:.4f}"


def test_flux_declines_with_time(model):
    """Fouling should reduce flux over time."""
    hours_list = [0, 8760, 26280, 52560]
    ffs = []
    for h in hours_list:
        r = model.predict({"operating_hours": h, "feed_temperature_degC": 25.0})
        ffs.append(float(np.atleast_1d(r["flux_decline_factor"])[0]))
    assert all(ffs[i] >= ffs[i + 1] - 1e-6 for i in range(len(ffs) - 1)), \
        f"Flux not declining: {ffs}"


def test_fouling_exponential(model):
    """At 1 year: factor = exp(-0.1) ~ 0.905."""
    r = model.predict({"operating_hours": 8760, "feed_temperature_degC": 25.0})
    ff = float(np.atleast_1d(r["flux_decline_factor"])[0])
    expected = np.exp(-0.10)
    assert abs(ff - expected) < 0.02, f"Flux factor at 1yr = {ff:.4f}, expected {expected:.4f}"


def test_warm_water_increases_flux(model):
    """Warmer water should increase flux (at clean membrane)."""
    r_cold = model.predict({"feed_temperature_degC": 15.0, "operating_hours": 0})
    r_warm = model.predict({"feed_temperature_degC": 35.0, "operating_hours": 0})
    ff_cold = float(np.atleast_1d(r_cold["flux_decline_factor"])[0])
    ff_warm = float(np.atleast_1d(r_warm["flux_decline_factor"])[0])
    assert ff_warm > ff_cold, f"Warm flux {ff_warm:.3f} not > cold {ff_cold:.3f}"


def test_permeate_flow_positive(model):
    """Permeate flow must be positive at reasonable conditions."""
    r = model.predict({"feed_salinity_ppm": 35000, "feed_pressure_bar": 60,
                        "recovery_ratio": 0.45, "operating_hours": 0})
    Q = float(np.atleast_1d(r["permeate_flow_m3_h"])[0])
    assert Q > 0, f"Permeate flow = {Q:.4f}"


def test_permeate_decreases_with_fouling(model):
    """Permeate flow should decrease with membrane age."""
    r_clean = model.predict({"operating_hours": 0})
    r_old = model.predict({"operating_hours": 43800})  # 5 years
    Q_clean = float(np.atleast_1d(r_clean["permeate_flow_m3_h"])[0])
    Q_old = float(np.atleast_1d(r_old["permeate_flow_m3_h"])[0])
    assert Q_old < Q_clean


def test_sec_reasonable_range(model):
    """SEC should be 2-6 kWh/m3 for SWRO."""
    r = model.predict({"feed_salinity_ppm": 35000, "feed_pressure_bar": 60,
                        "recovery_ratio": 0.45, "operating_hours": 0})
    sec = float(np.atleast_1d(r["sec_kwh_m3"])[0])
    assert 1.0 < sec < 10.0, f"SEC = {sec:.2f} kWh/m3"


def test_sec_increases_with_fouling(model):
    """SEC should increase slightly with membrane fouling."""
    r_clean = model.predict({"operating_hours": 0})
    r_old = model.predict({"operating_hours": 43800})
    sec_clean = float(np.atleast_1d(r_clean["sec_kwh_m3"])[0])
    sec_old = float(np.atleast_1d(r_old["sec_kwh_m3"])[0])
    assert sec_old >= sec_clean - 1e-6


def test_rejection_high(model):
    """Salt rejection should be >99% for clean membrane."""
    r = model.predict({"operating_hours": 0})
    rej = float(np.atleast_1d(r["rejection_pct"])[0])
    assert rej > 99.0, f"Rejection = {rej:.2f}%"


def test_rejection_degrades(model):
    """Rejection should decrease slightly over time."""
    r_clean = model.predict({"operating_hours": 0})
    r_old = model.predict({"operating_hours": 87600})
    rej_clean = float(np.atleast_1d(r_clean["rejection_pct"])[0])
    rej_old = float(np.atleast_1d(r_old["rejection_pct"])[0])
    assert rej_old <= rej_clean


def test_osmotic_pressure(model):
    """At S=35000 ppm: pi = 0.7 * 35 = 24.5 bar."""
    pi = model._model.osmotic_pressure(35000.0)
    assert abs(float(pi) - 24.5) < 0.5


def test_array_input(model):
    """Model should handle array operating_hours inputs."""
    hours = np.linspace(0, 87600, 10)
    r = model.predict({"operating_hours": hours})
    assert len(np.atleast_1d(r["flux_decline_factor"])) == 10


def test_benchmark(model):
    hours = np.random.uniform(0, 87600, 1000)
    start = time.perf_counter()
    model.predict({"operating_hours": hours})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed * 1000:.2f} ms")
    assert elapsed < 1.0
