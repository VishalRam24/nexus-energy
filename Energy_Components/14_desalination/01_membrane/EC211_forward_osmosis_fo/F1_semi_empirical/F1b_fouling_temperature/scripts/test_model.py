"""EC211 — Forward Osmosis (FO) — F1b Fouling + Temperature — Test Suite"""
import sys, time, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()


def test_predict_keys(model):
    r = model.predict({"pi_draw_bar": 60.0, "pi_feed_bar": 27.0, "operating_hours": 0})
    for k in ["permeate_flow_m3_h", "water_flux_lmh", "sec_total_kwh_m3",
              "salt_leakage_mg_l", "flux_decline_factor"]:
        assert k in r


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC211"
    assert info["fidelity"] == "F1b"


def test_flux_positive_with_driving_force(model):
    """Water flux must be positive when pi_draw > pi_feed."""
    r = model.predict({"pi_draw_bar": 60.0, "pi_feed_bar": 27.0,
                       "T_feed_degC": 25.0, "operating_hours": 0})
    J = float(np.atleast_1d(r["water_flux_lmh"])[0])
    assert J > 0, f"Water flux = {J:.4f} L/(m2*h)"


def test_zero_flux_when_no_driving_force(model):
    """No flux when pi_draw = pi_feed."""
    r = model.predict({"pi_draw_bar": 27.0, "pi_feed_bar": 27.0,
                       "T_feed_degC": 25.0, "operating_hours": 0})
    J = float(np.atleast_1d(r["water_flux_lmh"])[0])
    assert J < 0.1, f"Flux should be ~0 at equal osmotic pressures: {J:.4f}"


def test_flux_increases_with_driving_force(model):
    """Higher draw osmotic pressure → higher flux."""
    r_low  = model.predict({"pi_draw_bar": 40.0, "pi_feed_bar": 27.0, "operating_hours": 0})
    r_high = model.predict({"pi_draw_bar": 80.0, "pi_feed_bar": 27.0, "operating_hours": 0})
    J_low  = float(np.atleast_1d(r_low["water_flux_lmh"])[0])
    J_high = float(np.atleast_1d(r_high["water_flux_lmh"])[0])
    assert J_high > J_low, f"Flux not increasing with driving force: {J_low:.3f} vs {J_high:.3f}"


def test_flux_increases_with_temperature(model):
    """Higher T → higher water permeability → higher flux."""
    r_cold = model.predict({"pi_draw_bar": 60.0, "pi_feed_bar": 27.0,
                            "T_feed_degC": 10.0, "operating_hours": 0})
    r_warm = model.predict({"pi_draw_bar": 60.0, "pi_feed_bar": 27.0,
                            "T_feed_degC": 40.0, "operating_hours": 0})
    J_cold = float(np.atleast_1d(r_cold["water_flux_lmh"])[0])
    J_warm = float(np.atleast_1d(r_warm["water_flux_lmh"])[0])
    assert J_warm > J_cold, f"Flux not increasing with T: {J_cold:.3f} vs {J_warm:.3f}"


def test_flux_declines_with_fouling(model):
    """Flux should decline as membrane fouls."""
    r_new = model.predict({"pi_draw_bar": 60.0, "pi_feed_bar": 27.0,
                           "T_feed_degC": 25.0, "operating_hours": 0})
    r_old = model.predict({"pi_draw_bar": 60.0, "pi_feed_bar": 27.0,
                           "T_feed_degC": 25.0, "operating_hours": 52560})  # 6 years
    J_new = float(np.atleast_1d(r_new["water_flux_lmh"])[0])
    J_old = float(np.atleast_1d(r_old["water_flux_lmh"])[0])
    assert J_old < J_new, f"Flux not declining with fouling: {J_new:.3f} vs {J_old:.3f}"


def test_fouling_factor_at_zero(model):
    """At t=0: fouling factor = 1.0."""
    ff = float(np.atleast_1d(model._model._fouling_factor(0.0))[0])
    assert abs(ff - 1.0) < 1e-6


def test_sec_positive(model):
    r = model.predict({"pi_draw_bar": 60.0, "pi_feed_bar": 27.0, "operating_hours": 0})
    sec = float(np.atleast_1d(r["sec_total_kwh_m3"])[0])
    assert sec > 0


def test_sec_reasonable_range(model):
    """FO SEC: dominated by draw reconcentration, typically 1-5 kWh/m3 (McGinnis 2008)."""
    r = model.predict({"pi_draw_bar": 60.0, "pi_feed_bar": 27.0,
                       "T_feed_degC": 25.0, "operating_hours": 0})
    sec = float(np.atleast_1d(r["sec_total_kwh_m3"])[0])
    assert 0.5 < sec < 15.0, f"FO SEC = {sec:.3f} kWh/m3"


def test_salt_leakage_positive(model):
    r = model.predict({"pi_draw_bar": 60.0, "pi_feed_bar": 27.0, "operating_hours": 0})
    rss = float(np.atleast_1d(r["salt_leakage_mg_l"])[0])
    assert rss >= 0


def test_array_input(model):
    pi_draws = np.linspace(30.0, 100.0, 10)
    r = model.predict({"pi_draw_bar": pi_draws, "pi_feed_bar": 27.0, "operating_hours": 0})
    assert len(np.atleast_1d(r["water_flux_lmh"])) == 10


def test_benchmark(model):
    pi_draws = np.random.uniform(30.0, 100.0, 1000)
    start = time.perf_counter()
    model.predict({"pi_draw_bar": pi_draws, "pi_feed_bar": 27.0, "operating_hours": 0})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed * 1000:.2f} ms")
    assert elapsed < 1.0
