"""EC147 -- HTL -- F1b Feedstock Variation -- Test Suite"""
import sys, time, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()


def test_predict_keys(model):
    r = model.predict({"feedstock_type": "microalgae_chlorella", "temperature_degC": 330.0})
    for k in ["bio_crude_yield", "product_distribution", "energy_recovery",
              "LHV_eff_MJ_kg", "moisture_lhv_factor", "thermal_efficiency", "bio_crude_rate_kg_h"]:
        assert k in r, f"Missing key: {k}"


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC147"
    assert info["fidelity"] == "F1b"


def test_product_distribution_keys(model):
    r = model.predict({"feedstock_type": "microalgae_chlorella"})
    pd = r["product_distribution"]
    for k in ["bio_crude", "aqueous", "gas", "solid"]:
        assert k in pd


def test_product_distribution_sums_within_one(model):
    """
    Product distribution must sum to <= 1.0 (not all mass must be accounted for
    in the four phases — some small unaccounted fraction is acceptable).
    """
    for fs in ["microalgae_chlorella", "sewage_sludge", "wood_biomass"]:
        r = model.predict({"feedstock_type": fs, "temperature_degC": 330.0, "moisture_fraction": 0.80})
        total = sum(r["product_distribution"].values())
        assert total <= 1.001, f"{fs}: product distribution sums to {total:.4f} > 1"


def test_algae_higher_biocrude_than_wood(model):
    """Algae (high lipid) should give higher bio-crude yield than wood (low lipid)."""
    r_algae = model.predict({"feedstock_type": "microalgae_nannochloropsis", "temperature_degC": 330.0, "moisture_fraction": 0.80})
    r_wood  = model.predict({"feedstock_type": "wood_biomass",               "temperature_degC": 330.0, "moisture_fraction": 0.80})
    assert r_algae["bio_crude_yield"] > r_wood["bio_crude_yield"], \
        f"algae_bc={r_algae['bio_crude_yield']:.3f} not > wood_bc={r_wood['bio_crude_yield']:.3f}"


def test_biocrude_peaks_near_330_degC(model):
    """Bio-crude yield should be higher at 330 degC than at 270 or 380 degC."""
    r_270 = model.predict({"feedstock_type": "microalgae_chlorella", "temperature_degC": 270.0, "moisture_fraction": 0.80})
    r_330 = model.predict({"feedstock_type": "microalgae_chlorella", "temperature_degC": 330.0, "moisture_fraction": 0.80})
    r_380 = model.predict({"feedstock_type": "microalgae_chlorella", "temperature_degC": 380.0, "moisture_fraction": 0.80})
    assert r_330["bio_crude_yield"] > r_270["bio_crude_yield"], "Bio-crude should increase from 270 to 330 degC"
    assert r_330["bio_crude_yield"] > r_380["bio_crude_yield"], "Bio-crude should decrease from 330 to 380 degC"


def test_moisture_lhv_factor_bounds(model):
    """Moisture LHV factor must be in [0, 1] for high moisture feedstocks."""
    for M in [0.10, 0.50, 0.80]:
        r = model.predict({"feedstock_type": "microalgae_chlorella", "moisture_fraction": M})
        assert 0.0 <= r["moisture_lhv_factor"] <= 1.0, \
            f"M={M}: moisture_lhv_factor={r['moisture_lhv_factor']:.3f}"


def test_biocrude_yield_positive(model):
    """Bio-crude yield must be positive for all feedstocks."""
    for fs in ["microalgae_chlorella", "microalgae_nannochloropsis", "sewage_sludge", "wood_biomass", "macroalgae_laminaria"]:
        r = model.predict({"feedstock_type": fs, "temperature_degC": 330.0})
        assert r["bio_crude_yield"] > 0, f"{fs}: bio_crude_yield <= 0"


def test_energy_recovery_bounded(model):
    """Energy recovery must be in [0, 1]."""
    for fs in ["microalgae_chlorella", "sewage_sludge", "wood_biomass"]:
        r = model.predict({"feedstock_type": fs, "temperature_degC": 330.0, "moisture_fraction": 0.80})
        assert 0.0 <= r["energy_recovery"] <= 1.0, \
            f"{fs}: energy_recovery={r['energy_recovery']:.3f}"


def test_thermal_efficiency_reasonable(model):
    r = model.predict({"feedstock_type": "microalgae_chlorella", "temperature_degC": 330.0,
                       "moisture_fraction": 0.70, "PLR": 1.0})
    assert 0.10 <= r["thermal_efficiency"] <= 1.0


def test_unknown_feedstock_raises(model):
    with pytest.raises(ValueError, match="Unknown feedstock"):
        model.predict({"feedstock_type": "unicorn_algae"})


def test_biocrude_rate_positive(model):
    r = model.predict({"feedstock_type": "microalgae_chlorella", "temperature_degC": 330.0,
                       "moisture_fraction": 0.80, "PLR": 1.0, "feed_rate_kg_h": 1000.0})
    assert r["bio_crude_rate_kg_h"] > 0


def test_benchmark(model):
    start = time.perf_counter()
    for _ in range(200):
        model.predict({"feedstock_type": "microalgae_chlorella", "temperature_degC": 330.0})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 200 predictions in {elapsed*1000:.1f} ms")
    assert elapsed < 1.0
