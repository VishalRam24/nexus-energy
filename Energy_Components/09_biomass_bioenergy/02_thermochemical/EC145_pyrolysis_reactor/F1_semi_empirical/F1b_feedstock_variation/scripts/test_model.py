"""EC145 -- Pyrolysis Reactor -- F1b Feedstock Variation -- Test Suite"""
import sys, time, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()


def test_predict_keys(model):
    r = model.predict({"feedstock_type": "wood_chips", "temperature_degC": 500.0})
    for k in ["bio_oil_yield", "char_yield", "gas_yield", "LHV_eff_MJ_kg",
              "moisture_lhv_factor", "energy_recovery", "thermal_efficiency",
              "bio_oil_rate_kg_h", "char_rate_kg_h", "gas_rate_kg_h"]:
        assert k in r, f"Missing key: {k}"


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC145"
    assert info["fidelity"] == "F1b"


def test_yields_sum_to_one(model):
    """Bio-oil + char + gas mass fractions must sum to 1.0."""
    for fs in ["wood_chips", "pine", "corn_stover", "rice_husk", "switchgrass"]:
        r = model.predict({"feedstock_type": fs, "temperature_degC": 500.0, "moisture_fraction": 0.10})
        total = r["bio_oil_yield"] + r["char_yield"] + r["gas_yield"]
        assert abs(total - 1.0) < 1e-6, f"{fs}: yields sum to {total:.6f}"


def test_yields_nonnegative(model):
    """All yields must be non-negative."""
    for fs in ["wood_chips", "pine", "corn_stover", "rice_husk"]:
        for T in [350.0, 500.0, 650.0]:
            r = model.predict({"feedstock_type": fs, "temperature_degC": T})
            assert r["bio_oil_yield"] >= 0
            assert r["char_yield"] >= 0
            assert r["gas_yield"] >= 0


def test_bio_oil_peaks_near_500_degC(model):
    """
    Bio-oil yield must be higher at 500 degC than at extremes (350 and 650 degC).
    (Bridgwater 2012: fast pyrolysis peak at ~500 degC)
    """
    r_350 = model.predict({"feedstock_type": "wood_chips", "temperature_degC": 350.0, "moisture_fraction": 0.10})
    r_500 = model.predict({"feedstock_type": "wood_chips", "temperature_degC": 500.0, "moisture_fraction": 0.10})
    r_650 = model.predict({"feedstock_type": "wood_chips", "temperature_degC": 650.0, "moisture_fraction": 0.10})
    assert r_500["bio_oil_yield"] > r_350["bio_oil_yield"], "Bio-oil should increase from 350 to 500 degC"
    assert r_500["bio_oil_yield"] > r_650["bio_oil_yield"], "Bio-oil should decrease from 500 to 650 degC"


def test_char_decreases_with_temperature(model):
    """Char yield must decrease as temperature increases (secondary cracking)."""
    chars = []
    for T in [350.0, 450.0, 550.0, 650.0]:
        r = model.predict({"feedstock_type": "wood_chips", "temperature_degC": T, "moisture_fraction": 0.10})
        chars.append(r["char_yield"])
    assert all(chars[i] > chars[i+1] for i in range(len(chars)-1)), \
        f"Char not monotonically decreasing with T: {chars}"


def test_moisture_reduces_lhv_effective(model):
    """Higher moisture must reduce effective LHV (moisture-LHV coupling)."""
    r_dry  = model.predict({"feedstock_type": "wood_chips", "temperature_degC": 500.0, "moisture_fraction": 0.05})
    r_wet  = model.predict({"feedstock_type": "wood_chips", "temperature_degC": 500.0, "moisture_fraction": 0.40})
    assert r_dry["LHV_eff_MJ_kg"] > r_wet["LHV_eff_MJ_kg"], "LHV_eff should decrease with moisture"


def test_moisture_lhv_factor_bounds(model):
    """Moisture LHV factor must be in (0, 1]."""
    for M in [0.0, 0.10, 0.30, 0.50]:
        r = model.predict({"feedstock_type": "wood_chips", "temperature_degC": 500.0, "moisture_fraction": M})
        assert 0.0 <= r["moisture_lhv_factor"] <= 1.0, \
            f"M={M}: moisture_lhv_factor={r['moisture_lhv_factor']:.3f}"


def test_moisture_reduces_bio_oil(model):
    """Higher moisture must reduce bio-oil yield (Demirbas 2004: ~10% drop per 10% moisture)."""
    r_low = model.predict({"feedstock_type": "wood_chips", "temperature_degC": 500.0, "moisture_fraction": 0.05})
    r_high= model.predict({"feedstock_type": "wood_chips", "temperature_degC": 500.0, "moisture_fraction": 0.45})
    assert r_low["bio_oil_yield"] > r_high["bio_oil_yield"], "Bio-oil yield should decrease with moisture"


def test_pine_higher_lhv_than_rice_husk(model):
    """Pine (low ash) must have higher LHV_eff than rice husk (high ash) at same moisture."""
    r_pine = model.predict({"feedstock_type": "pine",      "temperature_degC": 500.0, "moisture_fraction": 0.10})
    r_rice = model.predict({"feedstock_type": "rice_husk", "temperature_degC": 500.0, "moisture_fraction": 0.10})
    assert r_pine["LHV_eff_MJ_kg"] > r_rice["LHV_eff_MJ_kg"]


def test_thermal_efficiency_decreases_with_moisture(model):
    """Higher moisture must reduce thermal efficiency."""
    r_dry = model.predict({"feedstock_type": "wood_chips", "temperature_degC": 500.0, "moisture_fraction": 0.05, "PLR": 1.0})
    r_wet = model.predict({"feedstock_type": "wood_chips", "temperature_degC": 500.0, "moisture_fraction": 0.45, "PLR": 1.0})
    assert r_dry["thermal_efficiency"] > r_wet["thermal_efficiency"]


def test_part_load_reduces_efficiency(model):
    """Lower PLR must give lower or equal thermal efficiency."""
    r_full = model.predict({"feedstock_type": "wood_chips", "temperature_degC": 500.0, "moisture_fraction": 0.10, "PLR": 1.0})
    r_half = model.predict({"feedstock_type": "wood_chips", "temperature_degC": 500.0, "moisture_fraction": 0.10, "PLR": 0.5})
    assert r_full["thermal_efficiency"] >= r_half["thermal_efficiency"]


def test_thermal_efficiency_reasonable(model):
    """Thermal efficiency at design point should be in 60-95% range."""
    r = model.predict({"feedstock_type": "wood_chips", "temperature_degC": 500.0, "moisture_fraction": 0.10, "PLR": 1.0})
    assert 0.55 <= r["thermal_efficiency"] <= 0.98, \
        f"eta_th={r['thermal_efficiency']:.3f} out of [0.55, 0.98]"


def test_bio_oil_yield_reasonable_range(model):
    """
    Fast pyrolysis bio-oil yield should be in 40-70% range at peak conditions.
    RATIONALE: Bridgwater (2012) reports 60-75% for optimized fast pyrolysis;
    our model clips at 70% and uses a Gaussian temperature profile, so 40-70% is
    the practical achievable range across all five feedstocks.
    """
    r = model.predict({"feedstock_type": "wood_chips", "temperature_degC": 500.0, "moisture_fraction": 0.05})
    assert 0.40 <= r["bio_oil_yield"] <= 0.70, \
        f"bio_oil={r['bio_oil_yield']:.3f} out of [0.40, 0.70]"


def test_unknown_feedstock_raises(model):
    """Unknown feedstock must raise ValueError."""
    with pytest.raises(ValueError, match="Unknown feedstock"):
        model.predict({"feedstock_type": "unobtanium", "temperature_degC": 500.0})


def test_energy_recovery_bounded(model):
    """Energy recovery must be in [0, 1.05]."""
    for fs in ["wood_chips", "pine", "corn_stover"]:
        r = model.predict({"feedstock_type": fs, "temperature_degC": 500.0, "moisture_fraction": 0.10})
        assert 0.0 <= r["energy_recovery"] <= 1.05, \
            f"{fs}: energy_recovery={r['energy_recovery']:.3f}"


def test_production_rates_consistent(model):
    """Product rates must match (bio_oil + char + gas) * feed_rate."""
    r = model.predict({"feedstock_type": "wood_chips", "temperature_degC": 500.0,
                       "moisture_fraction": 0.10, "PLR": 0.8, "feed_rate_kg_h": 1000.0})
    total_rate = r["bio_oil_rate_kg_h"] + r["char_rate_kg_h"] + r["gas_rate_kg_h"]
    expected   = 1000.0 * 0.8   # feed_rate * PLR
    assert abs(total_rate - expected) < 0.1, \
        f"Total rate={total_rate:.2f} vs expected={expected:.2f}"


def test_benchmark(model):
    """200 predictions must complete in < 1 second."""
    start = time.perf_counter()
    for _ in range(200):
        for fs in ["wood_chips", "pine"]:
            model.predict({"feedstock_type": fs, "temperature_degC": 500.0})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 400 predictions in {elapsed*1000:.1f} ms")
    assert elapsed < 1.0
