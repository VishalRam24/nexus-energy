"""EC150 -- FT BtL -- F1b Feedstock Variation -- Test Suite"""
import sys, time, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()


def test_predict_keys(model):
    r = model.predict({"feedstock_type": "wood_chips"})
    for k in ["ft_liquid_yield", "product_selectivity", "alpha", "co_conversion",
              "h2_co_ratio", "LHV_eff_MJ_kg", "moisture_lhv_factor",
              "thermal_efficiency", "ft_liquid_rate_kg_h", "diesel_rate_kg_h"]:
        assert k in r, f"Missing key: {k}"


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC150"
    assert info["fidelity"] == "F1b"


def test_product_selectivity_sums_to_one(model):
    """ASF product fractions must sum to 1.0."""
    for fs in ["wood_chips", "pine", "torrefied_wood"]:
        r = model.predict({"feedstock_type": fs})
        total = sum(r["product_selectivity"].values())
        assert abs(total - 1.0) < 1e-6, f"{fs}: selectivity sums to {total:.6f}"


def test_alpha_in_valid_range(model):
    """ASF alpha must be in [0.40, 0.95]."""
    for fs in ["wood_chips", "agricultural_residue", "municipal_solid_waste"]:
        r = model.predict({"feedstock_type": fs})
        assert 0.40 <= r["alpha"] <= 0.95, f"{fs}: alpha={r['alpha']:.3f}"


def test_alpha_decreases_with_temperature(model):
    """Higher FT temperature -> lower alpha (lighter products at higher T)."""
    r_low  = model.predict({"feedstock_type": "wood_chips", "temperature_degC": 200.0})
    r_high = model.predict({"feedstock_type": "wood_chips", "temperature_degC": 320.0})
    assert r_low["alpha"] > r_high["alpha"], \
        f"alpha not decreasing with T: {r_low['alpha']:.3f} -> {r_high['alpha']:.3f}"


def test_high_alpha_favors_wax(model):
    """
    High alpha (low T) must give higher wax selectivity than low alpha (high T).
    ASF distribution: high alpha shifts to heavy hydrocarbons.
    """
    r_lt = model.predict({"feedstock_type": "wood_chips", "temperature_degC": 200.0})
    r_ht = model.predict({"feedstock_type": "wood_chips", "temperature_degC": 300.0})
    assert r_lt["product_selectivity"]["wax"] > r_ht["product_selectivity"]["wax"], \
        "Wax selectivity should decrease at higher T (lower alpha)"


def test_torrefied_higher_h2co_than_msw(model):
    """Torrefied wood should have higher H2/CO ratio than MSW."""
    r_torr = model.predict({"feedstock_type": "torrefied_wood"})
    r_msw  = model.predict({"feedstock_type": "municipal_solid_waste"})
    assert r_torr["h2_co_ratio"] > r_msw["h2_co_ratio"], \
        f"torr H2/CO={r_torr['h2_co_ratio']:.2f} not > MSW={r_msw['h2_co_ratio']:.2f}"


def test_moisture_increases_h2co(model):
    """Higher moisture -> more steam -> water-gas shift -> higher H2/CO."""
    r_dry = model.predict({"feedstock_type": "wood_chips", "moisture_fraction": 0.05})
    r_wet = model.predict({"feedstock_type": "wood_chips", "moisture_fraction": 0.40})
    assert r_wet["h2_co_ratio"] > r_dry["h2_co_ratio"]


def test_moisture_reduces_lhv(model):
    r_dry = model.predict({"feedstock_type": "wood_chips", "moisture_fraction": 0.05})
    r_wet = model.predict({"feedstock_type": "wood_chips", "moisture_fraction": 0.40})
    assert r_dry["LHV_eff_MJ_kg"] > r_wet["LHV_eff_MJ_kg"]


def test_co_conversion_reasonable(model):
    """CO conversion must be in [0.20, 0.95]."""
    r = model.predict({"feedstock_type": "wood_chips", "temperature_degC": 230.0})
    assert 0.20 <= r["co_conversion"] <= 0.95, \
        f"co_conversion={r['co_conversion']:.3f}"


def test_ft_liquid_yield_positive(model):
    for fs in ["wood_chips", "pine", "torrefied_wood"]:
        r = model.predict({"feedstock_type": fs})
        assert r["ft_liquid_yield"] > 0, f"{fs}: ft_liquid_yield <= 0"


def test_diesel_rate_less_than_total(model):
    """Diesel rate must be less than total FT liquid rate."""
    r = model.predict({"feedstock_type": "wood_chips", "feed_rate_kg_h": 1000.0})
    assert r["diesel_rate_kg_h"] < r["ft_liquid_rate_kg_h"]


def test_unknown_feedstock_raises(model):
    with pytest.raises(ValueError, match="Unknown feedstock"):
        model.predict({"feedstock_type": "fossil_coal"})


def test_thermal_efficiency_reasonable(model):
    r = model.predict({"feedstock_type": "wood_chips", "PLR": 1.0, "moisture_fraction": 0.10})
    assert 0.30 <= r["thermal_efficiency"] <= 1.0


def test_benchmark(model):
    start = time.perf_counter()
    for _ in range(200):
        model.predict({"feedstock_type": "wood_chips", "temperature_degC": 230.0})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 200 predictions in {elapsed*1000:.1f} ms")
    assert elapsed < 1.0
