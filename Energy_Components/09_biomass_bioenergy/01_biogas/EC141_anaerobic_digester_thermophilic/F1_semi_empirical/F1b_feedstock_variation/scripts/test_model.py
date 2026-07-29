"""EC141 -- Anaerobic Digester Thermophilic -- F1b -- Test Suite"""
import sys, time, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()


def test_predict_keys(model):
    r = model.predict({"feedstock_type": "food_waste", "vs_loading_kg_m3_day": 3.0})
    for k in ["biogas_yield_m3_day", "methane_content_pct", "methane_yield_m3_day",
              "vs_removal_pct", "cn_ratio", "moisture_lhv_factor"]:
        assert k in r, f"Missing key: {k}"


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC141"
    assert info["fidelity"] == "F1b"


def test_all_feedstocks_produce_biogas(model):
    for fs in ["cattle_manure", "food_waste", "grass_silage", "sewage_sludge", "corn_silage"]:
        r = model.predict({"feedstock_type": fs, "vs_loading_kg_m3_day": 3.0})
        assert r["biogas_yield_m3_day"] > 0
        assert r["methane_yield_m3_day"] > 0


def test_food_waste_higher_than_manure(model):
    r_fw = model.predict({"feedstock_type": "food_waste", "vs_loading_kg_m3_day": 3.0})
    r_cm = model.predict({"feedstock_type": "cattle_manure", "vs_loading_kg_m3_day": 3.0})
    assert r_fw["methane_yield_m3_day"] > r_cm["methane_yield_m3_day"]


def test_methane_content_range(model):
    for fs in ["cattle_manure", "food_waste", "corn_silage"]:
        r = model.predict({"feedstock_type": fs, "vs_loading_kg_m3_day": 3.0})
        assert 50 <= r["methane_content_pct"] <= 75


def test_thermophilic_optimum(model):
    """
    Yield at 55 degC should be higher than at 37 degC (mesophilic).
    RATIONALE: Thermophilic Arrhenius factor at T_ref=55 degC = 1.0;
    at 37 degC it is lower (exp(-Ea/R*(1/310K - 1/328K)) << 1).
    """
    r_55 = model.predict({"feedstock_type": "food_waste", "vs_loading_kg_m3_day": 3.0,
                           "temperature_degC": 55.0})
    r_37 = model.predict({"feedstock_type": "food_waste", "vs_loading_kg_m3_day": 3.0,
                           "temperature_degC": 37.0})
    assert r_55["methane_yield_m3_day"] > r_37["methane_yield_m3_day"]


def test_inhibition_at_low_temp(model):
    """Below 45 degC should give lower yield than at 55 degC."""
    r_low  = model.predict({"feedstock_type": "food_waste", "vs_loading_kg_m3_day": 3.0,
                             "temperature_degC": 40.0})
    r_high = model.predict({"feedstock_type": "food_waste", "vs_loading_kg_m3_day": 3.0,
                             "temperature_degC": 55.0})
    assert r_high["methane_yield_m3_day"] > r_low["methane_yield_m3_day"]


def test_moisture_lhv_penalty(model):
    """High moisture should reduce effective yield due to LHV correction."""
    r_dry = model.predict({"feedstock_type": "food_waste", "vs_loading_kg_m3_day": 3.0,
                            "moisture_fraction": 0.0})
    r_wet = model.predict({"feedstock_type": "food_waste", "vs_loading_kg_m3_day": 3.0,
                            "moisture_fraction": 0.4})
    assert r_dry["methane_yield_m3_day"] > r_wet["methane_yield_m3_day"]
    assert r_dry["moisture_lhv_factor"] >= r_wet["moisture_lhv_factor"]


def test_moisture_lhv_factor_range(model):
    """LHV factor must be between 0 and 1."""
    r = model.predict({"feedstock_type": "food_waste", "vs_loading_kg_m3_day": 3.0,
                        "moisture_fraction": 0.3})
    assert 0.0 <= r["moisture_lhv_factor"] <= 1.0


def test_co_digestion_blend(model):
    r = model.predict({"feedstock_type": {"cattle_manure": 0.6, "food_waste": 0.4},
                        "vs_loading_kg_m3_day": 3.0})
    assert r["biogas_yield_m3_day"] > 0


def test_yield_increases_with_hrt(model):
    r_short = model.predict({"feedstock_type": "food_waste",
                              "vs_loading_kg_m3_day": 3.0, "hrt_days": 5.0})
    r_long  = model.predict({"feedstock_type": "food_waste",
                              "vs_loading_kg_m3_day": 3.0, "hrt_days": 25.0})
    assert r_long["methane_yield_m3_day"] > r_short["methane_yield_m3_day"]


def test_vs_removal_in_range(model):
    r = model.predict({"feedstock_type": "cattle_manure", "vs_loading_kg_m3_day": 3.0})
    assert 0 < r["vs_removal_pct"] < 100


def test_unknown_feedstock_raises(model):
    with pytest.raises(ValueError, match="Unknown feedstock"):
        model.predict({"feedstock_type": "unicorn_poop", "vs_loading_kg_m3_day": 3.0})


def test_benchmark(model):
    start = time.perf_counter()
    for _ in range(300):
        model.predict({"feedstock_type": "food_waste", "vs_loading_kg_m3_day": 3.0})
    elapsed = time.perf_counter() - start
    assert elapsed < 1.0
