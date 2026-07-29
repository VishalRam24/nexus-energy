"""EC140 -- Anaerobic Digester -- F1b Feedstock Variation -- Test Suite"""
import sys, time, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()


def test_predict_keys(model):
    r = model.predict({
        "feedstock_type": "cattle_manure",
        "vs_loading_kg_m3_day": 3.0,
    })
    for k in ["biogas_yield_m3_day", "methane_content_pct", "methane_yield_m3_day",
              "vs_removal_pct", "cn_ratio"]:
        assert k in r, f"Missing key: {k}"


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC140"
    assert info["fidelity"] == "F1b"


def test_all_feedstocks_produce_biogas(model):
    """Every feedstock must produce positive biogas yield."""
    feedstocks = ["cattle_manure", "food_waste", "grass_silage",
                  "sewage_sludge", "corn_silage"]
    for fs in feedstocks:
        r = model.predict({
            "feedstock_type": fs,
            "vs_loading_kg_m3_day": 3.0,
            "hrt_days": 20.0,
        })
        assert r["biogas_yield_m3_day"] > 0, f"{fs} must produce biogas"
        assert r["methane_yield_m3_day"] > 0, f"{fs} must produce methane"


def test_food_waste_higher_bmp_than_manure(model):
    """Food waste (BMP=400) should produce more methane than cattle manure (BMP=250)."""
    r_fw = model.predict({"feedstock_type": "food_waste", "vs_loading_kg_m3_day": 3.0})
    r_cm = model.predict({"feedstock_type": "cattle_manure", "vs_loading_kg_m3_day": 3.0})
    assert r_fw["methane_yield_m3_day"] > r_cm["methane_yield_m3_day"]


def test_methane_content_reasonable(model):
    """Methane content should be between 50-70%."""
    for fs in ["cattle_manure", "food_waste", "corn_silage"]:
        r = model.predict({"feedstock_type": fs, "vs_loading_kg_m3_day": 3.0})
        assert 50 <= r["methane_content_pct"] <= 70, \
            f"{fs}: CH4 content = {r['methane_content_pct']:.1f}%"


def test_cn_ratio_computed(model):
    """C/N ratio must be positive and reasonable."""
    r = model.predict({"feedstock_type": "cattle_manure", "vs_loading_kg_m3_day": 3.0})
    assert 5 < r["cn_ratio"] < 100


def test_co_digestion_blend(model):
    """Co-digestion blend must work and produce results."""
    r = model.predict({
        "feedstock_type": {"cattle_manure": 0.6, "food_waste": 0.4},
        "vs_loading_kg_m3_day": 3.0,
    })
    assert r["biogas_yield_m3_day"] > 0
    assert r["cn_ratio"] > 0


def test_co_digestion_synergy_effect(model):
    """
    Co-digestion of manure + food waste should benefit from synergy
    when C/N ratio approaches optimal range.
    """
    m = model._model
    # Check that blend has different CN than pure
    blend_props = m.blend_properties({"cattle_manure": 0.6, "food_waste": 0.4})
    single_props = m.blend_properties("cattle_manure")
    assert blend_props["cn_ratio"] != single_props["cn_ratio"]


def test_yield_increases_with_hrt(model):
    """Longer HRT should give higher methane yield (first-order kinetics)."""
    r_short = model.predict({
        "feedstock_type": "food_waste",
        "vs_loading_kg_m3_day": 3.0, "hrt_days": 5.0,
    })
    r_long = model.predict({
        "feedstock_type": "food_waste",
        "vs_loading_kg_m3_day": 3.0, "hrt_days": 30.0,
    })
    assert r_long["methane_yield_m3_day"] > r_short["methane_yield_m3_day"]


def test_yield_increases_with_vs_loading(model):
    """Higher VS loading should give more total biogas."""
    r_low = model.predict({
        "feedstock_type": "food_waste",
        "vs_loading_kg_m3_day": 1.0, "hrt_days": 20.0,
    })
    r_high = model.predict({
        "feedstock_type": "food_waste",
        "vs_loading_kg_m3_day": 5.0, "hrt_days": 20.0,
    })
    assert r_high["biogas_yield_m3_day"] > r_low["biogas_yield_m3_day"]


def test_temperature_effect(model):
    """Yield at 37 degC should be higher than at 25 degC (mesophilic optimum)."""
    r_37 = model.predict({
        "feedstock_type": "food_waste",
        "vs_loading_kg_m3_day": 3.0, "temperature_degC": 37.0,
    })
    r_25 = model.predict({
        "feedstock_type": "food_waste",
        "vs_loading_kg_m3_day": 3.0, "temperature_degC": 25.0,
    })
    assert r_37["methane_yield_m3_day"] > r_25["methane_yield_m3_day"]


def test_vs_removal_in_range(model):
    """VS removal must be between 0 and 100%."""
    r = model.predict({
        "feedstock_type": "cattle_manure",
        "vs_loading_kg_m3_day": 3.0, "hrt_days": 20.0,
    })
    assert 0 < r["vs_removal_pct"] < 100


def test_unknown_feedstock_raises(model):
    """Unknown feedstock should raise ValueError."""
    with pytest.raises(ValueError, match="Unknown feedstock"):
        model.predict({
            "feedstock_type": "unicorn_poop",
            "vs_loading_kg_m3_day": 3.0,
        })


def test_benchmark(model):
    """100 predictions must complete in < 1 second."""
    feedstocks = ["cattle_manure", "food_waste", "grass_silage"]
    start = time.perf_counter()
    for _ in range(100):
        for fs in feedstocks:
            model.predict({"feedstock_type": fs, "vs_loading_kg_m3_day": 3.0})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 300 predictions in {elapsed*1000:.1f} ms")
    assert elapsed < 1.0
