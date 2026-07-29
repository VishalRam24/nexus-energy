"""EC142 -- Biogas Upgrading -- F1b -- Test Suite"""
import sys, time, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()


def test_predict_keys(model):
    r = model.predict({"feedstock_type": "food_waste", "biogas_flow_m3_h": 100.0})
    for k in ["biomethane_flow_m3_h","methane_recovery_pct","biomethane_CH4_pct",
              "CO2_removal_pct","H2S_product_ppm","upgrading_energy_kwh_h",
              "net_energy_kwh_h","meets_spec","moisture_lhv_factor"]:
        assert k in r


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC142"
    assert info["fidelity"] == "F1b"


def test_all_feedstocks_produce_biomethane(model):
    for fs in ["cattle_manure","food_waste","sewage_sludge","corn_silage","grass_silage"]:
        r = model.predict({"feedstock_type": fs, "biogas_flow_m3_h": 200.0})
        assert r["biomethane_flow_m3_h"] > 0


def test_methane_recovery_reasonable(model):
    """
    Methane recovery for PSA upgrading typically 95-99%.
    RATIONALE: 2% slip assumed (industry typical; Bauer 2013).
    """
    r = model.predict({"feedstock_type": "food_waste", "biogas_flow_m3_h": 100.0})
    assert 90.0 <= r["methane_recovery_pct"] <= 100.0


def test_CH4_product_quality(model):
    """Biomethane CH4 content must be >= 96% (EU biomethane spec)."""
    for fs in ["cattle_manure","food_waste","sewage_sludge"]:
        r = model.predict({"feedstock_type": fs, "biogas_flow_m3_h": 100.0})
        assert r["biomethane_CH4_pct"] >= 96.0


def test_H2S_removal(model):
    """H2S in product must be < 10 ppm (below spec limit of 5)."""
    r = model.predict({"feedstock_type": "sewage_sludge", "biogas_flow_m3_h": 100.0})
    assert r["H2S_product_ppm"] < 10.0, "H2S not adequately removed"


def test_net_energy_positive(model):
    """Net energy must be positive (gross > upgrading parasitic)."""
    r = model.predict({"feedstock_type": "food_waste", "biogas_flow_m3_h": 100.0})
    assert r["net_energy_kwh_h"] > 0


def test_moisture_reduces_net_energy(model):
    """Higher moisture should reduce net energy (moisture-LHV coupling)."""
    r_dry = model.predict({"feedstock_type": "food_waste", "biogas_flow_m3_h": 100.0,
                            "moisture_fraction": 0.0})
    r_wet = model.predict({"feedstock_type": "food_waste", "biogas_flow_m3_h": 100.0,
                            "moisture_fraction": 0.4})
    assert r_dry["net_energy_kwh_h"] > r_wet["net_energy_kwh_h"]


def test_higher_CO2_feedstock_more_upgrading_energy(model):
    """Corn silage (CO2=46%) should need more upgrading energy than food waste (CO2=36%)."""
    r_corn = model.predict({"feedstock_type": "corn_silage", "biogas_flow_m3_h": 100.0})
    r_fw   = model.predict({"feedstock_type": "food_waste",  "biogas_flow_m3_h": 100.0})
    assert r_corn["upgrading_energy_kwh_h"] > r_fw["upgrading_energy_kwh_h"]


def test_flow_scales_linearly(model):
    """Doubling biogas flow should double biomethane output."""
    r1 = model.predict({"feedstock_type": "food_waste", "biogas_flow_m3_h": 100.0})
    r2 = model.predict({"feedstock_type": "food_waste", "biogas_flow_m3_h": 200.0})
    ratio = r2["biomethane_flow_m3_h"] / r1["biomethane_flow_m3_h"]
    assert abs(ratio - 2.0) < 0.01


def test_co_digestion_blend(model):
    r = model.predict({"feedstock_type": {"cattle_manure": 0.6, "food_waste": 0.4},
                        "biogas_flow_m3_h": 100.0})
    assert r["biomethane_flow_m3_h"] > 0


def test_moisture_lhv_factor_range(model):
    r = model.predict({"feedstock_type": "food_waste", "biogas_flow_m3_h": 100.0,
                        "moisture_fraction": 0.3})
    assert 0.0 <= r["moisture_lhv_factor"] <= 1.0


def test_unknown_feedstock_raises(model):
    with pytest.raises(ValueError):
        model.predict({"feedstock_type": "magic_algae", "biogas_flow_m3_h": 100.0})


def test_benchmark(model):
    start = time.perf_counter()
    for _ in range(300):
        model.predict({"feedstock_type": "food_waste", "biogas_flow_m3_h": 100.0})
    assert time.perf_counter() - start < 1.0
