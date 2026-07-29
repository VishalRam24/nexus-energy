"""EC149 -- Biodiesel Transesterification -- F1b Feedstock Variation -- Test Suite"""
import sys, time, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()


def test_predict_keys(model):
    r = model.predict({"feedstock_type": "soybean_oil"})
    for k in ["biodiesel_yield", "glycerol_yield", "oil_content",
              "temperature_factor", "ffa_penalty_factor", "LHV_eff_MJ_kg",
              "moisture_lhv_factor", "thermal_efficiency",
              "biodiesel_rate_kg_h", "glycerol_rate_kg_h"]:
        assert k in r, f"Missing key: {k}"


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC149"
    assert info["fidelity"] == "F1b"


def test_biodiesel_yield_positive(model):
    for fs in ["soybean_oil", "palm_oil", "rapeseed_oil", "waste_cooking_oil", "jatropha"]:
        r = model.predict({"feedstock_type": fs})
        assert r["biodiesel_yield"] > 0, f"{fs}: yield <= 0"


def test_rapeseed_higher_per_oil_yield_than_waste_oil(model):
    """
    On a per-unit-oil basis, rapeseed (FFA=0.3%) should outperform WCO (FFA=6%).
    RATIONALE: WCO has 90% oil content vs rapeseed 40%, so absolute FAME yield
    from raw feed is higher for WCO — but if we normalize by oil content (yield/oil),
    rapeseed is more efficient because it has lower FFA and thus less saponification.
    The correct industrial measure is conversion efficiency per unit of oil.
    """
    r_rape = model.predict({"feedstock_type": "rapeseed_oil", "temperature_degC": 60.0})
    r_wco  = model.predict({"feedstock_type": "waste_cooking_oil", "temperature_degC": 60.0})
    # Per-unit-oil FAME yield
    yield_per_oil_rape = r_rape["biodiesel_yield"] / r_rape["oil_content"]
    yield_per_oil_wco  = r_wco["biodiesel_yield"]  / r_wco["oil_content"]
    assert yield_per_oil_rape > yield_per_oil_wco, \
        f"rapeseed per-oil={yield_per_oil_rape:.3f} not > WCO per-oil={yield_per_oil_wco:.3f}"


def test_ffa_reduces_yield(model):
    """Higher FFA must reduce biodiesel yield via saponification penalty."""
    r_low_ffa  = model.predict({"feedstock_type": "soybean_oil", "ffa_pct": 0.5})
    r_high_ffa = model.predict({"feedstock_type": "soybean_oil", "ffa_pct": 8.0})
    assert r_low_ffa["biodiesel_yield"] > r_high_ffa["biodiesel_yield"], \
        "Yield should decrease with higher FFA"


def test_ffa_penalty_factor_below_one_for_high_ffa(model):
    """FFA penalty factor must be < 1 when FFA > 1%."""
    r = model.predict({"feedstock_type": "soybean_oil", "ffa_pct": 4.0})
    assert r["ffa_penalty_factor"] < 1.0


def test_ffa_penalty_factor_near_one_for_low_ffa(model):
    """FFA penalty factor must be ~1 when FFA <= 1%."""
    r = model.predict({"feedstock_type": "soybean_oil", "ffa_pct": 0.5})
    assert abs(r["ffa_penalty_factor"] - 1.0) < 0.01


def test_temperature_factor_peaks_near_60_degC(model):
    """Reaction rate factor must peak at ~55-65 degC."""
    f_40 = model.predict({"feedstock_type": "soybean_oil", "temperature_degC": 40.0})["temperature_factor"]
    f_60 = model.predict({"feedstock_type": "soybean_oil", "temperature_degC": 60.0})["temperature_factor"]
    f_80 = model.predict({"feedstock_type": "soybean_oil", "temperature_degC": 80.0})["temperature_factor"]
    assert f_60 > f_40, "f_T at 60°C should be higher than at 40°C"
    assert f_60 > f_80, "f_T at 60°C should be higher than at 80°C"


def test_moisture_reduces_lhv(model):
    r_dry = model.predict({"feedstock_type": "soybean_oil", "moisture_fraction": 0.01})
    r_wet = model.predict({"feedstock_type": "soybean_oil", "moisture_fraction": 0.20})
    assert r_dry["LHV_eff_MJ_kg"] > r_wet["LHV_eff_MJ_kg"]


def test_moisture_lhv_factor_bounds(model):
    for M in [0.01, 0.10, 0.20]:
        r = model.predict({"feedstock_type": "soybean_oil", "moisture_fraction": M})
        assert 0.0 < r["moisture_lhv_factor"] <= 1.0


def test_glycerol_is_coproduct(model):
    """Glycerol yield must be positive and less than biodiesel yield."""
    r = model.predict({"feedstock_type": "soybean_oil"})
    assert r["glycerol_yield"] > 0
    assert r["glycerol_yield"] < r["biodiesel_yield"]


def test_soybean_yield_realistic(model):
    """
    Soybean biodiesel yield should be ~0.16-0.20 kg FAME/kg raw seed.
    RATIONALE: soybean oil content 18%, conversion ~98%, stoich factor 1.02
    -> Y = 0.18 * 0.98 * 1.02 * f_T ~0.18; range broadened to 0.12-0.22
    to account for FFA and temperature effects.
    """
    r = model.predict({"feedstock_type": "soybean_oil", "temperature_degC": 60.0})
    assert 0.12 <= r["biodiesel_yield"] <= 0.22, \
        f"soybean biodiesel_yield={r['biodiesel_yield']:.3f} out of [0.12, 0.22]"


def test_unknown_feedstock_raises(model):
    with pytest.raises(ValueError, match="Unknown feedstock"):
        model.predict({"feedstock_type": "mystery_oil"})


def test_production_rates_positive(model):
    r = model.predict({"feedstock_type": "rapeseed_oil", "PLR": 0.8, "feed_rate_kg_h": 1000.0})
    assert r["biodiesel_rate_kg_h"] > 0
    assert r["glycerol_rate_kg_h"] > 0


def test_benchmark(model):
    start = time.perf_counter()
    for _ in range(200):
        model.predict({"feedstock_type": "soybean_oil", "temperature_degC": 60.0})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 200 predictions in {elapsed*1000:.1f} ms")
    assert elapsed < 1.0
