"""EC143 -- Biomass Gasifier -- F1b Feedstock Variation -- Test Suite"""
import sys, time, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()


def test_predict_keys(model):
    r = model.predict({
        "feedstock_type": "wood",
        "equivalence_ratio": 0.25,
    })
    for k in ["syngas_composition", "syngas_yield_nm3_kg",
              "cold_gas_efficiency", "tar_content_g_nm3", "lhv_syngas_mj_nm3"]:
        assert k in r, f"Missing key: {k}"


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC143"
    assert info["fidelity"] == "F1b"


def test_syngas_composition_sums_to_one(model):
    """Syngas mole fractions must sum to 1.0."""
    for fs in ["wood", "rice_husk", "pine", "corn_stover", "sewage_sludge"]:
        r = model.predict({"feedstock_type": fs, "equivalence_ratio": 0.25})
        comp = r["syngas_composition"]
        total = sum(comp.values())
        assert abs(total - 1.0) < 0.01, f"{fs}: fractions sum to {total:.4f}"


def test_all_feedstocks_produce_syngas(model):
    """Every feedstock must produce syngas with positive LHV."""
    for fs in ["wood", "rice_husk", "pine", "corn_stover", "sewage_sludge"]:
        r = model.predict({"feedstock_type": fs, "equivalence_ratio": 0.25})
        assert r["lhv_syngas_mj_nm3"] > 0, f"{fs}: LHV must be positive"
        assert r["cold_gas_efficiency"] > 0, f"{fs}: CGE must be positive"


def test_co_decreases_with_er(model):
    """Higher ER should decrease CO (more oxidation)."""
    r_low = model.predict({"feedstock_type": "wood", "equivalence_ratio": 0.20})
    r_high = model.predict({"feedstock_type": "wood", "equivalence_ratio": 0.40})
    assert r_low["syngas_composition"]["CO"] > r_high["syngas_composition"]["CO"]


def test_lhv_decreases_with_er(model):
    """Higher ER should decrease syngas LHV (more dilution)."""
    r_low = model.predict({"feedstock_type": "wood", "equivalence_ratio": 0.20})
    r_high = model.predict({"feedstock_type": "wood", "equivalence_ratio": 0.40})
    assert r_low["lhv_syngas_mj_nm3"] > r_high["lhv_syngas_mj_nm3"]


def test_tar_decreases_with_er(model):
    """Higher ER should decrease tar content."""
    r_low = model.predict({"feedstock_type": "wood", "equivalence_ratio": 0.20})
    r_high = model.predict({"feedstock_type": "wood", "equivalence_ratio": 0.40})
    assert r_low["tar_content_g_nm3"] > r_high["tar_content_g_nm3"]


def test_wood_higher_lhv_than_rice_husk(model):
    """Wood (higher C, lower ash) should give higher syngas LHV than rice husk."""
    r_wood = model.predict({"feedstock_type": "wood", "equivalence_ratio": 0.25})
    r_rh = model.predict({"feedstock_type": "rice_husk", "equivalence_ratio": 0.25})
    assert r_wood["lhv_syngas_mj_nm3"] > r_rh["lhv_syngas_mj_nm3"]


def test_moisture_reduces_cge(model):
    """Higher moisture should reduce cold gas efficiency."""
    r_dry = model.predict({
        "feedstock_type": "wood", "equivalence_ratio": 0.25,
        "moisture_content": 0.05,
    })
    r_wet = model.predict({
        "feedstock_type": "wood", "equivalence_ratio": 0.25,
        "moisture_content": 0.40,
    })
    assert r_dry["cold_gas_efficiency"] > r_wet["cold_gas_efficiency"]


def test_cge_in_reasonable_range(model):
    """Cold gas efficiency should be between 30-95%."""
    for fs in ["wood", "pine", "corn_stover"]:
        r = model.predict({"feedstock_type": fs, "equivalence_ratio": 0.25})
        assert 0.30 < r["cold_gas_efficiency"] < 0.95, \
            f"{fs}: CGE={r['cold_gas_efficiency']:.3f}"


def test_lhv_in_reasonable_range(model):
    """Syngas LHV should be between 2-8 MJ/Nm3 for air gasification."""
    for fs in ["wood", "rice_husk", "pine"]:
        r = model.predict({"feedstock_type": fs, "equivalence_ratio": 0.25})
        assert 2.0 < r["lhv_syngas_mj_nm3"] < 8.0, \
            f"{fs}: LHV={r['lhv_syngas_mj_nm3']:.2f} MJ/Nm3"


def test_composition_fractions_nonnegative(model):
    """All composition fractions must be non-negative."""
    for fs in ["wood", "rice_husk", "sewage_sludge"]:
        for er in [0.15, 0.25, 0.35, 0.45]:
            r = model.predict({"feedstock_type": fs, "equivalence_ratio": er})
            for species, val in r["syngas_composition"].items():
                assert val >= 0, f"{fs} ER={er}: {species}={val}"


def test_unknown_feedstock_raises(model):
    """Unknown feedstock should raise ValueError."""
    with pytest.raises(ValueError, match="Unknown feedstock"):
        model.predict({"feedstock_type": "unobtanium", "equivalence_ratio": 0.25})


def test_benchmark(model):
    """100 predictions must complete in < 1 second."""
    start = time.perf_counter()
    for _ in range(100):
        for fs in ["wood", "rice_husk"]:
            model.predict({"feedstock_type": fs, "equivalence_ratio": 0.25})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 200 predictions in {elapsed*1000:.1f} ms")
    assert elapsed < 1.0
