"""EC153 -- Binary Cycle Geothermal -- F1b Part-Load Ambient -- Test Suite"""
import sys, time, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()


def test_predict_keys(model):
    r = model.predict({
        "T_brine_degC": 150.0, "brine_flow_kg_s": 80.0,
        "T_ambient_degC": 25.0,
    })
    for k in ["power_output_kw", "efficiency", "resource_factor", "condenser_factor"]:
        assert k in r, f"Missing key: {k}"


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC153"
    assert info["fidelity"] == "F1b"


def test_power_positive_at_design(model):
    """At design conditions, power must be positive."""
    r = model.predict({
        "T_brine_degC": 150.0, "brine_flow_kg_s": 80.0,
        "T_ambient_degC": 30.0, "PLR": 1.0,
    })
    assert r["power_output_kw"] > 0


def test_power_bounded_by_rating(model):
    """Power output must not exceed rated capacity (5000 kW)."""
    r = model.predict({
        "T_brine_degC": 200.0, "brine_flow_kg_s": 200.0,
        "T_ambient_degC": 10.0, "PLR": 1.0,
    })
    assert r["power_output_kw"] <= 5000.0


def test_power_decreases_with_lower_plr(model):
    """Lower PLR should give lower power output."""
    r_full = model.predict({
        "T_brine_degC": 150.0, "brine_flow_kg_s": 80.0,
        "T_ambient_degC": 25.0, "PLR": 1.0,
    })
    r_half = model.predict({
        "T_brine_degC": 150.0, "brine_flow_kg_s": 80.0,
        "T_ambient_degC": 25.0, "PLR": 0.5,
    })
    assert r_full["power_output_kw"] > r_half["power_output_kw"]


def test_power_decreases_with_higher_ambient(model):
    """Higher ambient temperature should reduce power output."""
    r_cool = model.predict({
        "T_brine_degC": 150.0, "brine_flow_kg_s": 80.0,
        "T_ambient_degC": 15.0, "PLR": 1.0,
    })
    r_hot = model.predict({
        "T_brine_degC": 150.0, "brine_flow_kg_s": 80.0,
        "T_ambient_degC": 45.0, "PLR": 1.0,
    })
    assert r_cool["power_output_kw"] > r_hot["power_output_kw"]


def test_condenser_factor_unity_at_design(model):
    """Condenser factor should be ~1.0 at design T_cond=30 degC."""
    r = model.predict({
        "T_brine_degC": 150.0, "brine_flow_kg_s": 80.0,
        "T_ambient_degC": 30.0,
    })
    assert abs(r["condenser_factor"] - 1.0) < 0.01


def test_condenser_factor_below_one_hot(model):
    """Condenser factor should be < 1 when T_amb > T_cond_design."""
    r = model.predict({
        "T_brine_degC": 150.0, "brine_flow_kg_s": 80.0,
        "T_ambient_degC": 40.0,
    })
    assert r["condenser_factor"] < 1.0


def test_resource_factor_unity_at_year_zero(model):
    """Resource factor must be 1.0 at year 0."""
    r = model.predict({
        "T_brine_degC": 150.0, "brine_flow_kg_s": 80.0,
        "T_ambient_degC": 25.0, "years_operation": 0.0,
    })
    assert abs(r["resource_factor"] - 1.0) < 1e-6


def test_resource_factor_declines(model):
    """Resource factor must decrease with years."""
    r_0 = model.predict({
        "T_brine_degC": 150.0, "brine_flow_kg_s": 80.0,
        "T_ambient_degC": 25.0, "years_operation": 0.0,
    })
    r_20 = model.predict({
        "T_brine_degC": 150.0, "brine_flow_kg_s": 80.0,
        "T_ambient_degC": 25.0, "years_operation": 20.0,
    })
    assert r_20["resource_factor"] < r_0["resource_factor"]


def test_resource_factor_value_at_20yr(model):
    """At 20 years with 1.5%/yr decline: f = (1-0.015)^20 = 0.7386."""
    r = model.predict({
        "T_brine_degC": 150.0, "brine_flow_kg_s": 80.0,
        "T_ambient_degC": 25.0, "years_operation": 20.0,
    })
    expected = (1.0 - 0.015) ** 20
    assert abs(r["resource_factor"] - expected) < 0.001


def test_power_declines_with_years(model):
    """Power output should decrease over plant lifetime."""
    r_0 = model.predict({
        "T_brine_degC": 150.0, "brine_flow_kg_s": 80.0,
        "T_ambient_degC": 25.0, "years_operation": 0.0,
    })
    r_30 = model.predict({
        "T_brine_degC": 150.0, "brine_flow_kg_s": 80.0,
        "T_ambient_degC": 25.0, "years_operation": 30.0,
    })
    assert r_0["power_output_kw"] > r_30["power_output_kw"]


def test_efficiency_reasonable(model):
    """Efficiency should be in reasonable range for binary ORC (5-15%)."""
    r = model.predict({
        "T_brine_degC": 150.0, "brine_flow_kg_s": 80.0,
        "T_ambient_degC": 25.0, "PLR": 1.0,
    })
    assert 0.05 < r["efficiency"] < 0.20, \
        f"Efficiency = {r['efficiency']:.4f}"


def test_efficiency_nonnegative(model):
    """Efficiency must be non-negative."""
    r = model.predict({
        "T_brine_degC": 80.0, "brine_flow_kg_s": 10.0,
        "T_ambient_degC": 45.0, "PLR": 0.3, "years_operation": 30.0,
    })
    assert r["efficiency"] >= 0.0


def test_benchmark(model):
    """1000 predictions must complete in < 1 second."""
    start = time.perf_counter()
    for _ in range(1000):
        model.predict({
            "T_brine_degC": 150.0, "brine_flow_kg_s": 80.0,
            "T_ambient_degC": 25.0, "PLR": 0.8, "years_operation": 10.0,
        })
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed*1000:.1f} ms")
    assert elapsed < 1.0
