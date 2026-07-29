"""EC154 -- EGS -- F1b Reservoir Decline -- Test Suite"""
import sys
import time
import numpy as np
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()


def test_predict_keys(model):
    r = model.predict({
        "T_geo_degC": 200.0, "m_dot_kg_s": 50.0, "T_reject_degC": 25.0
    })
    for k in ["power_output_kw", "gross_efficiency", "net_efficiency",
              "resource_factor", "permeability_ratio", "pump_parasitic_frac",
              "effective_flow_kg_s", "T_out_degC"]:
        assert k in r, f"Missing key: {k}"


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC154"
    assert info["fidelity"] == "F1b"


# ---------- Power ----------
def test_power_positive_at_design(model):
    r = model.predict({
        "T_geo_degC": 200.0, "m_dot_kg_s": 50.0,
        "T_reject_degC": 25.0, "PLR": 1.0,
    })
    assert r["power_output_kw"] > 0


def test_power_nonnegative(model):
    r = model.predict({
        "T_geo_degC": 150.0, "m_dot_kg_s": 5.0,
        "T_reject_degC": 50.0, "PLR": 0.3, "years_operation": 40.0,
    })
    assert r["power_output_kw"] >= 0.0


def test_power_bounded_by_rating(model):
    r = model.predict({
        "T_geo_degC": 350.0, "m_dot_kg_s": 200.0,
        "T_reject_degC": -10.0, "PLR": 1.0,
    })
    assert r["power_output_kw"] <= 5000.0


# ---------- Thermal breakthrough ----------
def test_t_out_equals_t_geo_at_year_zero(model):
    """At year 0, no breakthrough: T_out should equal T_geo."""
    r = model.predict({
        "T_geo_degC": 200.0, "m_dot_kg_s": 50.0,
        "T_reject_degC": 25.0, "years_operation": 0.0,
    })
    assert abs(r["T_out_degC"] - 200.0) < 0.01


def test_t_out_declines_with_time(model):
    r0  = model.predict({"T_geo_degC": 200.0, "m_dot_kg_s": 50.0,
                          "T_reject_degC": 25.0, "years_operation": 0.0})
    r20 = model.predict({"T_geo_degC": 200.0, "m_dot_kg_s": 50.0,
                          "T_reject_degC": 25.0, "years_operation": 20.0})
    assert r20["T_out_degC"] < r0["T_out_degC"]


def test_t_out_above_injection_temperature(model):
    """T_out must always be >= T_inject (50 degC)."""
    r = model.predict({"T_geo_degC": 200.0, "m_dot_kg_s": 50.0,
                       "T_reject_degC": 25.0, "years_operation": 100.0})
    assert r["T_out_degC"] >= 50.0 - 1.0  # allow small numerical margin


# ---------- Resource factor ----------
def test_resource_factor_unity_at_year_zero(model):
    r = model.predict({"T_geo_degC": 200.0, "m_dot_kg_s": 50.0,
                       "T_reject_degC": 25.0, "years_operation": 0.0})
    assert abs(r["resource_factor"] - 1.0) < 1e-6


def test_resource_factor_declines_with_time(model):
    r0  = model.predict({"T_geo_degC": 200.0, "m_dot_kg_s": 50.0,
                          "T_reject_degC": 25.0, "years_operation": 0.0})
    r10 = model.predict({"T_geo_degC": 200.0, "m_dot_kg_s": 50.0,
                          "T_reject_degC": 25.0, "years_operation": 10.0})
    assert r10["resource_factor"] < r0["resource_factor"]


# ---------- Permeability ----------
def test_permeability_ratio_unity_at_year_zero(model):
    r = model.predict({"T_geo_degC": 200.0, "m_dot_kg_s": 50.0,
                       "T_reject_degC": 25.0, "years_operation": 0.0})
    assert abs(r["permeability_ratio"] - 1.0) < 1e-6


def test_permeability_declines_with_time(model):
    r0  = model.predict({"T_geo_degC": 200.0, "m_dot_kg_s": 50.0,
                          "T_reject_degC": 25.0, "years_operation": 0.0})
    r15 = model.predict({"T_geo_degC": 200.0, "m_dot_kg_s": 50.0,
                          "T_reject_degC": 25.0, "years_operation": 15.0})
    assert r15["permeability_ratio"] < r0["permeability_ratio"]


def test_permeability_bounded_above_minimum(model):
    """k_ratio must be >= 0.1 (bounded in model)."""
    r = model.predict({"T_geo_degC": 200.0, "m_dot_kg_s": 50.0,
                       "T_reject_degC": 25.0, "years_operation": 100.0})
    assert r["permeability_ratio"] >= 0.1


# ---------- Pump parasitic ----------
def test_pump_parasitic_at_year_zero(model):
    """Initial pump fraction = design value (0.10)."""
    r = model.predict({"T_geo_degC": 200.0, "m_dot_kg_s": 50.0,
                       "T_reject_degC": 25.0, "years_operation": 0.0})
    assert abs(r["pump_parasitic_frac"] - 0.10) < 0.001


def test_pump_parasitic_increases_with_perm_decline(model):
    r0  = model.predict({"T_geo_degC": 200.0, "m_dot_kg_s": 50.0,
                          "T_reject_degC": 25.0, "years_operation": 0.0})
    r20 = model.predict({"T_geo_degC": 200.0, "m_dot_kg_s": 50.0,
                          "T_reject_degC": 25.0, "years_operation": 20.0})
    assert r20["pump_parasitic_frac"] >= r0["pump_parasitic_frac"]


def test_net_efficiency_below_gross(model):
    r = model.predict({"T_geo_degC": 200.0, "m_dot_kg_s": 50.0,
                       "T_reject_degC": 25.0, "years_operation": 5.0})
    assert r["net_efficiency"] <= r["gross_efficiency"]


# ---------- Effective flow ----------
def test_flow_equals_design_at_year_zero(model):
    r = model.predict({"T_geo_degC": 200.0, "m_dot_kg_s": 50.0,
                       "T_reject_degC": 25.0, "years_operation": 0.0})
    assert abs(r["effective_flow_kg_s"] - 50.0) < 0.01


def test_flow_declines_with_permeability(model):
    r0  = model.predict({"T_geo_degC": 200.0, "m_dot_kg_s": 50.0,
                          "T_reject_degC": 25.0, "years_operation": 0.0})
    r20 = model.predict({"T_geo_degC": 200.0, "m_dot_kg_s": 50.0,
                          "T_reject_degC": 25.0, "years_operation": 20.0})
    assert r20["effective_flow_kg_s"] < r0["effective_flow_kg_s"]


# ---------- Ambient ----------
def test_power_decreases_with_higher_ambient(model):
    r_cool = model.predict({"T_geo_degC": 200.0, "m_dot_kg_s": 50.0,
                             "T_reject_degC": 10.0, "PLR": 1.0})
    r_hot  = model.predict({"T_geo_degC": 200.0, "m_dot_kg_s": 50.0,
                             "T_reject_degC": 45.0, "PLR": 1.0})
    assert r_cool["power_output_kw"] > r_hot["power_output_kw"]


# ---------- Efficiency ----------
def test_gross_efficiency_reasonable(model):
    """EGS gross efficiency: 5-15% typical (DiPippo 2015 Ch.16)."""
    r = model.predict({"T_geo_degC": 200.0, "m_dot_kg_s": 50.0,
                       "T_reject_degC": 25.0, "PLR": 1.0})
    assert 0.03 < r["gross_efficiency"] < 0.20, \
        f"Gross efficiency = {r['gross_efficiency']:.4f}"


def test_net_efficiency_nonnegative(model):
    r = model.predict({"T_geo_degC": 150.0, "m_dot_kg_s": 5.0,
                       "T_reject_degC": 50.0, "PLR": 0.3, "years_operation": 30.0})
    assert r["net_efficiency"] >= 0.0


# ---------- Benchmark ----------
def test_benchmark(model):
    start = time.perf_counter()
    for _ in range(1000):
        model.predict({
            "T_geo_degC": 200.0, "m_dot_kg_s": 50.0,
            "T_reject_degC": 25.0, "PLR": 0.8, "years_operation": 10.0,
        })
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed*1000:.1f} ms")
    assert elapsed < 1.0
