"""EC151 -- Dry Steam Geothermal Plant -- F1b Part-Load Ambient -- Test Suite"""
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


# ---------- Output keys ----------
def test_predict_keys(model):
    r = model.predict({
        "T_geo_degC": 200.0, "m_dot_steam_kg_s": 50.0, "T_reject_degC": 30.0
    })
    for k in ["power_output_kw", "efficiency", "resource_factor",
              "condenser_factor", "ncg_factor", "plr_factor"]:
        assert k in r, f"Missing key: {k}"


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC151"
    assert info["fidelity"] == "F1b"


# ---------- Power sanity ----------
def test_power_positive_at_design(model):
    r = model.predict({
        "T_geo_degC": 200.0, "m_dot_steam_kg_s": 50.0,
        "T_reject_degC": 30.0, "PLR": 1.0,
    })
    assert r["power_output_kw"] > 0


def test_power_bounded_by_rating(model):
    r = model.predict({
        "T_geo_degC": 280.0, "m_dot_steam_kg_s": 200.0,
        "T_reject_degC": 5.0, "PLR": 1.0,
    })
    assert r["power_output_kw"] <= 10000.0


def test_power_nonnegative(model):
    r = model.predict({
        "T_geo_degC": 150.0, "m_dot_steam_kg_s": 5.0,
        "T_reject_degC": 50.0, "PLR": 0.3, "years_operation": 40.0, "ncg_content_pct": 8.0,
    })
    assert r["power_output_kw"] >= 0.0


# ---------- Part-load ----------
def test_power_decreases_with_lower_plr(model):
    base = {"T_geo_degC": 200.0, "m_dot_steam_kg_s": 50.0, "T_reject_degC": 30.0}
    r_full = model.predict({**base, "PLR": 1.0})
    r_half = model.predict({**base, "PLR": 0.5})
    assert r_full["power_output_kw"] > r_half["power_output_kw"]


def test_plr_factor_unity_at_full_load(model):
    r = model.predict({
        "T_geo_degC": 200.0, "m_dot_steam_kg_s": 50.0,
        "T_reject_degC": 30.0, "PLR": 1.0,
    })
    assert abs(r["plr_factor"] - 1.0) < 1e-6


def test_plr_factor_less_than_one_at_part_load(model):
    r = model.predict({
        "T_geo_degC": 200.0, "m_dot_steam_kg_s": 50.0,
        "T_reject_degC": 30.0, "PLR": 0.5,
    })
    assert r["plr_factor"] < 1.0


# ---------- Ambient / condenser ----------
def test_power_decreases_with_higher_ambient(model):
    base = {"T_geo_degC": 200.0, "m_dot_steam_kg_s": 50.0, "PLR": 1.0}
    r_cool = model.predict({**base, "T_reject_degC": 15.0})
    r_hot  = model.predict({**base, "T_reject_degC": 45.0})
    assert r_cool["power_output_kw"] > r_hot["power_output_kw"]


def test_condenser_factor_near_one_at_design(model):
    """At T_reject = T_reject_design = 30 degC, condenser_factor ~ 1.0."""
    r = model.predict({
        "T_geo_degC": 200.0, "m_dot_steam_kg_s": 50.0, "T_reject_degC": 30.0,
    })
    assert 0.95 <= r["condenser_factor"] <= 1.05, \
        f"condenser_factor = {r['condenser_factor']:.4f} not near 1.0 at design"


def test_condenser_factor_below_one_hot(model):
    r = model.predict({
        "T_geo_degC": 200.0, "m_dot_steam_kg_s": 50.0, "T_reject_degC": 45.0,
    })
    assert r["condenser_factor"] < 1.0


def test_condenser_factor_above_one_cold(model):
    """Cold ambient improves condenser performance -> factor > 1."""
    r = model.predict({
        "T_geo_degC": 200.0, "m_dot_steam_kg_s": 50.0, "T_reject_degC": 10.0,
    })
    assert r["condenser_factor"] > 1.0


# ---------- Resource decline ----------
def test_resource_factor_unity_at_year_zero(model):
    r = model.predict({
        "T_geo_degC": 200.0, "m_dot_steam_kg_s": 50.0,
        "T_reject_degC": 30.0, "years_operation": 0.0,
    })
    assert abs(r["resource_factor"] - 1.0) < 1e-6


def test_resource_factor_declines_with_time(model):
    r0  = model.predict({"T_geo_degC": 200.0, "m_dot_steam_kg_s": 50.0,
                         "T_reject_degC": 30.0, "years_operation": 0.0})
    r20 = model.predict({"T_geo_degC": 200.0, "m_dot_steam_kg_s": 50.0,
                         "T_reject_degC": 30.0, "years_operation": 20.0})
    assert r20["resource_factor"] < r0["resource_factor"]


def test_resource_factor_value_at_20yr(model):
    """At 20 years with 1%/yr decline: f = (1-0.01)^20 = 0.8179."""
    r = model.predict({"T_geo_degC": 200.0, "m_dot_steam_kg_s": 50.0,
                       "T_reject_degC": 30.0, "years_operation": 20.0})
    expected = (1.0 - 0.01) ** 20
    assert abs(r["resource_factor"] - expected) < 0.001


# ---------- NCG ----------
def test_ncg_factor_one_at_baseline(model):
    """NCG factor should be 1.0 when ncg_content_pct equals the baseline (1.0 wt%)."""
    r = model.predict({
        "T_geo_degC": 200.0, "m_dot_steam_kg_s": 50.0,
        "T_reject_degC": 30.0, "ncg_content_pct": 1.0,
    })
    assert abs(r["ncg_factor"] - 1.0) < 1e-6


def test_ncg_factor_decreases_with_high_ncg(model):
    r_lo = model.predict({"T_geo_degC": 200.0, "m_dot_steam_kg_s": 50.0,
                          "T_reject_degC": 30.0, "ncg_content_pct": 1.0})
    r_hi = model.predict({"T_geo_degC": 200.0, "m_dot_steam_kg_s": 50.0,
                          "T_reject_degC": 30.0, "ncg_content_pct": 5.0})
    assert r_hi["ncg_factor"] < r_lo["ncg_factor"]


def test_power_decreases_with_higher_ncg(model):
    base = {"T_geo_degC": 200.0, "m_dot_steam_kg_s": 50.0, "T_reject_degC": 30.0}
    r_lo = model.predict({**base, "ncg_content_pct": 1.0})
    r_hi = model.predict({**base, "ncg_content_pct": 5.0})
    assert r_lo["power_output_kw"] > r_hi["power_output_kw"]


# ---------- Efficiency bounds ----------
def test_efficiency_reasonable_range(model):
    """
    Dry steam efficiency: 15-25% typical (higher than binary ORC).
    DiPippo (2015) Table 7.2: The Geysers units 15-21%.
    Allow slightly wider [0.10, 0.30] for extreme conditions.
    """
    r = model.predict({
        "T_geo_degC": 200.0, "m_dot_steam_kg_s": 50.0,
        "T_reject_degC": 30.0, "PLR": 1.0,
    })
    assert 0.10 < r["efficiency"] < 0.30, \
        f"Efficiency = {r['efficiency']:.4f}"


def test_efficiency_nonnegative(model):
    r = model.predict({
        "T_geo_degC": 150.0, "m_dot_steam_kg_s": 5.0,
        "T_reject_degC": 50.0, "PLR": 0.3,
        "years_operation": 40.0, "ncg_content_pct": 8.0,
    })
    assert r["efficiency"] >= 0.0


# ---------- Benchmark ----------
def test_benchmark(model):
    """1000 predictions must complete in < 1 second."""
    start = time.perf_counter()
    for _ in range(1000):
        model.predict({
            "T_geo_degC": 200.0, "m_dot_steam_kg_s": 50.0,
            "T_reject_degC": 30.0, "PLR": 0.8,
            "years_operation": 10.0, "ncg_content_pct": 2.0,
        })
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed*1000:.1f} ms")
    assert elapsed < 1.0
