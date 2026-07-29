"""EC152 -- Flash Steam Geothermal Plant -- F1b Part-Load Ambient -- Test Suite"""
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
        "T_brine_degC": 240.0, "m_dot_brine_kg_s": 100.0, "T_reject_degC": 30.0
    })
    for k in ["power_output_kw", "efficiency", "resource_factor",
              "condenser_factor", "scaling_factor", "flash_config", "steam_quality"]:
        assert k in r, f"Missing key: {k}"


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC152"
    assert info["fidelity"] == "F1b"


# ---------- Power ----------
def test_power_positive_at_design(model):
    r = model.predict({
        "T_brine_degC": 240.0, "m_dot_brine_kg_s": 100.0,
        "T_reject_degC": 30.0, "PLR": 1.0,
    })
    assert r["power_output_kw"] > 0


def test_power_nonnegative_extreme(model):
    r = model.predict({
        "T_brine_degC": 150.0, "m_dot_brine_kg_s": 10.0,
        "T_reject_degC": 50.0, "PLR": 0.3,
        "years_operation": 40.0, "TDS_g_L": 80.0,
    })
    assert r["power_output_kw"] >= 0.0


def test_power_bounded_by_rating(model):
    r = model.predict({
        "T_brine_degC": 300.0, "m_dot_brine_kg_s": 500.0,
        "T_reject_degC": 5.0, "PLR": 1.0,
    })
    assert r["power_output_kw"] <= 20000.0


# ---------- Part-load ----------
def test_power_decreases_with_lower_plr(model):
    base = {"T_brine_degC": 240.0, "m_dot_brine_kg_s": 100.0, "T_reject_degC": 30.0}
    r1 = model.predict({**base, "PLR": 1.0})
    r5 = model.predict({**base, "PLR": 0.5})
    assert r1["power_output_kw"] > r5["power_output_kw"]


# ---------- Ambient ----------
def test_power_decreases_with_higher_ambient(model):
    base = {"T_brine_degC": 240.0, "m_dot_brine_kg_s": 100.0, "PLR": 1.0}
    r_cool = model.predict({**base, "T_reject_degC": 10.0})
    r_hot  = model.predict({**base, "T_reject_degC": 45.0})
    assert r_cool["power_output_kw"] > r_hot["power_output_kw"]


def test_condenser_factor_near_one_at_design(model):
    r = model.predict({
        "T_brine_degC": 240.0, "m_dot_brine_kg_s": 100.0, "T_reject_degC": 30.0,
    })
    assert 0.95 <= r["condenser_factor"] <= 1.05


def test_condenser_factor_less_than_one_hot(model):
    r = model.predict({
        "T_brine_degC": 240.0, "m_dot_brine_kg_s": 100.0, "T_reject_degC": 45.0,
    })
    assert r["condenser_factor"] < 1.0


# ---------- Resource decline ----------
def test_resource_factor_unity_at_year_zero(model):
    r = model.predict({
        "T_brine_degC": 240.0, "m_dot_brine_kg_s": 100.0,
        "T_reject_degC": 30.0, "years_operation": 0.0,
    })
    assert abs(r["resource_factor"] - 1.0) < 1e-6


def test_resource_factor_declines(model):
    r0 = model.predict({"T_brine_degC": 240.0, "m_dot_brine_kg_s": 100.0,
                        "T_reject_degC": 30.0, "years_operation": 0.0})
    r20 = model.predict({"T_brine_degC": 240.0, "m_dot_brine_kg_s": 100.0,
                         "T_reject_degC": 30.0, "years_operation": 20.0})
    assert r20["resource_factor"] < r0["resource_factor"]


def test_resource_factor_value_20yr(model):
    """(1-0.015)^20 = 0.7386."""
    r = model.predict({"T_brine_degC": 240.0, "m_dot_brine_kg_s": 100.0,
                       "T_reject_degC": 30.0, "years_operation": 20.0})
    assert abs(r["resource_factor"] - (0.985 ** 20)) < 0.001


# ---------- Brine chemistry ----------
def test_scaling_factor_unity_at_base_tds(model):
    """At baseline TDS (10 g/L) and design T, scaling factor = 1."""
    r = model.predict({
        "T_brine_degC": 240.0, "m_dot_brine_kg_s": 100.0,
        "T_reject_degC": 30.0, "TDS_g_L": 10.0,
    })
    assert abs(r["scaling_factor"] - 1.0) < 0.02


def test_scaling_factor_decreases_high_tds(model):
    r_lo = model.predict({"T_brine_degC": 240.0, "m_dot_brine_kg_s": 100.0,
                          "T_reject_degC": 30.0, "TDS_g_L": 10.0})
    r_hi = model.predict({"T_brine_degC": 240.0, "m_dot_brine_kg_s": 100.0,
                          "T_reject_degC": 30.0, "TDS_g_L": 50.0})
    assert r_hi["scaling_factor"] < r_lo["scaling_factor"]


def test_power_decreases_with_high_tds(model):
    base = {"T_brine_degC": 240.0, "m_dot_brine_kg_s": 100.0, "T_reject_degC": 30.0}
    r_lo = model.predict({**base, "TDS_g_L": 10.0})
    r_hi = model.predict({**base, "TDS_g_L": 50.0})
    assert r_lo["power_output_kw"] > r_hi["power_output_kw"]


# ---------- Double flash ----------
def test_double_flash_activates_above_threshold(model):
    """At T_brine >= 200 degC, flash_config > 1.0."""
    r = model.predict({
        "T_brine_degC": 240.0, "m_dot_brine_kg_s": 100.0, "T_reject_degC": 30.0,
    })
    assert r["flash_config"] > 1.0


def test_single_flash_below_threshold(model):
    """At T_brine = 180 degC (< 200), flash_config = 1.0."""
    r = model.predict({
        "T_brine_degC": 180.0, "m_dot_brine_kg_s": 100.0, "T_reject_degC": 30.0,
    })
    assert abs(r["flash_config"] - 1.0) < 1e-6


def test_double_flash_gives_more_power(model):
    """At 240 degC, power must be higher than at 180 degC (double vs single flash)."""
    base = {"m_dot_brine_kg_s": 100.0, "T_reject_degC": 30.0, "PLR": 1.0}
    r_hi = model.predict({**base, "T_brine_degC": 240.0})
    r_lo = model.predict({**base, "T_brine_degC": 180.0})
    assert r_hi["power_output_kw"] > r_lo["power_output_kw"]


# ---------- Steam quality ----------
def test_steam_quality_between_zero_and_one(model):
    r = model.predict({
        "T_brine_degC": 240.0, "m_dot_brine_kg_s": 100.0, "T_reject_degC": 30.0,
    })
    assert 0.0 <= r["steam_quality"] <= 1.0


# ---------- Efficiency ----------
def test_efficiency_reasonable(model):
    """
    Flash steam plant efficiency: 10-20% typical (Zarrouk & Moon 2014).
    Allow [0.05, 0.30] to cover double-flash and wide T range.
    """
    r = model.predict({
        "T_brine_degC": 240.0, "m_dot_brine_kg_s": 100.0,
        "T_reject_degC": 30.0, "PLR": 1.0,
    })
    assert 0.05 < r["efficiency"] < 0.30, f"Efficiency = {r['efficiency']:.4f}"


def test_efficiency_nonnegative(model):
    r = model.predict({
        "T_brine_degC": 150.0, "m_dot_brine_kg_s": 10.0,
        "T_reject_degC": 50.0, "PLR": 0.3, "years_operation": 40.0,
    })
    assert r["efficiency"] >= 0.0


# ---------- Benchmark ----------
def test_benchmark(model):
    start = time.perf_counter()
    for _ in range(1000):
        model.predict({
            "T_brine_degC": 240.0, "m_dot_brine_kg_s": 100.0,
            "T_reject_degC": 30.0, "PLR": 0.8, "years_operation": 10.0, "TDS_g_L": 15.0,
        })
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed*1000:.1f} ms")
    assert elapsed < 1.0
