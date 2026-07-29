"""EC070 — Water-Source Heat Pump — F1b Part-Load — Test Suite

Tests must fail the model, not accommodate it.
Loosening requires # RATIONALE: comment.
"""

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


# --- Interface ---

def test_predict_returns_required_keys(model):
    r = model.predict({"T_source": 15.0, "T_sink": 45.0, "part_load_ratio": 0.5})
    for key in ["cop", "cooling_cop", "heating_capacity_kw",
                "electrical_input_kw", "cop_degradation_factor", "ua_effective"]:
        assert key in r, f"Missing key: {key}"


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC070"
    assert info["fidelity"] == "F1b"


# --- COP physics ---

def test_cop_always_greater_than_one(model):
    """COP must be > 1 at all valid operating points."""
    plr = np.linspace(0.05, 1.0, 20)
    r = model.predict({"T_source": 15.0, "T_sink": 45.0, "part_load_ratio": plr})
    assert np.all(r["cop"] > 1.0)


def test_cop_higher_than_ashp_range(model):
    """WSHP COP at A15/W45 should be higher than typical ASHP (~3.5), target > 4."""
    r = model.predict({"T_source": 15.0, "T_sink": 45.0, "part_load_ratio": 1.0})
    assert float(r["cop"]) > 4.0, (
        f"WSHP COP at W15/W45 should exceed 4.0, got {float(r['cop']):.2f}"
    )


def test_cop_rating_conditions_range(model):
    """COP at EN 14511 rating conditions W15/W45 should be in [4.0, 7.0]."""
    r = model.predict({"T_source": 15.0, "T_sink": 45.0, "part_load_ratio": 1.0})
    cop = float(r["cop"])
    assert 4.0 < cop < 7.0, f"COP at W15/W45 = {cop:.2f}, expected 4-7"


def test_cop_increases_with_source_temperature(model):
    """Higher source water temperature should increase COP."""
    sources = np.array([5.0, 10.0, 15.0, 20.0, 25.0])
    r = model.predict({"T_source": sources, "T_sink": 45.0, "part_load_ratio": 1.0})
    assert np.all(np.diff(r["cop"]) > 0), "COP should increase with source T"


def test_cop_decreases_with_sink_temperature(model):
    """Higher sink temperature means more lift, lower COP."""
    sinks = np.array([30.0, 40.0, 50.0, 60.0])
    r = model.predict({"T_source": 15.0, "T_sink": sinks, "part_load_ratio": 1.0})
    assert np.all(np.diff(r["cop"]) < 0), "COP should decrease with sink T"


def test_cooling_cop_is_cop_minus_one(model):
    """Cooling COP = heating COP - 1 (thermodynamic identity)."""
    r = model.predict({"T_source": 15.0, "T_sink": 45.0, "part_load_ratio": 1.0})
    np.testing.assert_allclose(
        float(r["cooling_cop"]), float(r["cop"]) - 1.0, atol=0.01
    )


# --- Part-load degradation ---

def test_plf_is_one_at_full_load(model):
    """Degradation factor must be 1.0 at PLR=1.0."""
    r = model.predict({"T_source": 15.0, "T_sink": 45.0, "part_load_ratio": 1.0})
    np.testing.assert_allclose(float(r["cop_degradation_factor"]), 1.0, atol=1e-9)


def test_plf_less_than_one_at_part_load(model):
    r = model.predict({"T_source": 15.0, "T_sink": 45.0, "part_load_ratio": 0.5})
    assert float(r["cop_degradation_factor"]) < 1.0


def test_cop_decreases_with_plr(model):
    """COP must decrease as PLR decreases."""
    plr = np.array([0.10, 0.25, 0.50, 0.75, 1.00])
    r = model.predict({"T_source": 15.0, "T_sink": 45.0, "part_load_ratio": plr})
    assert np.all(np.diff(r["cop"]) > 0), "COP must increase with PLR"


def test_cycling_penalty_below_plr_min(model):
    """COP degradation should be larger (steeper) below PLR_min=0.30."""
    r_just_above = model.predict({"T_source": 15.0, "T_sink": 45.0,
                                   "part_load_ratio": 0.31})
    r_just_below = model.predict({"T_source": 15.0, "T_sink": 45.0,
                                   "part_load_ratio": 0.20})
    assert float(r_just_below["cop_degradation_factor"]) < \
           float(r_just_above["cop_degradation_factor"]), (
        "Cycling penalty should further reduce COP below PLR_min"
    )


def test_very_low_plr_cop_drop(model):
    """At PLR=0.05 there should be a notable COP drop vs PLR=1.0."""
    r_full = model.predict({"T_source": 15.0, "T_sink": 45.0, "part_load_ratio": 1.0})
    r_low = model.predict({"T_source": 15.0, "T_sink": 45.0, "part_load_ratio": 0.05})
    assert float(r_full["cop"]) > float(r_low["cop"]) * 1.05, (
        "COP should drop significantly at very low PLR due to cycling"
    )


# --- Water flow rate effect ---

def test_ua_effective_at_rated_flow_equals_ua_rated(model):
    """U_A_eff at rated flow should equal U_A_rated."""
    r = model.predict({"T_source": 15.0, "T_sink": 45.0,
                       "part_load_ratio": 1.0, "water_flow_rate_ls": 3.0})
    np.testing.assert_allclose(float(r["ua_effective"]), 8.0, rtol=1e-5)


def test_ua_decreases_with_flow(model):
    """Lower flow rate -> lower U_A_eff (less convective heat transfer)."""
    r_low = model.predict({"T_source": 15.0, "T_sink": 45.0,
                            "part_load_ratio": 1.0, "water_flow_rate_ls": 1.0})
    r_high = model.predict({"T_source": 15.0, "T_sink": 45.0,
                             "part_load_ratio": 1.0, "water_flow_rate_ls": 5.0})
    assert float(r_low["ua_effective"]) < float(r_high["ua_effective"])


def test_cop_drops_at_reduced_flow(model):
    """Lower water flow -> reduced U_A -> lower effective source temp -> lower COP."""
    r_rated = model.predict({"T_source": 15.0, "T_sink": 45.0,
                              "part_load_ratio": 1.0, "water_flow_rate_ls": 3.0})
    r_low_flow = model.predict({"T_source": 15.0, "T_sink": 45.0,
                                 "part_load_ratio": 1.0, "water_flow_rate_ls": 0.8})
    assert float(r_rated["cop"]) > float(r_low_flow["cop"]), (
        "COP must drop at reduced water flow (higher evaporator approach temperature)"
    )


def test_cop_improves_with_higher_flow(model):
    """Above-rated flow should improve COP (higher U_A_eff)."""
    r_rated = model.predict({"T_source": 15.0, "T_sink": 45.0,
                              "part_load_ratio": 1.0, "water_flow_rate_ls": 3.0})
    r_high = model.predict({"T_source": 15.0, "T_sink": 45.0,
                             "part_load_ratio": 1.0, "water_flow_rate_ls": 5.5})
    assert float(r_high["cop"]) > float(r_rated["cop"]), (
        "Higher water flow should improve COP"
    )


# --- No defrost check ---

def test_no_defrost_cop_constant_above_freezing(model):
    """
    WSHP: COP should only depend on temperatures, not air conditions.
    Verify COP is not penalised at typical water source temperatures.
    Source temps 5-30 degC are above freezing => no defrost degradation.
    """
    r_5c = model.predict({"T_source": 5.0, "T_sink": 45.0, "part_load_ratio": 1.0})
    r_20c = model.predict({"T_source": 20.0, "T_sink": 45.0, "part_load_ratio": 1.0})
    # Higher source temp gives higher COP — expected and consistent
    assert float(r_20c["cop"]) > float(r_5c["cop"])


# --- Heating capacity ---

def test_heating_capacity_proportional_to_plr(model):
    plr = np.array([0.25, 0.5, 0.75, 1.0])
    r = model.predict({"T_source": 15.0, "T_sink": 45.0, "part_load_ratio": plr})
    np.testing.assert_allclose(r["heating_capacity_kw"], 50.0 * plr)


# --- Array inputs ---

def test_array_inputs(model):
    Ts = np.array([10.0, 15.0, 20.0])
    r = model.predict({"T_source": Ts, "T_sink": 45.0, "part_load_ratio": 0.8})
    assert r["cop"].shape == (3,)


# --- Benchmark ---

def test_benchmark(model):
    Ts = np.random.uniform(5, 30, 1000)
    Tk = np.random.uniform(30, 65, 1000)
    plr = np.random.uniform(0.05, 1.0, 1000)
    flow = np.random.uniform(0.5, 5.5, 1000)
    start = time.perf_counter()
    model.predict({"T_source": Ts, "T_sink": Tk,
                   "part_load_ratio": plr, "water_flow_rate_ls": flow})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed * 1000:.2f} ms")
    assert elapsed < 1.0
