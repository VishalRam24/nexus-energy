"""EC100 — Brayton Cycle Gas Turbine — F1b Part-Load + Ambient — Test Suite"""

import sys, time
import numpy as np
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()

ISO_T = 288.15  # K


# --- Interface ---

def test_predict_returns_all_keys(model):
    r = model.predict({"PLR": 1.0, "T_ambient_k": ISO_T})
    for key in ["efficiency", "power_output_kw", "heat_rate_kj_kwh",
                 "exhaust_temp_k", "f_amb_power"]:
        assert key in r, f"Missing key: {key}"


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC100"
    assert info["fidelity"] == "F1b"


# --- Efficiency physics ---

def test_efficiency_positive(model):
    r = model.predict({"PLR": 1.0, "T_ambient_k": ISO_T})
    assert float(r["efficiency"]) > 0


def test_efficiency_at_iso(model):
    """At ISO and PLR=1, efficiency should be close to rated (~38.5%)."""
    r = model.predict({"PLR": 1.0, "T_ambient_k": ISO_T})
    eta = float(r["efficiency"])
    assert 0.30 < eta < 0.44, f"ISO efficiency = {eta:.4f}"


def test_efficiency_drops_at_part_load(model):
    r_full = model.predict({"PLR": 1.0, "T_ambient_k": ISO_T})
    r_part = model.predict({"PLR": 0.5, "T_ambient_k": ISO_T})
    assert float(r_part["efficiency"]) < float(r_full["efficiency"])


def test_efficiency_drops_on_hot_day(model):
    """Higher ambient temperature reduces efficiency (less dense air)."""
    r_iso  = model.predict({"PLR": 1.0, "T_ambient_k": ISO_T})
    r_hot  = model.predict({"PLR": 1.0, "T_ambient_k": 313.15})  # 40C
    assert float(r_hot["efficiency"]) < float(r_iso["efficiency"])


# --- Power output ---

def test_power_at_iso_rated(model):
    """At ISO, PLR=1, power should be ~185 MW (185000 kW)."""
    r = model.predict({"PLR": 1.0, "T_ambient_k": ISO_T})
    P = float(r["power_output_kw"])
    assert abs(P - 185000.0) / 185000.0 < 0.05, f"P = {P/1000:.1f} MW"


def test_power_scales_with_plr(model):
    PLR = np.array([0.4, 0.6, 0.8, 1.0])
    r = model.predict({"PLR": PLR, "T_ambient_k": ISO_T})
    assert np.all(np.diff(r["power_output_kw"]) > 0)


def test_power_reduced_on_hot_day(model):
    """Hot-day power output < ISO power (density effect)."""
    r_iso = model.predict({"PLR": 1.0, "T_ambient_k": ISO_T})
    r_hot = model.predict({"PLR": 1.0, "T_ambient_k": 313.15})
    assert float(r_hot["power_output_kw"]) < float(r_iso["power_output_kw"])


# --- Ambient correction factor ---

def test_f_amb_unity_at_iso(model):
    """f_amb_power should be ~1.0 at ISO conditions."""
    r = model.predict({"PLR": 1.0, "T_ambient_k": ISO_T, "P_ambient_kpa": 101.325})
    assert float(r["f_amb_power"]) == pytest.approx(1.0, abs=0.01)


def test_f_amb_less_than_1_on_hot_day(model):
    r = model.predict({"PLR": 1.0, "T_ambient_k": 308.15})
    assert float(r["f_amb_power"]) < 1.0


# --- Heat rate ---

def test_heat_rate_positive(model):
    r = model.predict({"PLR": 1.0, "T_ambient_k": ISO_T})
    assert float(r["heat_rate_kj_kwh"]) > 0


def test_heat_rate_increases_at_part_load(model):
    r_full = model.predict({"PLR": 1.0, "T_ambient_k": ISO_T})
    r_part = model.predict({"PLR": 0.5, "T_ambient_k": ISO_T})
    assert float(r_part["heat_rate_kj_kwh"]) > float(r_full["heat_rate_kj_kwh"])


def test_heat_rate_consistency(model):
    """HR = 3600 / eta."""
    r = model.predict({"PLR": 0.8, "T_ambient_k": ISO_T})
    hr_calc = 3600.0 / float(r["efficiency"])
    assert abs(hr_calc - float(r["heat_rate_kj_kwh"])) < 1.0


# --- Exhaust temperature ---

def test_exhaust_temp_rises_at_part_load(model):
    """GT exhaust temperature rises when part-loaded."""
    r_full = model.predict({"PLR": 1.0, "T_ambient_k": ISO_T})
    r_part = model.predict({"PLR": 0.5, "T_ambient_k": ISO_T})
    assert float(r_part["exhaust_temp_k"]) > float(r_full["exhaust_temp_k"])


def test_exhaust_temp_reasonable(model):
    """Exhaust should be 500-650 degC (773-923 K) for simple cycle F-class GT."""
    r = model.predict({"PLR": 1.0, "T_ambient_k": ISO_T})
    T_exh_c = float(r["exhaust_temp_k"]) - 273.15
    assert 450 < T_exh_c < 700, f"Exhaust = {T_exh_c:.1f} degC"


# --- Array inputs ---

def test_array_inputs(model):
    PLR = np.array([0.5, 0.75, 1.0])
    r = model.predict({"PLR": PLR, "T_ambient_k": ISO_T})
    assert r["power_output_kw"].shape == (3,)


# --- Benchmark ---

def test_benchmark(model):
    PLR   = np.random.uniform(0.4, 1.0, 1000)
    T_amb = np.random.uniform(258, 318, 1000)
    start = time.perf_counter()
    model.predict({"PLR": PLR, "T_ambient_k": T_amb})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed*1000:.2f} ms")
    assert elapsed < 1.0
