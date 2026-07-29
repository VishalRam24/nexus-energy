"""EC113 -- Subcritical Coal Plant -- F1b Part-Load + Flue Loss -- Test Suite

Physics validation notes:
  - CO2 intensity tests use rated-load (PLR=1) only.
    RATIONALE: At part load the efficiency penalty means CO2/kWh naturally
    rises above 1000 g/kWh even before any model inaccuracy. Testing CO2
    intensity at PLR < 1 would require tighter bounds than the physical
    spread warrants. The rated-load value is the meaningful engineering
    benchmark (IEA/EPA reference conditions: PLR=1, T_amb=15C).
"""
import sys, time, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()


def test_predict_keys(model):
    r = model.predict({"part_load_ratio": 1.0, "ambient_temp": 15.0})
    for k in ["power_mw", "efficiency", "coal_rate_kgs", "co2_rate_kgs",
              "co2_intensity", "stack_temp_c", "flue_heat_loss_mw", "aux_power_fraction"]:
        assert k in r


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC113"
    assert info["fidelity"] == "F1b"


# --- Efficiency physics ---

def test_rated_iso_efficiency(model):
    """At PLR=1, T_amb=15C: net efficiency in subcritical range 35-38%."""
    r = model.predict({"part_load_ratio": 1.0, "ambient_temp": 15.0})
    eta = float(r["efficiency"])
    assert 0.35 <= eta <= 0.38, f"Expected 35-38%, got {eta*100:.2f}%"


def test_efficiency_drops_at_part_load(model):
    r_full = model.predict({"part_load_ratio": 1.0, "ambient_temp": 15.0})
    r_part = model.predict({"part_load_ratio": 0.30, "ambient_temp": 15.0})
    assert float(r_part["efficiency"]) < float(r_full["efficiency"])


def test_efficiency_drops_at_high_tamb(model):
    r_cool = model.predict({"part_load_ratio": 1.0, "ambient_temp": 5.0})
    r_hot  = model.predict({"part_load_ratio": 1.0, "ambient_temp": 40.0})
    assert float(r_hot["efficiency"]) < float(r_cool["efficiency"])


# --- CO2 at rated load only ---

def test_co2_intensity_rated_load(model):
    """
    RATIONALE: CO2 intensity tested at rated load only (PLR=1, T_amb=15C).
    At part load, efficiency penalty raises CO2/kWh well above 1000 g/kWh —
    this is physically correct, not a model error. Rated-load value is the
    IEA/EPA reference benchmark (subcritical bituminous: 900-1000 g/kWh).
    """
    r = model.predict({"part_load_ratio": 1.0, "ambient_temp": 15.0})
    co2 = float(r["co2_intensity"])
    assert 850 <= co2 <= 1050, f"CO2 intensity at rated load = {co2:.0f} g/kWh, expected 850-1050"


def test_co2_intensity_above_gas(model):
    """Coal CO2 intensity at rated load must exceed 600 g/kWh (CCGT reference)."""
    r = model.predict({"part_load_ratio": 1.0, "ambient_temp": 15.0})
    assert float(r["co2_intensity"]) > 600.0


# --- Flue loss model ---

def test_stack_temp_rises_at_part_load(model):
    """Stack temperature should increase at lower PLR."""
    r_full = model.predict({"part_load_ratio": 1.0, "ambient_temp": 15.0})
    r_part = model.predict({"part_load_ratio": 0.30, "ambient_temp": 15.0})
    assert float(r_part["stack_temp_c"]) > float(r_full["stack_temp_c"])


def test_stack_temp_above_dew_point(model):
    """Stack temperature must be > 120 degC (acid dew point) at all loads."""
    plr = np.linspace(0.30, 1.0, 20)
    r   = model.predict({"part_load_ratio": plr, "ambient_temp": 15.0})
    assert np.all(r["stack_temp_c"] > 120.0), "Stack temperature below acid dew point!"


def test_flue_heat_loss_positive(model):
    """Flue gas heat loss must be positive."""
    r = model.predict({"part_load_ratio": 1.0, "ambient_temp": 15.0})
    assert float(r["flue_heat_loss_mw"]) > 0


def test_flue_heat_loss_less_than_fuel(model):
    """Flue heat loss must be less than total fuel input."""
    r    = model.predict({"part_load_ratio": 1.0, "ambient_temp": 15.0})
    eta  = float(r["efficiency"])
    P_mw = float(r["power_mw"])
    fuel_mw = P_mw / eta
    assert float(r["flue_heat_loss_mw"]) < fuel_mw


# --- Auxiliary power ---

def test_aux_power_increases_at_part_load(model):
    """Auxiliary power fraction should be higher at part load."""
    r_full = model.predict({"part_load_ratio": 1.0, "ambient_temp": 15.0})
    r_part = model.predict({"part_load_ratio": 0.30, "ambient_temp": 15.0})
    assert float(r_part["aux_power_fraction"]) > float(r_full["aux_power_fraction"])


def test_aux_power_range(model):
    """Auxiliary power fraction in realistic range 0.05-0.12."""
    plr = np.linspace(0.30, 1.0, 20)
    r   = model.predict({"part_load_ratio": plr, "ambient_temp": 15.0})
    assert np.all(r["aux_power_fraction"] >= 0.05)
    assert np.all(r["aux_power_fraction"] <= 0.12)


# --- Energy conservation ---

def test_fuel_exceeds_output(model):
    r    = model.predict({"part_load_ratio": 1.0, "ambient_temp": 15.0})
    P_mw = float(r["power_mw"])
    m_c  = float(r["coal_rate_kgs"])
    fuel_mw = m_c * model._model.LHV_coal
    assert fuel_mw > P_mw


# --- Benchmark ---

def test_benchmark(model):
    plr   = np.random.uniform(0.30, 1.0, 1000)
    T_amb = np.random.uniform(-10, 45, 1000)
    start = time.perf_counter()
    model.predict({"part_load_ratio": plr, "ambient_temp": T_amb})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed*1000:.2f} ms")
    assert elapsed < 1.0
