"""EC109 — Simple Cycle Gas Turbine — F1a — Test Suite"""
import sys, time, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()


def test_predict_keys(model):
    r = model.predict({"part_load_ratio": 1.0, "ambient_temp_c": 15.0})
    for k in ["power_mw", "efficiency", "fuel_rate_kgs", "heat_rate_kjkwh"]:
        assert k in r


def test_get_info(model):
    assert model.get_info()["ec_id"] == "EC109"


def test_efficiency_below_45_percent(model):
    """Simple-cycle GT efficiency must be < 0.45 (physical limit)."""
    PLR   = np.linspace(0.3, 1.0, 50)
    T_amb = np.linspace(-20, 50, 50)
    r = model.predict({"part_load_ratio": PLR, "ambient_temp_c": T_amb})
    assert np.all(r["efficiency"] < 0.45), "Efficiency must be below 45%"


def test_efficiency_drops_at_low_plr(model):
    """Efficiency should be lower at partial load than at full load."""
    r_full = model.predict({"part_load_ratio": 1.0,  "ambient_temp_c": 15.0})
    r_part = model.predict({"part_load_ratio": 0.4,  "ambient_temp_c": 15.0})
    assert float(r_part["efficiency"]) < float(r_full["efficiency"]), \
        "Efficiency at PLR=0.4 must be less than at PLR=1.0"


def test_efficiency_drops_at_high_ambient(model):
    """Higher ambient temperature should reduce efficiency."""
    r_cold = model.predict({"part_load_ratio": 1.0, "ambient_temp_c": 5.0})
    r_hot  = model.predict({"part_load_ratio": 1.0, "ambient_temp_c": 40.0})
    assert float(r_hot["efficiency"]) < float(r_cold["efficiency"]), \
        "Efficiency must decrease with increasing ambient temperature"


def test_fuel_greater_than_power(model):
    """Fuel energy input must always exceed electrical output."""
    PLR   = np.linspace(0.3, 1.0, 20)
    r = model.predict({"part_load_ratio": PLR, "ambient_temp_c": 15.0})
    P_fuel_mw = r["fuel_rate_kgs"] * 50000.0 / 1000.0  # MW_th (LHV=50000 kJ/kg)
    assert np.all(P_fuel_mw > r["power_mw"]), "Fuel power must exceed electrical output"


def test_power_scales_with_plr(model):
    """Power output should scale linearly with PLR."""
    r50 = model.predict({"part_load_ratio": 0.5, "ambient_temp_c": 15.0})
    r100 = model.predict({"part_load_ratio": 1.0, "ambient_temp_c": 15.0})
    assert abs(float(r50["power_mw"]) - 0.5 * float(r100["power_mw"])) < 0.01


def test_heat_rate_reasonable(model):
    """Heat rate at full load ISO should be ~8780 kJ/kWh (eta=0.41)."""
    r = model.predict({"part_load_ratio": 1.0, "ambient_temp_c": 15.0})
    HR = float(r["heat_rate_kjkwh"])
    expected = 3600.0 / 0.41
    assert abs(HR - expected) / expected < 0.01, f"Heat rate = {HR:.0f}, expected ~{expected:.0f} kJ/kWh"


def test_efficiency_positive(model):
    """Efficiency must always be positive."""
    PLR = np.linspace(0.3, 1.0, 30)
    r = model.predict({"part_load_ratio": PLR, "ambient_temp_c": 15.0})
    assert np.all(r["efficiency"] > 0.0)


def test_fuel_positive(model):
    PLR = np.linspace(0.3, 1.0, 30)
    r = model.predict({"part_load_ratio": PLR, "ambient_temp_c": 15.0})
    assert np.all(r["fuel_rate_kgs"] > 0.0)


def test_benchmark(model):
    PLR   = np.random.uniform(0.3, 1.0, 1000)
    T_amb = np.random.uniform(-20, 50, 1000)
    start = time.perf_counter()
    model.predict({"part_load_ratio": PLR, "ambient_temp_c": T_amb})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed*1000:.2f} ms")
    assert elapsed < 1.0
