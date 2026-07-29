"""EC066 — Offshore Floating Wind — F1a — Test Suite"""

import sys, time
import numpy as np
import pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()


def test_predict_returns_dict(model):
    r = model.predict({"wind_speed": 11.0})
    for k in ["power_kw", "capacity_factor", "power_coefficient"]:
        assert k in r


def test_get_info(model):
    assert model.get_info()["ec_id"] == "EC066"


def test_zero_power_below_cut_in(model):
    r = model.predict({"wind_speed": 2.0})
    assert float(r["power_kw"]) == 0.0


def test_near_rated_power_at_rated_speed(model):
    """Should approach (slightly below due to platform loss) rated power."""
    r = model.predict({"wind_speed": 12.0})
    # 15 MW * (1 - 0.02) = 14700 kW
    assert 14000.0 < float(r["power_kw"]) <= 15000.0


def test_zero_power_above_cut_out(model):
    r = model.predict({"wind_speed": 26.0})
    assert float(r["power_kw"]) == 0.0


def test_power_increases_with_wind(model):
    v = np.array([4.0, 6.0, 8.0, 10.0])
    r = model.predict({"wind_speed": v})
    assert np.all(np.diff(r["power_kw"]) >= 0)


def test_capacity_factor_range(model):
    v = np.linspace(0, 25, 100)
    r = model.predict({"wind_speed": v})
    assert np.all(r["capacity_factor"] >= 0.0)
    assert np.all(r["capacity_factor"] <= 1.0)


def test_betz_limit(model):
    v = np.linspace(4, 25, 100)
    r = model.predict({"wind_speed": v})
    assert np.all(r["power_coefficient"] <= 0.6), "Cp exceeds Betz limit"


def test_platform_motion_penalty(model):
    """Floating turbine should produce slightly less than its raw curve due to motion penalty."""
    r = model.predict({"wind_speed": 12.0})
    # max possible 15000; with 2% loss, expect ~14700
    assert float(r["power_kw"]) < 15000.0
    assert float(r["power_kw"]) > 14000.0


def test_density_correction(model):
    r_low = model.predict({"wind_speed": 10.0, "air_density": 1.0})
    r_high = model.predict({"wind_speed": 10.0, "air_density": 1.3})
    assert float(r_high["power_kw"]) > float(r_low["power_kw"])


def test_large_capacity(model):
    """Floating turbines are large — should be ~10-15 MW class."""
    info = model.get_info()
    r = model.predict({"wind_speed": 13.0})
    assert float(r["power_kw"]) > 10000.0, "Floating wind expected >10 MW class"


def test_benchmark(model):
    v = np.random.uniform(0, 25, 1000)
    start = time.perf_counter()
    model.predict({"wind_speed": v})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed*1000:.2f} ms")
    assert elapsed < 1.0
