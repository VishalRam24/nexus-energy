"""EC069 — GSHP — F1b Part-Load — Test Suite"""
import sys, time, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()


def test_predict_keys(model):
    r = model.predict({"T_ground": 10.0, "T_sink": 35.0, "part_load_ratio": 0.5})
    for k in ["cop", "heating_capacity_kw", "electrical_input_kw", "T_ground"]:
        assert k in r


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC069"
    assert info["fidelity"] == "F1b"


def test_cop_greater_than_one(model):
    plr = np.linspace(0.1, 1.0, 20)
    r = model.predict({"T_ground": 10.0, "T_sink": 35.0, "part_load_ratio": plr})
    assert np.all(r["cop"] > 1.0)


def test_cop_decreases_with_lower_plr(model):
    plr = np.array([1.0, 0.75, 0.5, 0.25])
    r = model.predict({"T_ground": 10.0, "T_sink": 35.0, "part_load_ratio": plr})
    assert np.all(np.diff(r["cop"]) < 0), "COP should decrease with decreasing PLR"


def test_seasonal_temperature_variation(model):
    """Ground temperature should vary seasonally."""
    months = np.arange(1, 13)
    r = model.predict({"month": months, "T_sink": 35.0})
    T_gnd = r["T_ground"]
    assert float(np.max(T_gnd)) > float(np.min(T_gnd)), "Ground temp must vary"


def test_ground_temp_min_in_winter(model):
    """Minimum ground temperature should occur near month_min (Feb)."""
    months = np.arange(1, 13)
    r = model.predict({"month": months, "T_sink": 35.0})
    T_gnd = np.asarray(r["T_ground"])
    min_month = months[np.argmin(T_gnd)]
    assert min_month in [1, 2, 3], f"Min ground temp should be Jan-Mar, got month {min_month}"


def test_ground_temp_mean(model):
    """Annual mean ground temperature should be close to T_mean."""
    months = np.arange(1, 13)
    r = model.predict({"month": months, "T_sink": 35.0})
    T_mean = float(np.mean(r["T_ground"]))
    np.testing.assert_allclose(T_mean, 10.0, atol=0.5)


def test_cop_higher_than_ashp_at_same_conditions(model):
    """GSHP COP at 10C should be higher than ASHP would be at 7C."""
    r = model.predict({"T_ground": 10.0, "T_sink": 35.0, "part_load_ratio": 1.0})
    assert float(r["cop"]) > 3.5, f"GSHP COP at G10/W35 should be > 3.5, got {float(r['cop']):.2f}"


def test_month_input_equivalent_to_t_ground(model):
    """Using month=8 should give same as directly inputting T_ground for August."""
    r_month = model.predict({"month": 8, "T_sink": 35.0})
    T_aug = float(r_month["T_ground"])
    r_direct = model.predict({"T_ground": T_aug, "T_sink": 35.0})
    np.testing.assert_allclose(float(r_month["cop"]), float(r_direct["cop"]), rtol=1e-10)


def test_benchmark(model):
    months = np.random.uniform(1, 12, 1000)
    Tk = np.random.uniform(30, 55, 1000)
    plr = np.random.uniform(0.1, 1.0, 1000)
    start = time.perf_counter()
    model.predict({"month": months, "T_sink": Tk, "part_load_ratio": plr})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 in {elapsed*1000:.2f} ms")
    assert elapsed < 1.0
