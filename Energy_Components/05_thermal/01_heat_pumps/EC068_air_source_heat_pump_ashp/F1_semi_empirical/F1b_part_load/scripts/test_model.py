"""EC068 — ASHP — F1b Part-Load — Test Suite"""
import sys, time, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()


def test_predict_keys(model):
    r = model.predict({"T_source": 7.0, "T_sink": 35.0, "part_load_ratio": 0.5})
    for k in ["cop", "heating_capacity_kw", "electrical_input_kw", "cop_degradation_factor"]:
        assert k in r


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC068"
    assert info["fidelity"] == "F1b"


def test_cop_greater_than_one(model):
    """COP must be > 1 at all operating points."""
    plr = np.linspace(0.1, 1.0, 20)
    r = model.predict({"T_source": 7.0, "T_sink": 35.0, "part_load_ratio": plr})
    assert np.all(r["cop"] > 1.0)


def test_cop_decreases_with_lower_plr(model):
    """COP must decrease as PLR decreases (due to PLF degradation)."""
    plr = np.array([0.25, 0.5, 0.75, 1.0])
    r = model.predict({"T_source": 7.0, "T_sink": 35.0, "part_load_ratio": plr})
    assert np.all(np.diff(r["cop"]) > 0), "COP should increase with increasing PLR"


def test_plf_at_full_load(model):
    """PLF must be 1.0 at PLR=1.0 (no degradation at full load)."""
    r = model.predict({"T_source": 7.0, "T_sink": 35.0, "part_load_ratio": 1.0})
    np.testing.assert_allclose(float(r["cop_degradation_factor"]), 1.0, atol=1e-10)


def test_plf_less_than_one_at_part_load(model):
    """PLF < 1 at PLR < 1."""
    r = model.predict({"T_source": 7.0, "T_sink": 35.0, "part_load_ratio": 0.5})
    assert float(r["cop_degradation_factor"]) < 1.0


def test_cop_at_rating_conditions(model):
    """COP at A7/W35, PLR=1 should match F1a (~3.5)."""
    r = model.predict({"T_source": 7.0, "T_sink": 35.0, "part_load_ratio": 1.0})
    assert 2.5 < float(r["cop"]) < 5.0, f"COP at full load = {float(r['cop']):.2f}"


def test_cop_with_temp_lift(model):
    """COP decreases with increasing temperature lift."""
    sources = np.array([15.0, 7.0, 0.0, -7.0])
    r = model.predict({"T_source": sources, "T_sink": 35.0, "part_load_ratio": 0.8})
    assert np.all(np.diff(r["cop"]) < 0)


def test_cycling_penalty_below_plr_min(model):
    """COP degradation is larger below PLR_min due to cycling."""
    r_above = model.predict({"T_source": 7.0, "T_sink": 35.0, "part_load_ratio": 0.15})
    r_below = model.predict({"T_source": 7.0, "T_sink": 35.0, "part_load_ratio": 0.05})
    assert float(r_below["cop_degradation_factor"]) < float(r_above["cop_degradation_factor"])


def test_heating_capacity_proportional_to_plr(model):
    """Heating capacity = rated_capacity * PLR."""
    plr = np.array([0.25, 0.5, 0.75, 1.0])
    r = model.predict({"T_source": 7.0, "T_sink": 35.0, "part_load_ratio": plr})
    np.testing.assert_allclose(r["heating_capacity_kw"], 10.0 * plr)


def test_benchmark(model):
    Ts = np.random.uniform(-15, 30, 1000)
    Tk = np.random.uniform(30, 55, 1000)
    plr = np.random.uniform(0.1, 1.0, 1000)
    start = time.perf_counter()
    model.predict({"T_source": Ts, "T_sink": Tk, "part_load_ratio": plr})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 in {elapsed*1000:.2f} ms")
    assert elapsed < 1.0
