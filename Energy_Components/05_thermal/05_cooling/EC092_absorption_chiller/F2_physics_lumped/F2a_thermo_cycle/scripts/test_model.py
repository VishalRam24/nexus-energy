"""EC092 — Absorption Chiller — F2a — Test Suite"""
import sys, time, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()


def test_predict_keys(model):
    r = model.predict({"T_gen_degC": 90, "T_cond_degC": 35, "T_evap_degC": 7, "T_abs_degC": 35})
    for k in ["cop", "cooling_kw", "heat_input_kw", "pump_power_kw", "solution_flow_kg_s"]:
        assert k in r


def test_get_info(model):
    assert model.get_info()["ec_id"] == "EC092"


def test_cop_typical_range(model):
    """Single-effect LiBr COP should be 0.5-0.85."""
    r = model.predict({"T_gen_degC": 90, "T_cond_degC": 35, "T_evap_degC": 7, "T_abs_degC": 35})
    assert 0.3 < r["cop"] < 1.0, f"COP={r['cop']:.3f}"


def test_cop_increases_with_tgen(model):
    """Higher generator temp -> higher COP (up to a point)."""
    cops = []
    for Tg in [80, 85, 90, 95, 100, 110]:
        r = model.predict({"T_gen_degC": Tg, "T_cond_degC": 35, "T_evap_degC": 7, "T_abs_degC": 35})
        cops.append(r["cop"])
    # Should generally increase (may plateau)
    assert cops[-1] >= cops[0], f"COP should increase with T_gen: {cops}"


def test_pump_power_small(model):
    """Solution pump power should be << cooling capacity."""
    r = model.predict({"T_gen_degC": 90, "T_cond_degC": 35, "T_evap_degC": 7, "T_abs_degC": 35})
    assert r["pump_power_kw"] < r["cooling_kw"] * 0.05


def test_circulation_ratio(model):
    """Circulation ratio should be 5-30 for typical single-effect."""
    r = model.predict({"T_gen_degC": 90, "T_cond_degC": 35, "T_evap_degC": 7, "T_abs_degC": 35})
    assert 3.0 < r["circulation_ratio"] < 50.0


def test_heat_input_larger_than_cooling(model):
    """For COP < 1, heat input must be larger than cooling."""
    r = model.predict({"T_gen_degC": 90, "T_cond_degC": 35, "T_evap_degC": 7, "T_abs_degC": 35})
    if r["cop"] < 1.0:
        assert r["heat_input_kw"] > r["cooling_kw"]


def test_benchmark(model):
    start = time.perf_counter()
    for _ in range(100):
        model.predict({"T_gen_degC": 90, "T_cond_degC": 35, "T_evap_degC": 7, "T_abs_degC": 35})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 100 cycles in {elapsed*1000:.1f} ms")
    assert elapsed < 5.0
