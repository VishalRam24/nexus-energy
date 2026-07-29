"""EC197 — DME Synthesis Reactor — F1a Conversion Efficiency — Test Suite"""
import sys, time, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()


def test_predict_keys(model):
    r = model.predict({"temperature_C": 260.0, "pressure_bar": 40.0})
    for k in ["co_conversion", "selectivity_dme", "dme_production_mol_s",
              "energy_efficiency", "heat_released_kW"]:
        assert k in r, f"Missing: {k}"


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC197"
    assert info["fidelity"] == "F1a"


def test_conversion_positive_at_design(model):
    X = float(model.model.conversion(260.0, 40.0))
    assert X > 0.5, f"CO conversion at design = {X:.4f}"


def test_conversion_bounded(model):
    X = float(model.model.conversion(260.0, 40.0))
    assert 0.0 <= X <= 1.0


def test_conversion_decreases_far_from_optimal(model):
    X_opt = float(model.model.conversion(260.0, 40.0))
    X_far = float(model.model.conversion(400.0, 40.0))
    assert X_opt > X_far


def test_conversion_increases_with_pressure(model):
    X_low  = float(model.model.conversion(260.0, 20.0))
    X_high = float(model.model.conversion(260.0, 50.0))
    assert X_high > X_low


def test_selectivity_peak_near_T_S_opt(model):
    """Selectivity maximum near T_S_opt = 265 degC."""
    S_opt = float(model.model.selectivity_dme(265.0))
    S_low = float(model.model.selectivity_dme(200.0))
    S_high = float(model.model.selectivity_dme(330.0))
    assert S_opt > S_low, "S_DME at 265 must exceed S at 200"
    assert S_opt > S_high, "S_DME at 265 must exceed S at 330"


def test_selectivity_bounded(model):
    S = float(model.model.selectivity_dme(260.0))
    assert 0.0 <= S <= 1.0


def test_dme_production_positive(model):
    dme = float(model.model.dme_production_mol_s(260.0, 40.0, n_co_in=1.0))
    assert dme > 0.0


def test_heat_released_positive(model):
    Q = float(model.model.heat_released_kW(260.0, 40.0, n_co_in=1.0))
    assert Q > 0.0


def test_energy_efficiency_positive(model):
    eta = float(model.model.energy_efficiency(260.0, 40.0))
    assert eta > 0.0


def test_energy_efficiency_bounded(model):
    eta = float(model.model.energy_efficiency(260.0, 40.0))
    assert eta <= 1.0


def test_vectorized(model):
    T = np.linspace(200, 330, 20)
    r = model.predict({"temperature_C": T, "pressure_bar": 40.0})
    assert r["co_conversion"].shape == (20,)


def test_benchmark(model):
    T = np.random.uniform(200, 330, 1000)
    start = time.perf_counter()
    model.predict({"temperature_C": T, "pressure_bar": 40.0})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed*1000:.2f} ms")
    assert elapsed < 5.0
