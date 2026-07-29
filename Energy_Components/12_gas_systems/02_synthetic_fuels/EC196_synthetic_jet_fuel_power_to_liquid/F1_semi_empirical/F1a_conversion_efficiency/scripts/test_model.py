"""EC196 — Synthetic Jet Fuel — F1a Conversion Efficiency — Test Suite"""
import sys, time, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()


def test_predict_keys(model):
    r = model.predict({"temperature_C": 220.0, "pressure_bar": 25.0})
    for k in ["co_conversion", "selectivity_jet_C8_C16", "jet_fuel_mol_s",
              "energy_efficiency", "heat_released_kW"]:
        assert k in r, f"Missing key: {k}"


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC196"
    assert info["fidelity"] == "F1a"


def test_conversion_at_optimal_temp_high(model):
    """At T_opt=220C and reference pressure, conversion should be high (>0.7)."""
    X = float(model.model.conversion(220.0, 25.0))
    assert X > 0.7, f"CO conversion at design = {X:.4f}, expected > 0.7"


def test_conversion_bounded(model):
    X = float(model.model.conversion(220.0, 25.0))
    assert 0.0 <= X <= 1.0


def test_conversion_decreases_far_from_optimal_T(model):
    """Conversion at T_opt > at T_opt+80."""
    X_opt = float(model.model.conversion(220.0, 25.0))
    X_far = float(model.model.conversion(300.0, 25.0))
    assert X_opt > X_far, "Conversion must peak at T_opt"


def test_conversion_increases_with_pressure(model):
    """Higher pressure → more favorable FT equilibrium."""
    X_low = float(model.model.conversion(220.0, 15.0))
    X_high = float(model.model.conversion(220.0, 35.0))
    assert X_high > X_low


def test_asf_selectivity_range(model):
    """For alpha=0.90, C8-C16 selectivity should be ~20-40%."""
    S = model.model.asf_selectivity_jet()
    assert 0.10 < S < 0.60, f"S_jet={S:.4f} out of expected range"


def test_asf_selectivity_increases_with_alpha(model):
    """Higher alpha (longer chain) → more C8-C16."""
    S_low  = model.model.asf_selectivity_jet(alpha=0.75)
    S_high = model.model.asf_selectivity_jet(alpha=0.92)
    assert S_high > S_low


def test_energy_efficiency_positive(model):
    eta = float(model.model.energy_efficiency(220.0, 25.0))
    assert eta > 0.0


def test_energy_efficiency_less_than_one(model):
    """Efficiency < 1 (we don't get more energy out than in via FT alone)."""
    eta = float(model.model.energy_efficiency(220.0, 25.0))
    assert eta <= 1.0


def test_heat_released_positive(model):
    Q = float(model.model.heat_released_kW(220.0, 25.0, n_co_in=1.0))
    assert Q > 0.0


def test_heat_released_scales_with_flow(model):
    Q1 = float(model.model.heat_released_kW(220.0, 25.0, n_co_in=1.0))
    Q2 = float(model.model.heat_released_kW(220.0, 25.0, n_co_in=2.0))
    assert abs(Q2 - 2.0 * Q1) < 0.01


def test_vectorized(model):
    T = np.linspace(180, 260, 20)
    r = model.predict({"temperature_C": T, "pressure_bar": 25.0})
    assert r["co_conversion"].shape == (20,)


def test_benchmark(model):
    T = np.random.uniform(180, 260, 1000)
    start = time.perf_counter()
    model.predict({"temperature_C": T, "pressure_bar": 25.0})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed*1000:.2f} ms")
    assert elapsed < 5.0
