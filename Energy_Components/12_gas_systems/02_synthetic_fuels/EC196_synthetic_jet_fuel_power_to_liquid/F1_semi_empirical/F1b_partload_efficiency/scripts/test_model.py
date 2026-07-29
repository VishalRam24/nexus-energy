"""EC196 — Synthetic Jet Fuel — F1b Part-Load + Thermal — Test Suite"""
import sys, time, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()


def test_predict_keys(model):
    r = model.predict({"T_set": 220.0, "pressure_bar": 25.0})
    for k in ["co_conversion", "effective_temperature_C", "alpha_ASF",
              "selectivity_jet_C8_C16", "jet_fuel_mol_s",
              "heat_recovery_kW", "energy_efficiency", "deactivation_factor"]:
        assert k in r, f"Missing: {k}"


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC196"
    assert info["fidelity"] == "F1b"


def test_effective_temperature_decreases_at_part_load(model):
    T_full = float(model.model._effective_temperature(220.0, 1.0))
    T_half = float(model.model._effective_temperature(220.0, 0.5))
    assert T_half < T_full, "Lower PLR → lower effective bed temperature"


def test_conversion_decreases_at_part_load(model):
    X_full = float(model.model.conversion(220.0, 25.0, plr=1.0))
    X_half = float(model.model.conversion(220.0, 25.0, plr=0.5))
    assert X_half < X_full, "Part-load conversion must be lower than full-load"


def test_conversion_bounded(model):
    X = float(model.model.conversion(220.0, 25.0, plr=0.6))
    assert 0.0 <= X <= 1.0


def test_alpha_increases_at_lower_temperature(model):
    """Lower bed T → longer chain → higher alpha (more wax, less jet)."""
    alpha_hot  = float(model.model._alpha_at_T(230.0))
    alpha_cold = float(model.model._alpha_at_T(200.0))
    assert alpha_cold > alpha_hot, "Alpha must increase at lower T (more wax in LTFT)"


def test_deactivation_increases_with_hours(model):
    d0 = float(model.model.deactivation_factor(0.0))
    d10k = float(model.model.deactivation_factor(10000.0))
    assert d0 > d10k, "Catalyst activity must decrease with operating hours"


def test_deactivation_bounded_below_20pct(model):
    """Deactivation capped at 20% loss."""
    d = float(model.model.deactivation_factor(200000.0))  # extreme
    assert d >= 0.80, f"Deactivation floor at 0.80, got {d:.4f}"


def test_heat_recovery_positive_at_full_load(model):
    Q = float(model.model.heat_recovery_kW(220.0, 25.0))
    assert Q > 0.0


def test_heat_recovery_scales_with_plr(model):
    """Higher PLR → more throughput → more heat released."""
    Q_high = float(model.model.heat_recovery_kW(220.0, 25.0, plr=1.0))
    Q_low  = float(model.model.heat_recovery_kW(220.0, 25.0, plr=0.5))
    assert Q_high > Q_low


def test_energy_efficiency_positive(model):
    eta = float(model.model.energy_efficiency(220.0, 25.0))
    assert eta > 0.0


def test_energy_efficiency_bounded(model):
    eta = float(model.model.energy_efficiency(220.0, 25.0))
    assert 0.0 < eta <= 1.0


def test_vectorized(model):
    plr = np.linspace(0.2, 1.0, 20)
    r = model.predict({"T_set": 220.0, "pressure_bar": 25.0, "plr": plr})
    assert r["co_conversion"].shape == (20,)


def test_benchmark(model):
    T = np.random.uniform(190, 250, 1000)
    start = time.perf_counter()
    model.predict({"T_set": T, "pressure_bar": 25.0})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed*1000:.2f} ms")
    assert elapsed < 5.0
