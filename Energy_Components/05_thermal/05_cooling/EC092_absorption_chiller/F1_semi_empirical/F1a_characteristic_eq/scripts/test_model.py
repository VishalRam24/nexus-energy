"""EC092 — Absorption Chiller — F1a — Test Suite"""
import sys, time, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()


def test_predict_keys(model):
    r = model.predict({"T_generator": 90.0, "T_condenser": 35.0, "T_evaporator": 7.0})
    for k in ["cop", "cooling_kw", "heat_input_kw", "heat_rejection_kw"]:
        assert k in r, f"Missing key: {k}"


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC092"
    assert info["fidelity"] == "F1a"


def test_cop_below_single_effect_limit(model):
    """Single-effect LiBr-H2O COP must never exceed 0.80."""
    T_gens = np.linspace(70, 120, 30)
    for T_cond in [25.0, 35.0, 45.0]:
        r = model.predict({"T_generator": T_gens, "T_condenser": T_cond, "T_evaporator": 7.0})
        cop = np.asarray(r["cop"])
        assert np.all(cop <= 0.80 + 1e-9), f"COP exceeded 0.80: max={cop.max():.4f}"


def test_cop_increases_with_T_gen(model):
    """Higher generator temperature -> higher COP (more driving force)."""
    T_gens = np.array([75.0, 80.0, 85.0, 90.0, 95.0, 100.0])
    r = model.predict({"T_generator": T_gens, "T_condenser": 35.0, "T_evaporator": 7.0})
    cop = np.asarray(r["cop"])
    assert np.all(np.diff(cop) >= 0), f"COP did not increase monotonically with T_gen: {cop}"


def test_cop_decreases_with_T_cond(model):
    """Higher condenser temperature -> lower COP (less driving force)."""
    T_conds = np.array([28.0, 32.0, 35.0, 38.0, 42.0])
    r = model.predict({"T_generator": 90.0, "T_condenser": T_conds, "T_evaporator": 7.0})
    cop = np.asarray(r["cop"])
    assert np.all(np.diff(cop) <= 0), f"COP did not decrease with T_cond: {cop}"


def test_energy_balance(model):
    """Q_reject = Q_generator + Q_cool (first law of thermodynamics)."""
    r = model.predict({"T_generator": 90.0, "T_condenser": 35.0, "T_evaporator": 7.0})
    Q_gen = float(r["heat_input_kw"])
    Q_cool = float(r["cooling_kw"])
    Q_rej = float(r["heat_rejection_kw"])
    assert abs(Q_rej - (Q_gen + Q_cool)) < 1e-6, \
        f"Energy balance failed: Q_rej={Q_rej:.2f}, Q_gen+Q_cool={Q_gen+Q_cool:.2f}"


def test_cop_at_rated_conditions(model):
    """COP at rated conditions (T_gen=90C, T_cond=35C) should be ~0.70."""
    r = model.predict({"T_generator": 90.0, "T_condenser": 35.0, "T_evaporator": 7.0})
    cop = float(r["cop"])
    assert 0.60 < cop <= 0.80, f"COP at rated = {cop:.4f}, expected ~0.70"


def test_cop_positive(model):
    """COP must be non-negative across all valid operating range."""
    T_gens = np.linspace(70, 120, 20)
    T_conds = np.linspace(25, 45, 20)
    T_gen_grid, T_cond_grid = np.meshgrid(T_gens, T_conds)
    r = model.predict({"T_generator": T_gen_grid.ravel(),
                       "T_condenser": T_cond_grid.ravel(),
                       "T_evaporator": 7.0})
    assert np.all(np.asarray(r["cop"]) >= 0.0)


def test_heat_rejection_exceeds_cooling(model):
    """Heat rejected to cooling tower must exceed cooling delivered."""
    r = model.predict({"T_generator": 90.0, "T_condenser": 35.0, "T_evaporator": 7.0})
    assert float(r["heat_rejection_kw"]) > float(r["cooling_kw"])


def test_benchmark(model):
    T_gen = np.random.uniform(70, 120, 1000)
    T_cond = np.random.uniform(25, 45, 1000)
    start = time.perf_counter()
    model.predict({"T_generator": T_gen, "T_condenser": T_cond, "T_evaporator": 7.0})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed*1000:.2f} ms")
    assert elapsed < 1.0
