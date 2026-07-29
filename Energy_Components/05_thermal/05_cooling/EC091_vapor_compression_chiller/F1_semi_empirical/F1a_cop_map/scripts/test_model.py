"""EC091 — Vapor Compression Chiller — F1a — Test Suite"""
import sys, time, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()


def test_predict_keys(model):
    r = model.predict({"T_chw_supply": 5.0, "T_cond": 35.0})
    for k in ["cop", "cooling_kw", "electrical_kw", "heat_rejection_kw"]:
        assert k in r


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC091"
    assert "cop" in info["outputs"]


def test_cop_greater_than_one(model):
    """Chiller COP must always be > 1."""
    T_evap = np.linspace(4, 12, 20)
    T_cond = np.linspace(25, 45, 20)
    for te in T_evap:
        r = model.predict({"T_chw_supply": te, "T_cond": 35.0})
        assert float(r["cop"]) > 1.0, f"COP={float(r['cop']):.2f} at T_evap={te}"


def test_cop_decreases_with_higher_T_cond(model):
    """COP must decrease as condenser temperature rises (higher lift)."""
    T_conds = np.array([25.0, 30.0, 35.0, 40.0, 45.0])
    cops = np.array([
        float(model.predict({"T_chw_supply": 5.0, "T_cond": tc})["cop"])
        for tc in T_conds
    ])
    assert np.all(np.diff(cops) < 0), f"COP not monotonically decreasing: {cops}"


def test_cop_increases_with_higher_T_evap(model):
    """COP must increase as evaporator temperature rises (lower lift)."""
    T_evaps = np.array([4.0, 6.0, 8.0, 10.0, 12.0])
    cops = np.array([
        float(model.predict({"T_chw_supply": te, "T_cond": 35.0})["cop"])
        for te in T_evaps
    ])
    assert np.all(np.diff(cops) > 0), f"COP not monotonically increasing with T_evap: {cops}"


def test_energy_balance(model):
    """Q_reject = Q_cooling + W_comp (first law of thermodynamics)."""
    r = model.predict({"T_chw_supply": 5.0, "T_cond": 35.0, "part_load_ratio": 1.0})
    q = float(r["cooling_kw"])
    w = float(r["electrical_kw"])
    q_rej = float(r["heat_rejection_kw"])
    assert abs(q_rej - (q + w)) < 0.1, f"Energy balance failed: {q_rej:.1f} != {q:.1f}+{w:.1f}"


def test_part_load_cooling_scales(model):
    """Cooling capacity must scale linearly with PLR."""
    r_full = model.predict({"T_chw_supply": 5.0, "T_cond": 35.0, "part_load_ratio": 1.0})
    r_half = model.predict({"T_chw_supply": 5.0, "T_cond": 35.0, "part_load_ratio": 0.5})
    assert abs(float(r_half["cooling_kw"]) - float(r_full["cooling_kw"]) * 0.5) < 0.5


def test_part_load_cop_degradation(model):
    """COP at part load should be less than full-load COP (PLR factor < 1 at PLR < 1)."""
    r_full = model.predict({"T_chw_supply": 5.0, "T_cond": 35.0, "part_load_ratio": 1.0})
    r_part = model.predict({"T_chw_supply": 5.0, "T_cond": 35.0, "part_load_ratio": 0.3})
    assert float(r_part["cop"]) < float(r_full["cop"]), "COP should degrade at part load"


def test_cop_at_nominal_conditions(model):
    """COP at T_evap=5C, T_cond=35C, PLR=1 should be near rated COP~5.5."""
    r = model.predict({"T_chw_supply": 5.0, "T_cond": 35.0, "part_load_ratio": 1.0})
    cop = float(r["cop"])
    assert 4.0 < cop < 7.0, f"Nominal COP={cop:.2f} outside expected range [4,7]"


def test_benchmark(model):
    T_evap = np.random.uniform(4, 12, 1000)
    T_cond = np.random.uniform(25, 45, 1000)
    plr = np.random.uniform(0.1, 1.0, 1000)
    start = time.perf_counter()
    model.predict({"T_chw_supply": T_evap, "T_cond": T_cond, "part_load_ratio": plr})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed*1000:.2f} ms")
    assert elapsed < 1.0
