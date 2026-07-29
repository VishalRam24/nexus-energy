"""EC098 — Organic Rankine Cycle (ORC) — F1a — Test Suite"""
import sys, time, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()


def test_predict_keys(model):
    r = model.predict({"T_hot": 150.0, "T_cold": 30.0})
    for k in ["efficiency", "power_kw", "heat_input_kw", "heat_rejection_kw"]:
        assert k in r, f"Missing key: {k}"


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC098"
    assert info["fidelity"] == "F1a"


def test_eta_below_carnot(model):
    """Actual efficiency must always be less than Carnot efficiency."""
    T_hots = np.linspace(80, 280, 30)
    for T_cold in [15.0, 30.0, 45.0]:
        r = model.predict({"T_hot": T_hots, "T_cold": T_cold, "part_load_ratio": 1.0})
        T_hot_K = T_hots + 273.15
        T_cold_K = T_cold + 273.15
        eta_carnot = 1.0 - T_cold_K / T_hot_K
        eta_actual = np.asarray(r["efficiency"])
        assert np.all(eta_actual <= eta_carnot + 1e-9), \
            f"Efficiency exceeded Carnot at T_hot={T_hots[eta_actual > eta_carnot]}"


def test_eta_below_25pct(model):
    """ORC efficiency should be below 25% for sub-300C heat sources."""
    T_hots = np.linspace(80, 250, 20)
    r = model.predict({"T_hot": T_hots, "T_cold": 30.0})
    assert np.all(np.asarray(r["efficiency"]) < 0.25)


def test_eta_increases_with_T_hot(model):
    """Higher source temperature -> higher efficiency."""
    T_hots = np.array([100.0, 120.0, 150.0, 180.0, 200.0])
    r = model.predict({"T_hot": T_hots, "T_cold": 30.0})
    eta = np.asarray(r["efficiency"])
    assert np.all(np.diff(eta) >= 0), f"Efficiency not monotone with T_hot: {eta}"


def test_eta_decreases_with_T_cold(model):
    """Higher sink temperature -> lower efficiency."""
    T_colds = np.array([15.0, 20.0, 30.0, 40.0, 50.0])
    r = model.predict({"T_hot": 150.0, "T_cold": T_colds})
    eta = np.asarray(r["efficiency"])
    assert np.all(np.diff(eta) <= 0), f"Efficiency not monotone with T_cold: {eta}"


def test_power_less_than_heat_input(model):
    """Power output must be less than heat input (second law)."""
    r = model.predict({"T_hot": 150.0, "T_cold": 30.0})
    assert float(r["power_kw"]) < float(r["heat_input_kw"])


def test_energy_balance(model):
    """Q_heat_in = P_out + Q_reject (first law)."""
    r = model.predict({"T_hot": 150.0, "T_cold": 30.0})
    lhs = float(r["heat_input_kw"])
    rhs = float(r["power_kw"]) + float(r["heat_rejection_kw"])
    assert abs(lhs - rhs) < 1e-6, f"Energy balance: Q_in={lhs:.3f}, P+Q_rej={rhs:.3f}"


def test_part_load_reduces_efficiency(model):
    """Lower PLR reduces effective efficiency (part-load factor < 1 at PLR < 1)."""
    r_full = model.predict({"T_hot": 150.0, "T_cold": 30.0, "part_load_ratio": 1.0})
    r_part = model.predict({"T_hot": 150.0, "T_cold": 30.0, "part_load_ratio": 0.5})
    assert float(r_full["efficiency"]) > float(r_part["efficiency"])


def test_positive_efficiency(model):
    """Efficiency must be non-negative across valid range."""
    T_hots = np.linspace(80, 300, 20)
    T_colds = np.linspace(10, 50, 20)
    for T_h in T_hots:
        for T_c in T_colds:
            if T_h > T_c + 5:
                r = model.predict({"T_hot": T_h, "T_cold": T_c})
                assert float(r["efficiency"]) >= 0.0


def test_benchmark(model):
    T_hots = np.random.uniform(80, 280, 1000)
    T_colds = np.random.uniform(15, 45, 1000)
    plrs = np.random.uniform(0.3, 1.0, 1000)
    start = time.perf_counter()
    model.predict({"T_hot": T_hots, "T_cold": T_colds, "part_load_ratio": plrs})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed*1000:.2f} ms")
    assert elapsed < 1.0
