"""EC175 — Induction Motor/Generator — F1a — Test Suite"""
import sys, time, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()


def test_predict_keys(model):
    r = model.predict({"load_fraction": 1.0})
    for k in ["efficiency", "input_power_kw", "output_power_kw", "losses_kw", "slip"]:
        assert k in r


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC175"
    assert info["fidelity"] == "F1a"


def test_efficiency_less_than_one(model):
    """Efficiency must never exceed 1."""
    plr = np.linspace(0.05, 1.2, 100)
    r = model.predict({"load_fraction": plr})
    assert np.all(r["efficiency"] < 1.0), "Efficiency must be < 1 everywhere"


def test_efficiency_positive(model):
    """Efficiency must be positive."""
    plr = np.linspace(0.05, 1.2, 100)
    r = model.predict({"load_fraction": plr})
    assert np.all(r["efficiency"] > 0.0)


def test_efficiency_peaks_near_full_load(model):
    """Efficiency should be highest near full load (PLR ~0.75-1.0)."""
    plr = np.linspace(0.1, 1.1, 200)
    r = model.predict({"load_fraction": plr})
    peak_idx = np.argmax(r["efficiency"])
    peak_plr = plr[peak_idx]
    assert 0.5 <= peak_plr <= 1.05, f"Peak efficiency at PLR={peak_plr:.2f}, expected near 1.0"


def test_rated_efficiency(model):
    """At PLR=1.0, efficiency should be close to rated (0.917 for IE3)."""
    r = model.predict({"load_fraction": 1.0})
    eta = float(r["efficiency"])
    assert abs(eta - 0.917) < 0.001, f"eta at rated = {eta:.4f}, expected ~0.917"


def test_losses_positive(model):
    """Losses must always be positive."""
    plr = np.linspace(0.05, 1.2, 50)
    r = model.predict({"load_fraction": plr})
    assert np.all(r["losses_kw"] > 0.0), "Losses must be positive"


def test_power_balance(model):
    """Input power = output power + losses."""
    plr = np.linspace(0.1, 1.2, 50)
    r = model.predict({"load_fraction": plr})
    diff = np.abs(r["input_power_kw"] - r["output_power_kw"] - r["losses_kw"])
    assert np.all(diff < 1e-9), "Power balance violated: P_in != P_out + P_loss"


def test_slip_in_range(model):
    """Slip must be in [0, 1]."""
    plr = np.linspace(0.0, 1.2, 50)
    r = model.predict({"load_fraction": plr})
    assert np.all(r["slip"] >= 0.0) and np.all(r["slip"] <= 1.0)


def test_slip_increases_with_load(model):
    """Slip increases with load (linear approximation)."""
    plr = np.array([0.25, 0.5, 0.75, 1.0])
    r = model.predict({"load_fraction": plr})
    assert np.all(np.diff(r["slip"]) > 0), "Slip should increase with load"


def test_output_power_proportional_to_plr(model):
    """Output power = PLR * rated_power (by definition)."""
    plr = np.array([0.25, 0.50, 0.75, 1.0])
    r = model.predict({"load_fraction": plr})
    expected = plr * 15.0  # 15 kW rated
    np.testing.assert_allclose(r["output_power_kw"], expected, rtol=1e-9)


def test_vectorized_input(model):
    """Model must accept array inputs."""
    plr = np.linspace(0.1, 1.2, 50)
    r = model.predict({"load_fraction": plr})
    assert len(r["efficiency"]) == 50


def test_benchmark(model):
    plr = np.random.uniform(0.05, 1.2, 1000)
    start = time.perf_counter()
    model.predict({"load_fraction": plr})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed*1000:.2f} ms")
    assert elapsed < 1.0
