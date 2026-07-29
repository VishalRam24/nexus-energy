"""EC085 — Natural Gas Boiler — F1b Part-Load — Test Suite"""
import sys, time, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()


def test_predict_keys(model):
    r = model.predict({"PLR": 0.5})
    for k in ["efficiency", "heat_output_kw", "fuel_input_kw", "flue_loss_kw", "standby_loss_kw"]:
        assert k in r


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC085"
    assert info["fidelity"] == "F1b"


def test_efficiency_peaks_near_0_9(model):
    """With a0=0.75, a1=0.45, a2=-0.25, peak is at PLR=0.9, eta=0.9525."""
    plr = np.linspace(0.1, 1.0, 100)
    r = model.predict({"PLR": plr})
    eta = np.asarray(r["efficiency"])
    peak_plr = plr[np.argmax(eta)]
    assert 0.8 < peak_plr < 1.0, f"Peak efficiency at PLR={peak_plr:.2f}"
    assert float(np.max(eta)) > 0.90


def test_efficiency_drops_at_low_plr(model):
    """Efficiency at PLR=0.2 should be lower than at PLR=0.9."""
    r_low = model.predict({"PLR": 0.2})
    r_high = model.predict({"PLR": 0.9})
    assert float(r_low["efficiency"]) < float(r_high["efficiency"])


def test_efficiency_bounded(model):
    """Efficiency must be in [0, 1]."""
    plr = np.linspace(0.1, 1.0, 50)
    r = model.predict({"PLR": plr})
    assert np.all(r["efficiency"] >= 0.0) and np.all(r["efficiency"] <= 1.0)


def test_fuel_exceeds_heat_output(model):
    """Fuel input must be >= heat output (since eta <= 1)."""
    plr = np.linspace(0.1, 1.0, 50)
    r = model.predict({"PLR": plr})
    assert np.all(r["fuel_input_kw"] >= r["heat_output_kw"] - 1e-6)


def test_flue_loss_positive(model):
    """Flue gas loss must be positive when boiler is firing."""
    r = model.predict({"PLR": 0.5})
    assert float(r["flue_loss_kw"]) > 0.0


def test_flue_loss_increases_with_plr(model):
    """Higher firing rate = higher flue loss."""
    plr = np.array([0.2, 0.5, 0.8, 1.0])
    r = model.predict({"PLR": plr})
    assert np.all(np.diff(r["flue_loss_kw"]) > 0)


def test_standby_loss_constant(model):
    """Standby loss is a constant fraction of rated capacity."""
    plr = np.array([0.2, 0.5, 1.0])
    r = model.predict({"PLR": plr})
    standby = np.asarray(r["standby_loss_kw"])
    np.testing.assert_allclose(standby, 0.005 * 50.0, atol=1e-6)


def test_heat_output_proportional_to_plr(model):
    """Q_out = PLR * Q_rated."""
    plr = np.array([0.2, 0.5, 1.0])
    r = model.predict({"PLR": plr})
    np.testing.assert_allclose(r["heat_output_kw"], plr * 50.0, atol=1e-6)


def test_custom_flue_temp(model):
    """Providing flue_gas_temp should override automatic calculation."""
    r1 = model.predict({"PLR": 0.5})
    r2 = model.predict({"PLR": 0.5, "flue_gas_temp": 200.0})
    assert float(r1["flue_loss_kw"]) != float(r2["flue_loss_kw"])


def test_benchmark(model):
    plr = np.random.uniform(0.1, 1.0, 1000)
    start = time.perf_counter()
    model.predict({"PLR": plr})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 in {elapsed*1000:.2f} ms")
    assert elapsed < 1.0
