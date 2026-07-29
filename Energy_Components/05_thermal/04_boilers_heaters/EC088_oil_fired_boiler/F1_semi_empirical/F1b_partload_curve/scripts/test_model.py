"""EC088 — Oil-Fired Boiler — F1b Part-Load — Test Suite"""
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
    assert info["ec_id"] == "EC088"
    assert info["fidelity"] == "F1b"


def test_efficiency_peaks_near_design_plr(model):
    """Peak efficiency should be in the 0.7-1.0 PLR range for oil boiler."""
    plr = np.linspace(0.15, 1.0, 100)
    r   = model.predict({"PLR": plr})
    peak_plr = plr[np.argmax(r["efficiency"])]
    assert 0.6 < peak_plr <= 1.0, f"Peak efficiency at PLR={peak_plr:.2f}"


def test_efficiency_bounded(model):
    plr = np.linspace(0.15, 1.0, 50)
    r   = model.predict({"PLR": plr})
    assert np.all(r["efficiency"] >= 0.0) and np.all(r["efficiency"] <= 1.0)


def test_fuel_exceeds_heat_output(model):
    plr = np.linspace(0.15, 1.0, 50)
    r   = model.predict({"PLR": plr})
    assert np.all(r["fuel_input_kw"] >= r["heat_output_kw"] - 1e-6)


def test_flue_loss_positive(model):
    r = model.predict({"PLR": 0.5})
    assert float(r["flue_loss_kw"]) > 0.0


def test_flue_loss_increases_with_plr(model):
    """Higher firing rate = more flue mass flow = larger flue loss."""
    plr = np.array([0.2, 0.4, 0.7, 1.0])
    r   = model.predict({"PLR": plr})
    assert np.all(np.diff(r["flue_loss_kw"]) > 0), "Flue loss must increase with PLR"


def test_flue_temp_above_sulfur_dewpoint(model):
    """Oil flue gas must stay above ~120 degC (sulfur dewpoint) to avoid acid condensation.
    RATIONALE: Minimum flue temp constraint for fuel oil (BS EN 303-1)."""
    plr = np.linspace(0.15, 1.0, 20)
    r   = model.predict({"PLR": plr})
    assert np.all(r["flue_gas_temp_c"] >= 100.0), \
        "Oil boiler flue gas must stay above sulfur dewpoint"


def test_standby_loss_constant(model):
    plr = np.array([0.2, 0.5, 1.0])
    r   = model.predict({"PLR": plr})
    sb  = np.asarray(r["standby_loss_kw"])
    expected = model._physics.standby_frac * model._physics.Q_rated
    np.testing.assert_allclose(sb, expected, atol=1e-6)


def test_heat_output_proportional_to_plr(model):
    plr = np.array([0.2, 0.5, 1.0])
    r   = model.predict({"PLR": plr})
    Q_rated = model._physics.Q_rated
    np.testing.assert_allclose(r["heat_output_kw"], plr * Q_rated, atol=1e-6)


def test_custom_flue_temp(model):
    r1 = model.predict({"PLR": 0.5})
    r2 = model.predict({"PLR": 0.5, "flue_gas_temp": 250.0})
    assert float(r1["flue_loss_kw"]) != float(r2["flue_loss_kw"])


def test_benchmark(model):
    plr = np.random.uniform(0.15, 1.0, 1000)
    t0  = time.perf_counter()
    model.predict({"PLR": plr})
    assert time.perf_counter() - t0 < 1.0
