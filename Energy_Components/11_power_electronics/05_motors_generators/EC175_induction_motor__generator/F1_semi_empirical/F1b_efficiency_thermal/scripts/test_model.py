"""EC175 — Induction Motor/Generator — F1b Thermal — Test Suite"""
import sys, time, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()


# --- Interface Tests ---

def test_predict_keys(model):
    r = model.predict({"load_fraction": 1.0})
    for k in ["efficiency", "input_power_kw", "output_power_kw", "losses_kw",
              "current_A", "derating_factor", "slip"]:
        assert k in r


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC175"
    assert info["fidelity"] == "F1b"


# --- Physics Sanity ---

def test_efficiency_less_than_one(model):
    plr = np.linspace(0.05, 1.2, 100)
    for T_w in [25, 75, 120, 155]:
        r = model.predict({"load_fraction": plr, "winding_temperature": T_w})
        assert np.all(r["efficiency"] < 1.0), f"eta >= 1 at T_w={T_w}"


def test_efficiency_positive(model):
    plr = np.linspace(0.05, 1.2, 100)
    r = model.predict({"load_fraction": plr, "winding_temperature": 75.0})
    assert np.all(r["efficiency"] > 0.0)


def test_losses_positive(model):
    plr = np.linspace(0.05, 1.2, 50)
    r = model.predict({"load_fraction": plr, "winding_temperature": 100.0})
    assert np.all(r["losses_kw"] > 0.0)


def test_power_balance(model):
    plr = np.linspace(0.1, 1.2, 50)
    r = model.predict({"load_fraction": plr, "winding_temperature": 80.0})
    diff = np.abs(r["input_power_kw"] - r["output_power_kw"] - r["losses_kw"])
    assert np.all(diff < 1e-9), "Power balance violated"


# --- Thermal Monotonicity ---

def test_efficiency_decreases_with_temperature(model):
    """Higher winding temperature -> higher copper losses -> lower efficiency."""
    plr = 1.0
    temps = [25, 50, 75, 100, 120, 155]
    etas = []
    for T_w in temps:
        r = model.predict({"load_fraction": plr, "winding_temperature": T_w})
        etas.append(float(r["efficiency"]))
    for i in range(len(etas) - 1):
        assert etas[i] > etas[i + 1], (
            f"eta should decrease: eta({temps[i]}C)={etas[i]:.4f} "
            f"<= eta({temps[i+1]}C)={etas[i+1]:.4f}"
        )


def test_losses_increase_with_temperature(model):
    """Higher T -> more copper losses."""
    plr = 1.0
    loss_25 = float(model.predict({"load_fraction": plr, "winding_temperature": 25.0})["losses_kw"])
    loss_155 = float(model.predict({"load_fraction": plr, "winding_temperature": 155.0})["losses_kw"])
    assert loss_155 > loss_25, "Losses must increase with winding temperature"


def test_current_increases_with_temperature(model):
    """More losses -> more input power -> higher current at same output."""
    plr = 1.0
    I_25 = float(model.predict({"load_fraction": plr, "winding_temperature": 25.0})["current_A"])
    I_155 = float(model.predict({"load_fraction": plr, "winding_temperature": 155.0})["current_A"])
    assert I_155 > I_25, "Current must increase with temperature"


# --- Derating Tests ---

def test_derating_unity_below_threshold(model):
    """No derating below 40C ambient."""
    r = model.predict({"load_fraction": 1.0, "ambient_temperature": 25.0})
    assert float(r["derating_factor"]) == 1.0


def test_derating_below_unity_above_threshold(model):
    """Derating above 40C ambient."""
    r = model.predict({"load_fraction": 1.0, "ambient_temperature": 50.0})
    assert float(r["derating_factor"]) < 1.0


def test_derating_monotonic(model):
    """Derating factor decreases with ambient temperature above threshold."""
    from model import InductionMotorF1b
    import json
    with open(Path(__file__).parent.parent / "data" / "parameters.json") as f:
        params = json.load(f)
    m = InductionMotorF1b(params)
    T_amb = np.array([25.0, 40.0, 45.0, 50.0, 55.0, 60.0])
    d = m.derating_factor(T_amb)
    # Should be non-increasing
    assert np.all(np.diff(d) <= 0), f"Derating should be non-increasing: {d}"


# --- Reference Point ---

def test_rated_efficiency_at_reference_temperature(model):
    """At T_ref=25C, F1b should match F1a rated efficiency."""
    r = model.predict({"load_fraction": 1.0, "winding_temperature": 25.0})
    eta = float(r["efficiency"])
    assert abs(eta - 0.917) < 0.001, f"eta at T_ref should be ~0.917, got {eta:.4f}"


# --- Edge Cases ---

def test_extreme_temperature(model):
    """Model should not crash at extreme temperature."""
    r = model.predict({"load_fraction": 0.5, "winding_temperature": 180.0})
    assert 0 < float(r["efficiency"]) < 1.0


def test_vectorized(model):
    plr = np.linspace(0.1, 1.2, 50)
    T_w = np.linspace(25, 155, 50)
    r = model.predict({"load_fraction": plr, "winding_temperature": T_w})
    assert len(r["efficiency"]) == 50


def test_benchmark(model):
    plr = np.random.uniform(0.05, 1.2, 1000)
    T_w = np.random.uniform(25, 155, 1000)
    start = time.perf_counter()
    model.predict({"load_fraction": plr, "winding_temperature": T_w})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed*1000:.2f} ms")
    assert elapsed < 1.0
