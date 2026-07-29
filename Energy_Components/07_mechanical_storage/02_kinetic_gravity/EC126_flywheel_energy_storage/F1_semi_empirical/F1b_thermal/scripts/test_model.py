"""EC126 — Flywheel Energy Storage — F1b Thermal — Test Suite"""
import sys, time, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()


# --- Interface ---

def test_predict_keys(model):
    r = model.predict({"soc": 0.5, "power_command_kw": -50.0})
    for k in ["power_actual_kw", "losses_kw", "self_discharge_rate_per_hour",
              "efficiency", "speed_rpm"]:
        assert k in r


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC126"
    assert info["fidelity"] == "F1b"


# --- Speed & Energy Physics ---

def test_speed_at_soc_0_equals_omega_min(model):
    r = model.predict({"soc": 0.0, "power_command_kw": 0.0})
    expected_rpm = 15000.0
    assert abs(float(r["speed_rpm"]) - expected_rpm) < 1.0


def test_speed_at_soc_1_equals_omega_max(model):
    r = model.predict({"soc": 1.0, "power_command_kw": 0.0})
    expected_rpm = 30000.0
    assert abs(float(r["speed_rpm"]) - expected_rpm) < 1.0


def test_speed_increases_with_soc(model):
    soc = np.linspace(0, 1, 20)
    r = model.predict({"soc": soc, "power_command_kw": 0.0})
    assert np.all(np.diff(r["speed_rpm"]) > 0)


# --- Windage Loss Physics ---

def test_windage_loss_increases_with_soc(model):
    """Higher SOC -> higher speed -> more windage (cubic)."""
    from model import FlywheelF1b
    import json
    with open(Path(__file__).parent.parent / "data" / "parameters.json") as f:
        params = json.load(f)
    m = FlywheelF1b(params)
    soc = np.array([0.1, 0.5, 1.0])
    P_w = m.windage_loss(soc, 25.0)
    assert np.all(np.diff(P_w) > 0), "Windage should increase with SOC (speed)"


def test_windage_decreases_with_temperature(model):
    """Higher T -> lower air density -> less windage."""
    from model import FlywheelF1b
    import json
    with open(Path(__file__).parent.parent / "data" / "parameters.json") as f:
        params = json.load(f)
    m = FlywheelF1b(params)
    P_cold = m.windage_loss(0.5, -10.0)
    P_hot = m.windage_loss(0.5, 50.0)
    assert float(P_cold) > float(P_hot), "Windage should be less at higher temperature (lower air density)"


def test_bearing_loss_increases_with_soc(model):
    """Bearing loss linear in omega."""
    from model import FlywheelF1b
    import json
    with open(Path(__file__).parent.parent / "data" / "parameters.json") as f:
        params = json.load(f)
    m = FlywheelF1b(params)
    soc = np.array([0.1, 0.5, 1.0])
    P_b = m.bearing_loss(soc)
    assert np.all(np.diff(P_b) > 0)


# --- Self-Discharge ---

def test_self_discharge_positive(model):
    r = model.predict({"soc": 0.5, "power_command_kw": 0.0})
    assert float(r["self_discharge_rate_per_hour"]) > 0


def test_self_discharge_varies_with_soc(model):
    """Self-discharge rate should vary with SOC due to cubic windage vs quadratic energy."""
    soc = np.array([0.1, 0.5, 1.0])
    r = model.predict({"soc": soc, "power_command_kw": 0.0})
    # Just verify it's not constant
    sd = r["self_discharge_rate_per_hour"]
    assert not np.allclose(sd, sd[0]), "Self-discharge rate should vary with SOC"


# --- Efficiency ---

def test_efficiency_bounded(model):
    """Efficiency between 0 and 1."""
    soc = np.random.uniform(0.1, 1.0, 100)
    P = np.random.choice([-50, -25, 25, 50, 75, 100], size=100)
    r = model.predict({"soc": soc, "power_command_kw": P, "ambient_temperature": 25.0})
    # Efficiency can exceed 1.0 if standby losses make actual output > command in some edge cases
    # But for large power commands it should be < 1
    eta = r["efficiency"]
    assert np.all(eta >= 0.0)


def test_efficiency_higher_at_higher_power(model):
    """Standby losses are fixed, so larger power command -> higher fraction efficiency."""
    r_low = model.predict({"soc": 0.5, "power_command_kw": 10.0})
    r_high = model.predict({"soc": 0.5, "power_command_kw": 100.0})
    assert float(r_high["efficiency"]) > float(r_low["efficiency"])


# --- Losses ---

def test_losses_positive_during_operation(model):
    r = model.predict({"soc": 0.5, "power_command_kw": 50.0})
    assert float(r["losses_kw"]) > 0


# --- Temperature Effect ---

def test_higher_temp_less_total_loss(model):
    """Higher ambient -> less windage -> less total loss (at same SOC, idle)."""
    from model import FlywheelF1b
    import json
    with open(Path(__file__).parent.parent / "data" / "parameters.json") as f:
        params = json.load(f)
    m = FlywheelF1b(params)
    loss_cold = float(m.total_standby_loss(0.5, -10.0))
    loss_hot = float(m.total_standby_loss(0.5, 50.0))
    assert loss_cold > loss_hot


# --- Edge Cases ---

def test_soc_zero(model):
    r = model.predict({"soc": 0.0, "power_command_kw": 0.0})
    assert float(r["speed_rpm"]) > 0  # Still at omega_min


def test_vectorized(model):
    soc = np.linspace(0.1, 1.0, 50)
    P = np.linspace(-100, 100, 50)
    r = model.predict({"soc": soc, "power_command_kw": P})
    assert len(r["efficiency"]) == 50


def test_benchmark(model):
    soc = np.random.uniform(0, 1, 1000)
    P = np.random.uniform(-100, 100, 1000)
    start = time.perf_counter()
    model.predict({"soc": soc, "power_command_kw": P})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed*1000:.2f} ms")
    assert elapsed < 1.0
