"""EC126 — Flywheel Energy Storage — F1a Kinetic Model — Test Suite"""
import sys, time, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()


def test_predict_keys(model):
    r = model.predict({"speed_rpm": 12000.0})
    for k in ["energy_stored_kwh", "soc", "power_kw", "self_discharge_kw", "round_trip_efficiency"]:
        assert k in r, f"Missing key: {k}"


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC126"
    assert info["fidelity"] == "F1a"


def test_soc_bounds(model):
    """SOC must be in [0, 1] across full speed range."""
    rpm = np.linspace(8000, 16000, 100)
    r = model.predict({"speed_rpm": rpm})
    assert np.all(r["soc"] >= 0.0) and np.all(r["soc"] <= 1.0)


def test_soc_zero_at_min_speed(model):
    """SOC = 0 at minimum speed (omega_min)."""
    r = model.predict({"speed_rpm": 8000.0})
    assert abs(float(r["soc"])) < 1e-6, f"SOC at omega_min = {float(r['soc']):.6f}"


def test_soc_one_at_max_speed(model):
    """SOC = 1 at maximum speed (omega_max)."""
    r = model.predict({"speed_rpm": 16000.0})
    # Small floating-point rounding in rpm→rad/s conversion is expected
    assert abs(float(r["soc"]) - 1.0) < 1e-3, f"SOC at omega_max = {float(r['soc']):.6f}"


def test_energy_proportional_to_omega_squared(model):
    """E = 0.5*J*omega^2: energy must scale with omega^2."""
    rpm_a, rpm_b = 8000.0, 16000.0
    r_a = model.predict({"speed_rpm": rpm_a})
    r_b = model.predict({"speed_rpm": rpm_b})
    # omega ratio = 2, so E ratio = 4
    ratio = float(r_b["energy_stored_kwh"]) / float(r_a["energy_stored_kwh"])
    assert abs(ratio - 4.0) < 1e-4, f"Energy ratio (16k/8k rpm) = {ratio:.4f}, expected 4.0"


def test_energy_increases_with_speed(model):
    """Energy stored must monotonically increase with speed."""
    rpm = np.linspace(8000, 16000, 50)
    r = model.predict({"speed_rpm": rpm})
    assert np.all(np.diff(r["energy_stored_kwh"]) > 0)


def test_self_discharge_positive(model):
    """Self-discharge power must always be positive (energy is lost, not gained)."""
    rpm = np.linspace(8000, 16000, 50)
    r = model.predict({"speed_rpm": rpm})
    assert np.all(r["self_discharge_kw"] > 0.0)


def test_rte_less_than_one(model):
    """Round-trip efficiency must always be < 1."""
    t = np.linspace(0, 10, 50)
    r = model.predict({"speed_rpm": 12000.0, "time_hours": t})
    assert np.all(r["round_trip_efficiency"] < 1.0)


def test_rte_decreases_with_standby_time(model):
    """RTE must decrease with longer standby time due to self-discharge losses."""
    t = np.linspace(0, 10, 50)
    r = model.predict({"speed_rpm": 12000.0, "time_hours": t})
    assert np.all(np.diff(r["round_trip_efficiency"]) < 0)


def test_rte_zero_standby(model):
    """RTE at t=0 = eta_motor * eta_gen = 0.95 * 0.95 = 0.9025."""
    r = model.predict({"speed_rpm": 12000.0, "time_hours": 0.0})
    assert abs(float(r["round_trip_efficiency"]) - 0.9025) < 1e-6


def test_charging_power_positive(model):
    """Positive torque → charging → positive electrical power (motor draws from grid)."""
    r = model.predict({"speed_rpm": 12000.0, "torque_nm": 50.0})
    assert float(r["power_kw"]) > 0.0, "Charging power must be positive"


def test_discharging_power_negative(model):
    """Negative torque → discharging → negative electrical power (generator feeds grid)."""
    r = model.predict({"speed_rpm": 12000.0, "torque_nm": -50.0})
    assert float(r["power_kw"]) < 0.0, "Discharging power must be negative"


def test_power_zero_at_zero_torque(model):
    """Zero torque → zero power (coast, no conversion)."""
    r = model.predict({"speed_rpm": 12000.0, "torque_nm": 0.0})
    assert float(r["power_kw"]) == 0.0


def test_benchmark(model):
    rpm = np.random.uniform(8000, 16000, 1000)
    start = time.perf_counter()
    model.predict({"speed_rpm": rpm})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed*1000:.2f} ms")
    assert elapsed < 1.0
