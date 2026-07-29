"""EC078 — Hot Water Tank TES — F1a — Test Suite"""
import sys, time, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()


def test_predict_keys(model):
    r = model.predict({"temperature": 60.0})
    for k in ["dT_dt", "energy_stored_kwh", "soc", "heat_loss_w"]:
        assert k in r


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC078"
    assert info["fidelity"] == "F1a"


def test_energy_nonnegative(model):
    """Stored energy must be >= 0 at T >= T_min."""
    T = np.linspace(30.0, 90.0, 50)   # T_min = 30
    r = model.predict({"temperature": T})
    assert np.all(r["energy_stored_kwh"] >= 0.0), "Energy stored is negative"


def test_heat_loss_positive_when_hot(model):
    """Heat loss must be positive whenever T > T_amb."""
    r = model.predict({"temperature": 60.0, "t_ambient": 20.0})
    assert float(r["heat_loss_w"]) > 0.0, "Heat loss should be positive when T > T_amb"


def test_heat_loss_zero_at_ambient(model):
    """Heat loss must be zero when T == T_amb."""
    r = model.predict({"temperature": 20.0, "t_ambient": 20.0})
    assert abs(float(r["heat_loss_w"])) < 1e-9


def test_heat_loss_negative_below_ambient(model):
    """Heat flows into the tank (negative loss) when T < T_amb."""
    r = model.predict({"temperature": 10.0, "t_ambient": 20.0})
    assert float(r["heat_loss_w"]) < 0.0


def test_dTdt_negative_without_charge(model):
    """Tank cools down when there is no charging input and T > T_amb."""
    r = model.predict({
        "temperature": 60.0,
        "q_charge": 0.0,
        "q_discharge": 0.0,
        "t_ambient": 20.0,
    })
    assert float(r["dT_dt"]) < 0.0, "dT/dt must be negative when only losing heat"


def test_dTdt_increases_with_charge(model):
    """Adding charge power must increase dT/dt."""
    r1 = model.predict({"temperature": 60.0, "q_charge":    0.0, "q_discharge": 0.0})
    r2 = model.predict({"temperature": 60.0, "q_charge": 5000.0, "q_discharge": 0.0})
    assert float(r2["dT_dt"]) > float(r1["dT_dt"])


def test_soc_bounds(model):
    """SOC must be 0 at T_min and 1 at T_max."""
    r_min = model.predict({"temperature": 30.0})   # T_min
    r_max = model.predict({"temperature": 90.0})   # T_max
    np.testing.assert_allclose(float(r_min["soc"]), 0.0, atol=1e-10)
    np.testing.assert_allclose(float(r_max["soc"]), 1.0, atol=1e-10)


def test_soc_monotonic(model):
    """SOC must increase monotonically with temperature."""
    T = np.linspace(30.0, 90.0, 50)
    r = model.predict({"temperature": T})
    assert np.all(np.diff(r["soc"]) > 0)


def test_energy_scales_with_temperature(model):
    """Energy stored at 90C must be greater than at 30C."""
    r_low  = model.predict({"temperature": 30.0})
    r_high = model.predict({"temperature": 90.0})
    assert float(r_high["energy_stored_kwh"]) > float(r_low["energy_stored_kwh"])


def test_benchmark(model):
    T     = np.random.uniform(30.0, 90.0, 1000)
    Q_in  = np.random.uniform(0, 10000, 1000)
    Q_out = np.random.uniform(0, 10000, 1000)
    T_a   = np.random.uniform(15, 25, 1000)
    start = time.perf_counter()
    model.predict({"temperature": T, "q_charge": Q_in, "q_discharge": Q_out, "t_ambient": T_a})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed*1000:.2f} ms")
    assert elapsed < 1.0
