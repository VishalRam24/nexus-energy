"""EC036 -- VRFB -- F2a Stack Model -- Test Suite"""
import sys, time, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()


def test_predict_keys(model):
    r = model.predict({
        "current_A": 50.0, "flow_rate_L_min": 10.0,
        "dt": 10.0, "duration_s": 60.0, "soc_init": 0.5,
    })
    for k in ["t", "voltage", "soc", "power_stack", "power_pump", "net_power", "efficiency"]:
        assert k in r, f"Missing key: {k}"


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC036"
    assert info["fidelity"] == "F2a"


def test_soc_decreases_on_discharge(model):
    """SOC must decrease during discharge (positive current)."""
    r = model.predict({
        "current_A": 50.0, "flow_rate_L_min": 10.0,
        "dt": 10.0, "duration_s": 600.0, "soc_init": 0.8,
    })
    assert r["soc"][-1] < r["soc"][0], "SOC did not decrease during discharge"


def test_soc_increases_on_charge(model):
    """SOC must increase during charge (negative current)."""
    r = model.predict({
        "current_A": -50.0, "flow_rate_L_min": 10.0,
        "dt": 10.0, "duration_s": 600.0, "soc_init": 0.3,
    })
    assert r["soc"][-1] > r["soc"][0], "SOC did not increase during charge"


def test_voltage_positive_during_discharge(model):
    """Stack voltage should be positive during moderate discharge."""
    r = model.predict({
        "current_A": 50.0, "flow_rate_L_min": 10.0,
        "dt": 60.0, "duration_s": 600.0, "soc_init": 0.5,
    })
    assert np.all(r["voltage"] > 0), "Negative stack voltage during discharge"


def test_pump_power_positive(model):
    """Pump power must always be >= 0."""
    r = model.predict({
        "current_A": 50.0, "flow_rate_L_min": 10.0,
        "dt": 60.0, "duration_s": 600.0, "soc_init": 0.5,
    })
    assert np.all(r["power_pump"] >= 0), "Negative pump power"


def test_pump_power_increases_with_flow(model):
    """Higher flow rate -> higher pump power."""
    m = model._model
    P_low = float(m.pump_power_w(5.0))
    P_high = float(m.pump_power_w(20.0))
    assert P_high > P_low


def test_higher_flow_reduces_conc_overpotential(model):
    """Higher flow rate should give higher voltage (less conc overpotential)."""
    m = model._model
    V_low = float(m.cell_voltage(0.5, 60.0, 3.0))
    V_high = float(m.cell_voltage(0.5, 60.0, 20.0))
    assert V_high > V_low, "Higher flow did not increase cell voltage"


def test_efficiency_between_0_and_1(model):
    r = model.predict({
        "current_A": 50.0, "flow_rate_L_min": 10.0,
        "dt": 60.0, "duration_s": 600.0, "soc_init": 0.5,
    })
    assert np.all(r["efficiency"] >= 0.0)
    assert np.all(r["efficiency"] <= 1.0)


def test_nernst_at_50pct_soc(model):
    """At SOC=0.5, E_nernst should equal E0."""
    E = float(model._model.e_nernst(0.5))
    np.testing.assert_allclose(E, model._model.E0, atol=1e-6)


def test_all_finite(model):
    r = model.predict({
        "current_A": 50.0, "flow_rate_L_min": 10.0,
        "dt": 10.0, "duration_s": 300.0, "soc_init": 0.5,
    })
    for k, v in r.items():
        assert np.all(np.isfinite(v)), f"Non-finite values in {k}"


def test_benchmark(model):
    start = time.perf_counter()
    model.predict({
        "current_A": 50.0, "flow_rate_L_min": 10.0,
        "dt": 1.0, "duration_s": 3600.0, "soc_init": 0.5,
    })
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 3600-step simulation in {elapsed*1000:.1f} ms")
    assert elapsed < 30.0
