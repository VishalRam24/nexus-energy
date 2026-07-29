"""EC078 — Hot Water Tank TES — F1b Stratified — Test Suite"""
import sys, time, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()


def test_predict_keys(model):
    r = model.predict({
        "T_inlet_hot": 80.0, "T_inlet_cold": 15.0,
        "flow_rate_charge": 0.1, "flow_rate_discharge": 0.0,
        "T_ambient": 20.0, "duration_s": 600.0,
    })
    for k in ["T_nodes", "T_outlet_hot", "T_outlet_cold", "stored_energy_kwh", "stratification_efficiency"]:
        assert k in r


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC078"
    assert info["fidelity"] == "F1b"


def test_node_count(model):
    """Should return 10 node temperatures."""
    r = model.predict({"duration_s": 60.0})
    assert len(r["T_nodes"]) == 10


def test_charging_heats_top_first(model):
    """After charging, top nodes should be hotter than bottom nodes."""
    r = model.predict({
        "T_inlet_hot": 80.0, "T_inlet_cold": 15.0,
        "flow_rate_charge": 0.2, "flow_rate_discharge": 0.0,
        "T_ambient": 20.0, "duration_s": 1800.0,
    })
    T = r["T_nodes"]
    # Top (index 0) should be >= bottom (index 9) due to buoyancy
    assert T[0] >= T[-1], f"Top={T[0]:.1f} should be >= Bottom={T[-1]:.1f}"


def test_stratification_develops(model):
    """Temperature difference between top and bottom should increase with charging."""
    r = model.predict({
        "T_inlet_hot": 80.0, "T_inlet_cold": 15.0,
        "flow_rate_charge": 0.15, "flow_rate_discharge": 0.0,
        "T_ambient": 20.0, "duration_s": 1800.0,
    })
    T = r["T_nodes"]
    dT = T[0] - T[-1]
    assert dT > 5.0, f"Stratification dT={dT:.1f}C should be >5C after 30min charge"


def test_energy_increases_with_charging(model):
    """Stored energy should increase during charging."""
    r_start = model.predict({"duration_s": 1.0})
    r_charged = model.predict({
        "T_inlet_hot": 80.0, "flow_rate_charge": 0.2,
        "T_ambient": 20.0, "duration_s": 3600.0,
    })
    assert r_charged["stored_energy_kwh"] > r_start["stored_energy_kwh"]


def test_discharge_cools_tank(model):
    """Discharging from a hot tank should reduce stored energy."""
    # First charge
    r_charge = model.predict({
        "T_inlet_hot": 80.0, "flow_rate_charge": 0.2,
        "T_ambient": 20.0, "duration_s": 3600.0,
    })
    # Then discharge
    r_discharge = model.predict({
        "T_inlet_cold": 15.0, "flow_rate_discharge": 0.2,
        "T_ambient": 20.0, "duration_s": 1800.0,
        "T_initial": r_charge["T_nodes"],
    })
    assert r_discharge["stored_energy_kwh"] < r_charge["stored_energy_kwh"]


def test_outlet_hot_temperature_reasonable(model):
    """Hot outlet temperature should be near top node temperature."""
    r = model.predict({
        "T_inlet_hot": 80.0, "flow_rate_charge": 0.1,
        "T_ambient": 20.0, "duration_s": 3600.0,
    })
    assert r["T_outlet_hot"] == r["T_nodes"][0]


def test_standby_heat_loss(model):
    """Tank should cool down during standby (no flow)."""
    T_hot = [70.0] * 10
    r = model.predict({
        "T_ambient": 20.0, "duration_s": 7200.0,
        "T_initial": T_hot,
    })
    assert r["T_nodes"][0] < 70.0, "Tank should lose heat during standby"


def test_stratification_efficiency_bounded(model):
    """Stratification efficiency must be in [0, 1]."""
    r = model.predict({
        "T_inlet_hot": 80.0, "flow_rate_charge": 0.1,
        "T_ambient": 20.0, "duration_s": 3600.0,
    })
    assert 0.0 <= r["stratification_efficiency"] <= 1.0


def test_buoyancy_correction(model):
    """Nodes should always be monotonically non-increasing (top to bottom)."""
    r = model.predict({
        "T_inlet_hot": 80.0, "flow_rate_charge": 0.3,
        "T_ambient": 20.0, "duration_s": 1800.0,
    })
    T = r["T_nodes"]
    for i in range(len(T) - 1):
        assert T[i] >= T[i + 1] - 0.01, \
            f"Buoyancy violated: node {i}={T[i]:.1f} < node {i+1}={T[i+1]:.1f}"


def test_benchmark(model):
    """Benchmark 1h simulation with 10s timestep."""
    start = time.perf_counter()
    model.predict({
        "T_inlet_hot": 80.0, "flow_rate_charge": 0.1,
        "T_ambient": 20.0, "duration_s": 3600.0, "dt": 10.0,
    })
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1h sim in {elapsed*1000:.1f} ms")
    assert elapsed < 5.0
