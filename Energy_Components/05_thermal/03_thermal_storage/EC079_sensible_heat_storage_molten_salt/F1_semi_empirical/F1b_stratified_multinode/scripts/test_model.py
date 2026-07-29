"""EC079 -- Molten Salt TES -- F1b Stratified -- Test Suite"""
import sys, time, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()


def test_predict_keys(model):
    r = model.predict({
        "T_charge_degC": 565.0, "T_discharge_degC": 290.0,
        "flow_rate_kg_s": 500.0, "mode": "charge", "duration_s": 3600.0,
        "T_nodes_init": [400.0] * 10,
    })
    for k in ["T_nodes", "T_outlet_degC", "stored_energy_kwh",
              "thermal_efficiency", "freeze_warning"]:
        assert k in r, f"Missing key: {k}"


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC079"
    assert info["fidelity"] == "F1b"


def test_t_nodes_length(model):
    """Output must have exactly 10 node temperatures."""
    r = model.predict({
        "mode": "idle", "flow_rate_kg_s": 0.0, "duration_s": 3600.0,
        "T_nodes_init": [400.0] * 10,
    })
    assert len(r["T_nodes"]) == 10


def test_charging_increases_top_node_temperature(model):
    """During charging, top node (node 0) should increase."""
    T_init = [300.0] * 10
    r = model.predict({
        "T_charge_degC": 565.0, "flow_rate_kg_s": 500.0,
        "mode": "charge", "duration_s": 3600.0,
        "T_nodes_init": T_init,
    })
    assert r["T_nodes"][0] > T_init[0], "Top node must heat up during charging"


def test_discharging_cools_bottom_node(model):
    """During discharging, bottom node (node 9) should cool."""
    T_init = [500.0] * 10
    r = model.predict({
        "T_discharge_degC": 290.0, "flow_rate_kg_s": 500.0,
        "mode": "discharge", "duration_s": 3600.0,
        "T_nodes_init": T_init,
    })
    assert r["T_nodes"][9] < T_init[9], "Bottom node must cool during discharging"


def test_stored_energy_nonnegative(model):
    """Stored energy must be non-negative."""
    r = model.predict({
        "mode": "idle", "flow_rate_kg_s": 0.0, "duration_s": 3600.0,
        "T_nodes_init": [290.0] * 10,
    })
    assert r["stored_energy_kwh"] >= 0.0


def test_stored_energy_increases_with_temperature(model):
    """Higher uniform temperature -> more stored energy."""
    r_cold = model.predict({
        "mode": "idle", "flow_rate_kg_s": 0.0, "duration_s": 1.0,
        "T_nodes_init": [300.0] * 10,
    })
    r_hot = model.predict({
        "mode": "idle", "flow_rate_kg_s": 0.0, "duration_s": 1.0,
        "T_nodes_init": [500.0] * 10,
    })
    assert r_hot["stored_energy_kwh"] > r_cold["stored_energy_kwh"]


def test_freeze_warning_near_freeze_point(model):
    """Freeze warning should trigger when any node is near 220 degC."""
    r = model.predict({
        "mode": "idle", "flow_rate_kg_s": 0.0, "duration_s": 1.0,
        "T_nodes_init": [225.0] * 10,
    })
    assert r["freeze_warning"] is True


def test_no_freeze_warning_at_normal_temps(model):
    """No freeze warning at normal operating temperatures."""
    r = model.predict({
        "mode": "idle", "flow_rate_kg_s": 0.0, "duration_s": 1.0,
        "T_nodes_init": [400.0] * 10,
    })
    assert r["freeze_warning"] is False


def test_idle_heat_loss(model):
    """During idle, stored energy should decrease (heat loss to ambient)."""
    T_init = [450.0] * 10
    r = model.predict({
        "mode": "idle", "flow_rate_kg_s": 0.0, "duration_s": 3600.0,
        "T_nodes_init": T_init, "T_ambient_degC": 25.0,
    })
    # Final temps should be slightly lower
    assert np.mean(r["T_nodes"]) < np.mean(T_init)


def test_stratification_develops_during_charge(model):
    """Charging from uniform cold should develop stratification (top hotter than bottom)."""
    T_init = [290.0] * 10
    r = model.predict({
        "T_charge_degC": 565.0, "flow_rate_kg_s": 500.0,
        "mode": "charge", "duration_s": 3600.0,
        "T_nodes_init": T_init,
    })
    # Top nodes should be warmer than bottom nodes
    assert r["T_nodes"][0] > r["T_nodes"][9], \
        "Charging should create stratification: top hotter than bottom"


def test_temperature_dependent_density(model):
    """Salt density must decrease with temperature (rho = 2090 - 0.636*T)."""
    m = model._model
    rho_300 = m.rho(300.0)
    rho_500 = m.rho(500.0)
    assert rho_300 > rho_500, "Density must decrease with temperature"
    # Check specific values
    assert abs(rho_300 - (2090 - 0.636 * 300)) < 0.01
    assert abs(rho_500 - (2090 - 0.636 * 500)) < 0.01


def test_temperature_dependent_cp(model):
    """Salt cp must increase with temperature (cp = 1443 + 0.172*T)."""
    m = model._model
    cp_300 = m.cp(300.0)
    cp_500 = m.cp(500.0)
    assert cp_500 > cp_300, "cp must increase with temperature"
    assert abs(cp_300 - (1443 + 0.172 * 300)) < 0.01


def test_outlet_temp_charge_below_inlet(model):
    """During charge, outlet (bottom) should be cooler than charge inlet (top)."""
    T_init = [350.0] * 10
    r = model.predict({
        "T_charge_degC": 565.0, "flow_rate_kg_s": 500.0,
        "mode": "charge", "duration_s": 3600.0,
        "T_nodes_init": T_init,
    })
    assert r["T_outlet_degC"] < 565.0, "Outlet must be cooler than charge inlet"


def test_benchmark(model):
    """Single prediction must complete in < 2 seconds."""
    start = time.perf_counter()
    for _ in range(10):
        model.predict({
            "T_charge_degC": 565.0, "flow_rate_kg_s": 500.0,
            "mode": "charge", "duration_s": 3600.0,
            "T_nodes_init": [350.0] * 10,
        })
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 10 predictions in {elapsed*1000:.1f} ms")
    assert elapsed < 20.0
