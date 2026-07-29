"""EC153 — Binary Cycle Geothermal Plant — F1a Exergy Model — Test Suite"""
import sys, time, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()


def test_predict_keys(model):
    r = model.predict({"T_geothermal": 150.0, "T_rejection": 25.0, "flow_rate_kgs": 50.0})
    for k in ["power_kw", "efficiency", "heat_input_kw", "T_reinjection_c"]:
        assert k in r


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC153"
    assert "fidelity" in info


def test_efficiency_less_than_carnot(model):
    """Plant efficiency must be less than Carnot efficiency."""
    T_geos = np.linspace(80, 200, 20)
    for T_geo in T_geos:
        r = model.predict({"T_geothermal": float(T_geo), "T_rejection": 25.0, "flow_rate_kgs": 50.0})
        T_geo_K = T_geo + 273.15
        T_rej_K = 25.0 + 273.15
        eta_carnot = 1.0 - T_rej_K / T_geo_K
        eta_plant = float(r["efficiency"])
        assert eta_plant < eta_carnot + 1e-9, f"T_geo={T_geo}: eta_plant={eta_plant:.4f} >= eta_Carnot={eta_carnot:.4f}"


def test_efficiency_less_than_015(model):
    """Typical binary geothermal efficiency < 15% for T_geo < 200°C."""
    r = model.predict({"T_geothermal": 150.0, "T_rejection": 25.0, "flow_rate_kgs": 50.0})
    eta = float(r["efficiency"])
    assert eta < 0.15, f"Efficiency = {eta:.4f}, expected < 0.15"


def test_power_proportional_to_flow(model):
    """Power should scale linearly with flow rate."""
    flows = np.array([10.0, 20.0, 40.0, 80.0])
    powers = np.array([float(model.predict({
        "T_geothermal": 150.0, "T_rejection": 25.0, "flow_rate_kgs": float(f)
    })["power_kw"]) for f in flows])
    # Linear scaling: power / flow should be approximately constant
    ratios = powers / flows
    assert np.std(ratios) / np.mean(ratios) < 0.01, "Power not proportional to flow rate"


def test_power_increases_with_T_geo(model):
    """Higher T_geo → more available exergy → more power."""
    T_geos = np.array([100.0, 120.0, 140.0, 160.0, 180.0])
    powers = np.array([float(model.predict({
        "T_geothermal": float(T), "T_rejection": 25.0, "flow_rate_kgs": 50.0
    })["power_kw"]) for T in T_geos])
    assert np.all(np.diff(powers) > 0), f"Power not increasing with T_geo: {powers}"


def test_reinjection_temperature(model):
    """T_reinjection should be T_rejection + 10 degC."""
    r = model.predict({"T_geothermal": 150.0, "T_rejection": 25.0, "flow_rate_kgs": 50.0})
    T_reinj = float(r["T_reinjection_c"])
    assert abs(T_reinj - 35.0) < 0.01, f"T_reinject = {T_reinj:.2f}°C, expected 35.0°C"


def test_design_point_power(model):
    """At design point (T_geo=150°C, T_rej=25°C, 50 kg/s), power should be ~1–10 MW range."""
    r = model.predict({"T_geothermal": 150.0, "T_rejection": 25.0, "flow_rate_kgs": 50.0})
    P_kw = float(r["power_kw"])
    assert 500 < P_kw < 15000, f"Design power = {P_kw:.1f} kW"


def test_power_decreases_with_high_rejection_temp(model):
    """Higher rejection temperature → lower efficiency → less power."""
    T_rejs = np.array([10.0, 20.0, 30.0, 40.0])
    powers = np.array([float(model.predict({
        "T_geothermal": 150.0, "T_rejection": float(T), "flow_rate_kgs": 50.0
    })["power_kw"]) for T in T_rejs])
    assert np.all(np.diff(powers) < 0), f"Power not decreasing with T_reject: {powers}"


def test_array_input(model):
    """Model should handle array inputs."""
    T_geos = np.linspace(100, 200, 15)
    r = model.predict({"T_geothermal": T_geos, "T_rejection": 25.0, "flow_rate_kgs": 50.0})
    assert len(r["power_kw"]) == 15


def test_benchmark(model):
    T_geos = np.random.uniform(80, 200, 1000)
    T_rejs = np.random.uniform(10, 40, 1000)
    flows  = np.random.uniform(10, 100, 1000)
    start = time.perf_counter()
    model.predict({"T_geothermal": T_geos, "T_rejection": T_rejs, "flow_rate_kgs": flows})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed*1000:.2f} ms")
    assert elapsed < 1.0
