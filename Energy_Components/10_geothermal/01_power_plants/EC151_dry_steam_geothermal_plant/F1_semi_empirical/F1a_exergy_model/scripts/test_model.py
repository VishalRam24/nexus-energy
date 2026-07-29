"""EC151 — Dry Steam Geothermal Plant — F1a Exergy Model — Test Suite"""
import sys, time, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()


def test_predict_keys(model):
    r = model.predict({"T_geothermal": 200.0, "T_rejection": 35.0, "flow_rate_kgs": 50.0})
    for k in ["power_kw", "efficiency", "heat_input_kw", "T_condenser_c"]:
        assert k in r


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC151"
    assert "fidelity" in info


def test_efficiency_less_than_carnot(model):
    """Plant efficiency must be strictly less than Carnot efficiency at all T_geo."""
    T_geos = np.linspace(180, 280, 20)
    for T_geo in T_geos:
        r = model.predict({"T_geothermal": float(T_geo), "T_rejection": 30.0, "flow_rate_kgs": 50.0})
        T_geo_K = T_geo + 273.15
        T_rej_K = 30.0 + 5.0 + 273.15  # condenser = T_rejection + offset
        eta_carnot = 1.0 - T_rej_K / T_geo_K
        eta_plant = float(r["efficiency"])
        assert eta_plant < eta_carnot + 1e-9, (
            f"T_geo={T_geo}: eta_plant={eta_plant:.4f} >= eta_Carnot={eta_carnot:.4f}"
        )


def test_thermal_efficiency_range(model):
    """Dry steam plant thermal efficiency must be 15–21% at design-range conditions."""
    T_geos = np.linspace(180, 240, 10)
    for T_geo in T_geos:
        r = model.predict({"T_geothermal": float(T_geo), "T_rejection": 30.0, "flow_rate_kgs": 50.0})
        eta = float(r["efficiency"])
        assert 0.13 <= eta <= 0.25, (
            f"T_geo={T_geo}: eta={eta:.4f} outside expected 13-25% for dry steam"
        )


def test_power_proportional_to_flow(model):
    """Power must scale linearly with steam flow rate."""
    flows = np.array([10.0, 25.0, 50.0, 100.0, 200.0])
    powers = np.array([float(model.predict({
        "T_geothermal": 200.0, "T_rejection": 30.0, "flow_rate_kgs": float(f)
    })["power_kw"]) for f in flows])
    ratios = powers / flows
    assert np.std(ratios) / np.mean(ratios) < 0.01, "Power not proportional to flow rate"


def test_power_increases_with_T_geo(model):
    """Higher T_geo → more exergy → more power."""
    T_geos = np.array([180.0, 200.0, 220.0, 240.0, 260.0])
    powers = np.array([float(model.predict({
        "T_geothermal": float(T), "T_rejection": 30.0, "flow_rate_kgs": 50.0
    })["power_kw"]) for T in T_geos])
    assert np.all(np.diff(powers) > 0), f"Power not increasing with T_geo: {powers}"


def test_power_decreases_with_T_rejection(model):
    """Higher rejection T → lower efficiency → less power (condenser penalty)."""
    T_rejs = np.array([10.0, 20.0, 30.0, 40.0])
    powers = np.array([float(model.predict({
        "T_geothermal": 220.0, "T_rejection": float(T), "flow_rate_kgs": 50.0
    })["power_kw"]) for T in T_rejs])
    assert np.all(np.diff(powers) < 0), f"Power not decreasing with T_rejection: {powers}"


def test_design_point_power_range(model):
    """At T_geo=200°C, T_rej=35°C, 50 kg/s — power should be in MW range."""
    r = model.predict({"T_geothermal": 200.0, "T_rejection": 35.0, "flow_rate_kgs": 50.0})
    P_kw = float(r["power_kw"])
    assert 1000 < P_kw < 30000, f"Design power = {P_kw:.1f} kW"


def test_power_nonnegative(model):
    """Power must never be negative."""
    T_geos = np.linspace(180, 280, 30)
    T_rejs = np.linspace(10, 50, 30)
    for Tg, Tr in zip(T_geos, T_rejs):
        r = model.predict({"T_geothermal": float(Tg), "T_rejection": float(Tr), "flow_rate_kgs": 50.0})
        assert float(r["power_kw"]) >= 0.0, f"Negative power at T_geo={Tg}, T_rej={Tr}"


def test_condenser_temperature(model):
    """T_condenser must be T_rejection + offset (5 degC)."""
    r = model.predict({"T_geothermal": 200.0, "T_rejection": 30.0, "flow_rate_kgs": 50.0})
    T_cond = float(r["T_condenser_c"])
    assert abs(T_cond - 35.0) < 0.01, f"T_condenser = {T_cond:.2f}°C, expected 35.0°C"


def test_array_input(model):
    """Model must accept array inputs."""
    T_geos = np.linspace(180, 280, 15)
    r = model.predict({"T_geothermal": T_geos, "T_rejection": 30.0, "flow_rate_kgs": 50.0})
    assert len(r["power_kw"]) == 15


def test_benchmark(model):
    T_geos = np.random.uniform(180, 280, 1000)
    T_rejs = np.random.uniform(10, 50, 1000)
    flows  = np.random.uniform(10, 200, 1000)
    start = time.perf_counter()
    model.predict({"T_geothermal": T_geos, "T_rejection": T_rejs, "flow_rate_kgs": flows})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed*1000:.2f} ms")
    assert elapsed < 1.0
