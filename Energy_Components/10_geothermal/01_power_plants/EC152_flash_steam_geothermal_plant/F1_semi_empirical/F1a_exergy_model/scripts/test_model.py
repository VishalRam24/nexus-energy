"""EC152 — Flash Steam Geothermal Plant — F1a Exergy Model — Test Suite"""
import sys, time, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()


def test_predict_keys(model):
    r = model.predict({"T_geothermal": 240.0, "T_rejection": 40.0, "flow_rate_kgs": 100.0})
    for k in ["power_kw", "efficiency", "heat_input_kw", "T_flash_c", "steam_quality", "T_condenser_c"]:
        assert k in r


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC152"
    assert "fidelity" in info


def test_efficiency_less_than_carnot(model):
    """Plant efficiency must be strictly less than Carnot efficiency."""
    T_geos = np.linspace(200, 320, 20)
    for T_geo in T_geos:
        r = model.predict({"T_geothermal": float(T_geo), "T_rejection": 40.0, "flow_rate_kgs": 100.0})
        T_geo_K  = T_geo + 273.15
        T_cond_K = 40.0 + 5.0 + 273.15  # T_rejection + condenser_offset
        eta_carnot = 1.0 - T_cond_K / T_geo_K
        eta_plant = float(r["efficiency"])
        assert eta_plant < eta_carnot + 1e-9, (
            f"T_geo={T_geo}: eta_plant={eta_plant:.4f} >= eta_Carnot={eta_carnot:.4f}"
        )


def test_thermal_efficiency_range(model):
    """Flash steam thermal efficiency must be 10–20% at design-range conditions."""
    T_geos = np.linspace(200, 280, 10)
    for T_geo in T_geos:
        r = model.predict({"T_geothermal": float(T_geo), "T_rejection": 40.0, "flow_rate_kgs": 100.0})
        eta = float(r["efficiency"])
        assert 0.08 <= eta <= 0.25, (
            f"T_geo={T_geo}: eta={eta:.4f} outside expected 8-25% for flash steam"
        )


def test_optimal_flash_temperature_between_source_and_condenser(model):
    """Optimal T_flash must be between condenser temperature and T_geo."""
    T_geos = np.linspace(200, 320, 15)
    for T_geo in T_geos:
        r = model.predict({"T_geothermal": float(T_geo), "T_rejection": 40.0, "flow_rate_kgs": 100.0})
        T_fl   = float(r["T_flash_c"])
        T_cond = float(r["T_condenser_c"])
        assert T_cond < T_fl < T_geo, (
            f"T_flash={T_fl:.1f} not between T_cond={T_cond:.1f} and T_geo={T_geo:.1f}"
        )


def test_steam_quality_physical(model):
    """Steam quality from flash must be in [0, 1] and > 0 when T_geo > T_flash."""
    T_geos = np.linspace(200, 320, 15)
    for T_geo in T_geos:
        r = model.predict({"T_geothermal": float(T_geo), "T_rejection": 40.0, "flow_rate_kgs": 100.0})
        x = float(r["steam_quality"])
        assert 0.0 < x <= 1.0, f"T_geo={T_geo}: steam quality={x:.3f} outside (0,1]"


def test_power_proportional_to_flow(model):
    """Power must scale linearly with brine flow rate."""
    flows = np.array([10.0, 50.0, 100.0, 250.0, 500.0])
    powers = np.array([float(model.predict({
        "T_geothermal": 240.0, "T_rejection": 40.0, "flow_rate_kgs": float(f)
    })["power_kw"]) for f in flows])
    ratios = powers / flows
    assert np.std(ratios) / np.mean(ratios) < 0.01, "Power not proportional to flow rate"


def test_power_increases_with_T_geo(model):
    """Higher T_geo → more exergy → more power."""
    T_geos = np.array([200.0, 220.0, 240.0, 270.0, 300.0])
    powers = np.array([float(model.predict({
        "T_geothermal": float(T), "T_rejection": 40.0, "flow_rate_kgs": 100.0
    })["power_kw"]) for T in T_geos])
    assert np.all(np.diff(powers) > 0), f"Power not increasing with T_geo: {powers}"


def test_power_decreases_with_T_rejection(model):
    """Higher rejection T → lower Carnot efficiency → less power."""
    T_rejs = np.array([10.0, 25.0, 40.0, 55.0])
    powers = np.array([float(model.predict({
        "T_geothermal": 250.0, "T_rejection": float(T), "flow_rate_kgs": 100.0
    })["power_kw"]) for T in T_rejs])
    assert np.all(np.diff(powers) < 0), f"Power not decreasing with T_rejection: {powers}"


def test_power_nonnegative(model):
    """Power must never be negative."""
    T_geos = np.linspace(200, 320, 20)
    for Tg in T_geos:
        r = model.predict({"T_geothermal": float(Tg), "T_rejection": 40.0, "flow_rate_kgs": 100.0})
        assert float(r["power_kw"]) >= 0.0, f"Negative power at T_geo={Tg}"


def test_flash_T_lower_than_source(model):
    """Flash temperature must always be lower than geothermal source T."""
    T_geos = np.linspace(200, 320, 15)
    T_rejs = np.linspace(10, 60, 15)
    for Tg, Tr in zip(T_geos, T_rejs):
        r = model.predict({"T_geothermal": float(Tg), "T_rejection": float(Tr), "flow_rate_kgs": 100.0})
        assert float(r["T_flash_c"]) < float(Tg), (
            f"T_flash={r['T_flash_c']:.1f} >= T_geo={Tg}"
        )


def test_array_input(model):
    """Model must accept array inputs."""
    T_geos = np.linspace(200, 320, 15)
    r = model.predict({"T_geothermal": T_geos, "T_rejection": 40.0, "flow_rate_kgs": 100.0})
    assert len(r["power_kw"]) == 15


def test_benchmark(model):
    T_geos = np.random.uniform(200, 320, 1000)
    T_rejs = np.random.uniform(10, 60, 1000)
    flows  = np.random.uniform(10, 500, 1000)
    start = time.perf_counter()
    model.predict({"T_geothermal": T_geos, "T_rejection": T_rejs, "flow_rate_kgs": flows})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed*1000:.2f} ms")
    assert elapsed < 1.0
