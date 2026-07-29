"""EC154 — Enhanced Geothermal System (EGS) — F1a Exergy Model — Test Suite"""
import sys, time, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()


def test_predict_keys(model):
    r = model.predict({"T_geothermal": 200.0, "T_rejection": 25.0, "flow_rate_kgs": 50.0})
    for k in ["power_net_kw", "power_gross_kw", "parasitic_kw",
              "efficiency_net", "efficiency_gross", "heat_input_kw", "T_reinjection_c"]:
        assert k in r


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC154"
    assert "fidelity" in info


def test_net_power_less_than_gross(model):
    """Net power must be less than gross power due to parasitic pump load."""
    T_geos = np.linspace(150, 350, 20)
    for T_geo in T_geos:
        r = model.predict({"T_geothermal": float(T_geo), "T_rejection": 25.0, "flow_rate_kgs": 50.0})
        P_net   = float(r["power_net_kw"])
        P_gross = float(r["power_gross_kw"])
        assert P_net < P_gross + 1e-9, (
            f"T_geo={T_geo}: P_net={P_net:.1f} >= P_gross={P_gross:.1f}"
        )


def test_parasitic_equals_gross_minus_net(model):
    """Parasitic = P_gross - P_net (energy balance check)."""
    r = model.predict({"T_geothermal": 200.0, "T_rejection": 25.0, "flow_rate_kgs": 50.0})
    P_par_expected = float(r["power_gross_kw"]) - float(r["power_net_kw"])
    P_par_actual   = float(r["parasitic_kw"])
    assert abs(P_par_actual - P_par_expected) < 1e-6, (
        f"Energy balance failed: parasitic={P_par_actual:.3f}, expected={P_par_expected:.3f}"
    )


def test_gross_efficiency_less_than_carnot(model):
    """Gross efficiency must be less than Carnot efficiency."""
    T_geos = np.linspace(150, 350, 20)
    for T_geo in T_geos:
        r = model.predict({"T_geothermal": float(T_geo), "T_rejection": 25.0, "flow_rate_kgs": 50.0})
        T_geo_K   = T_geo + 273.15
        T_reinj_K = float(r["T_reinjection_c"]) + 273.15
        eta_carnot  = 1.0 - T_reinj_K / T_geo_K
        eta_gross   = float(r["efficiency_gross"])
        assert eta_gross < eta_carnot + 1e-9, (
            f"T_geo={T_geo}: eta_gross={eta_gross:.4f} >= eta_Carnot={eta_carnot:.4f}"
        )


def test_net_efficiency_less_than_gross(model):
    """Net efficiency must be less than gross efficiency."""
    r = model.predict({"T_geothermal": 200.0, "T_rejection": 25.0, "flow_rate_kgs": 50.0})
    assert float(r["efficiency_net"]) < float(r["efficiency_gross"])


def test_power_proportional_to_flow(model):
    """Net power must scale linearly with flow rate."""
    flows = np.array([5.0, 25.0, 50.0, 100.0, 200.0])
    powers = np.array([float(model.predict({
        "T_geothermal": 200.0, "T_rejection": 25.0, "flow_rate_kgs": float(f)
    })["power_net_kw"]) for f in flows])
    ratios = powers / flows
    assert np.std(ratios) / np.mean(ratios) < 0.01, "Net power not proportional to flow"


def test_power_increases_with_T_geo(model):
    """Higher T_rock → more available exergy → more net power."""
    T_geos = np.array([150.0, 200.0, 250.0, 300.0, 350.0])
    powers = np.array([float(model.predict({
        "T_geothermal": float(T), "T_rejection": 25.0, "flow_rate_kgs": 50.0
    })["power_net_kw"]) for T in T_geos])
    assert np.all(np.diff(powers) > 0), f"Net power not increasing with T_geo: {powers}"


def test_power_decreases_with_T_rejection(model):
    """Higher rejection T → lower efficiency → less power."""
    T_rejs = np.array([10.0, 20.0, 30.0, 40.0])
    powers = np.array([float(model.predict({
        "T_geothermal": 250.0, "T_rejection": float(T), "flow_rate_kgs": 50.0
    })["power_net_kw"]) for T in T_rejs])
    assert np.all(np.diff(powers) < 0), f"Power not decreasing with T_rejection: {powers}"


def test_power_nonnegative(model):
    """Net power must be non-negative."""
    T_geos = np.linspace(150, 350, 30)
    for Tg in T_geos:
        r = model.predict({"T_geothermal": float(Tg), "T_rejection": 25.0, "flow_rate_kgs": 50.0})
        assert float(r["power_net_kw"]) >= 0.0, f"Negative net power at T_geo={Tg}"


def test_reinjection_temperature(model):
    """T_reinjection must equal T_rejection + 30 degC (offset from parameters)."""
    r = model.predict({"T_geothermal": 200.0, "T_rejection": 25.0, "flow_rate_kgs": 50.0})
    T_reinj = float(r["T_reinjection_c"])
    assert abs(T_reinj - 55.0) < 0.01, f"T_reinject={T_reinj:.2f}°C, expected 55.0°C"


def test_array_input(model):
    """Model must accept array inputs."""
    T_geos = np.linspace(150, 350, 15)
    r = model.predict({"T_geothermal": T_geos, "T_rejection": 25.0, "flow_rate_kgs": 50.0})
    assert len(r["power_net_kw"]) == 15


def test_benchmark(model):
    T_geos = np.random.uniform(150, 350, 1000)
    T_rejs = np.random.uniform(10, 40, 1000)
    flows  = np.random.uniform(5, 200, 1000)
    start = time.perf_counter()
    model.predict({"T_geothermal": T_geos, "T_rejection": T_rejs, "flow_rate_kgs": flows})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed*1000:.2f} ms")
    assert elapsed < 1.0
