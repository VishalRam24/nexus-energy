"""EC207 — CO2 Compression & Pipeline — F1a — Test Suite"""
import sys, time, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()


def test_predict_keys(model):
    r = model.predict({"P_inlet_bar": 1.5, "P_outlet_bar": 150.0, "m_dot_kg_s": 100.0})
    for k in ["sec_kwh_per_tco2", "shaft_power_kw", "stage_pressure_ratio",
               "stage_discharge_T_K", "pipeline_dp_bar", "pipeline_outlet_P_bar",
               "is_supercritical_in", "is_supercritical_out"]:
        assert k in r


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC207"
    assert info["fidelity"] == "F1a"


def test_sec_in_ipcc_range(model):
    """IPCC (2005): full CO2 compression to pipeline pressure ~80-120 kWh/tCO2."""
    r = model.predict({"P_inlet_bar": 1.5, "P_outlet_bar": 150.0, "T_inlet_K": 308.15})
    sec = float(r["sec_kwh_per_tco2"])
    assert 50.0 < sec < 200.0, f"SEC={sec:.1f} kWh/tCO2 outside IPCC credible range"


def test_sec_increases_with_outlet_pressure(model):
    P_out = np.array([80.0, 100.0, 120.0, 150.0, 180.0])
    r = model.predict({"P_inlet_bar": 1.5, "P_outlet_bar": P_out})
    assert np.all(np.diff(r["sec_kwh_per_tco2"]) > 0)


def test_sec_decreases_with_inlet_pressure(model):
    P_in = np.array([1.0, 2.0, 3.0, 5.0])
    r = model.predict({"P_inlet_bar": P_in, "P_outlet_bar": 150.0})
    assert np.all(np.diff(r["sec_kwh_per_tco2"]) < 0)


def test_shaft_power_scales_with_mass_flow(model):
    r1 = model.predict({"P_inlet_bar": 1.5, "P_outlet_bar": 150.0, "m_dot_kg_s": 50.0})
    r2 = model.predict({"P_inlet_bar": 1.5, "P_outlet_bar": 150.0, "m_dot_kg_s": 100.0})
    ratio = float(r2["shaft_power_kw"]) / float(r1["shaft_power_kw"])
    assert ratio == pytest.approx(2.0, rel=1e-6)


def test_outlet_supercritical(model):
    """150 bar, 308 K must be supercritical (above T_crit=304.2 K, P_crit=73.8 bar)."""
    r = model.predict({"P_inlet_bar": 1.5, "P_outlet_bar": 150.0, "T_inlet_K": 308.15})
    assert bool(r["is_supercritical_out"]) is True


def test_inlet_not_supercritical(model):
    """1.5 bar inlet is gaseous, not supercritical."""
    r = model.predict({"P_inlet_bar": 1.5, "P_outlet_bar": 150.0, "T_inlet_K": 308.15})
    assert bool(r["is_supercritical_in"]) is False


def test_stage_ratio_consistent(model):
    """N=4 stages: PR_stage^4 = PR_total."""
    r = model.predict({"P_inlet_bar": 1.5, "P_outlet_bar": 150.0})
    PR_stage = float(r["stage_pressure_ratio"])
    PR_total = 150.0 / 1.5
    assert PR_stage ** 4 == pytest.approx(PR_total, rel=1e-6)


def test_stage_discharge_above_inlet(model):
    r = model.predict({"P_inlet_bar": 1.5, "P_outlet_bar": 150.0, "T_inlet_K": 308.15})
    assert float(r["stage_discharge_T_K"]) > 308.15


def test_pipeline_dp_positive(model):
    r = model.predict({"P_inlet_bar": 1.5, "P_outlet_bar": 150.0,
                        "m_dot_kg_s": 100.0, "pipeline_length_km": 100.0,
                        "pipeline_diameter_m": 0.3})
    assert float(r["pipeline_dp_bar"]) > 0


def test_pipeline_dp_zero_for_zero_length(model):
    r = model.predict({"P_inlet_bar": 1.5, "P_outlet_bar": 150.0,
                        "m_dot_kg_s": 100.0, "pipeline_length_km": 0.0})
    assert float(r["pipeline_dp_bar"]) == pytest.approx(0.0, abs=1e-6)


def test_pipeline_dp_increases_with_length(model):
    L = np.array([50.0, 100.0, 200.0, 300.0])
    r = model.predict({"P_inlet_bar": 1.5, "P_outlet_bar": 150.0,
                        "m_dot_kg_s": 100.0, "pipeline_length_km": L,
                        "pipeline_diameter_m": 0.3})
    assert np.all(np.diff(r["pipeline_dp_bar"]) > 0)


def test_pipeline_dp_decreases_with_diameter(model):
    D = np.array([0.15, 0.2, 0.3, 0.4, 0.5])
    r = model.predict({"P_inlet_bar": 1.5, "P_outlet_bar": 150.0,
                        "m_dot_kg_s": 100.0, "pipeline_length_km": 100.0,
                        "pipeline_diameter_m": D})
    assert np.all(np.diff(r["pipeline_dp_bar"]) < 0)


def test_no_compression_when_pressures_equal(model):
    r = model.predict({"P_inlet_bar": 50.0, "P_outlet_bar": 50.0})
    assert float(r["sec_kwh_per_tco2"]) == pytest.approx(0.0, abs=1e-6)


def test_benchmark(model):
    rng = np.random.default_rng(42)
    P_in = rng.uniform(1.0, 5.0, 1000)
    P_out = rng.uniform(100.0, 200.0, 1000)
    start = time.perf_counter()
    model.predict({"P_inlet_bar": P_in, "P_outlet_bar": P_out, "m_dot_kg_s": 100.0})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed*1000:.2f} ms")
    assert elapsed < 1.0
