"""EC207 — CO2 Compression & Pipeline — F1b Injection Pressure + Leakage — Test Suite"""
import sys, time, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()


def test_predict_keys(model):
    r = model.predict({"P_inlet_bar": 1.5, "P_outlet_bar": 150.0, "m_dot_kg_s": 100.0})
    for k in ["sec_kwh_per_tco2", "shaft_power_kw", "pipeline_dp_bar",
              "pipeline_outlet_P_bar", "polytropic_efficiency",
              "seal_leakage_fraction", "pipeline_leakage_fraction",
              "net_co2_delivered_kg_s"]:
        assert k in r


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC207"
    assert info["fidelity"] == "F1b"


def test_sec_in_range_at_design(model):
    """SEC at design: 50-200 kWh/tCO2."""
    r = model.predict({"P_inlet_bar": 1.5, "P_outlet_bar": 150.0, "operating_hours": 0})
    sec = float(np.atleast_1d(r["sec_kwh_per_tco2"])[0])
    assert 50.0 < sec < 200.0, f"SEC = {sec:.1f} kWh/tCO2"


def test_sec_increases_with_outlet_pressure(model):
    P_outs = [100.0, 120.0, 150.0, 180.0]
    secs = [float(np.atleast_1d(model.predict({"P_outlet_bar": p, "operating_hours": 0})["sec_kwh_per_tco2"])[0])
            for p in P_outs]
    assert all(secs[i] <= secs[i + 1] for i in range(len(secs) - 1))


def test_sec_increases_with_degradation(model):
    """Degraded compressor requires more energy per tonne CO2."""
    sec_fresh = float(np.atleast_1d(model.predict({"operating_hours": 0})["sec_kwh_per_tco2"])[0])
    sec_aged = float(np.atleast_1d(model.predict({"operating_hours": 50000})["sec_kwh_per_tco2"])[0])
    assert sec_aged >= sec_fresh - 1e-6


def test_efficiency_degrades(model):
    """Polytropic efficiency decreases over time."""
    eta_fresh = float(np.atleast_1d(model.predict({"operating_hours": 0})["polytropic_efficiency"])[0])
    eta_aged = float(np.atleast_1d(model.predict({"operating_hours": 50000})["polytropic_efficiency"])[0])
    assert eta_aged < eta_fresh


def test_efficiency_floor(model):
    """Efficiency must not drop below 0.60."""
    r = model.predict({"operating_hours": 1e7})
    eta = float(np.atleast_1d(r["polytropic_efficiency"])[0])
    assert eta >= 0.60


def test_no_leakage_at_start(model):
    """No seal or pipeline leakage at t=0."""
    r = model.predict({"operating_hours": 0})
    assert float(np.atleast_1d(r["seal_leakage_fraction"])[0]) == pytest.approx(0.0, abs=1e-10)
    assert float(np.atleast_1d(r["pipeline_leakage_fraction"])[0]) == pytest.approx(0.0, abs=1e-10)


def test_leakage_grows_with_time(model):
    """Leakage increases with operating hours."""
    f1 = float(np.atleast_1d(model.predict({"operating_hours": 8760})["pipeline_leakage_fraction"])[0])
    f2 = float(np.atleast_1d(model.predict({"operating_hours": 43800})["pipeline_leakage_fraction"])[0])
    assert f2 > f1


def test_net_delivered_less_than_input(model):
    """Net delivered CO2 must be <= input flow (after leakage)."""
    r = model.predict({"m_dot_kg_s": 100.0, "operating_hours": 10000})
    m_net = float(np.atleast_1d(r["net_co2_delivered_kg_s"])[0])
    assert m_net <= 100.0
    assert m_net > 0.0


def test_net_delivered_equals_input_at_start(model):
    """At t=0, no leakage → net delivery = input."""
    r = model.predict({"m_dot_kg_s": 100.0, "operating_hours": 0})
    m_net = float(np.atleast_1d(r["net_co2_delivered_kg_s"])[0])
    assert abs(m_net - 100.0) < 1e-6


def test_dp_zero_for_zero_length(model):
    r = model.predict({"pipeline_length_km": 0.0, "m_dot_kg_s": 100.0, "operating_hours": 0})
    assert float(np.atleast_1d(r["pipeline_dp_bar"])[0]) == pytest.approx(0.0, abs=1e-6)


def test_dp_positive_for_nonzero_length(model):
    r = model.predict({"pipeline_length_km": 100.0, "m_dot_kg_s": 100.0, "operating_hours": 0})
    assert float(np.atleast_1d(r["pipeline_dp_bar"])[0]) > 0


def test_array_input(model):
    P_outs = np.linspace(100.0, 200.0, 10)
    r = model.predict({"P_outlet_bar": P_outs, "operating_hours": 0})
    assert len(np.atleast_1d(r["sec_kwh_per_tco2"])) == 10


def test_benchmark(model):
    rng = np.random.default_rng(42)
    P_in = rng.uniform(1.0, 5.0, 1000)
    P_out = rng.uniform(100.0, 200.0, 1000)
    start = time.perf_counter()
    model.predict({"P_inlet_bar": P_in, "P_outlet_bar": P_out, "operating_hours": 5000})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed * 1000:.2f} ms")
    assert elapsed < 1.0
