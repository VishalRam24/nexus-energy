"""EC201 — DAC Solid Sorbent — F1a — Test Suite"""
import sys, time, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()


def test_predict_keys(model):
    r = model.predict({"air_flow_m3h": 1e6, "relative_humidity": 0.5})
    for k in ["co2_captured_tpa", "thermal_energy_mwh_pa", "electrical_energy_mwh_pa",
              "specific_thermal_kwht", "specific_electric_kwhe"]:
        assert k in r, f"Missing output key: {k}"


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC201"
    assert info["fidelity"] == "F1a"


def test_thermal_exceeds_electric(model):
    """E_thermal must be greater than E_electric for solid sorbent DAC."""
    r = model.predict({"air_flow_m3h": 1e6, "relative_humidity": 0.5})
    eth = float(r["specific_thermal_kwht"])
    eel = float(r["specific_electric_kwhe"])
    assert eth > eel, f"E_thermal ({eth:.0f}) must exceed E_electric ({eel:.0f})"


def test_capture_increases_with_flow(model):
    """Higher air flow -> more CO2 captured."""
    flows = np.array([1e5, 5e5, 1e6, 5e6, 1e7])
    r = model.predict({"air_flow_m3h": flows, "relative_humidity": 0.5})
    co2 = r["co2_captured_tpa"]
    assert np.all(np.diff(co2) > 0), "CO2 captured must increase with air flow"


def test_humidity_affects_thermal_energy(model):
    """Higher humidity lowers specific thermal energy (better sorbent performance)."""
    rh_vals = np.array([0.2, 0.4, 0.6, 0.8])
    r = model.predict({"air_flow_m3h": 1e6, "relative_humidity": rh_vals})
    eth = r["specific_thermal_kwht"]
    assert np.all(np.diff(eth) < 0), "Specific thermal energy should decrease with increasing RH"


def test_specific_electric_is_constant(model):
    """Electrical energy per tCO2 should be constant across conditions."""
    flows = np.array([1e5, 1e6, 1e7])
    r = model.predict({"air_flow_m3h": flows, "relative_humidity": 0.5})
    eel = r["specific_electric_kwhe"]
    assert np.allclose(eel, eel[0]), "Specific electric should be constant"


def test_energy_range(model):
    """Specific thermal should be in realistic range for solid sorbent DAC (1000-3000 kWh/tCO2)."""
    r = model.predict({"air_flow_m3h": 1e6, "relative_humidity": 0.5})
    eth = float(r["specific_thermal_kwht"])
    assert 1000 < eth < 3000, f"Specific thermal {eth:.0f} outside realistic range"


def test_annual_thermal_exceeds_electric_total(model):
    """Annual thermal MWh must exceed annual electrical MWh."""
    r = model.predict({"air_flow_m3h": 1e6, "relative_humidity": 0.5})
    assert float(r["thermal_energy_mwh_pa"]) > float(r["electrical_energy_mwh_pa"])


def test_benchmark(model):
    flows = np.random.uniform(1e5, 1e7, 1000)
    rh = np.random.uniform(0.1, 0.9, 1000)
    start = time.perf_counter()
    model.predict({"air_flow_m3h": flows, "relative_humidity": rh})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed*1000:.2f} ms")
    assert elapsed < 1.0
