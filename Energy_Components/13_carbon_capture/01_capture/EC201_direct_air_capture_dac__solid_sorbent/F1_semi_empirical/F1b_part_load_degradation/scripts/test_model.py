"""EC201 — DAC Solid Sorbent — F1b Part-Load Degradation — Test Suite"""
import sys, time, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()


def test_predict_keys(model):
    r = model.predict({"PLR": 1.0, "n_cycles": 0})
    for k in ["co2_captured_kg_h", "thermal_energy_kwh_ton",
              "electrical_energy_kwh_ton", "sorbent_capacity_pct"]:
        assert k in r


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC201"
    assert info["fidelity"] == "F1b"


def test_capacity_100pct_at_start(model):
    """Fresh sorbent should have 100% capacity."""
    r = model.predict({"n_cycles": 0})
    cap = float(np.atleast_1d(r["sorbent_capacity_pct"])[0])
    assert abs(cap - 100.0) < 0.1


def test_capacity_degrades_with_cycles(model):
    """Capacity should decrease with cycles."""
    cycles = [0, 1000, 5000, 10000]
    caps = []
    for n in cycles:
        r = model.predict({"n_cycles": n})
        caps.append(float(np.atleast_1d(r["sorbent_capacity_pct"])[0]))
    assert all(caps[i] >= caps[i + 1] for i in range(len(caps) - 1)), \
        f"Capacity not decreasing: {caps}"


def test_degradation_rate(model):
    """At 10000 cycles: capacity = 100*(1 - 5e-5*10000) = 50%."""
    r = model.predict({"n_cycles": 10000})
    cap = float(np.atleast_1d(r["sorbent_capacity_pct"])[0])
    assert abs(cap - 50.0) < 1.0, f"Capacity at 10k cycles = {cap:.1f}%, expected 50%"


def test_co2_captured_positive(model):
    """CO2 captured must be positive."""
    r = model.predict({"air_flow_m3_s": 10.0, "PLR": 0.5, "n_cycles": 0})
    co2 = float(np.atleast_1d(r["co2_captured_kg_h"])[0])
    assert co2 > 0


def test_co2_decreases_with_degradation(model):
    """CO2 captured should decrease as sorbent degrades."""
    co2_vals = []
    for n in [0, 5000, 10000]:
        r = model.predict({"air_flow_m3_s": 10.0, "n_cycles": n})
        co2_vals.append(float(np.atleast_1d(r["co2_captured_kg_h"])[0]))
    assert all(co2_vals[i] >= co2_vals[i + 1] - 1e-6 for i in range(len(co2_vals) - 1)), \
        f"CO2 not decreasing with degradation: {co2_vals}"


def test_thermal_energy_increases_cold_ambient(model):
    """Colder ambient -> larger temperature swing -> more thermal energy."""
    r_warm = model.predict({"T_ambient_degC": 30.0, "n_cycles": 0})
    r_cold = model.predict({"T_ambient_degC": 0.0, "n_cycles": 0})
    E_warm = float(np.atleast_1d(r_warm["thermal_energy_kwh_ton"])[0])
    E_cold = float(np.atleast_1d(r_cold["thermal_energy_kwh_ton"])[0])
    assert E_cold > E_warm, f"E_th cold={E_cold:.0f} not > warm={E_warm:.0f}"


def test_thermal_energy_increases_with_degradation(model):
    """Degraded sorbent needs more energy per ton CO2."""
    r_fresh = model.predict({"n_cycles": 0})
    r_old = model.predict({"n_cycles": 10000})
    E_fresh = float(np.atleast_1d(r_fresh["thermal_energy_kwh_ton"])[0])
    E_old = float(np.atleast_1d(r_old["thermal_energy_kwh_ton"])[0])
    assert E_old > E_fresh


def test_electrical_increases_at_part_load(model):
    """Specific electrical energy should increase at part-load."""
    r_full = model.predict({"PLR": 1.0, "n_cycles": 0})
    r_part = model.predict({"PLR": 0.3, "n_cycles": 0})
    E_full = float(np.atleast_1d(r_full["electrical_energy_kwh_ton"])[0])
    E_part = float(np.atleast_1d(r_part["electrical_energy_kwh_ton"])[0])
    assert E_part > E_full


def test_humidity_affects_capture(model):
    """Higher humidity should improve capture (for amine sorbents)."""
    r_dry = model.predict({"relative_humidity": 0.2, "air_flow_m3_s": 10.0, "n_cycles": 0})
    r_wet = model.predict({"relative_humidity": 0.7, "air_flow_m3_s": 10.0, "n_cycles": 0})
    co2_dry = float(np.atleast_1d(r_dry["co2_captured_kg_h"])[0])
    co2_wet = float(np.atleast_1d(r_wet["co2_captured_kg_h"])[0])
    assert co2_wet >= co2_dry - 1e-6


def test_array_input(model):
    """Model should handle array PLR inputs."""
    PLRs = np.linspace(0.3, 1.0, 10)
    r = model.predict({"PLR": PLRs, "n_cycles": 0})
    assert len(np.atleast_1d(r["co2_captured_kg_h"])) == 10


def test_benchmark(model):
    PLRs = np.random.uniform(0.3, 1.0, 1000)
    start = time.perf_counter()
    model.predict({"PLR": PLRs, "n_cycles": 5000})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed * 1000:.2f} ms")
    assert elapsed < 1.0
