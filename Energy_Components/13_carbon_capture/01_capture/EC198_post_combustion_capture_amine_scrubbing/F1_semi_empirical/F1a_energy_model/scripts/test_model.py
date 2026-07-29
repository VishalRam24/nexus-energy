"""EC198 — Post-Combustion Capture (Amine Scrubbing) — F1a Energy Model — Test Suite"""
import sys, time, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()


def test_predict_keys(model):
    r = model.predict({"flue_gas_rate": 500.0, "co2_fraction": 0.12, "capture_rate": 0.90})
    for k in ["co2_captured_kgs", "reboiler_duty_mw", "electricity_mw", "specific_energy_gjt"]:
        assert k in r


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC198"
    assert "fidelity" in info


def test_specific_energy_in_range(model):
    """Specific energy must be 2.5–5.0 GJ/tCO2 across the operating range (Abu-Zahra 2007)."""
    crs = np.linspace(0.80, 0.95, 10)
    for cr in crs:
        r = model.predict({"flue_gas_rate": 500.0, "co2_fraction": 0.12, "capture_rate": float(cr)})
        E = float(r["specific_energy_gjt"])
        assert 2.5 < E < 5.0, f"Capture rate {cr:.2f}: specific energy = {E:.2f} GJ/t"


def test_co2_captured_less_than_co2_in(model):
    """CO2 captured must not exceed CO2 in the flue gas."""
    for cr in [0.80, 0.90, 0.95]:
        r = model.predict({"flue_gas_rate": 500.0, "co2_fraction": 0.12, "capture_rate": cr})
        co2_cap = float(r["co2_captured_kgs"])
        # CO2 in flue gas (rough: mol fraction * MW_CO2/MW_flue * flue_rate)
        xCO2 = 0.12
        MW_CO2, MW_air = 0.04401, 0.02897
        MW_flue = xCO2 * MW_CO2 + (1.0 - xCO2) * MW_air
        co2_in = 500.0 * xCO2 * MW_CO2 / MW_flue
        assert co2_cap <= co2_in + 1e-6, f"Capture rate {cr}: captured ({co2_cap:.2f}) > CO2_in ({co2_in:.2f})"


def test_reboiler_positive(model):
    """Reboiler duty must be positive."""
    r = model.predict({"flue_gas_rate": 500.0, "co2_fraction": 0.12})
    assert float(r["reboiler_duty_mw"]) > 0


def test_electricity_positive(model):
    """Electricity demand must be positive."""
    r = model.predict({"flue_gas_rate": 500.0, "co2_fraction": 0.12})
    assert float(r["electricity_mw"]) > 0


def test_energy_increases_at_high_capture(model):
    """Specific energy should increase at high capture rates (harder to capture last CO2)."""
    r_90 = model.predict({"flue_gas_rate": 500.0, "co2_fraction": 0.12, "capture_rate": 0.90})
    r_95 = model.predict({"flue_gas_rate": 500.0, "co2_fraction": 0.12, "capture_rate": 0.95})
    E_90 = float(r_90["specific_energy_gjt"])
    E_95 = float(r_95["specific_energy_gjt"])
    assert E_95 > E_90, f"Energy at 95% ({E_95:.2f}) not > energy at 90% ({E_90:.2f})"


def test_co2_proportional_to_flow(model):
    """CO2 captured should scale linearly with flue gas rate."""
    flows = np.array([200.0, 400.0, 600.0, 800.0])
    co2s  = np.array([float(model.predict({
        "flue_gas_rate": float(f), "co2_fraction": 0.12, "capture_rate": 0.90
    })["co2_captured_kgs"]) for f in flows])
    ratios = co2s / flows
    assert np.std(ratios) / np.mean(ratios) < 0.01, "CO2 not proportional to flow"


def test_design_point_specific_energy(model):
    """At design (cr=0.90), specific energy should be ~3.2–4.0 GJ/t (Abu-Zahra 2007)."""
    r = model.predict({"flue_gas_rate": 500.0, "co2_fraction": 0.12, "capture_rate": 0.90})
    E = float(r["specific_energy_gjt"])
    assert 3.0 < E < 4.0, f"Design specific energy = {E:.2f} GJ/tCO2"


def test_array_input(model):
    """Model should handle array inputs."""
    crs = np.linspace(0.80, 0.95, 12)
    r = model.predict({"flue_gas_rate": 500.0, "co2_fraction": 0.12, "capture_rate": crs})
    assert len(r["specific_energy_gjt"]) == 12


def test_benchmark(model):
    flows = np.random.uniform(100, 1000, 1000)
    xCO2s = np.random.uniform(0.04, 0.15, 1000)
    crs   = np.random.uniform(0.80, 0.95, 1000)
    start = time.perf_counter()
    model.predict({"flue_gas_rate": flows, "co2_fraction": xCO2s, "capture_rate": crs})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed*1000:.2f} ms")
    assert elapsed < 1.0
