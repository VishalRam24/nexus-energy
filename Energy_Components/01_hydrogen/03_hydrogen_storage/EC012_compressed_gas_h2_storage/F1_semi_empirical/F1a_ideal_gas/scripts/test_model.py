"""EC012 — Compressed Gas H2 Storage — F1a — Test Suite"""
import sys, time, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()


def test_predict_keys(model):
    r = model.predict({"pressure": 350, "temperature": 298.15})
    for k in ["stored_mass_kg", "energy_stored_MJ", "fill_fraction",
              "compression_work_kJ_per_kg", "gravimetric_wt_pct",
              "volumetric_kg_per_m3", "compressibility_Z"]:
        assert k in r


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC012"
    assert info["fidelity"] == "F1a"


def test_mass_increases_with_pressure(model):
    """More pressure => more stored H2."""
    P = np.array([100, 200, 350, 500, 700])
    r = model.predict({"pressure": P, "temperature": 298.15})
    assert np.all(np.diff(r["stored_mass_kg"]) > 0)


def test_mass_decreases_with_temperature(model):
    """Higher temperature => lower density => less stored mass at same P."""
    T = np.array([253.15, 273.15, 298.15, 323.15, 353.15])
    r = model.predict({"pressure": 700.0, "temperature": T})
    assert np.all(np.diff(r["stored_mass_kg"]) < 0)


def test_energy_positive(model):
    r = model.predict({"pressure": 350, "temperature": 298.15})
    assert float(r["energy_stored_MJ"]) > 0


def test_compression_work_positive(model):
    """Compression work must be positive (work input)."""
    r = model.predict({"pressure": 700, "temperature": 298.15})
    assert float(r["compression_work_kJ_per_kg"]) > 0


def test_compression_work_increases_with_pressure(model):
    P = np.array([50, 100, 200, 350, 500, 700])
    r = model.predict({"pressure": P, "temperature": 298.15})
    assert np.all(np.diff(r["compression_work_kJ_per_kg"]) > 0)


def test_compressibility_above_one(model):
    """Z > 1 for H2 at high pressure (repulsive interactions dominate)."""
    r = model.predict({"pressure": 700, "temperature": 298.15})
    assert float(r["compressibility_Z"]) > 1.0


def test_fill_fraction_bounds(model):
    r_low = model.predict({"pressure": 50.0, "temperature": 298.15})
    r_high = model.predict({"pressure": 700.0, "temperature": 298.15})
    assert 0.0 < float(r_low["fill_fraction"]) < 1.0
    assert abs(float(r_high["fill_fraction"]) - 1.0) < 0.01


def test_gravimetric_density_reasonable(model):
    """Type IV at 700 bar should give ~5-6 wt%."""
    r = model.predict({"pressure": 700, "temperature": 298.15})
    wt = float(r["gravimetric_wt_pct"])
    assert 2.0 < wt < 15.0, f"Gravimetric density {wt:.1f}% seems unreasonable"


def test_volumetric_density_reasonable(model):
    """At 700 bar ~40 kg/m3."""
    r = model.predict({"pressure": 700, "temperature": 298.15})
    rho = float(r["volumetric_kg_per_m3"])
    assert 10.0 < rho < 80.0, f"Volumetric density {rho:.1f} kg/m3 seems unreasonable"


def test_benchmark(model):
    P = np.random.uniform(50, 700, 1000)
    T = np.random.uniform(250, 350, 1000)
    start = time.perf_counter()
    model.predict({"pressure": P, "temperature": T})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed*1000:.2f} ms")
    assert elapsed < 1.0
