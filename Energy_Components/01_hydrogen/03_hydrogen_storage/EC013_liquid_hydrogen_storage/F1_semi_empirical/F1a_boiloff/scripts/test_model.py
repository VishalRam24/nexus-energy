"""EC013 — Liquid H2 Storage — F1a — Test Suite"""
import sys, time, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()


def test_predict_keys(model):
    r = model.predict({"fill_fraction": 0.8, "T_ambient": 298.15})
    for k in ["stored_mass_kg", "energy_stored_MJ", "heat_leak_W",
              "boiloff_kg_per_day", "boiloff_pct_per_day",
              "time_to_empty_days", "gravimetric_wt_pct", "volumetric_kg_per_m3"]:
        assert k in r


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC013"
    assert info["fidelity"] == "F1a"


def test_mass_increases_with_fill(model):
    f = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    r = model.predict({"fill_fraction": f, "T_ambient": 298.15})
    assert np.all(np.diff(r["stored_mass_kg"]) > 0)


def test_boiloff_positive(model):
    r = model.predict({"fill_fraction": 0.8, "T_ambient": 298.15})
    assert float(r["boiloff_kg_per_day"]) > 0
    assert float(r["boiloff_pct_per_day"]) > 0
    assert float(r["heat_leak_W"]) > 0


def test_boiloff_increases_with_ambient_temperature(model):
    """Higher ambient T => larger heat leak => more boil-off."""
    T = np.array([253.15, 273.15, 298.15, 313.15])
    r = model.predict({"fill_fraction": 0.8, "T_ambient": T})
    assert np.all(np.diff(r["heat_leak_W"]) > 0)
    assert np.all(np.diff(r["boiloff_kg_per_day"]) > 0)


def test_bor_pct_decreases_with_fill(model):
    """Boil-off mass rate is fixed by heat leak; %/day decreases as fill increases."""
    f = np.array([0.2, 0.4, 0.6, 0.8, 0.95])
    r = model.predict({"fill_fraction": f, "T_ambient": 298.15})
    assert np.all(np.diff(r["boiloff_pct_per_day"]) < 0)


def test_bor_in_realistic_range(model):
    """Typical small dewars: ~0.1-2 %/day at room T."""
    r = model.predict({"fill_fraction": 0.8, "T_ambient": 298.15})
    bor = float(r["boiloff_pct_per_day"])
    assert 0.05 < bor < 5.0, f"BOR={bor:.3f} %/day outside typical range"


def test_time_to_empty_positive(model):
    r = model.predict({"fill_fraction": 0.8, "T_ambient": 298.15})
    assert float(r["time_to_empty_days"]) > 0


def test_energy_positive(model):
    r = model.predict({"fill_fraction": 0.8, "T_ambient": 298.15})
    assert float(r["energy_stored_MJ"]) > 0


def test_volumetric_density_below_lh2_density(model):
    """Volumetric density at full fill < pure LH2 density (~70.85 kg/m3)."""
    r = model.predict({"fill_fraction": 0.95, "T_ambient": 298.15})
    rho = float(r["volumetric_kg_per_m3"])
    assert 50.0 < rho < 75.0


def test_zero_fill(model):
    r = model.predict({"fill_fraction": 0.0, "T_ambient": 298.15})
    assert float(r["stored_mass_kg"]) == pytest.approx(0.0, abs=1e-12)
    assert float(r["energy_stored_MJ"]) == pytest.approx(0.0, abs=1e-12)


def test_benchmark(model):
    f_arr = np.random.uniform(0.05, 0.95, 1000)
    T_arr = np.random.uniform(240, 320, 1000)
    start = time.perf_counter()
    model.predict({"fill_fraction": f_arr, "T_ambient": T_arr})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed*1000:.2f} ms")
    assert elapsed < 1.0
