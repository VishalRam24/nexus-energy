"""EC131 — Tidal Barrage — F1b Tidal Range — Test Suite"""
import sys, time, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()


def test_predict_keys(model):
    r = model.predict({"tidal_range_m": 8.0})
    for k in ["power_kw", "power_mw", "turbine_efficiency", "seawater_density_kg_m3",
              "theoretical_power_kw", "energy_per_cycle_mwh"]:
        assert k in r


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC131"
    assert info["fidelity"] == "F1b"


def test_design_efficiency_near_peak(model):
    m = model._model
    eta = float(m.turbine_efficiency(m.h_design))
    assert abs(eta - m.eta_peak) < 0.001


def test_efficiency_drops_at_half_head(model):
    m = model._model
    eta_design = float(m.turbine_efficiency(m.h_design))
    eta_half   = float(m.turbine_efficiency(m.h_design * 0.5))
    assert eta_design > eta_half


def test_efficiency_zero_below_hmin(model):
    m = model._model
    eta = float(m.turbine_efficiency(m.h_min * 0.5))
    assert eta == 0.0


def test_power_increases_with_range(model):
    R_arr = np.linspace(4, 12, 10)
    r = model.predict({"tidal_range_m": R_arr})
    assert np.all(np.diff(r["power_kw"]) > 0)


def test_power_zero_below_hmin(model):
    m = model._model
    R = m.h_min * 2.0 * 0.5  # tidal range giving h < h_min
    r = model.predict({"tidal_range_m": R})
    assert float(r["power_kw"]) == 0.0


def test_power_proportional_to_area(model):
    r1 = model.predict({"tidal_range_m": 8.0, "basin_area_m2": 1e8})
    r2 = model.predict({"tidal_range_m": 8.0, "basin_area_m2": 2e8})
    ratio = float(r2["power_kw"]) / float(r1["power_kw"])
    assert abs(ratio - 2.0) < 0.01


def test_seawater_density_increases_with_salinity(model):
    m = model._model
    rho1 = float(m.seawater_density(T_C=12.0, S_psu=30.0))
    rho2 = float(m.seawater_density(T_C=12.0, S_psu=38.0))
    assert rho2 > rho1


def test_seawater_density_decreases_with_temperature(model):
    m = model._model
    rho_cold = float(m.seawater_density(T_C=5.0, S_psu=35.0))
    rho_warm = float(m.seawater_density(T_C=20.0, S_psu=35.0))
    assert rho_cold > rho_warm


def test_higher_density_more_power(model):
    r_cold = model.predict({"tidal_range_m": 8.0, "T_C": 5.0, "S_psu": 38.0})
    r_warm = model.predict({"tidal_range_m": 8.0, "T_C": 20.0, "S_psu": 30.0})
    assert float(r_cold["power_kw"]) > float(r_warm["power_kw"])


def test_pumping_increases_power(model):
    r_no = model.predict({"tidal_range_m": 8.0, "pumping_mode": False})
    r_pump = model.predict({"tidal_range_m": 8.0, "pumping_mode": True})
    assert float(r_pump["power_kw"]) > float(r_no["power_kw"])


def test_vectorized(model):
    R = np.linspace(3, 12, 50)
    r = model.predict({"tidal_range_m": R})
    assert len(r["power_kw"]) == 50


def test_benchmark(model):
    R = np.random.uniform(3, 12, 1000)
    start = time.perf_counter()
    model.predict({"tidal_range_m": R})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed*1000:.2f} ms")
    assert elapsed < 1.0
