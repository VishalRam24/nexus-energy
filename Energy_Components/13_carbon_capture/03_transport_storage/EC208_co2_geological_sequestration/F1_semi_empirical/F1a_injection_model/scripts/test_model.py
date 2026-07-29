"""EC208 — CO2 Geological Sequestration — F1a — Test Suite"""
import sys, time, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()


def test_predict_keys(model):
    r = model.predict({})
    for k in ["bottomhole_P_bar", "injection_rate_kg_per_s", "injection_rate_tco2_per_day",
               "storage_capacity_tco2", "pore_volume_m3", "years_to_fill"]:
        assert k in r


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC208"
    assert info["fidelity"] == "F1a"


def test_bottomhole_pressure_exceeds_wellhead(model):
    """BHP = wellhead + hydrostatic column: always > wellhead for positive depth."""
    r = model.predict({"P_wellhead_bar": 150.0, "depth_m": 2000.0})
    assert float(r["bottomhole_P_bar"]) > 150.0


def test_bottomhole_hydrostatic_formula(model):
    """BHP = P_wh + rho*g*depth; rho=700 kg/m³, g=9.81."""
    r = model.predict({"P_wellhead_bar": 150.0, "depth_m": 2000.0})
    expected = 150.0 + 700.0 * 9.81 * 2000.0 / 1e5  # bar
    assert float(r["bottomhole_P_bar"]) == pytest.approx(expected, rel=1e-6)


def test_injection_rate_positive(model):
    """With BHP > P_res, injection rate must be positive."""
    r = model.predict({"P_wellhead_bar": 150.0, "depth_m": 2000.0})
    assert float(r["injection_rate_kg_per_s"]) > 0
    assert float(r["injection_rate_tco2_per_day"]) > 0


def test_injection_rate_increases_with_wellhead_pressure(model):
    """Higher wellhead → higher BHP → larger dP drive → more injection."""
    P_wh = np.array([100.0, 125.0, 150.0, 175.0, 200.0])
    r = model.predict({"P_wellhead_bar": P_wh})
    assert np.all(np.diff(r["injection_rate_kg_per_s"]) > 0)


def test_injection_rate_increases_with_permeability(model):
    """Higher k → more flow (linear in Darcy equation)."""
    k = np.array([5.0, 10.0, 50.0, 200.0, 1000.0])
    r = model.predict({"k_mD": k})
    assert np.all(np.diff(r["injection_rate_kg_per_s"]) > 0)


def test_injection_rate_linear_in_permeability(model):
    """Darcy: Q ∝ k → doubling k doubles injection rate."""
    r1 = model.predict({"k_mD": 25.0})
    r2 = model.predict({"k_mD": 50.0})
    ratio = float(r2["injection_rate_kg_per_s"]) / float(r1["injection_rate_kg_per_s"])
    assert ratio == pytest.approx(2.0, rel=1e-6)


def test_injection_rate_linear_in_thickness(model):
    """Darcy: Q ∝ h → doubling thickness doubles injection."""
    r1 = model.predict({"h_m": 50.0})
    r2 = model.predict({"h_m": 100.0})
    ratio = float(r2["injection_rate_kg_per_s"]) / float(r1["injection_rate_kg_per_s"])
    assert ratio == pytest.approx(2.0, rel=1e-6)


def test_zero_injection_when_no_pressure_drive(model):
    """If BHP = P_res → zero injection."""
    # P_res = 200 bar; BHP = P_wh + rho*g*depth/1e5
    # Set P_wh so BHP = P_res: P_wh = 200 - 700*9.81*2000/1e5 = 200 - 137.34 = 62.66 bar
    rho, g, d = 700.0, 9.81, 2000.0
    P_res_bar = 200.0
    P_wh_zero = P_res_bar - rho * g * d / 1e5
    r = model.predict({"P_wellhead_bar": P_wh_zero})
    assert float(r["injection_rate_kg_per_s"]) == pytest.approx(0.0, abs=1e-6)


def test_storage_capacity_positive(model):
    r = model.predict({})
    assert float(r["storage_capacity_tco2"]) > 0
    assert float(r["pore_volume_m3"]) > 0


def test_storage_capacity_scales_with_area(model):
    """V_pore ∝ area: doubling area doubles capacity."""
    r1 = model.predict({"area_km2": 50.0})
    r2 = model.predict({"area_km2": 100.0})
    ratio = float(r2["storage_capacity_tco2"]) / float(r1["storage_capacity_tco2"])
    assert ratio == pytest.approx(2.0, rel=1e-6)


def test_storage_capacity_scales_with_efficiency(model):
    """M_stored ∝ E: doubling efficiency doubles capacity."""
    r1 = model.predict({"storage_efficiency": 0.01})
    r2 = model.predict({"storage_efficiency": 0.02})
    ratio = float(r2["storage_capacity_tco2"]) / float(r1["storage_capacity_tco2"])
    assert ratio == pytest.approx(2.0, rel=1e-6)


def test_storage_efficiency_ipcc_range(model):
    """IPCC (2005) saline aquifer efficiency: 1–4%. Test both extremes are physically plausible."""
    r_low = model.predict({"storage_efficiency": 0.01})
    r_high = model.predict({"storage_efficiency": 0.04})
    assert float(r_high["storage_capacity_tco2"]) == pytest.approx(
        4.0 * float(r_low["storage_capacity_tco2"]), rel=1e-6)


def test_years_to_fill_positive(model):
    r = model.predict({})
    assert float(r["years_to_fill"]) > 0


def test_benchmark(model):
    rng = np.random.default_rng(42)
    P_wh = rng.uniform(100, 250, 1000)
    k = rng.uniform(1, 500, 1000)
    start = time.perf_counter()
    model.predict({"P_wellhead_bar": P_wh, "k_mD": k})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed*1000:.2f} ms")
    assert elapsed < 1.0
