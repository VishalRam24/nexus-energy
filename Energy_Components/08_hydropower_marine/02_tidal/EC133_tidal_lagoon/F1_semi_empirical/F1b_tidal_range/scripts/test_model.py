"""EC133 — Tidal Lagoon — F1b Tidal Range / Efficiency — Test Suite

Physics rules:
  - Power scales with tidal_range^2 (basin energy equation)
  - At design head, turbine_efficiency = eta_peak
  - Turbine efficiency degrades below and above design head
  - Salinity increase → density increase → power increase
  - Temperature increase → density decrease → power decrease
  - Pumping mode gives more power than non-pumping
  - Power = 0 when tidal range < 2*h_min
  - Spring range gives more power than neap range
"""
import sys, time, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()


def test_predict_keys(model):
    r = model.predict({"tidal_range_m": 9.0})
    for k in ["power_mw", "turbine_efficiency", "energy_per_cycle_mwh", "seawater_density_kgm3"]:
        assert k in r, f"Missing key: {k}"


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC133"
    assert info["fidelity"] == "F1b"


def test_power_positive_at_design_range(model):
    """Power must be positive at design tidal range."""
    r = model.predict({"tidal_range_m": 9.0})
    assert float(r["power_mw"]) > 0.0


def test_power_scales_with_range_squared(model):
    """Power ∝ h^2 → (2*h)^2 = 4*h^2 → doubling range gives 4x theoretical power."""
    m = model._model
    P1 = float(m.avg_power_mw(5.0))
    P2 = float(m.avg_power_mw(10.0))
    # Ratio should be ~4 (h doubles from 2.5 to 5 m, P ∝ h^2)
    ratio = P2 / P1 if P1 > 0 else 0
    assert 3.5 <= ratio <= 4.5, f"Power ratio at 2x range: {ratio:.2f}, expected ~4"


def test_turbine_efficiency_at_design(model):
    """At design head (h=h_design), turbine efficiency must equal eta_peak."""
    m = model._model
    eta = float(m.turbine_efficiency(m.h_design))
    assert abs(eta - m.eta_peak) < 1e-9, f"eta at design head {eta:.4f} != eta_peak {m.eta_peak:.4f}"


def test_turbine_efficiency_degrades_at_low_head(model):
    """Efficiency at neap tide (h < h_design) must be < eta_peak."""
    m = model._model
    eta_neap = float(m.turbine_efficiency(m.h_design * 0.7))
    assert eta_neap < m.eta_peak, "Efficiency must degrade at sub-design head"


def test_power_zero_below_h_min(model):
    """No power generation when tidal amplitude < h_min."""
    m = model._model
    P = float(m.avg_power_mw(2 * m.h_min * 0.9))  # range = 2*h, set amplitude just below h_min
    assert P == 0.0, f"Power should be 0 below h_min, got {P:.3f} MW"


def test_pumping_mode_adds_power(model):
    """Pumping mode must generate more power than ebb-only at same tidal range."""
    r_base = model.predict({"tidal_range_m": 9.0, "pumping_mode": False})
    r_pump = model.predict({"tidal_range_m": 9.0, "pumping_mode": True})
    assert float(r_pump["power_mw"]) > float(r_base["power_mw"])


def test_salinity_increases_density(model):
    """Higher salinity → higher seawater density."""
    m = model._model
    rho_low  = float(m.seawater_density(T_C=12.0, S_psu=30.0))
    rho_high = float(m.seawater_density(T_C=12.0, S_psu=37.0))
    assert rho_high > rho_low


def test_temperature_decreases_density(model):
    """Higher temperature → lower seawater density."""
    m = model._model
    rho_cold = float(m.seawater_density(T_C=5.0,  S_psu=35.0))
    rho_warm = float(m.seawater_density(T_C=20.0, S_psu=35.0))
    assert rho_warm < rho_cold


def test_salinity_increases_power(model):
    """Higher salinity (denser water) → higher power output."""
    r_low  = model.predict({"tidal_range_m": 9.0, "T_C": 12.0, "S_psu": 30.0})
    r_high = model.predict({"tidal_range_m": 9.0, "T_C": 12.0, "S_psu": 37.0})
    assert float(r_high["power_mw"]) > float(r_low["power_mw"])


def test_spring_vs_neap_power(model):
    """Spring tidal range must give significantly more power than neap."""
    m = model._model
    R_spring = 2 * m.h_design * (1 + m.sn_amp)
    R_neap   = 2 * m.h_design * (1 - m.sn_amp)
    P_spring = float(m.avg_power_mw(R_spring))
    P_neap   = float(m.avg_power_mw(R_neap))
    assert P_spring > P_neap, f"Spring ({P_spring:.1f} MW) must exceed neap ({P_neap:.1f} MW)"


def test_reference_density_sensible(model):
    """Reference density at (T_ref, S_ref) should equal rho_ref exactly."""
    m = model._model
    rho = float(m.seawater_density(m.T_ref, m.S_ref))
    assert abs(rho - m.rho_ref) < 1e-6


def test_energy_per_cycle_consistent_with_power(model):
    """Energy per cycle = P_avg * T_tide."""
    m = model._model
    P_mw = float(m.avg_power_mw(9.0))
    E_mwh = float(m.energy_per_cycle_mwh(9.0))
    T_h = m.T_tide / 3600.0
    assert abs(E_mwh - P_mw * T_h) < 1e-6


def test_benchmark(model):
    rng = np.random.default_rng(42)
    R   = rng.uniform(1.0, 12.0, 1000)
    start = time.perf_counter()
    model.predict({"tidal_range_m": R})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed*1000:.2f} ms")
    assert elapsed < 1.0
