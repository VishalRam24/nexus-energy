"""EC182 — Distribution Line — F1b Thermal Ampacity — Test Suite"""
import sys, time, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()


def test_predict_keys(model):
    r = model.predict({"V_s_kV": 11.0, "P_load_kW": 500.0, "Q_load_kVAR": 150.0})
    for k in ["V_r_kV", "I_line_A", "P_loss_kW", "Q_loss_kVAR", "P_s_kW",
              "efficiency", "voltage_drop_kV", "voltage_drop_pct", "power_factor_load",
              "R_ac_ohm_km", "skin_factor", "I_max_A", "ampacity_margin",
              "congestion_factor", "derating_factor"]:
        assert k in r, f"Missing key: {k}"


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC182"
    assert info["fidelity"] == "F1b"


def test_r_increases_with_temperature(model):
    r20 = model.model.r_ac_ohm_per_km(20.0)
    r75 = model.model.r_ac_ohm_per_km(75.0)
    assert r75 > r20, f"R(75C)={r75:.4f} must be > R(20C)={r20:.4f}"


def test_r_correction_magnitude(model):
    """Al: R(75C)/R(20C) ≈ 1 + 0.00403*55 = 1.222"""
    r20 = model.model.r_ac_ohm_per_km(20.0)
    r75 = model.model.r_ac_ohm_per_km(75.0)
    ratio = r75 / r20
    assert 1.15 < ratio < 1.35, f"R ratio {ratio:.3f}"


def test_skin_factor_applied(model):
    """R_ac >= R_dc at same temperature."""
    R_ac = model.model.r_ac_ohm_per_km(20.0)
    R_dc = model.model.R_dc_ohm_per_km
    assert R_ac >= R_dc


def test_ampacity_positive(model):
    assert model.model.thermal_ampacity_A(25.0) > 0.0


def test_ampacity_decreases_at_higher_ambient(model):
    I_low  = model.model.thermal_ampacity_A(10.0)
    I_high = model.model.thermal_ampacity_A(40.0)
    assert float(I_high) < float(I_low)


def test_derating_factor_less_than_one_hot_day(model):
    df = model.model.ampacity_derating_factor(40.0)
    assert float(df) < 1.0


def test_derating_factor_greater_than_one_cold_day(model):
    df = model.model.ampacity_derating_factor(0.0)
    assert float(df) > 1.0


def test_hot_conductor_more_losses(model):
    r_hot  = model.predict({"V_s_kV": 11.0, "P_load_kW": 500.0, "T_cond_C": 75.0})
    r_cold = model.predict({"V_s_kV": 11.0, "P_load_kW": 500.0, "T_cond_C": 20.0})
    assert float(r_hot["P_loss_kW"]) > float(r_cold["P_loss_kW"])


def test_voltage_drop_positive(model):
    r = model.predict({"V_s_kV": 11.0, "P_load_kW": 500.0, "Q_load_kVAR": 150.0})
    assert float(r["voltage_drop_kV"]) > 0.0


def test_efficiency_range(model):
    r = model.predict({"V_s_kV": 11.0, "P_load_kW": 500.0, "Q_load_kVAR": 150.0})
    assert 0.0 < float(r["efficiency"]) < 1.0


def test_ampacity_margin_positive_light_load(model):
    r = model.predict({"V_s_kV": 11.0, "P_load_kW": 50.0})
    assert float(r["ampacity_margin"]) > 0.0


def test_congestion_factor_range(model):
    r = model.predict({"V_s_kV": 11.0, "P_load_kW": 500.0})
    assert 0.0 <= float(r["congestion_factor"]) <= 2.0   # can exceed 1 if overloaded


def test_vectorized(model):
    P = np.linspace(100, 1500, 20)
    r = model.predict({"V_s_kV": 11.0, "P_load_kW": P, "Q_load_kVAR": P * 0.3})
    assert r["I_line_A"].shape == (20,)


def test_benchmark(model):
    P = np.random.uniform(100, 2000, 1000)
    start = time.perf_counter()
    model.predict({"V_s_kV": 11.0, "P_load_kW": P, "Q_load_kVAR": P * 0.3})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed*1000:.2f} ms")
    assert elapsed < 5.0
