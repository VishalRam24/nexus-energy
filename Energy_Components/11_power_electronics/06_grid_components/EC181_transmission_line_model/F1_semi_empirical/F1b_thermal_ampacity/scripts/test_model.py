"""EC181 — Transmission Line — F1b Thermal Ampacity — Test Suite"""
import sys, time, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()


def test_predict_keys(model):
    r = model.predict({"V_s_pu": 1.0, "delta_s_rad": 0.0,
                       "P_load_pu": 0.5, "Q_load_pu": 0.2})
    expected = ["V_r_pu", "delta_r_rad", "I_series_pu", "I_series_A",
                "P_loss_pu", "Q_loss_pu", "P_s_pu", "Q_s_pu",
                "efficiency", "voltage_drop_pu",
                "R_ac_pu_total", "skin_factor", "I_max_A",
                "ampacity_margin", "derating_factor"]
    for k in expected:
        assert k in r, f"Missing key: {k}"


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC181"
    assert info["fidelity"] == "F1b"


def test_resistance_increases_with_temperature(model):
    """R(T) must increase with conductor temperature (positive alpha)."""
    r20 = model.model.r_ac_pu_per_km(20.0)
    r75 = model.model.r_ac_pu_per_km(75.0)
    assert r75 > r20, f"R(75C)={r75:.6f} should be > R(20C)={r20:.6f}"


def test_r_t_correction_magnitude(model):
    """R(75C)/R(20C) should be ~1.22 for aluminium (alpha=0.00403, dT=55 K)."""
    r20 = model.model.r_ac_pu_per_km(20.0)
    r75 = model.model.r_ac_pu_per_km(75.0)
    ratio = r75 / r20
    assert 1.15 < ratio < 1.35, f"R ratio {ratio:.3f}, expected ~1.22 for Al with alpha=0.00403"


def test_skin_factor_above_one(model):
    """Skin effect factor must be >= 1.0."""
    assert model.model.skin_factor >= 1.0, "skin_factor must be >= 1"


def test_skin_factor_applied(model):
    """R_ac > R_dc at same temperature (skin effect)."""
    R_ac = model.model.r_ac_pu_per_km(20.0)
    R_dc = model.model.R_dc_pu_per_km
    assert R_ac >= R_dc, "R_ac must be >= R_dc"


def test_ampacity_positive(model):
    """Thermal ampacity must be positive."""
    I_max = model.model.thermal_ampacity_A(T_amb_C=25.0)
    assert float(I_max) > 0.0, f"I_max={float(I_max)}"


def test_ampacity_decreases_with_ambient_temp(model):
    """Higher ambient → less heat dissipation → lower ampacity."""
    I_low  = model.model.thermal_ampacity_A(T_amb_C=10.0)
    I_high = model.model.thermal_ampacity_A(T_amb_C=40.0)
    assert float(I_high) < float(I_low), \
        f"I_max(40C)={float(I_high):.1f} A should be < I_max(10C)={float(I_low):.1f} A"


def test_derating_factor_below_one_at_high_ambient(model):
    """Derating factor must be < 1 at T_amb > 25 degC."""
    df = model.model.ampacity_derating_factor(40.0)
    assert float(df) < 1.0, f"Derating at 40C={float(df):.4f}"


def test_derating_factor_above_one_at_low_ambient(model):
    """Derating factor > 1 at T_amb < 25 degC (more cooling capacity)."""
    df = model.model.ampacity_derating_factor(0.0)
    assert float(df) > 1.0, f"Derating at 0C={float(df):.4f}"


def test_hot_conductor_higher_loss(model):
    """Hotter conductor → higher R → higher P_loss for same current."""
    r_hot  = model.predict({"V_s_pu": 1.0, "delta_s_rad": 0.0,
                             "P_load_pu": 0.6, "Q_load_pu": 0.2,
                             "T_cond_C": 75.0})
    r_cold = model.predict({"V_s_pu": 1.0, "delta_s_rad": 0.0,
                             "P_load_pu": 0.6, "Q_load_pu": 0.2,
                             "T_cond_C": 20.0})
    assert float(r_hot["P_loss_pu"]) > float(r_cold["P_loss_pu"]), \
        "Hotter conductor must have higher losses"


def test_ampacity_margin_positive_for_light_load(model):
    """Light load should be well within ampacity."""
    r = model.predict({"V_s_pu": 1.0, "P_load_pu": 0.1, "Q_load_pu": 0.05})
    assert float(r["ampacity_margin"]) > 0.0, "Light load should have positive ampacity margin"


def test_voltage_drop_positive_inductive_load(model):
    r = model.predict({"V_s_pu": 1.0, "P_load_pu": 0.6, "Q_load_pu": 0.3})
    assert float(r["voltage_drop_pu"]) > 0.0


def test_efficiency_range(model):
    r = model.predict({"V_s_pu": 1.0, "P_load_pu": 0.6, "Q_load_pu": 0.3})
    assert 0.0 < float(r["efficiency"]) < 1.0


def test_vectorized_input(model):
    P = np.linspace(0.1, 1.0, 20)
    r = model.predict({"V_s_pu": 1.0, "P_load_pu": P, "Q_load_pu": P * 0.3})
    assert r["V_r_pu"].shape == (20,)
    assert r["P_loss_pu"].shape == (20,)


def test_benchmark(model):
    P = np.random.uniform(0.1, 1.2, 1000)
    start = time.perf_counter()
    model.predict({"V_s_pu": 1.0, "P_load_pu": P, "Q_load_pu": P * 0.3})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed*1000:.2f} ms")
    assert elapsed < 5.0
