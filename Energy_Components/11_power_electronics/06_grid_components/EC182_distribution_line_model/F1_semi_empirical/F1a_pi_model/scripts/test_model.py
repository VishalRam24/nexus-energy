"""EC182 — Distribution Line — F1a R+jX Model — Test Suite"""
import sys, time, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()


def test_predict_keys(model):
    r = model.predict({"V_s_kV": 11.0, "P_load_kW": 1000.0, "Q_load_kVAR": 400.0})
    for k in ["V_r_kV", "I_line_A", "P_loss_kW", "Q_loss_kVAR",
              "efficiency", "voltage_drop_kV", "voltage_drop_pct", "power_factor_load"]:
        assert k in r


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC182"
    assert info["fidelity"] == "F1a"


def test_V_r_less_than_V_s_for_inductive_load(model):
    """V_r < V_s for positive P and lagging Q."""
    r = model.predict({"V_s_kV": 11.0, "P_load_kW": 1500.0, "Q_load_kVAR": 700.0})
    assert float(r["V_r_kV"]) < 11.0, f"V_r={float(r['V_r_kV']):.3f} must be < 11 kV"


def test_P_loss_positive(model):
    """Active loss must be positive for nonzero load."""
    r = model.predict({"V_s_kV": 11.0, "P_load_kW": 1000.0, "Q_load_kVAR": 400.0})
    assert float(r["P_loss_kW"]) > 0.0


def test_power_balance(model):
    """P_s = P_load + P_loss."""
    r = model.predict({"V_s_kV": 11.0, "P_load_kW": 1000.0, "Q_load_kVAR": 400.0})
    balance = float(r["P_s_kW"]) - 1000.0 - float(r["P_loss_kW"])
    assert abs(balance) < 1e-6, f"Power balance error: {balance:.6f} kW"


def test_longer_line_more_loss(model):
    """More resistance (longer line) → more P_loss."""
    r2 = model.predict({"V_s_kV": 11.0, "P_load_kW": 1000.0,
                        "Q_load_kVAR": 400.0, "length_km": 2.0})
    r10 = model.predict({"V_s_kV": 11.0, "P_load_kW": 1000.0,
                         "Q_load_kVAR": 400.0, "length_km": 10.0})
    assert float(r10["P_loss_kW"]) > float(r2["P_loss_kW"])


def test_voltage_drop_increases_with_load(model):
    r_low = model.predict({"V_s_kV": 11.0, "P_load_kW": 200.0, "Q_load_kVAR": 80.0})
    r_high = model.predict({"V_s_kV": 11.0, "P_load_kW": 2000.0, "Q_load_kVAR": 800.0})
    assert float(r_high["voltage_drop_pct"]) > float(r_low["voltage_drop_pct"])


def test_efficiency_below_one(model):
    r = model.predict({"V_s_kV": 11.0, "P_load_kW": 1000.0, "Q_load_kVAR": 400.0})
    assert float(r["efficiency"]) < 1.0


def test_zero_load_near_zero_loss(model):
    """Zero load → near-zero current → near-zero loss (no shunt, pure series model)."""
    r = model.predict({"V_s_kV": 11.0, "P_load_kW": 0.0, "Q_load_kVAR": 0.0})
    assert float(r["P_loss_kW"]) < 1e-6


def test_power_factor_computed_correctly(model):
    """pf = P / sqrt(P^2 + Q^2)."""
    P, Q = 1000.0, 700.0
    r = model.predict({"V_s_kV": 11.0, "P_load_kW": P, "Q_load_kVAR": Q})
    expected_pf = P / np.sqrt(P**2 + Q**2)
    assert abs(float(r["power_factor_load"]) - expected_pf) < 1e-9


def test_vectorized_input(model):
    P = np.linspace(100, 2000, 30)
    Q = P * 0.4
    r = model.predict({"V_s_kV": 11.0, "P_load_kW": P, "Q_load_kVAR": Q})
    assert r["V_r_kV"].shape == (30,)
    assert np.all(r["P_loss_kW"] >= 0)


def test_benchmark(model):
    P = np.random.uniform(100, 3000, 1000)
    Q = P * 0.4
    start = time.perf_counter()
    model.predict({"V_s_kV": 11.0, "P_load_kW": P, "Q_load_kVAR": Q})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed*1000:.2f} ms")
    assert elapsed < 5.0
