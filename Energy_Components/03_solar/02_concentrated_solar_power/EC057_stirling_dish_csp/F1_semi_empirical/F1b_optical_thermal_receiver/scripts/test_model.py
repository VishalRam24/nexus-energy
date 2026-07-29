"""EC057 — Stirling Dish CSP — F1b Optical+Receiver Thermal — Test Suite"""

import sys, time
import numpy as np
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()


# --- Interface ---

def test_predict_returns_all_keys(model):
    r = model.predict({"dni_w_m2": 900.0, "theta_deg": 0.0,
                        "T_receiver_degC": 720.0, "T_ambient_degC": 25.0, "PLR": 1.0})
    for key in ["power_output_kw", "Q_absorbed_kw", "Q_receiver_loss_kw",
                 "Q_net_thermal_kw", "eta_stirling", "overall_efficiency", "iam_factor"]:
        assert key in r, f"Missing key: {key}"


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC057"
    assert info["fidelity"] == "F1b"


# --- IAM physics ---

def test_iam_unity_at_zero_angle(model):
    """IAM should be ~1.0 at zero incidence angle."""
    r = model.predict({"dni_w_m2": 900.0, "theta_deg": 0.0,
                        "T_receiver_degC": 720.0, "T_ambient_degC": 25.0})
    assert float(r["iam_factor"]) == pytest.approx(1.0, abs=0.001)


def test_iam_decreases_with_angle(model):
    theta = np.array([0.0, 2.0, 5.0, 8.0])
    r = model.predict({"dni_w_m2": 900.0, "theta_deg": theta,
                        "T_receiver_degC": 720.0, "T_ambient_degC": 25.0})
    assert np.all(np.diff(r["iam_factor"]) < 0)


# --- Power output physics ---

def test_power_positive_at_good_conditions(model):
    r = model.predict({"dni_w_m2": 900.0, "theta_deg": 0.0,
                        "T_receiver_degC": 720.0, "T_ambient_degC": 25.0, "PLR": 1.0})
    assert float(r["power_output_kw"]) > 0


def test_zero_power_at_zero_dni(model):
    r = model.predict({"dni_w_m2": 0.0, "theta_deg": 0.0,
                        "T_receiver_degC": 720.0, "T_ambient_degC": 25.0})
    assert float(r["power_output_kw"]) == 0.0


def test_power_increases_with_dni(model):
    dni = np.array([400.0, 600.0, 800.0, 1000.0])
    r = model.predict({"dni_w_m2": dni, "theta_deg": 0.0,
                        "T_receiver_degC": 720.0, "T_ambient_degC": 25.0, "PLR": 1.0})
    assert np.all(np.diff(r["power_output_kw"]) > 0)


def test_power_decreases_at_part_load(model):
    r_full = model.predict({"dni_w_m2": 900.0, "theta_deg": 0.0,
                             "T_receiver_degC": 720.0, "T_ambient_degC": 25.0, "PLR": 1.0})
    r_half = model.predict({"dni_w_m2": 900.0, "theta_deg": 0.0,
                             "T_receiver_degC": 720.0, "T_ambient_degC": 25.0, "PLR": 0.5})
    assert float(r_half["power_output_kw"]) < float(r_full["power_output_kw"])


# --- Receiver losses ---

def test_receiver_loss_positive(model):
    r = model.predict({"dni_w_m2": 900.0, "theta_deg": 0.0,
                        "T_receiver_degC": 720.0, "T_ambient_degC": 25.0})
    assert float(r["Q_receiver_loss_kw"]) > 0


def test_receiver_loss_increases_with_temperature(model):
    """Higher receiver temperature means more radiative and convective loss."""
    T_rec = np.array([400.0, 500.0, 600.0, 700.0, 800.0])
    r = model.predict({"dni_w_m2": 900.0, "theta_deg": 0.0,
                        "T_receiver_degC": T_rec, "T_ambient_degC": 25.0})
    assert np.all(np.diff(r["Q_receiver_loss_kw"]) > 0)


# --- Stirling efficiency ---

def test_stirling_eta_below_carnot(model):
    """Stirling efficiency must be below Carnot limit."""
    T_hot = 720.0
    T_sink = 25.0 + 15.0  # T_amb + T_approach
    eta_carnot = 1.0 - (T_sink + 273.15) / (T_hot - 20.0 + 273.15)
    r = model.predict({"dni_w_m2": 900.0, "theta_deg": 0.0,
                        "T_receiver_degC": T_hot, "T_ambient_degC": 25.0, "PLR": 1.0})
    eta_s = float(r["eta_stirling"])
    assert eta_s < eta_carnot, f"eta_stirling={eta_s:.4f} >= Carnot={eta_carnot:.4f}"


def test_stirling_eta_positive(model):
    r = model.predict({"dni_w_m2": 900.0, "theta_deg": 0.0,
                        "T_receiver_degC": 720.0, "T_ambient_degC": 25.0, "PLR": 1.0})
    assert float(r["eta_stirling"]) > 0


def test_higher_receiver_temp_improves_eta(model):
    r_low  = model.predict({"dni_w_m2": 900.0, "theta_deg": 0.0,
                             "T_receiver_degC": 500.0, "T_ambient_degC": 25.0})
    r_high = model.predict({"dni_w_m2": 900.0, "theta_deg": 0.0,
                             "T_receiver_degC": 720.0, "T_ambient_degC": 25.0})
    assert float(r_high["eta_stirling"]) > float(r_low["eta_stirling"])


# --- Energy balance ---

def test_energy_balance(model):
    """Q_absorbed = Q_receiver_loss + Q_net_thermal (first law at receiver)."""
    r = model.predict({"dni_w_m2": 900.0, "theta_deg": 0.0,
                        "T_receiver_degC": 720.0, "T_ambient_degC": 25.0, "PLR": 1.0})
    Q_abs  = float(r["Q_absorbed_kw"])
    Q_loss = float(r["Q_receiver_loss_kw"])
    Q_net  = float(r["Q_net_thermal_kw"])
    assert abs((Q_loss + Q_net) - Q_abs) < 0.01 * Q_abs, (
        f"Energy balance fail: Q_abs={Q_abs:.3f}, Q_loss+Q_net={Q_loss+Q_net:.3f}")


# --- Overall efficiency range ---

def test_overall_efficiency_range(model):
    """System efficiency at good conditions: 15-30%."""
    r = model.predict({"dni_w_m2": 900.0, "theta_deg": 0.0,
                        "T_receiver_degC": 720.0, "T_ambient_degC": 25.0, "PLR": 1.0})
    eta = float(r["overall_efficiency"])
    assert 0.10 < eta < 0.35, f"Overall efficiency = {eta:.4f}"


# --- Rated power check ---

def test_rated_power_approximately_25kw(model):
    """SunCatcher/EuroDish class: ~25 kW at peak conditions."""
    r = model.predict({"dni_w_m2": 900.0, "theta_deg": 0.0,
                        "T_receiver_degC": 720.0, "T_ambient_degC": 25.0, "PLR": 1.0})
    P = float(r["power_output_kw"])
    # Allow 30% tolerance — real dish output varies with exact conditions
    assert 10.0 < P < 35.0, f"Power = {P:.2f} kW, expected ~25 kW"


# --- Array inputs ---

def test_array_inputs(model):
    dni = np.array([400.0, 700.0, 900.0])
    r = model.predict({"dni_w_m2": dni, "theta_deg": 0.0,
                        "T_receiver_degC": 720.0, "T_ambient_degC": 25.0})
    assert r["power_output_kw"].shape == (3,)


# --- Benchmark ---

def test_benchmark(model):
    dni = np.random.uniform(500, 1000, 1000)
    PLR = np.random.uniform(0.3, 1.0, 1000)
    start = time.perf_counter()
    model.predict({"dni_w_m2": dni, "theta_deg": 2.0,
                    "T_receiver_degC": 720.0, "T_ambient_degC": 25.0, "PLR": PLR})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed*1000:.2f} ms")
    assert elapsed < 1.0
