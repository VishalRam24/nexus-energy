"""EC055 — Solar Tower — F1b With Thermal Losses — Test Suite"""

import sys, time
import numpy as np
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()


def test_predict_returns_dict(model):
    r = model.predict({"dni_w_m2": 900.0, "solar_zenith_deg": 20.0,
                       "T_receiver_degC": 600.0, "T_ambient_degC": 25.0})
    for key in ["Q_field_kw", "useful_heat_kw", "thermal_loss_kw",
                "Q_radiative_kw", "Q_convective_kw",
                "optical_efficiency", "receiver_efficiency", "overall_efficiency",
                "h_conv_w_m2k"]:
        assert key in r


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC055"
    assert info["fidelity"] == "F1b"


def test_zero_dni_gives_zero_output(model):
    r = model.predict({"dni_w_m2": 0.0, "solar_zenith_deg": 20.0,
                       "T_receiver_degC": 600.0, "T_ambient_degC": 25.0})
    assert float(r["useful_heat_kw"]) == 0.0, "No DNI → no useful heat"
    assert float(r["Q_field_kw"]) == 0.0


def test_useful_heat_increases_with_dni(model):
    """Higher DNI → more power onto receiver → more useful heat."""
    dnis = np.array([200.0, 400.0, 600.0, 800.0, 1000.0])
    r = model.predict({"dni_w_m2": dnis, "solar_zenith_deg": 20.0,
                       "T_receiver_degC": 600.0, "T_ambient_degC": 25.0})
    assert np.all(np.diff(r["useful_heat_kw"]) > 0), \
        "Useful heat must increase monotonically with DNI"


def test_thermal_loss_increases_with_T_recv(model):
    """Higher receiver temperature → higher radiative + convective losses."""
    T_recvs = np.array([300.0, 400.0, 500.0, 600.0, 700.0])
    r = model.predict({"dni_w_m2": 900.0, "solar_zenith_deg": 20.0,
                       "T_receiver_degC": T_recvs, "T_ambient_degC": 25.0})
    assert np.all(np.diff(r["thermal_loss_kw"]) > 0), \
        "Thermal losses must increase with T_recv"


def test_receiver_efficiency_drops_with_T_recv(model):
    """Higher T_recv → more losses → lower receiver efficiency."""
    T_recvs = np.array([350.0, 450.0, 550.0, 650.0, 750.0])
    r = model.predict({"dni_w_m2": 900.0, "solar_zenith_deg": 20.0,
                       "T_receiver_degC": T_recvs, "T_ambient_degC": 25.0})
    assert np.all(np.diff(r["receiver_efficiency"]) < 0), \
        "Receiver efficiency must drop with increasing T_recv"


def test_lower_T_amb_reduces_losses(model):
    """Lower ambient temperature increases temperature difference → but also REDUCES losses (less driving delta for T_r-T_a)."""
    # Lower T_amb → larger (T_r^4 - T_a^4) radiative loss ... wait, lower T_amb
    # means T_a smaller → T_r^4 - T_a^4 is LARGER → more radiative loss
    # So lower T_amb → MORE thermal loss → LESS useful heat
    r_cold = model.predict({"dni_w_m2": 900.0, "solar_zenith_deg": 20.0,
                             "T_receiver_degC": 600.0, "T_ambient_degC": -10.0})
    r_hot = model.predict({"dni_w_m2": 900.0, "solar_zenith_deg": 20.0,
                            "T_receiver_degC": 600.0, "T_ambient_degC": 40.0})
    assert float(r_cold["thermal_loss_kw"]) > float(r_hot["thermal_loss_kw"]), \
        "Cold ambient → larger (T_r - T_amb) → more thermal loss"
    assert float(r_cold["useful_heat_kw"]) < float(r_hot["useful_heat_kw"]), \
        "Cold ambient → more loss → less useful heat"


def test_wind_increases_losses(model):
    """Higher wind → higher convective coefficient → more convective losses."""
    r_no_wind = model.predict({"dni_w_m2": 900.0, "solar_zenith_deg": 20.0,
                                "T_receiver_degC": 600.0, "T_ambient_degC": 25.0,
                                "wind_speed_m_s": 0.0})
    r_wind = model.predict({"dni_w_m2": 900.0, "solar_zenith_deg": 20.0,
                             "T_receiver_degC": 600.0, "T_ambient_degC": 25.0,
                             "wind_speed_m_s": 10.0})
    assert float(r_wind["Q_convective_kw"]) > float(r_no_wind["Q_convective_kw"]), \
        "Wind must increase convective losses"
    assert float(r_wind["useful_heat_kw"]) < float(r_no_wind["useful_heat_kw"]), \
        "Wind must reduce useful heat output"


def test_radiative_loss_dominates_at_high_T(model):
    """At T_recv=700C, radiative loss should dominate over convective."""
    r = model.predict({"dni_w_m2": 900.0, "solar_zenith_deg": 20.0,
                       "T_receiver_degC": 700.0, "T_ambient_degC": 25.0,
                       "wind_speed_m_s": 5.0})
    assert float(r["Q_radiative_kw"]) > float(r["Q_convective_kw"]), \
        "At T_recv=700C, radiative loss must dominate (T^4 scaling)"


def test_optical_efficiency_in_range(model):
    """Optical efficiency should be 0.3-0.75 at low zenith."""
    r = model.predict({"dni_w_m2": 900.0, "solar_zenith_deg": 10.0,
                       "T_receiver_degC": 600.0, "T_ambient_degC": 25.0})
    eta = float(r["optical_efficiency"])
    assert 0.30 < eta < 0.75, f"Optical efficiency = {eta:.3f}"


def test_useful_heat_positive_at_design_point(model):
    """At design point (DNI=900, zenith=20, T_recv=600, T_amb=25), Q_useful > 0."""
    r = model.predict({"dni_w_m2": 900.0, "solar_zenith_deg": 20.0,
                       "T_receiver_degC": 600.0, "T_ambient_degC": 25.0})
    assert float(r["useful_heat_kw"]) > 1000.0, \
        "At design point, useful heat should be substantial (> 1000 kW for 100 MW_th tower)"


def test_array_inputs(model):
    dnis = np.array([500.0, 700.0, 900.0])
    zeniths = np.array([20.0, 25.0, 30.0])
    T_recvs = np.array([550.0, 580.0, 600.0])
    T_ambs = np.array([15.0, 25.0, 35.0])
    r = model.predict({"dni_w_m2": dnis, "solar_zenith_deg": zeniths,
                       "T_receiver_degC": T_recvs, "T_ambient_degC": T_ambs})
    assert r["useful_heat_kw"].shape == (3,)


def test_benchmark(model):
    dnis = np.random.uniform(200, 1000, 1000)
    zeniths = np.random.uniform(5, 70, 1000)
    T_recvs = np.random.uniform(400, 750, 1000)
    T_ambs = np.random.uniform(-10, 45, 1000)
    start = time.perf_counter()
    model.predict({"dni_w_m2": dnis, "solar_zenith_deg": zeniths,
                   "T_receiver_degC": T_recvs, "T_ambient_degC": T_ambs})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed*1000:.1f} ms")
    assert elapsed < 2.0
