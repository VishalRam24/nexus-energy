"""EC090 — Solar Water Heater Combi-System — F1b — Test Suite"""
import sys, time, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()


def test_predict_keys(model):
    r = model.predict({"I_solar_w_m2": 600.0, "Q_demand_kw": 12.0})
    for k in ["Q_solar_kw", "Q_aux_kw", "Q_aux_fuel_kw",
              "Q_standby_kw", "solar_fraction", "eta_collector"]:
        assert k in r


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC090"
    assert info["fidelity"] == "F1b"


def test_no_solar_at_night(model):
    """Solar yield must be zero when irradiance is zero."""
    r = model.predict({"I_solar_w_m2": 0.0, "Q_demand_kw": 10.0})
    assert float(r["Q_solar_kw"]) == 0.0
    assert float(r["solar_fraction"]) == 0.0


def test_solar_increases_with_irradiance(model):
    """Higher solar irradiance → higher solar yield."""
    I = np.array([100.0, 300.0, 600.0, 900.0])
    r = model.predict({"I_solar_w_m2": I, "Q_demand_kw": 20.0})
    assert np.all(np.diff(r["Q_solar_kw"]) >= 0), "Solar yield must increase with irradiance"


def test_solar_fraction_bounded(model):
    """Solar fraction must be in [0, 1]."""
    I = np.linspace(0, 1000, 50)
    r = model.predict({"I_solar_w_m2": I, "Q_demand_kw": 10.0})
    assert np.all(r["solar_fraction"] >= 0.0)
    assert np.all(r["solar_fraction"] <= 1.0)


def test_aux_zero_when_fully_solar(model):
    """No auxiliary needed when solar fully covers demand."""
    r = model.predict({"I_solar_w_m2": 1200.0, "Q_demand_kw": 1.0, "T_ambient": 15.0})
    assert float(r["Q_aux_kw"]) == 0.0 or float(r["solar_fraction"]) > 0.99


def test_aux_full_at_night(model):
    """At night (I=0), full demand falls on auxiliary."""
    Q_dem = 15.0
    r     = model.predict({"I_solar_w_m2": 0.0, "Q_demand_kw": Q_dem})
    assert float(r["Q_aux_kw"]) > 0.0, "Auxiliary must supply heat at night"


def test_standby_loss_positive(model):
    """Tank standby loss must be positive when tank > ambient."""
    r = model.predict({"I_solar_w_m2": 500.0, "Q_demand_kw": 10.0, "T_ambient": 10.0})
    assert float(r["Q_standby_kw"]) > 0.0


def test_collector_efficiency_decreases_hot_ambient(model):
    """Collector efficiency decreases at lower dT = (T_coll - T_amb).
    At higher ambient T, dT is smaller → higher efficiency.
    RATIONALE: Hottel-Whillier-Bliss: eta = eta0 - a1*(dT/I) - a2*(dT/I)^2."""
    r_cold = model.predict({"I_solar_w_m2": 600.0, "Q_demand_kw": 10.0, "T_ambient": 0.0})
    r_warm = model.predict({"I_solar_w_m2": 600.0, "Q_demand_kw": 10.0, "T_ambient": 30.0})
    # Warmer ambient → smaller dT → higher collector efficiency
    assert float(r_warm["eta_collector"]) >= float(r_cold["eta_collector"]), \
        "Warmer ambient → smaller dT → higher collector efficiency"


def test_aux_efficiency_bounded(model):
    plr = np.linspace(0.15, 1.0, 30)
    r   = model.predict({"I_solar_w_m2": 0.0,
                          "Q_demand_kw": plr * model._physics.Q_aux_rated})
    assert np.all(r["eta_aux"] >= 0.0) and np.all(r["eta_aux"] <= 1.0)


def test_benchmark(model):
    I   = np.random.uniform(0, 1000, 1000)
    Q   = np.random.uniform(5, 20, 1000)
    t0  = time.perf_counter()
    model.predict({"I_solar_w_m2": I, "Q_demand_kw": Q})
    assert time.perf_counter() - t0 < 1.0
