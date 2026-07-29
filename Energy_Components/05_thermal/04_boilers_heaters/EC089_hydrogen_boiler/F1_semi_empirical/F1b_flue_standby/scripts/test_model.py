"""EC089 — Hydrogen Boiler — F1b H2O-rich Flue + Condensing — Test Suite

Tests MUST fail the model, not accommodate it.
"""
import sys, time, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()  # condensing=True by default


@pytest.fixture
def model_noncond():
    return ComponentModel({"condensing": False})


# --- Interface ---

def test_predict_keys(model):
    r = model.predict({"PLR": 0.5})
    for k in ["efficiency", "heat_output_kw", "fuel_input_kw", "flue_loss_kw",
              "latent_recovery_kw", "standby_loss_kw", "h2_flow_kg_s",
              "flue_gas_temp_c", "condensing"]:
        assert k in r, f"Missing key: {k}"


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC089"
    assert info["fidelity"] == "F1b"


# --- Efficiency ---

def test_efficiency_drops_at_low_plr(model):
    r_low  = model.predict({"PLR": 0.1})
    r_high = model.predict({"PLR": 0.9})
    assert float(r_low["efficiency"]) < float(r_high["efficiency"])


def test_efficiency_bounded(model):
    plr = np.linspace(0.1, 1.0, 50)
    r = model.predict({"PLR": plr})
    eta = np.asarray(r["efficiency"])
    assert np.all(eta >= 0.0) and np.all(eta <= 1.0)


def test_fuel_exceeds_heat_output(model):
    plr = np.linspace(0.1, 1.0, 50)
    r = model.predict({"PLR": plr})
    assert np.all(np.asarray(r["fuel_input_kw"]) >= np.asarray(r["heat_output_kw"]) - 1e-6)


# --- Flue gas — H2O-rich ---

def test_flue_loss_positive(model):
    """H2 combustion must produce hot H2O-rich exhaust — flue loss > 0."""
    r = model.predict({"PLR": 0.5})
    assert float(r["flue_loss_kw"]) > 0.0


def test_flue_loss_increases_with_plr(model):
    plr = np.array([0.2, 0.4, 0.6, 0.8, 1.0])
    r = model.predict({"PLR": plr})
    flue = np.asarray(r["flue_loss_kw"])
    assert np.all(np.diff(flue) > 0), "Flue loss must increase with firing rate"


def test_flue_temp_above_ambient(model):
    """Flue temperature must always exceed ambient."""
    plr = np.linspace(0.1, 1.0, 50)
    r = model.predict({"PLR": plr})
    T_flue = np.asarray(r["flue_gas_temp_c"])
    T_amb = model._params["T_ambient"]
    assert np.all(T_flue > T_amb), "Flue temperature must exceed ambient"


def test_condensing_lower_flue_temp_than_noncond(model, model_noncond):
    """Condensing mode: flue exits at lower temperature than non-condensing."""
    r_c = model.predict({"PLR": 1.0})
    r_n = model_noncond.predict({"PLR": 1.0})
    assert float(r_c["flue_gas_temp_c"]) < float(r_n["flue_gas_temp_c"]), \
        "Condensing flue temp must be lower than non-condensing"


def test_condensing_lower_flue_loss(model, model_noncond):
    """Condensing mode recovers latent heat, net flue loss should be lower."""
    r_c = model.predict({"PLR": 1.0})
    r_n = model_noncond.predict({"PLR": 1.0})
    assert float(r_c["flue_loss_kw"]) < float(r_n["flue_loss_kw"]), \
        "Condensing boiler should have lower flue loss"


# --- Latent recovery ---

def test_latent_recovery_nonzero_condensing(model):
    """In condensing mode, latent heat recovery must be positive."""
    r = model.predict({"PLR": 0.5})
    assert float(r["latent_recovery_kw"]) > 0.0, \
        "Condensing mode must recover latent heat"


def test_latent_recovery_zero_noncond(model_noncond):
    """Non-condensing mode: latent recovery must be zero."""
    r = model_noncond.predict({"PLR": 0.5})
    assert float(r["latent_recovery_kw"]) == 0.0, \
        "Non-condensing mode must have zero latent recovery"


def test_latent_recovery_increases_with_plr(model):
    """More H2 burned => more H2O produced => more condensate => more recovery."""
    plr = np.array([0.2, 0.5, 0.8, 1.0])
    r = model.predict({"PLR": plr})
    lat = np.asarray(r["latent_recovery_kw"])
    assert np.all(np.diff(lat) > 0), "Latent recovery must increase with PLR"


# --- Standby loss ---

def test_standby_loss_positive(model):
    r = model.predict({"PLR": 0.5})
    assert float(r["standby_loss_kw"]) > 0.0


def test_standby_constant(model):
    plr = np.array([0.1, 0.5, 1.0])
    r = model.predict({"PLR": plr})
    sb = np.asarray(r["standby_loss_kw"])
    assert np.allclose(sb, sb[0])


# --- H2 physics ---

def test_h2_flow_increases_with_plr(model):
    plr = np.array([0.2, 0.5, 1.0])
    r = model.predict({"PLR": plr})
    h2 = np.asarray(r["h2_flow_kg_s"])
    assert np.all(np.diff(h2) > 0)


# --- Benchmark ---

def test_benchmark(model):
    plr = np.random.uniform(0.1, 1.0, 1000)
    start = time.perf_counter()
    model.predict({"PLR": plr})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed*1000:.2f} ms")
    assert elapsed < 1.0
