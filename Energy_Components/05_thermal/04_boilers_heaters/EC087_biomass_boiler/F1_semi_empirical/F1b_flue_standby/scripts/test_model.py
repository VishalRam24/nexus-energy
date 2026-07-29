"""EC087 — Biomass Boiler — F1b Flue + Moisture + Cycling Standby — Test Suite

Tests MUST fail the model, not accommodate it.
"""
import sys, time, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()


# --- Interface ---

def test_predict_keys(model):
    r = model.predict({"PLR": 0.5})
    for k in ["efficiency", "heat_output_kw", "fuel_input_kw", "flue_loss_kw",
              "standby_loss_kw", "cycling_loss_kw", "flue_gas_temp_c"]:
        assert k in r, f"Missing key: {k}"


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC087"
    assert info["fidelity"] == "F1b"


# --- Efficiency ---

def test_efficiency_peaks_near_full_load(model):
    """With a2 < 0, efficiency peaks mid/high load, not at extremes."""
    plr = np.linspace(0.15, 1.0, 100)
    r = model.predict({"PLR": plr})
    eta = np.asarray(r["efficiency"])
    peak_idx = np.argmax(eta)
    # Peak should not be at the minimum PLR
    assert peak_idx > 5, f"Peak efficiency at low PLR={plr[peak_idx]:.2f} — unexpected"
    assert float(np.max(eta)) > 0.80, "Max efficiency should exceed 0.80"


def test_efficiency_drops_at_very_low_plr(model):
    """Efficiency at PLR=0.15 should be lower than at PLR=0.9."""
    r_low  = model.predict({"PLR": 0.15})
    r_high = model.predict({"PLR": 0.9})
    assert float(r_low["efficiency"]) < float(r_high["efficiency"])


def test_efficiency_bounded(model):
    plr = np.linspace(0.15, 1.0, 50)
    r = model.predict({"PLR": plr})
    assert np.all(np.asarray(r["efficiency"]) >= 0.0)
    assert np.all(np.asarray(r["efficiency"]) <= 1.0)


# --- Flue gas loss ---

def test_flue_loss_positive(model):
    """Biomass combustion always produces hot flue gas — loss must be > 0."""
    r = model.predict({"PLR": 0.5})
    assert float(r["flue_loss_kw"]) > 0.0


def test_flue_loss_increases_with_plr(model):
    """Higher firing rate = higher fuel flow = more flue gas = higher loss."""
    plr = np.array([0.2, 0.4, 0.6, 0.8, 1.0])
    r = model.predict({"PLR": plr})
    flue = np.asarray(r["flue_loss_kw"])
    assert np.all(np.diff(flue) > 0), "Flue loss must be monotonically increasing with PLR"


def test_flue_temp_increases_with_plr(model):
    plr = np.array([0.2, 0.5, 0.8, 1.0])
    r = model.predict({"PLR": plr})
    temps = np.asarray(r["flue_gas_temp_c"])
    assert np.all(np.diff(temps) > 0), "Flue temperature must rise with PLR"


def test_moisture_correction_reduces_efficiency(model):
    """Higher moisture content should reduce efficiency (moisture evaporation energy loss)."""
    m_dry  = ComponentModel({"moisture_content": 0.05})
    m_wet  = ComponentModel({"moisture_content": 0.30})
    r_dry  = m_dry.predict({"PLR": 0.8})
    r_wet  = m_wet.predict({"PLR": 0.8})
    assert float(r_dry["efficiency"]) > float(r_wet["efficiency"]), \
        "Dry fuel should have higher efficiency than wet fuel"


def test_moisture_increases_flue_loss(model):
    """More moisture => more flue mass flow => higher flue loss."""
    m_dry = ComponentModel({"moisture_content": 0.05})
    m_wet = ComponentModel({"moisture_content": 0.30})
    r_dry = m_dry.predict({"PLR": 0.8})
    r_wet = m_wet.predict({"PLR": 0.8})
    assert float(r_wet["flue_loss_kw"]) > float(r_dry["flue_loss_kw"]), \
        "Wetter fuel should produce higher flue loss"


# --- Standby loss ---

def test_standby_loss_positive(model):
    r = model.predict({"PLR": 0.5})
    assert float(r["standby_loss_kw"]) > 0.0


def test_standby_loss_constant(model):
    """Standby is a constant fraction of Q_rated — PLR-independent."""
    plr = np.array([0.2, 0.5, 1.0])
    r = model.predict({"PLR": plr})
    sb = np.asarray(r["standby_loss_kw"])
    assert np.allclose(sb, sb[0]), "Standby loss must be constant across PLR"


# --- Cycling loss ---

def test_cycling_loss_higher_at_low_plr(model):
    """Low PLR = frequent cycling = higher cycling loss."""
    r_low  = model.predict({"PLR": 0.15})
    r_high = model.predict({"PLR": 1.0})
    assert float(r_low["cycling_loss_kw"]) > float(r_high["cycling_loss_kw"]), \
        "Cycling loss should be higher at low PLR"


def test_cycling_loss_zero_at_plr_one(model):
    """At PLR=1 there is no cycling (continuous firing)."""
    r = model.predict({"PLR": 1.0})
    assert float(r["cycling_loss_kw"]) == pytest.approx(0.0, abs=1e-9)


# --- Energy balance ---

def test_fuel_exceeds_heat_output(model):
    plr = np.linspace(0.15, 1.0, 50)
    r = model.predict({"PLR": plr})
    assert np.all(np.asarray(r["fuel_input_kw"]) >= np.asarray(r["heat_output_kw"]) - 1e-6)


# --- Custom flue temp ---

def test_flue_override_changes_loss(model):
    r1 = model.predict({"PLR": 0.5})
    r2 = model.predict({"PLR": 0.5, "T_flue_override": 250.0})
    assert float(r2["flue_loss_kw"]) > float(r1["flue_loss_kw"]), \
        "Higher overridden flue temp must increase flue loss"


# --- Benchmark ---

def test_benchmark(model):
    plr = np.random.uniform(0.15, 1.0, 1000)
    start = time.perf_counter()
    model.predict({"PLR": plr})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed*1000:.2f} ms")
    assert elapsed < 1.0
