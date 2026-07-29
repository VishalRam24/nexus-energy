"""EC086 — Electric Boiler — F1b Standby Loss + Ambient — Test Suite

Tests MUST fail the model, not accommodate it.
Loosening requires # RATIONALE: comment.
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
    for k in ["efficiency", "electrical_input_kw", "heat_output_kw",
              "standby_loss_kw", "flue_loss_kw", "controls_kw"]:
        assert k in r, f"Missing key: {k}"


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC086"
    assert info["fidelity"] == "F1b"


# --- Flue loss zero ---

def test_flue_loss_is_zero(model):
    """Electric boiler has no combustion — flue loss must be identically zero."""
    plr = np.linspace(0.05, 1.0, 50)
    r = model.predict({"PLR": plr})
    assert np.all(np.asarray(r["flue_loss_kw"]) == 0.0), \
        "flue_loss_kw must be zero for electric boiler"


# --- Standby loss ---

def test_standby_loss_positive_at_design(model):
    """Standby loss from thermal mass must be > 0 at design conditions (T_fluid > T_amb)."""
    r = model.predict({"PLR": 0.5})
    assert float(r["standby_loss_kw"]) > 0.0, "Standby loss should be positive"


def test_standby_increases_with_cold_ambient(model):
    """Colder ambient => larger dT => higher standby loss."""
    r_warm = model.predict({"PLR": 0.5, "T_ambient": 25.0})
    r_cold = model.predict({"PLR": 0.5, "T_ambient": -10.0})
    assert float(r_cold["standby_loss_kw"]) > float(r_warm["standby_loss_kw"]), \
        "Cold ambient should increase standby loss"


def test_standby_zero_at_zero_dT(model):
    """When T_fluid == T_ambient, standby loss is zero."""
    T = 20.0
    r = model.predict({"PLR": 0.5, "T_ambient": T, "T_fluid": T})
    assert abs(float(r["standby_loss_kw"])) < 1e-9, \
        "Standby loss must be zero when dT=0"


def test_standby_not_plr_dependent(model):
    """Standby loss depends only on temperatures, not PLR."""
    r1 = model.predict({"PLR": 0.2, "T_ambient": 10.0})
    r2 = model.predict({"PLR": 0.8, "T_ambient": 10.0})
    assert abs(float(r1["standby_loss_kw"]) - float(r2["standby_loss_kw"])) < 1e-9, \
        "Standby loss must be PLR-independent"


# --- Efficiency ---

def test_efficiency_near_eta_nom_at_full_load(model):
    """At high PLR, effective eta should be close to eta_nom (standby is small fraction)."""
    r = model.predict({"PLR": 1.0})
    eta = float(r["efficiency"])
    assert eta > 0.95, f"At full load eta={eta:.4f} should be close to eta_nom=0.99"


def test_efficiency_drops_with_cold_ambient(model):
    """Cold ambient raises standby loss, reducing effective efficiency."""
    r_warm = model.predict({"PLR": 0.2, "T_ambient": 20.0})
    r_cold = model.predict({"PLR": 0.2, "T_ambient": -10.0})
    assert float(r_cold["efficiency"]) < float(r_warm["efficiency"]), \
        "Cold ambient should degrade effective efficiency"


def test_efficiency_bounded(model):
    """Efficiency in [0, 1] across entire PLR range."""
    plr = np.linspace(0.05, 1.0, 50)
    r = model.predict({"PLR": plr})
    eta = np.asarray(r["efficiency"])
    assert np.all(eta >= 0.0) and np.all(eta <= 1.0), "Efficiency out of [0,1]"


# --- Energy balance ---

def test_heat_output_less_than_electrical_input(model):
    """Q_out <= P_in (first law; no COP > 1 for resistance heater)."""
    plr = np.linspace(0.05, 1.0, 50)
    r = model.predict({"PLR": plr})
    Q = np.asarray(r["heat_output_kw"])
    P = np.asarray(r["electrical_input_kw"])
    assert np.all(Q <= P + 1e-9), "Heat output exceeds electrical input"


def test_heat_output_increases_with_plr(model):
    """Higher PLR => higher heat output."""
    plr = np.array([0.1, 0.3, 0.5, 0.8, 1.0])
    r = model.predict({"PLR": plr})
    Q = np.asarray(r["heat_output_kw"])
    assert np.all(np.diff(Q) > 0), "Heat output must increase with PLR"


def test_controls_constant(model):
    """Controls parasitic is constant regardless of PLR."""
    plr = np.array([0.1, 0.5, 1.0])
    r = model.predict({"PLR": plr})
    ctrl = np.asarray(r["controls_kw"])
    assert np.allclose(ctrl, ctrl[0]), "Controls draw must be constant"


# --- Benchmark ---

def test_benchmark(model):
    plr = np.random.uniform(0.05, 1.0, 1000)
    start = time.perf_counter()
    model.predict({"PLR": plr})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed*1000:.2f} ms")
    assert elapsed < 1.0
