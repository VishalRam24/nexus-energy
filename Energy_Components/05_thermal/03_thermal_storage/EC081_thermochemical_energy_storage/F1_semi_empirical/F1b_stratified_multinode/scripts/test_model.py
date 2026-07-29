"""EC081 — Thermochemical TES — F1b Stratified — Test Suite"""
import sys, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()


def test_predict_keys(model):
    r = model.predict({"q_charge_W": 80000.0, "q_discharge_W": 0.0,
                       "t_ambient": 20.0, "duration_s": 3600.0})
    for k in ["x_nodes", "soc", "Q_actual_charge_kw",
              "Q_actual_discharge_kw", "Q_loss_kw", "stratification_index"]:
        assert k in r, f"Missing key: {k}"


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC081"
    assert info["fidelity"] == "F1b"


def test_soc_increases_during_charge(model):
    """SOC (dehydration extent) must increase during pure charging."""
    r = model.predict({"q_charge_W": 100000.0, "q_discharge_W": 0.0,
                       "t_ambient": 20.0, "duration_s": 3600.0,
                       "x_initial": 0.0})
    assert r["soc"] > 0.0, f"SOC must increase during charging, got {r['soc']}"


def test_soc_decreases_during_discharge(model):
    """SOC must decrease during pure discharge from charged state."""
    r = model.predict({"q_charge_W": 0.0, "q_discharge_W": 80000.0,
                       "t_ambient": 20.0, "duration_s": 3600.0,
                       "x_initial": 0.8})
    assert r["soc"] < 0.8, f"SOC should decrease, got {r['soc']:.3f}"


def test_soc_bounded(model):
    """SOC must stay in [0, 1]."""
    r = model.predict({"q_charge_W": 300000.0, "q_discharge_W": 0.0,
                       "t_ambient": 20.0, "duration_s": 10 * 3600,
                       "x_initial": 1.0})
    assert 0.0 <= r["soc"] <= 1.0
    assert np.all(r["x_nodes"] >= 0.0) and np.all(r["x_nodes"] <= 1.0)


def test_charge_fills_bottom_first(model):
    """Charging fills bottom nodes first (reaction front propagates up).
    RATIONALE: Inlet flow enters at base of sorbent bed; dehydration begins at inlet."""
    r = model.predict({"q_charge_W": 80000.0, "q_discharge_W": 0.0,
                       "t_ambient": 20.0, "duration_s": 3 * 3600,
                       "x_initial": 0.0, "dt": 60.0})
    x = r["x_nodes"]
    N = len(x)
    x_bottom = np.mean(x[:N // 2])
    x_top    = np.mean(x[N // 2:])
    assert x_bottom > x_top, \
        f"Charge should fill bottom first: x_bottom={x_bottom:.3f}, x_top={x_top:.3f}"


def test_discharge_draws_from_top(model):
    """Discharge draws from top nodes first.
    RATIONALE: Hot hydration reactant contacts top of bed first."""
    N = model._model.N
    r = model.predict({"q_charge_W": 0.0, "q_discharge_W": 100000.0,
                       "t_ambient": 20.0, "duration_s": 2 * 3600,
                       "x_initial": 1.0, "dt": 60.0})
    x = r["x_nodes"]
    x_top    = np.mean(x[N // 2:])
    x_bottom = np.mean(x[:N // 2])
    assert x_top < x_bottom, \
        f"Discharge should draw top-down: x_top={x_top:.3f}, x_bottom={x_bottom:.3f}"


def test_eta_rt_applied_charge_side_only(model):
    """Round-trip efficiency reduces effective charge power, not discharge.
    RATIONALE: η_rt on charge side only avoids double-counting (EC082 fix pattern)."""
    N = model._model.N
    eta_rt = model._model.eta_rt

    # Charge 1 hour at 100 kW, then check how much was actually stored
    r_ch = model.predict({"q_charge_W": 100000.0, "q_discharge_W": 0.0,
                          "t_ambient": 0.0, "duration_s": 3600.0,
                          "x_initial": 0.0, "dt": 60.0})
    # Effective charge power should be reduced by eta_rt
    assert r_ch["Q_actual_charge_kw"] < 100.0, \
        "η_rt should reduce effective charge power below input"
    # Approximate: Q_actual ~ 100 kW * eta_rt (minus some loss)
    assert r_ch["Q_actual_charge_kw"] <= 100.0 * eta_rt + 1.0, \
        f"Effective charge Q should not exceed 100*eta_rt={100*eta_rt:.1f} kW"


def test_thermal_loss_positive_when_hot(model):
    """Heat loss should be positive when reaction temperature > ambient."""
    r = model.predict({"q_charge_W": 0.0, "q_discharge_W": 0.0,
                       "t_ambient": 10.0, "duration_s": 3600.0,
                       "x_initial": 0.5})
    assert r["Q_loss_kw"] > 0.0, "Heat loss must be positive when reactor > ambient"


def test_stratification_index_range(model):
    """Stratification index must be in [0, 1]."""
    for x_init in [0.0, 0.3, 0.6, 1.0]:
        r = model.predict({"q_charge_W": 60000.0, "q_discharge_W": 0.0,
                           "t_ambient": 20.0, "duration_s": 1800.0,
                           "x_initial": x_init})
        SI = r["stratification_index"]
        assert 0.0 <= SI <= 1.0, f"SI={SI:.3f} out of range for x_init={x_init}"


def test_x_history_shape(model):
    """x_history must have shape (n_steps+1, N_nodes)."""
    r = model.predict({"q_charge_W": 50000.0, "q_discharge_W": 0.0,
                       "t_ambient": 20.0, "duration_s": 600.0, "dt": 60.0})
    xh = r["x_history"]
    N  = model._model.N
    assert xh.shape[1] == N
    assert xh.shape[0] == 11  # 600/60 + 1


def test_no_charge_no_discharge_stable(model):
    """With no charge/discharge and minimal loss, SOC stays ~constant."""
    x0 = 0.4
    # Set ambient = reactor T to eliminate loss
    r = model.predict({"q_charge_W": 0.0, "q_discharge_W": 0.0,
                       "t_ambient": model._model.T_amb_default + 80.0,
                       "duration_s": 3600.0,
                       "x_initial": x0, "dt": 60.0})
    assert abs(r["soc"] - x0) < 0.05, f"SOC drifted unexpectedly: {r['soc']:.3f} vs {x0}"
