"""EC083 — Borehole TES — F1b Stratified — Test Suite"""
import sys, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()


def test_predict_keys(model):
    r = model.predict({"q_charge_W": 100000.0, "q_discharge_W": 0.0,
                       "t_ambient": 10.0, "duration_s": 3600.0})
    for k in ["T_nodes", "soc", "Q_actual_charge_kw",
              "Q_actual_discharge_kw", "Q_loss_kw", "stratification_index"]:
        assert k in r, f"Missing key: {k}"


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC083"
    assert info["fidelity"] == "F1b"


def test_soc_increases_during_charge(model):
    """SOC must increase during charging from undisturbed ground temperature."""
    T_gnd = model._model.T_ground_far
    r = model.predict({"q_charge_W": 200000.0, "q_discharge_W": 0.0,
                       "t_ambient": T_gnd, "duration_s": 3600.0,
                       "T_initial": T_gnd})
    assert r["soc"] > 0.0, f"SOC must increase during charging, got {r['soc']:.4f}"


def test_soc_decreases_during_discharge(model):
    """SOC must decrease during discharge."""
    T_gnd = model._model.T_ground_far
    T_init = np.linspace(T_gnd + 25, T_gnd + 5, model._model.N)
    r = model.predict({"q_charge_W": 0.0, "q_discharge_W": 150000.0,
                       "t_ambient": T_gnd, "duration_s": 3600.0,
                       "T_initial": list(T_init)})
    soc_before = float(np.clip(
        np.sum((T_init - T_gnd) * model._model.C_node) / (model._model.capacity_kwh * 3.6e6),
        0.0, 1.0))
    assert r["soc"] < soc_before, f"SOC should decrease: {r['soc']:.3f} vs {soc_before:.3f}"


def test_soc_bounded(model):
    """SOC must be in [0, 1]."""
    r = model.predict({"q_charge_W": 1000000.0, "q_discharge_W": 0.0,
                       "t_ambient": 10.0, "duration_s": 10 * 3600})
    assert 0.0 <= r["soc"] <= 1.0


def test_hot_top_after_charge(model):
    """After charging, top nodes should be hotter than bottom nodes.
    RATIONALE: Hot fluid injected at top → upper ground heats first."""
    T_gnd = model._model.T_ground_far
    r = model.predict({"q_charge_W": 300000.0, "q_discharge_W": 0.0,
                       "t_ambient": T_gnd, "duration_s": 5 * 3600,
                       "T_initial": T_gnd, "dt": 300.0})
    T = r["T_nodes"]
    N = len(T)
    T_top = np.mean(T[:N // 2])
    T_bot = np.mean(T[N // 2:])
    assert T_top > T_bot, f"Top should be hotter: T_top={T_top:.1f}, T_bot={T_bot:.1f}"


def test_loss_positive_when_warm(model):
    """Heat loss should be positive when ground is warmer than undisturbed."""
    T_gnd = model._model.T_ground_far
    T_init = T_gnd + 15.0
    r = model.predict({"q_charge_W": 0.0, "q_discharge_W": 0.0,
                       "t_ambient": T_gnd, "duration_s": 3600.0,
                       "T_initial": T_init})
    assert r["Q_loss_kw"] > 0.0, "Positive heat loss when ground warmer than undisturbed"


def test_eta_rt_reduces_effective_charge(model):
    """η_rt (charge-side only) should reduce stored energy vs ideal."""
    eta_rt = model._model.eta_rt
    T_gnd  = model._model.T_ground_far
    r = model.predict({"q_charge_W": 200000.0, "q_discharge_W": 0.0,
                       "t_ambient": T_gnd, "duration_s": 3600.0,
                       "T_initial": T_gnd})
    # Effective charge < input * 1h
    assert r["Q_actual_charge_kw"] <= 200.0, "Effective charge should not exceed input"
    assert r["Q_actual_charge_kw"] > 0.0, "Some charge must occur"


def test_stratification_index_range(model):
    """SI must be in [0, 1]."""
    T_gnd = model._model.T_ground_far
    r = model.predict({"q_charge_W": 200000.0, "q_discharge_W": 0.0,
                       "t_ambient": T_gnd, "duration_s": 3600.0,
                       "T_initial": T_gnd})
    assert 0.0 <= r["stratification_index"] <= 1.0


def test_T_history_shape(model):
    """T_history must have shape (n_steps+1, N)."""
    r = model.predict({"q_charge_W": 100000.0, "q_discharge_W": 0.0,
                       "t_ambient": 10.0, "duration_s": 1800.0, "dt": 300.0})
    Th = r["T_history"]
    N  = model._model.N
    assert Th.shape[1] == N
    assert Th.shape[0] == 7  # 1800/300 + 1
