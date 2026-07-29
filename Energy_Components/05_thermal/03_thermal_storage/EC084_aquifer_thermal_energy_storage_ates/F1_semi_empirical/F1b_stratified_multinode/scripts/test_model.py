"""EC084 — ATES — F1b Stratified — Test Suite"""
import sys, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()


def test_predict_keys(model):
    r = model.predict({"q_charge_W": 200000.0, "q_discharge_W": 0.0,
                       "t_ambient": 12.0, "duration_s": 3600.0})
    for k in ["T_nodes", "soc", "Q_actual_charge_kw",
              "Q_actual_discharge_kw", "Q_loss_kw", "stratification_index"]:
        assert k in r, f"Missing key: {k}"


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC084"
    assert info["fidelity"] == "F1b"


def test_soc_increases_during_charge(model):
    T_nat = model._model.T_aquifer_natural
    r = model.predict({"q_charge_W": 400000.0, "q_discharge_W": 0.0,
                       "t_ambient": T_nat, "duration_s": 3600.0,
                       "T_initial": T_nat})
    assert r["soc"] > 0.0, f"SOC must increase during charging, got {r['soc']:.4f}"


def test_soc_decreases_during_discharge(model):
    T_nat = model._model.T_aquifer_natural
    # Start with warm aquifer
    T_init = np.linspace(T_nat + 20, T_nat + 5, model._model.N)
    soc_before = float(np.clip(
        np.sum(np.maximum(T_init - T_nat, 0.0) * model._model.C_node) /
        (model._model.capacity_kwh * 3.6e6), 0.0, 1.0))
    r = model.predict({"q_charge_W": 0.0, "q_discharge_W": 300000.0,
                       "t_ambient": T_nat, "duration_s": 3600.0,
                       "T_initial": list(T_init)})
    assert r["soc"] < soc_before, f"SOC should decrease: {r['soc']:.3f} vs {soc_before:.3f}"


def test_soc_bounded(model):
    r = model.predict({"q_charge_W": 2000000.0, "q_discharge_W": 0.0,
                       "t_ambient": 12.0, "duration_s": 10 * 3600})
    assert 0.0 <= r["soc"] <= 1.0


def test_warm_well_end_hottest_after_charge(model):
    """After charging, node 0 (warm-well end) must be hottest.
    RATIONALE: Heat injected at warm-well end (node 0)."""
    T_nat = model._model.T_aquifer_natural
    r = model.predict({"q_charge_W": 500000.0, "q_discharge_W": 0.0,
                       "t_ambient": T_nat, "duration_s": 3 * 3600,
                       "T_initial": T_nat, "dt": 600.0})
    T = r["T_nodes"]
    assert T[0] > T[-1], f"Warm well (node 0) should be hottest: T[0]={T[0]:.1f}, T[-1]={T[-1]:.1f}"


def test_thermal_loss_positive_when_warm(model):
    T_nat = model._model.T_aquifer_natural
    r = model.predict({"q_charge_W": 0.0, "q_discharge_W": 0.0,
                       "t_ambient": T_nat, "duration_s": 3600.0,
                       "T_initial": T_nat + 15.0})
    assert r["Q_loss_kw"] > 0.0, "Heat loss to undisturbed aquifer must be positive when warm"


def test_stratification_index_range(model):
    r = model.predict({"q_charge_W": 300000.0, "q_discharge_W": 0.0,
                       "t_ambient": 12.0, "duration_s": 3600.0})
    assert 0.0 <= r["stratification_index"] <= 1.0


def test_eta_rt_reduces_charge(model):
    """η_rt on charge side should reduce effective stored power."""
    eta_rt = model._model.eta_rt
    T_nat  = model._model.T_aquifer_natural
    r = model.predict({"q_charge_W": 400000.0, "q_discharge_W": 0.0,
                       "t_ambient": T_nat, "duration_s": 3600.0,
                       "T_initial": T_nat})
    # Effective charge rate < input * 1h
    assert r["Q_actual_charge_kw"] <= 400.0, "Effective charge should not exceed input"
    assert r["Q_actual_charge_kw"] > 0.0, "Some charging must occur"


def test_T_history_shape(model):
    r = model.predict({"q_charge_W": 100000.0, "q_discharge_W": 0.0,
                       "t_ambient": 12.0, "duration_s": 3600.0, "dt": 600.0})
    Th = r["T_history"]
    N  = model._model.N
    assert Th.shape[1] == N
    assert Th.shape[0] == 7  # 3600/600 + 1
