"""EC082 — Ice TES — F1b Stratified — Test Suite"""
import sys, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()


def test_predict_keys(model):
    r = model.predict({"q_charge_W": 50000.0, "q_discharge_W": 0.0,
                       "t_ambient": 20.0, "duration_s": 3600.0})
    for k in ["f_nodes", "soc", "Q_actual_charge_kw",
              "Q_actual_discharge_kw", "Q_loss_kw", "stratification_index"]:
        assert k in r, f"Missing key: {k}"


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC082"
    assert info["fidelity"] == "F1b"


def test_ice_forms_bottom_up_during_charge(model):
    """During charging, bottom nodes (index 0) should have MORE ice than top nodes.
    RATIONALE: Coils at base → freezing front propagates upward (ASHRAE ch.51,
    MacPhee & Dincer 2009)."""
    r = model.predict({"q_charge_W": 80000.0, "q_discharge_W": 0.0,
                       "t_ambient": 20.0, "duration_s": 3 * 3600,
                       "f_initial": 0.0, "dt": 60.0})
    f = r["f_nodes"]
    N = len(f)
    # Bottom half should have more ice than top half on average
    f_bottom = np.mean(f[:N // 2])
    f_top    = np.mean(f[N // 2:])
    assert f_bottom > f_top, \
        f"Ice should be more at bottom: f_bottom={f_bottom:.3f}, f_top={f_top:.3f}"


def test_ice_melts_top_down_during_discharge(model):
    """During discharging, top nodes lose ice first.
    RATIONALE: Warm return fluid contacts upper ice surface first
    (Jekel et al. 1993, ASHRAE TES design)."""
    N = model._model.N
    f_full = np.ones(N)   # start fully frozen

    r = model.predict({"q_charge_W": 0.0, "q_discharge_W": 100000.0,
                       "t_ambient": 20.0, "duration_s": 2 * 3600,
                       "f_initial": f_full, "dt": 60.0})
    f = r["f_nodes"]
    # Top nodes should have less ice than bottom nodes
    f_top    = np.mean(f[N // 2:])
    f_bottom = np.mean(f[:N // 2])
    assert f_top < f_bottom, \
        f"Ice should melt top-down: f_top={f_top:.3f}, f_bottom={f_bottom:.3f}"


def test_soc_increases_during_charge(model):
    """SOC must increase during pure charging."""
    r0 = model.predict({"q_charge_W": 0.0, "q_discharge_W": 0.0,
                        "t_ambient": 20.0, "duration_s": 60.0,
                        "f_initial": 0.0})
    r1 = model.predict({"q_charge_W": 80000.0, "q_discharge_W": 0.0,
                        "t_ambient": 20.0, "duration_s": 3600.0,
                        "f_initial": 0.0})
    assert r1["soc"] > r0["soc"], "SOC must increase during charging"


def test_soc_decreases_during_discharge(model):
    """SOC must decrease during pure discharge from charged state."""
    N = model._model.N
    f_half = np.zeros(N)
    f_half[:N // 2] = 1.0   # bottom half frozen

    r = model.predict({"q_charge_W": 0.0, "q_discharge_W": 80000.0,
                       "t_ambient": 20.0, "duration_s": 3600.0,
                       "f_initial": f_half})
    assert r["soc"] < 0.5, f"SOC should decrease; got {r['soc']:.3f}"


def test_soc_bounded(model):
    """SOC must stay in [0, 1] at all times."""
    r = model.predict({"q_charge_W": 200000.0, "q_discharge_W": 0.0,
                       "t_ambient": 20.0, "duration_s": 10 * 3600,
                       "f_initial": 1.0})
    assert 0.0 <= r["soc"] <= 1.0
    f = r["f_nodes"]
    assert np.all(f >= 0.0) and np.all(f <= 1.0)


def test_heat_loss_positive_when_ambient_warm(model):
    """Heat loss from warm ambient must be positive (ambient melts ice)."""
    r = model.predict({"q_charge_W": 0.0, "q_discharge_W": 0.0,
                       "t_ambient": 20.0, "duration_s": 3600.0,
                       "f_initial": 0.5})
    assert r["Q_loss_kw"] > 0.0, "Heat infiltration must be positive when T_amb > T_pc"


def test_stratification_index_range(model):
    """Stratification index must be in [0, 1]."""
    N = model._model.N
    for f_init in [0.0, 0.5, 1.0]:
        r = model.predict({"q_charge_W": 60000.0, "q_discharge_W": 0.0,
                           "t_ambient": 20.0, "duration_s": 3600.0,
                           "f_initial": f_init})
        SI = r["stratification_index"]
        assert 0.0 <= SI <= 1.0, f"SI={SI:.3f} out of range at f_init={f_init}"


def test_stratification_higher_after_charge_than_uniform(model):
    """After charging from zero, bottom-heavy distribution should give SI > 0."""
    r = model.predict({"q_charge_W": 80000.0, "q_discharge_W": 0.0,
                       "t_ambient": 20.0, "duration_s": 2 * 3600,
                       "f_initial": 0.0, "dt": 60.0})
    assert r["stratification_index"] > 0.0, \
        "Bottom-up charging should yield positive stratification index"


def test_no_charge_no_discharge_soc_constant(model):
    """With no charge/discharge and minimal heat loss, SOC should stay ~constant."""
    # Use T_amb = 0 (same as T_phase_change) to avoid heat loss melting ice
    r = model.predict({"q_charge_W": 0.0, "q_discharge_W": 0.0,
                       "t_ambient": 0.0, "duration_s": 3600.0,
                       "f_initial": 0.3, "dt": 60.0})
    # SOC should be very close to 0.3
    assert abs(r["soc"] - 0.3) < 0.05, f"SOC drifted: {r['soc']:.3f} vs 0.30"


def test_f_history_shape(model):
    """f_history must have shape (n_steps+1, N_nodes)."""
    r = model.predict({"q_charge_W": 50000.0, "q_discharge_W": 0.0,
                       "t_ambient": 20.0, "duration_s": 600.0, "dt": 60.0})
    fh = r["f_history"]
    N  = model._model.N
    assert fh.shape[1] == N
    assert fh.shape[0] == 11   # 600/60 + 1
