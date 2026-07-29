"""EC130 — Small/Micro Hydropower — F1b Head-Flow — Test Suite

Phase 7 note: use 30% flow for part-load test (70% flow risks hitting P_rated cap).
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
    r = model.predict({"flow_rate_m3s": 0.9, "gross_head_m": 45.0})
    for k in ["power_kw", "efficiency", "capacity_factor", "net_head_m",
              "head_loss_fraction", "cavitation_derate", "turbine_type_used"]:
        assert k in r


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC130"
    assert info["fidelity"] == "F1b"


# --- Turbine auto-selection ---

def test_auto_pelton_high_head(model):
    r = model.predict({"flow_rate_m3s": 0.5, "gross_head_m": 200.0})
    assert r["turbine_type_used"] == "pelton"


def test_auto_kaplan_low_head(model):
    r = model.predict({"flow_rate_m3s": 2.0, "gross_head_m": 5.0})
    assert r["turbine_type_used"] == "kaplan"


def test_auto_francis_mid_head(model):
    r = model.predict({"flow_rate_m3s": 1.3, "gross_head_m": 45.0})
    assert r["turbine_type_used"] == "francis"


# --- Part-load (30% flow, Phase 7 validated) ---

def test_part_load_30pct_flow_positive_power(model):
    """
    Phase 7 note: 70% flow hits P_rated cap. Use 30% flow for part-load test.
    30% flow should give positive power well below rated.
    """
    m = model._model
    Q_30 = 0.30 * m.Q_design
    r = model.predict({"flow_rate_m3s": Q_30, "gross_head_m": 45.0})
    P = float(r["power_kw"])
    assert 0 < P < m.P_rated, f"30% flow: P={P:.1f} should be in (0, P_rated={m.P_rated})"


def test_full_load_power_capped_at_rated(model):
    m = model._model
    r = model.predict({"flow_rate_m3s": m.Q_design * 1.5, "gross_head_m": 45.0})
    assert float(r["power_kw"]) <= m.P_rated


# --- Hill chart physics ---

def test_efficiency_peaks_near_design_flow(model):
    m = model._model
    H_gross = m.H_design * 1.03  # slightly above design to cancel f_loss
    Q_arr = np.linspace(m.Q_design * 0.25, m.Q_design * 1.05, 50)
    r = model.predict({"flow_rate_m3s": Q_arr, "gross_head_m": H_gross})
    peak_Q = Q_arr[np.argmax(r["efficiency"])]
    assert abs(peak_Q - m.Q_design) < m.Q_design * 0.15


def test_efficiency_bounded(model):
    Q = np.random.uniform(0.1, 2.0, 200)
    H = np.random.uniform(5, 200, 200)
    r = model.predict({"flow_rate_m3s": Q, "gross_head_m": H})
    assert np.all(r["efficiency"] >= 0.0)
    assert np.all(r["efficiency"] <= 1.0)


def test_power_increases_with_head(model):
    H_arr = np.linspace(30.0, 60.0, 10)
    r = model.predict({"flow_rate_m3s": 0.8, "gross_head_m": H_arr})
    assert np.all(np.diff(r["power_kw"]) >= 0)


# --- Head losses ---

def test_head_loss_increases_with_flow(model):
    m = model._model
    f1 = float(m.penstock_head_loss_fraction(0.3))
    f2 = float(m.penstock_head_loss_fraction(m.Q_design))
    f3 = float(m.penstock_head_loss_fraction(m.Q_design * 1.5))
    assert f1 < f2 < f3


def test_net_head_less_than_gross(model):
    H_net = float(model._model.net_head(45.0, 1.3))
    assert H_net < 45.0 and H_net > 0.0


# --- Cavitation ---

def test_cavitation_safe_at_design_head(model):
    m = model._model
    derate = float(m.cavitation_derate(m.H_design, "francis"))
    assert derate == 1.0, f"Derate={derate:.4f} at design head — should be 1.0"


def test_pelton_cavitation_always_safe(model):
    """Pelton (impulse) turbine has very low sigma_c, rarely cavitates."""
    derate = float(model._model.cavitation_derate(500.0, "pelton"))
    assert derate == 1.0


# --- Ecological flow ---

def test_eco_flow_zero_output(model):
    m = model._model
    r = model.predict({"flow_rate_m3s": m.Q_eco * 0.5, "gross_head_m": 45.0})
    assert float(r["power_kw"]) == 0.0


# --- Temperature ---

def test_cold_water_higher_density(model):
    from model import _water_density
    rho_cold = float(_water_density(4.0))
    rho_warm = float(_water_density(20.0))
    assert rho_cold > rho_warm


# --- Vectorized ---

def test_vectorized(model):
    Q = np.linspace(0.2, 1.5, 50)
    H = np.full(50, 45.0)
    r = model.predict({"flow_rate_m3s": Q, "gross_head_m": H})
    assert len(r["power_kw"]) == 50


# --- Benchmark ---

def test_benchmark(model):
    Q = np.random.uniform(0.1, 2.0, 1000)
    H = np.random.uniform(5, 200, 1000)
    start = time.perf_counter()
    model.predict({"flow_rate_m3s": Q, "gross_head_m": H})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed*1000:.2f} ms")
    assert elapsed < 1.0
