"""EC129 — Run-of-River Hydropower — F1b Head-Flow — Test Suite"""
import sys, time, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()


# --- Interface ---

def test_predict_keys(model):
    r = model.predict({"flow_rate_m3s": 60.0, "gross_head_m": 8.5})
    for k in ["power_kw", "efficiency", "capacity_factor", "net_head_m",
              "head_loss_fraction", "flow_ratio", "cavitation_derate"]:
        assert k in r


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC129"
    assert info["fidelity"] == "F1b"


# --- Design point ---

def test_design_efficiency_near_peak(model):
    """At design Q and H, efficiency should be near eta_peak * eta_gen."""
    m = model._model
    r = model.predict({"flow_rate_m3s": m.Q_design, "gross_head_m": m.H_design / (1 - m.f_loss_ref)})
    eta = float(np.mean(r["efficiency"]))
    expected_max = m.eta_peak * m.eta_gen
    assert eta >= 0.8 * expected_max, f"eta={eta:.4f} too low vs expected {expected_max:.4f}"


# --- Hill chart physics ---

def test_efficiency_peaks_near_design_flow(model):
    """Efficiency should peak near Q_design."""
    m = model._model
    H_gross = m.H_design / (1 - m.f_loss_ref)
    Q_arr = np.linspace(m.Q_design * 0.3, m.Q_design * 1.1, 50)
    r = model.predict({"flow_rate_m3s": Q_arr, "gross_head_m": H_gross})
    peak_idx = np.argmax(r["efficiency"])
    peak_Q = Q_arr[peak_idx]
    assert abs(peak_Q - m.Q_design) < m.Q_design * 0.15, \
        f"Peak at Q={peak_Q:.1f}, expected ~{m.Q_design}"


def test_efficiency_bounded(model):
    Q = np.random.uniform(10, 100, 200)
    H = np.random.uniform(3, 15, 200)
    r = model.predict({"flow_rate_m3s": Q, "gross_head_m": H})
    assert np.all(r["efficiency"] >= 0.0)
    assert np.all(r["efficiency"] <= 1.0)


def test_efficiency_drops_off_design_head(model):
    m = model._model
    H_design_gross = m.H_design / (1 - m.f_loss_ref)
    r_design = model.predict({"flow_rate_m3s": m.Q_design, "gross_head_m": H_design_gross})
    r_off    = model.predict({"flow_rate_m3s": m.Q_design, "gross_head_m": H_design_gross * 0.6})
    assert float(np.mean(r_design["efficiency"])) > float(np.mean(r_off["efficiency"]))


# --- Head losses ---

def test_head_loss_increases_with_flow(model):
    """Head loss fraction should increase as Q^2."""
    f1 = float(model._model.penstock_head_loss_fraction(30.0))
    f2 = float(model._model.penstock_head_loss_fraction(75.0))
    f3 = float(model._model.penstock_head_loss_fraction(100.0))
    assert f1 < f2 < f3


def test_net_head_less_than_gross(model):
    H_net = float(model._model.net_head(10.0, 75.0))
    assert H_net < 10.0


# --- Power physics ---

def test_power_increases_with_flow(model):
    """Below P_rated, power should increase with flow."""
    Q_arr = np.linspace(20, 70, 10)
    r = model.predict({"flow_rate_m3s": Q_arr, "gross_head_m": 8.5})
    # At least monotonic increase over most of range
    P = r["power_kw"]
    assert P[-1] > P[0]


def test_power_increases_with_head(model):
    """At constant flow, power increases with head."""
    H_arr = np.linspace(5, 12, 10)
    r = model.predict({"flow_rate_m3s": 60.0, "gross_head_m": H_arr})
    assert np.all(np.diff(r["power_kw"]) >= 0)


def test_power_capped_at_rated(model):
    m = model._model
    r = model.predict({"flow_rate_m3s": 200.0, "gross_head_m": 20.0})
    assert float(r["power_kw"]) <= m.P_rated


def test_zero_flow_zero_power(model):
    r = model.predict({"flow_rate_m3s": 0.0, "gross_head_m": 8.0})
    assert float(r["power_kw"]) == 0.0


# --- Ecological flow ---

def test_below_eco_flow_zero_power(model):
    """Flow below ecological minimum → zero available power."""
    m = model._model
    r = model.predict({"flow_rate_m3s": m.Q_eco * 0.5, "gross_head_m": 8.0})
    assert float(r["power_kw"]) == 0.0


def test_power_uses_available_not_total_flow(model):
    """Power with Q = Q_eco + delta should increase with delta."""
    m = model._model
    r1 = model.predict({"flow_rate_m3s": m.Q_eco + 5.0, "gross_head_m": 8.5})
    r2 = model.predict({"flow_rate_m3s": m.Q_eco + 20.0, "gross_head_m": 8.5})
    assert float(r2["power_kw"]) > float(r1["power_kw"])


# --- Water temperature ---

def test_cold_water_slightly_more_power(model):
    """Cold water is denser → marginally more power."""
    r_cold = model.predict({"flow_rate_m3s": 60.0, "gross_head_m": 8.5, "T_water_C": 5.0})
    r_warm = model.predict({"flow_rate_m3s": 60.0, "gross_head_m": 8.5, "T_water_C": 20.0})
    assert float(r_cold["power_kw"]) >= float(r_warm["power_kw"])


# --- Cavitation ---

def test_cavitation_derate_at_design_is_one(model):
    """At design head, cavitation derate should be 1.0 (safe operation)."""
    m = model._model
    derate = float(m.cavitation_derate(m.H_design))
    assert derate == 1.0, f"Cavitation derate at design={derate:.4f}, expected 1.0"


def test_cavitation_derate_decreases_with_low_head(model):
    """Very low head → risk of cavitation → derate < 1."""
    m = model._model
    d1 = float(m.cavitation_derate(m.H_design))
    # Extremely low head relative to H_atm - H_vapor
    d_low = float(m.cavitation_derate(0.5))
    assert d1 >= d_low


# --- Vectorized ---

def test_vectorized(model):
    Q = np.linspace(20, 100, 50)
    H = np.linspace(5, 12, 50)
    r = model.predict({"flow_rate_m3s": Q, "gross_head_m": H})
    assert len(r["power_kw"]) == 50


# --- Benchmark ---

def test_benchmark(model):
    Q = np.random.uniform(10, 120, 1000)
    H = np.random.uniform(3, 15, 1000)
    start = time.perf_counter()
    model.predict({"flow_rate_m3s": Q, "gross_head_m": H})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed*1000:.2f} ms")
    assert elapsed < 1.0
