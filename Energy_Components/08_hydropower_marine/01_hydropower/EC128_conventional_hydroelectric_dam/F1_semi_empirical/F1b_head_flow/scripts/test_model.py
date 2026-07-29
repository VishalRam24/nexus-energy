"""EC128 — Conventional Hydro Dam — F1b Head-Flow — Test Suite"""
import sys, time, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()


# --- Interface ---

def test_predict_keys(model):
    r = model.predict({"flow_rate_m3s": 50.0, "head_m": 100.0})
    for k in ["power_kw", "efficiency", "specific_speed", "flow_ratio"]:
        assert k in r


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC128"
    assert info["fidelity"] == "F1b"


# --- Francis Turbine ---

def test_francis_peak_efficiency_at_design(model):
    """At design point (Q=50, H=100), efficiency should be near eta_peak * eta_gen."""
    r = model.predict({"flow_rate_m3s": 50.0, "head_m": 100.0, "turbine_type": "francis"})
    eta = float(r["efficiency"])
    expected = 0.93 * 0.98  # eta_peak * eta_gen = 0.9114
    assert abs(eta - expected) < 0.001, f"Francis design eta={eta:.4f}, expected {expected:.4f}"


def test_francis_power_at_design(model):
    """P = eta * rho * g * Q * H / 1000, capped at P_rated."""
    r = model.predict({"flow_rate_m3s": 50.0, "head_m": 100.0, "turbine_type": "francis"})
    P = float(r["power_kw"])
    expected = 0.93 * 0.98 * 1000 * 9.81 * 50 * 100 / 1000  # ~44704 kW
    assert abs(P - expected) < 100, f"P={P:.0f}kW, expected {expected:.0f}kW"


def test_francis_zero_below_qmin(model):
    """Below minimum flow ratio, efficiency and power should be 0."""
    r = model.predict({"flow_rate_m3s": 5.0, "head_m": 100.0, "turbine_type": "francis"})
    # q = 5/50 = 0.1 < q_min=0.2
    assert float(r["efficiency"]) == 0.0
    assert float(r["power_kw"]) == 0.0


# --- Kaplan Turbine ---

def test_kaplan_design_point(model):
    r = model.predict({"flow_rate_m3s": 100.0, "head_m": 20.0, "turbine_type": "kaplan"})
    eta = float(r["efficiency"])
    expected = 0.91 * 0.98
    assert abs(eta - expected) < 0.001


def test_kaplan_flatter_efficiency_curve(model):
    """Kaplan (adjustable blades) should have flatter efficiency than Francis."""
    # Compare efficiency at off-design flow
    r_f = model.predict({"flow_rate_m3s": 25.0, "head_m": 100.0, "turbine_type": "francis"})
    r_k = model.predict({"flow_rate_m3s": 50.0, "head_m": 20.0, "turbine_type": "kaplan"})
    # Both at q=0.5: Francis k_q=0.4, Kaplan k_q=0.25
    # Kaplan should have less drop-off at same q
    eta_f = float(r_f["efficiency"])
    eta_k = float(r_k["efficiency"])
    # eta_f at q=0.5: 0.93*(1-0.4*0.25)*0.98 = 0.93*0.9*0.98 ≈ 0.820
    # eta_k at q=0.5: 0.91*(1-0.25*0.25)*0.98 = 0.91*0.9375*0.98 ≈ 0.836
    # Kaplan efficiency at q=0.5 relative to peak should be higher
    assert eta_k / (0.91 * 0.98) > eta_f / (0.93 * 0.98), \
        "Kaplan should have flatter efficiency curve"


# --- Pelton Turbine ---

def test_pelton_design_point(model):
    r = model.predict({"flow_rate_m3s": 10.0, "head_m": 600.0, "turbine_type": "pelton"})
    eta = float(r["efficiency"])
    expected = 0.91 * 0.98
    assert abs(eta - expected) < 0.001


# --- Efficiency Hill Chart ---

def test_efficiency_bounded(model):
    Q = np.random.uniform(5, 60, 200)
    H = np.random.uniform(50, 150, 200)
    r = model.predict({"flow_rate_m3s": Q, "head_m": H, "turbine_type": "francis"})
    assert np.all(r["efficiency"] >= 0.0)
    assert np.all(r["efficiency"] <= 1.0)


def test_efficiency_peaks_at_design_point(model):
    """Efficiency should be highest at design Q and H."""
    Q_arr = np.linspace(10, 55, 100)
    r = model.predict({"flow_rate_m3s": Q_arr, "head_m": 100.0, "turbine_type": "francis"})
    peak_idx = np.argmax(r["efficiency"])
    peak_Q = Q_arr[peak_idx]
    assert abs(peak_Q - 50.0) < 5.0, f"Peak at Q={peak_Q:.1f}, expected ~50"


def test_efficiency_drops_off_design_head(model):
    """Efficiency at off-design head should be lower than at design head."""
    r_design = model.predict({"flow_rate_m3s": 50.0, "head_m": 100.0, "turbine_type": "francis"})
    r_off = model.predict({"flow_rate_m3s": 50.0, "head_m": 70.0, "turbine_type": "francis"})
    assert float(r_design["efficiency"]) > float(r_off["efficiency"])


# --- Power Physics ---

def test_power_increases_with_flow(model):
    """At constant head, power should increase with flow (within valid range, below cap)."""
    Q_arr = np.linspace(15, 45, 10)  # Stay below design to avoid P_rated cap
    r = model.predict({"flow_rate_m3s": Q_arr, "head_m": 100.0, "turbine_type": "francis"})
    assert np.all(np.diff(r["power_kw"]) > 0)


def test_power_increases_with_head(model):
    """At constant flow, power increases with head (within operating range)."""
    H_arr = np.linspace(70, 110, 10)  # Stay within valid head range
    Q = 30.0  # Below design flow to avoid P_rated cap
    r = model.predict({"flow_rate_m3s": Q, "head_m": H_arr, "turbine_type": "francis"})
    assert np.all(np.diff(r["power_kw"]) > 0)


def test_power_capped_at_rated(model):
    """Power should not exceed P_rated."""
    r = model.predict({"flow_rate_m3s": 55.0, "head_m": 125.0, "turbine_type": "francis"})
    assert float(r["power_kw"]) <= 45000.0


# --- Flow Ratio ---

def test_flow_ratio_at_design(model):
    r = model.predict({"flow_rate_m3s": 50.0, "head_m": 100.0, "turbine_type": "francis"})
    assert abs(float(r["flow_ratio"]) - 1.0) < 0.001


# --- Environmental Flow ---

def test_environmental_flow(model):
    from model import HydroelectricDamF1b
    import json
    with open(Path(__file__).parent.parent / "data" / "parameters.json") as f:
        params = json.load(f)
    m = HydroelectricDamF1b(params)
    Q_env = m.environmental_flow("francis")
    assert abs(Q_env - 5.0) < 0.01  # 10% of Q_rated=50


def test_available_flow_after_env(model):
    from model import HydroelectricDamF1b
    import json
    with open(Path(__file__).parent.parent / "data" / "parameters.json") as f:
        params = json.load(f)
    m = HydroelectricDamF1b(params)
    Q_avail = m.available_flow(50.0, "francis")
    assert abs(Q_avail - 45.0) < 0.01


# --- Edge Cases ---

def test_zero_flow(model):
    r = model.predict({"flow_rate_m3s": 0.0, "head_m": 100.0})
    assert float(r["power_kw"]) == 0.0


def test_vectorized(model):
    Q = np.linspace(10, 55, 50)
    H = np.linspace(70, 130, 50)
    r = model.predict({"flow_rate_m3s": Q, "head_m": H, "turbine_type": "francis"})
    assert len(r["power_kw"]) == 50


def test_benchmark(model):
    Q = np.random.uniform(5, 60, 1000)
    H = np.random.uniform(50, 150, 1000)
    start = time.perf_counter()
    model.predict({"flow_rate_m3s": Q, "head_m": H, "turbine_type": "francis"})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed*1000:.2f} ms")
    assert elapsed < 1.0
