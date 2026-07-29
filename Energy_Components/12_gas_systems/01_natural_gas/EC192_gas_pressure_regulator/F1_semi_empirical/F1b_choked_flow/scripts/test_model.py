"""EC192 — Gas Pressure Regulator — F1b Choked Flow — Test Suite"""
import sys, time, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()


def test_predict_keys(model):
    r = model.predict({"P_up_bar": 70.0, "P_down_bar": 10.0})
    for k in ["flow_std_m3_per_h", "flow_kg_per_s", "T_downstream_K",
              "is_choked", "expansion_factor_Y"]:
        assert k in r, f"Missing key: {k}"


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC192"
    assert info["fidelity"] == "F1b"


def test_choke_detection(model):
    """Large pressure drop should trigger choke."""
    r = model.predict({"P_up_bar": 70.0, "P_down_bar": 5.0})
    choked = bool(np.atleast_1d(r["is_choked"])[0])
    assert choked, "Expected choke at large ΔP"


def test_no_choke_small_dp(model):
    """Small pressure drop should not choke."""
    r = model.predict({"P_up_bar": 70.0, "P_down_bar": 68.0})
    choked = bool(np.atleast_1d(r["is_choked"])[0])
    assert not choked, "Expected no choke at small ΔP"


def test_choked_flow_independent_of_downstream(model):
    """
    Phase 7 fix: At choke, flow must NOT increase with further ΔP increase.
    Flow should be identical when ΔP_actual > ΔP_choke.
    """
    r1 = model.predict({"P_up_bar": 70.0, "P_down_bar": 10.0})
    r2 = model.predict({"P_up_bar": 70.0, "P_down_bar": 2.0})
    Q1 = float(np.atleast_1d(r1["flow_std_m3_per_h"])[0])
    Q2 = float(np.atleast_1d(r2["flow_std_m3_per_h"])[0])
    # Both should be choked → same flow (within numerical tolerance)
    assert abs(Q1 - Q2) < 1e-6, \
        f"Choked flow not independent of downstream P: Q1={Q1:.4f}, Q2={Q2:.4f}"


def test_expansion_factor_clamp(model):
    """Y must be ≥ 2/3 always."""
    for P_down in [5.0, 10.0, 30.0, 60.0, 69.0]:
        r = model.predict({"P_up_bar": 70.0, "P_down_bar": P_down})
        Y = float(np.atleast_1d(r["expansion_factor_Y"])[0])
        assert Y >= 2.0/3.0 - 1e-9, f"Y = {Y:.4f} < 2/3 at P_down={P_down}"


def test_y_equals_two_thirds_at_choke(model):
    """At choke, Y must equal exactly 2/3."""
    r = model.predict({"P_up_bar": 70.0, "P_down_bar": 5.0})
    Y = float(np.atleast_1d(r["expansion_factor_Y"])[0])
    assert abs(Y - 2.0/3.0) < 1e-9, f"Y at choke = {Y:.6f}, expected 2/3"


def test_flow_positive(model):
    """Flow must be positive for P_up > P_down."""
    r = model.predict({"P_up_bar": 70.0, "P_down_bar": 10.0})
    Q = float(np.atleast_1d(r["flow_std_m3_per_h"])[0])
    assert Q > 0, f"Flow = {Q}"


def test_jt_cooling(model):
    """Downstream temperature must be lower than upstream (JT cooling)."""
    r = model.predict({"P_up_bar": 70.0, "P_down_bar": 10.0, "T_up_K": 300.0})
    T_down = float(np.atleast_1d(r["T_downstream_K"])[0])
    assert T_down < 300.0, f"JT cooling not occurring: T_down={T_down:.2f} K"


def test_valve_travel_reduces_flow(model):
    """Partial valve opening reduces flow."""
    r_full = model.predict({"P_up_bar": 70.0, "P_down_bar": 60.0, "valve_travel": 1.0})
    r_half = model.predict({"P_up_bar": 70.0, "P_down_bar": 60.0, "valve_travel": 0.5})
    Q_full = float(np.atleast_1d(r_full["flow_std_m3_per_h"])[0])
    Q_half = float(np.atleast_1d(r_half["flow_std_m3_per_h"])[0])
    assert Q_half < Q_full, f"Partial valve not reducing flow: {Q_full:.2f} vs {Q_half:.2f}"


def test_array_input(model):
    P_downs = np.linspace(5.0, 65.0, 10)
    r = model.predict({"P_up_bar": 70.0, "P_down_bar": P_downs})
    assert len(np.atleast_1d(r["flow_std_m3_per_h"])) == 10


def test_benchmark(model):
    P_downs = np.random.uniform(5.0, 65.0, 1000)
    start = time.perf_counter()
    model.predict({"P_up_bar": 70.0, "P_down_bar": P_downs})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed * 1000:.2f} ms")
    assert elapsed < 1.0
