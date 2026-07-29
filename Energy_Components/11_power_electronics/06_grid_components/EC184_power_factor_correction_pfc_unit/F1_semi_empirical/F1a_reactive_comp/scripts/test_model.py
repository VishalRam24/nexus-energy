"""EC184 — PFC Unit — F1a Reactive Compensation — Test Suite"""
import sys, time, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()


def test_predict_keys(model):
    r = model.predict({"P_kW": 1000.0, "pf_initial": 0.75, "pf_target": 0.95})
    for k in ["Q_load_kVAR", "Q_required_kVAR", "Q_compensated_kVAR",
              "Q_residual_kVAR", "pf_achieved", "P_loss_kW", "bank_utilization"]:
        assert k in r


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC184"
    assert info["fidelity"] == "F1a"


def test_pf_improves(model):
    """Achieved PF must be higher than initial PF."""
    r = model.predict({"P_kW": 2000.0, "pf_initial": 0.75, "pf_target": 0.95})
    assert float(r["pf_achieved"]) > 0.75, \
        f"pf_achieved={float(r['pf_achieved']):.4f} should exceed 0.75"


def test_pf_achieved_not_exceeds_one(model):
    """Power factor cannot exceed 1."""
    r = model.predict({"P_kW": 2000.0, "pf_initial": 0.75, "pf_target": 0.95})
    assert float(r["pf_achieved"]) <= 1.0


def test_Q_required_formula(model):
    """Q_req = P*(tan(phi1) - tan(phi2))."""
    P, pf1, pf2 = 1000.0, 0.75, 0.95
    phi1 = np.arccos(pf1)
    phi2 = np.arccos(pf2)
    expected = P * (np.tan(phi1) - np.tan(phi2))
    r = model.predict({"P_kW": P, "pf_initial": pf1, "pf_target": pf2})
    assert abs(float(r["Q_required_kVAR"]) - expected) < 1e-6


def test_Q_comp_capped_at_rated(model):
    """Q_compensated must never exceed Q_rated."""
    r = model.predict({"P_kW": 10000.0, "pf_initial": 0.5, "pf_target": 0.95})
    Q_rated = model._model.Q_rated_kVAR
    assert float(r["Q_compensated_kVAR"]) <= Q_rated + 1e-9, \
        f"Q_comp={float(r['Q_compensated_kVAR']):.1f} > Q_rated={Q_rated}"


def test_losses_positive_when_compensating(model):
    """Non-zero Q_comp → P_loss > 0."""
    r = model.predict({"P_kW": 1000.0, "pf_initial": 0.75, "pf_target": 0.95})
    assert float(r["P_loss_kW"]) > 0.0


def test_Q_residual_decreases_after_compensation(model):
    """Q_residual < Q_load (compensation reduces reactive demand)."""
    r = model.predict({"P_kW": 1000.0, "pf_initial": 0.75, "pf_target": 0.95})
    assert float(r["Q_residual_kVAR"]) < float(r["Q_load_kVAR"])


def test_bank_utilization_between_zero_and_one(model):
    r = model.predict({"P_kW": 500.0, "pf_initial": 0.80, "pf_target": 0.95})
    u = float(r["bank_utilization"])
    assert 0.0 <= u <= 1.0, f"bank_utilization={u:.4f}"


def test_unity_pf_load_needs_no_compensation(model):
    """pf=1.0 load has no reactive component, Q_required≈0."""
    r = model.predict({"P_kW": 1000.0, "pf_initial": 0.9999, "pf_target": 0.95})
    # Initial pf already >= target → Q_required <= 0 → Q_comp = 0
    assert float(r["Q_compensated_kVAR"]) < 1e-3


def test_vectorized_P(model):
    P = np.linspace(100, 5000, 50)
    r = model.predict({"P_kW": P, "pf_initial": 0.75, "pf_target": 0.95})
    assert r["Q_compensated_kVAR"].shape == (50,)
    assert np.all(r["pf_achieved"] >= 0.0)


def test_benchmark(model):
    P = np.random.uniform(100, 5000, 1000)
    pf = np.random.uniform(0.55, 0.94, 1000)
    start = time.perf_counter()
    model.predict({"P_kW": P, "pf_initial": pf, "pf_target": 0.95})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed*1000:.2f} ms")
    assert elapsed < 1.0
