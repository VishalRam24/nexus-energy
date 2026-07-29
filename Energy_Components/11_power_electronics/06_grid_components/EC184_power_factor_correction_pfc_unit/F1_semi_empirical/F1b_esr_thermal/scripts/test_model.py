"""EC184 -- PFC Unit -- F1b ESR Thermal -- Test Suite"""
import sys, time, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()


# --- Interface ---

def test_predict_keys(model):
    r = model.predict({"P_kW": 1000.0, "pf_initial": 0.8})
    for k in ["Q_compensated_kVAR", "pf_achieved", "P_ESR_kW",
              "P_dielectric_kW", "P_loss_kW", "I_cap_A",
              "ESR_Ohm", "tan_delta", "Q_rated_available_kVAR"]:
        assert k in r, f"missing key {k}"


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC184"
    assert info["fidelity"] == "F1b"


# --- Physics: ESR(T) ---

def test_esr_changes_with_temperature(model):
    """ESR has temperature dependence (PP film: negative alpha means lower ESR at higher T)."""
    r_cold = float(model.predict({"P_kW": 1000.0, "pf_initial": 0.8, "T_cap": 25.0})["ESR_Ohm"])
    r_hot = float(model.predict({"P_kW": 1000.0, "pf_initial": 0.8, "T_cap": 65.0})["ESR_Ohm"])
    # ESR_alpha = -0.003 => ESR decreases with T
    assert r_hot < r_cold, f"PP film ESR should decrease with T: {r_cold:.4e} vs {r_hot:.4e}"


def test_tan_delta_increases_with_temperature(model):
    """tan_delta increases with temperature (aging / dielectric heating)."""
    td_25 = float(model.predict({"P_kW": 1000.0, "pf_initial": 0.8, "T_cap": 25.0})["tan_delta"])
    td_70 = float(model.predict({"P_kW": 1000.0, "pf_initial": 0.8, "T_cap": 70.0})["tan_delta"])
    assert td_70 > td_25


def test_losses_positive(model):
    """Losses must be positive when compensating."""
    r = model.predict({"P_kW": 2000.0, "pf_initial": 0.75, "T_cap": 40.0})
    assert float(r["P_loss_kW"]) > 0.0


def test_loss_equals_esr_plus_dielectric(model):
    """P_loss = P_ESR + P_dielectric."""
    r = model.predict({"P_kW": 2000.0, "pf_initial": 0.75, "T_cap": 40.0})
    diff = abs(float(r["P_loss_kW"]) - float(r["P_ESR_kW"]) - float(r["P_dielectric_kW"]))
    assert diff < 1e-10, f"P_loss breakdown mismatch: {diff}"


def test_zero_loss_at_zero_compensation(model):
    """When Q_comp = 0 (e.g. pf_initial = pf_target), losses ~ 0."""
    r = model.predict({"P_kW": 1000.0, "pf_initial": 0.95,
                       "pf_target": 0.95, "T_cap": 25.0})
    assert float(r["P_loss_kW"]) < 1e-6


# --- PF correction sanity ---

def test_pf_improves(model):
    """PF after compensation must be >= initial PF."""
    r = model.predict({"P_kW": 5000.0, "pf_initial": 0.75, "T_cap": 25.0})
    assert float(r["pf_achieved"]) >= 0.75


def test_pf_not_exceed_one(model):
    r = model.predict({"P_kW": 5000.0, "pf_initial": 0.75, "T_cap": 25.0})
    assert float(r["pf_achieved"]) <= 1.0 + 1e-9


# --- Thermal derating ---

def test_q_rated_available_below_T_max(model):
    """No derating below T_max = 70 C."""
    r = model.predict({"P_kW": 1000.0, "pf_initial": 0.8, "T_cap": 50.0})
    assert abs(float(r["Q_rated_available_kVAR"]) - 1000.0) < 1e-6


def test_q_rated_derated_above_T_max(model):
    """Q_rated_available < Q_rated when T > T_max."""
    r = model.predict({"P_kW": 1000.0, "pf_initial": 0.8, "T_cap": 75.0})
    assert float(r["Q_rated_available_kVAR"]) < 1000.0


# --- Vectorised ---

def test_vectorised(model):
    P = np.linspace(100, 5000, 50)
    pf = np.linspace(0.7, 0.9, 50)
    r = model.predict({"P_kW": P, "pf_initial": pf, "T_cap": 40.0})
    assert len(r["P_loss_kW"]) == 50


# --- Benchmark ---

def test_benchmark(model):
    P = np.random.uniform(100, 5000, 1000)
    pf = np.random.uniform(0.6, 0.95, 1000)
    start = time.perf_counter()
    model.predict({"P_kW": P, "pf_initial": pf, "T_cap": 40.0})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed*1000:.2f} ms")
    assert elapsed < 1.0
