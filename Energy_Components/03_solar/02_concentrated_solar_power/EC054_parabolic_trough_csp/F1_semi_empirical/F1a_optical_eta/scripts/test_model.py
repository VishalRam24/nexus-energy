"""EC054 — Parabolic Trough CSP — F1a — Test Suite"""
import sys, time, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()


def test_predict_keys(model):
    r = model.predict({"dni": 850.0, "T_absorber": 300.0, "T_ambient": 25.0, "incidence_angle": 0.0})
    for k in ["useful_heat_kw", "optical_efficiency", "thermal_loss_kw", "overall_efficiency"]:
        assert k in r, f"Missing key: {k}"


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC054"
    assert info["fidelity"] == "F1a"


def test_zero_dni_gives_zero_useful_heat(model):
    """No solar resource -> no useful heat (but heat loss continues)."""
    r = model.predict({"dni": 0.0, "T_absorber": 300.0, "T_ambient": 25.0, "incidence_angle": 0.0})
    assert float(r["useful_heat_kw"]) == 0.0
    assert float(r["optical_efficiency"]) == 0.0


def test_IAM_unity_at_normal_incidence(model):
    """At theta=0, IAM=1, so optical efficiency equals peak eta_optical."""
    r = model.predict({"dni": 1000.0, "T_absorber": 100.0, "T_ambient": 25.0, "incidence_angle": 0.0})
    # With very low heat loss at T_abs=100C, Q_abs/P_incident ~ eta_optical
    eta_opt = float(r["optical_efficiency"])
    assert abs(eta_opt - 0.75) < 1e-9, f"eta_opt at theta=0 should be 0.75, got {eta_opt:.4f}"


def test_IAM_decreases_with_angle(model):
    """Optical efficiency decreases as incidence angle increases."""
    thetas = np.array([0.0, 15.0, 30.0, 45.0, 60.0])
    r = model.predict({"dni": 800.0, "T_absorber": 100.0, "T_ambient": 25.0, "incidence_angle": thetas})
    eta_opt = np.asarray(r["optical_efficiency"])
    assert np.all(np.diff(eta_opt) <= 0), f"Optical efficiency should decrease with angle: {eta_opt}"


def test_heat_loss_increases_with_T_absorber(model):
    """Higher absorber temperature -> higher thermal heat loss."""
    T_abs = np.array([150.0, 200.0, 250.0, 300.0, 350.0, 400.0])
    r = model.predict({"dni": 800.0, "T_absorber": T_abs, "T_ambient": 25.0, "incidence_angle": 0.0})
    q_loss = np.asarray(r["thermal_loss_kw"])
    assert np.all(np.diff(q_loss) > 0), f"Heat loss must increase with T_abs: {q_loss}"


def test_useful_heat_below_absorbed(model):
    """Useful heat <= absorbed solar power (always, due to heat losses)."""
    r = model.predict({"dni": 850.0, "T_absorber": 300.0, "T_ambient": 25.0, "incidence_angle": 0.0})
    Q_useful = float(r["useful_heat_kw"])
    # Compute absorbed power manually
    A = 864.0   # m2
    eta_opt = 0.75
    Q_abs = 850.0 * A * eta_opt / 1000.0  # kW
    assert Q_useful <= Q_abs + 1e-6


def test_overall_eta_below_optical_eta(model):
    """Overall efficiency must be <= optical efficiency (heat losses reduce output)."""
    r = model.predict({"dni": 800.0, "T_absorber": 300.0, "T_ambient": 25.0, "incidence_angle": 0.0})
    assert float(r["overall_efficiency"]) <= float(r["optical_efficiency"]) + 1e-9


def test_useful_heat_non_negative(model):
    """Q_useful is clamped at 0 — cannot be negative."""
    r = model.predict({"dni": 0.0, "T_absorber": 400.0, "T_ambient": 10.0, "incidence_angle": 0.0})
    assert float(r["useful_heat_kw"]) >= 0.0


def test_useful_heat_increases_with_dni(model):
    """More DNI -> more useful heat output."""
    dnis = np.array([200.0, 400.0, 600.0, 800.0, 1000.0])
    r = model.predict({"dni": dnis, "T_absorber": 200.0, "T_ambient": 25.0, "incidence_angle": 0.0})
    q_useful = np.asarray(r["useful_heat_kw"])
    assert np.all(np.diff(q_useful) > 0), f"Useful heat must increase with DNI: {q_useful}"


def test_benchmark(model):
    dnis = np.random.uniform(0, 1000, 1000)
    T_abs = np.random.uniform(100, 400, 1000)
    T_amb = np.random.uniform(0, 50, 1000)
    thetas = np.random.uniform(0, 75, 1000)
    start = time.perf_counter()
    model.predict({"dni": dnis, "T_absorber": T_abs, "T_ambient": T_amb, "incidence_angle": thetas})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed*1000:.2f} ms")
    assert elapsed < 1.0
