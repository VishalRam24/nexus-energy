"""EC127 — Gravity Energy Storage — F1b Losses — Test Suite"""
import sys
import time
import numpy as np
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()


# --- Interface ---

def test_predict_discharge_keys(model):
    r = model.predict({"soc": 0.5, "velocity_mps": 1.0, "mode": "discharge"})
    for k in ["power_kw", "efficiency", "round_trip_efficiency",
              "friction_loss_kw", "drag_loss_kw", "bearing_loss_kw",
              "total_mech_loss_kw", "motor_efficiency", "generator_efficiency"]:
        assert k in r, f"Missing key: {k}"


def test_predict_losses_keys(model):
    r = model.predict({"velocity_mps": 1.5, "mode": "losses"})
    for k in ["friction_kw", "drag_kw", "bearing_kw", "total_mech_loss_kw"]:
        assert k in r, f"Missing key: {k}"


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC127"
    assert info["fidelity"] == "F1b"


# --- Mechanical losses scale with speed ---

def test_friction_loss_zero_at_zero_velocity(model):
    """No friction loss at standstill."""
    r = model.predict({"velocity_mps": 0.0, "mode": "losses"})
    assert float(r["friction_kw"]) == 0.0


def test_friction_loss_linear_with_velocity(model):
    """Coulomb friction P = mu*m*g*v — doubling v should double friction."""
    r1 = model.predict({"velocity_mps": 0.5, "mode": "losses"})
    r2 = model.predict({"velocity_mps": 1.0, "mode": "losses"})
    ratio = float(r2["friction_kw"]) / float(r1["friction_kw"])
    assert abs(ratio - 2.0) < 0.01, f"Friction ratio should be 2.0 (linear), got {ratio:.4f}"


def test_drag_loss_quadratic_with_velocity(model):
    """Aerodynamic drag P = k*v^2 — doubling v should 4x drag."""
    r1 = model.predict({"velocity_mps": 0.5, "mode": "losses"})
    r2 = model.predict({"velocity_mps": 1.0, "mode": "losses"})
    ratio = float(r2["drag_kw"]) / float(r1["drag_kw"])
    assert abs(ratio - 4.0) < 0.01, f"Drag ratio should be 4.0 (quadratic), got {ratio:.4f}"


def test_total_losses_increase_with_velocity(model):
    """Total losses must monotonically increase with velocity."""
    vs = np.array([0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
    losses = [float(model.predict({"velocity_mps": v, "mode": "losses"})["total_mech_loss_kw"])
              for v in vs]
    assert all(losses[i] < losses[i + 1] for i in range(len(losses) - 1)), \
        f"Total losses should increase with velocity: {losses}"


def test_losses_all_positive(model):
    """All loss components must be non-negative."""
    for v in [0.1, 1.0, 2.0, 3.0]:
        r = model.predict({"velocity_mps": v, "mode": "losses"})
        for k in ["friction_kw", "drag_kw", "bearing_kw", "total_mech_loss_kw"]:
            assert float(r[k]) >= 0, f"{k} should be non-negative at v={v}"


def test_total_loss_equals_sum_of_components(model):
    """Total = friction + drag + bearing."""
    v = 1.5
    r = model.predict({"velocity_mps": v, "mode": "losses"})
    expected = float(r["friction_kw"]) + float(r["drag_kw"]) + float(r["bearing_kw"])
    actual = float(r["total_mech_loss_kw"])
    assert abs(actual - expected) < 1e-9, \
        f"Total loss {actual:.6f} != sum of components {expected:.6f}"


# --- Part-load motor/generator efficiency ---

def test_motor_efficiency_at_full_load_near_rated(model):
    """At PLF=1, motor efficiency should be approximately the rated value."""
    m = model._model
    eta = float(m.motor_efficiency(1.0))
    # It won't be exactly eta_rated due to model form, but should be close
    assert abs(eta - m.eta_motor_rated) < 0.02, \
        f"Motor efficiency at full load {eta:.4f} should be near rated {m.eta_motor_rated:.4f}"


def test_motor_efficiency_zero_at_zero_load(model):
    """At PLF→0, motor/generator efficiency approaches 0."""
    m = model._model
    eta_tiny = float(m.motor_efficiency(0.001))
    assert eta_tiny < 0.5, \
        f"Motor efficiency at near-zero load should be low: {eta_tiny:.4f}"


def test_efficiency_increases_toward_rated_with_plf(model):
    """Motor efficiency should generally increase from low PLF toward rated PLF."""
    m = model._model
    plfs = [0.1, 0.25, 0.5, 0.75, 1.0]
    etas = [float(m.motor_efficiency(plf)) for plf in plfs]
    # Check that η at 1.0 > η at 0.1
    assert etas[-1] > etas[0], \
        f"Motor efficiency at PLF=1 ({etas[-1]:.4f}) should exceed at PLF=0.1 ({etas[0]:.4f})"


def test_generator_efficiency_less_than_1(model):
    """Generator efficiency must always be < 1."""
    m = model._model
    for plf in [0.1, 0.5, 1.0]:
        eta = float(m.generator_efficiency(plf))
        assert eta < 1.0, f"Generator efficiency should be < 1 at PLF={plf}, got {eta:.4f}"


def test_generator_efficiency_positive(model):
    """Generator efficiency must be positive for PLF > 0."""
    m = model._model
    for plf in [0.1, 0.5, 1.0]:
        eta = float(m.generator_efficiency(plf))
        assert eta > 0, f"Generator efficiency should be positive at PLF={plf}"


# --- Power and efficiency ---

def test_discharge_power_positive(model):
    r = model.predict({"soc": 0.5, "velocity_mps": 1.0, "mode": "discharge"})
    assert float(r["power_kw"]) > 0


def test_charge_power_positive(model):
    r = model.predict({"soc": 0.5, "velocity_mps": 1.0, "mode": "charge"})
    assert float(r["power_kw"]) > 0


def test_charge_power_greater_than_discharge(model):
    """For same velocity, charge power > discharge power (losses add to both sides)."""
    r_ch = model.predict({"soc": 0.5, "velocity_mps": 1.5, "mode": "charge"})
    r_dis = model.predict({"soc": 0.5, "velocity_mps": 1.5, "mode": "discharge"})
    assert float(r_ch["power_kw"]) > float(r_dis["power_kw"]), \
        "Charge power (input) should exceed discharge power (output)"


def test_rte_less_than_1(model):
    """Round-trip efficiency < 1 due to mechanical losses."""
    r = model.predict({"soc": 0.5, "velocity_mps": 1.0})
    rte = float(r["round_trip_efficiency"])
    assert rte < 1.0, f"RTE should be < 1, got {rte:.4f}"


def test_rte_decreases_with_velocity(model):
    """Higher velocity → more friction and drag losses → lower RTE."""
    vs = [0.5, 1.0, 2.0, 3.0]
    rtes = [float(model.predict({"soc": 0.5, "velocity_mps": v})["round_trip_efficiency"])
            for v in vs]
    assert all(rtes[i] >= rtes[i + 1] for i in range(len(rtes) - 1)), \
        f"RTE should decrease with velocity: {rtes}"


def test_rte_reasonable_range(model):
    """Gravity storage RTE typically 0.80–0.95 at moderate load."""
    r = model.predict({"soc": 0.5, "velocity_mps": 1.0})
    rte = float(r["round_trip_efficiency"])
    assert 0.70 <= rte <= 0.98, f"RTE={rte:.4f} outside expected [0.70, 0.98]"


def test_discharge_power_increases_with_velocity(model):
    """Higher velocity → more gravitational power → more discharge output."""
    vs = np.array([0.1, 0.5, 1.0, 2.0])
    Ps = [float(model.predict({"soc": 0.5, "velocity_mps": v, "mode": "discharge"})["power_kw"])
          for v in vs]
    assert all(Ps[i] < Ps[i + 1] for i in range(len(Ps) - 1)), \
        f"Discharge power should increase with velocity: {Ps}"


def test_efficiency_bounded_0_1(model):
    """Efficiency must be in [0, 1]."""
    for v in np.linspace(0.1, 3.0, 20):
        for mode in ["charge", "discharge"]:
            r = model.predict({"soc": 0.5, "velocity_mps": v, "mode": mode})
            eta = float(r["efficiency"])
            assert 0.0 <= eta <= 1.0, f"Efficiency {eta:.4f} out of [0,1] at v={v:.1f}, mode={mode}"


def test_zero_velocity_zero_power(model):
    """Zero velocity → zero power and zero losses."""
    r = model.predict({"soc": 0.5, "velocity_mps": 0.0, "mode": "discharge"})
    assert float(r["power_kw"]) == 0.0
    r_loss = model.predict({"velocity_mps": 0.0, "mode": "losses"})
    assert float(r_loss["friction_kw"]) == 0.0


# --- Vectorized ---

def test_vectorized(model):
    v = np.linspace(0.1, 3.0, 50)
    r = model.predict({"soc": 0.5, "velocity_mps": v, "mode": "discharge"})
    assert len(r["power_kw"]) == 50


# --- Benchmark ---

def test_benchmark_1000(model):
    v = np.random.uniform(0.1, 3.0, 1000)
    soc = np.random.uniform(0, 1, 1000)
    start = time.perf_counter()
    model.predict({"soc": soc, "velocity_mps": v, "mode": "discharge"})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed * 1000:.2f} ms")
    assert elapsed < 0.2
