"""EC185 -- SVC -- F1b Conduction+Switching -- Test Suite"""
import sys, time, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()


# --- Interface ---

def test_predict_keys(model):
    r = model.predict({"Q_demand_MVAR": 50.0})
    for k in ["Q_out_MVAR", "Q_effective_MVAR", "P_thyristor_MW",
              "P_reactor_MW", "P_ESR_TSC_MW", "P_cooling_MW",
              "P_loss_MW", "I_rms_A", "operating_mode"]:
        assert k in r, f"missing key {k}"


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC185"
    assert info["fidelity"] == "F1b"


# --- Physics: Mode-specific losses ---

def test_inductive_mode_has_reactor_losses(model):
    """TCR mode (Q < 0) must have non-zero reactor losses."""
    r = model.predict({"Q_demand_MVAR": -40.0})
    assert float(r["P_reactor_MW"]) > 0.0


def test_capacitive_mode_no_reactor_losses(model):
    """TSC mode (Q > 0) must NOT have reactor losses."""
    r = model.predict({"Q_demand_MVAR": 80.0})
    assert float(r["P_reactor_MW"]) == 0.0


def test_capacitive_mode_has_ESR_losses(model):
    """TSC mode must have non-zero ESR losses."""
    r = model.predict({"Q_demand_MVAR": 80.0})
    assert float(r["P_ESR_TSC_MW"]) > 0.0


def test_inductive_mode_no_ESR_losses(model):
    """TCR mode must NOT have TSC ESR losses."""
    r = model.predict({"Q_demand_MVAR": -40.0})
    assert float(r["P_ESR_TSC_MW"]) == 0.0


def test_cooling_always_on(model):
    """Cooling loss must be present in all modes."""
    for Q in [-50, 0, 50, 100]:
        r = model.predict({"Q_demand_MVAR": float(Q)})
        assert float(r["P_cooling_MW"]) > 0.0


def test_standby_losses_equal_cooling_only(model):
    """At Q=0, only cooling losses (thyristors not conducting)."""
    r = model.predict({"Q_demand_MVAR": 0.0})
    assert float(r["P_thyristor_MW"]) == 0.0
    assert float(r["P_reactor_MW"]) == 0.0
    assert float(r["P_ESR_TSC_MW"]) == 0.0


def test_total_loss_equals_sum(model):
    """P_loss = P_thy + P_reactor + P_ESR + P_cooling."""
    for Q in [-50, -25, 0, 50, 100]:
        r = model.predict({"Q_demand_MVAR": float(Q)})
        total = (float(r["P_thyristor_MW"]) + float(r["P_reactor_MW"])
                 + float(r["P_ESR_TSC_MW"]) + float(r["P_cooling_MW"]))
        assert abs(float(r["P_loss_MW"]) - total) < 1e-10


def test_losses_increase_with_Q_magnitude(model):
    """Losses must increase as |Q| increases (more current)."""
    losses = []
    for Q in [10, 25, 50, 100]:
        r = model.predict({"Q_demand_MVAR": float(Q)})
        losses.append(float(r["P_loss_MW"]))
    for i in range(len(losses) - 1):
        assert losses[i] < losses[i + 1], f"Loss should increase: {losses}"


# --- Voltage scaling ---

def test_lower_voltage_reduces_output(model):
    """Q_effective < Q_out at V < 1.0 (Q scales with V^2)."""
    r = model.predict({"Q_demand_MVAR": 100.0, "V_pu": 0.9})
    assert float(r["Q_effective_MVAR"]) < float(r["Q_out_MVAR"])


def test_higher_voltage_increases_output(model):
    r = model.predict({"Q_demand_MVAR": 100.0, "V_pu": 1.1})
    assert float(r["Q_effective_MVAR"]) > float(r["Q_out_MVAR"])


# --- Clamping ---

def test_clamped_at_Q_max(model):
    r = model.predict({"Q_demand_MVAR": 200.0})
    assert float(r["Q_out_MVAR"]) <= 100.0


def test_clamped_at_Q_min(model):
    r = model.predict({"Q_demand_MVAR": -100.0})
    assert float(r["Q_out_MVAR"]) >= -50.0


# --- Vectorised ---

def test_vectorised(model):
    Q = np.linspace(-50, 100, 50)
    r = model.predict({"Q_demand_MVAR": Q})
    assert len(r["P_loss_MW"]) == 50


# --- Benchmark ---

def test_benchmark(model):
    Q = np.random.uniform(-50, 100, 1000)
    start = time.perf_counter()
    model.predict({"Q_demand_MVAR": Q})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed*1000:.2f} ms")
    assert elapsed < 1.0
