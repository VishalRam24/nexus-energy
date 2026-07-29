"""EC186 -- STATCOM -- F1b Conduction+Switching -- Test Suite"""
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
    for k in ["Q_out_MVAR", "P_cond_MW", "P_sw_MW", "P_diode_MW",
              "P_transformer_MW", "P_standby_MW", "P_total_loss_MW",
              "I_rms_A", "r_ce_Ohm", "operating_mode"]:
        assert k in r, f"missing key {k}"


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC186"
    assert info["fidelity"] == "F1b"


# --- Physics ---

def test_total_loss_equals_sum(model):
    """P_total = P_cond + P_sw + P_diode + P_transformer + P_standby."""
    for Q in [-100, -50, 0, 50, 100]:
        r = model.predict({"Q_demand_MVAR": float(Q)})
        computed = (float(r["P_cond_MW"]) + float(r["P_sw_MW"])
                    + float(r["P_diode_MW"]) + float(r["P_transformer_MW"])
                    + float(r["P_standby_MW"]))
        assert abs(float(r["P_total_loss_MW"]) - computed) < 1e-9


def test_standby_always_present(model):
    for Q in [-100, 0, 100]:
        r = model.predict({"Q_demand_MVAR": float(Q)})
        assert float(r["P_standby_MW"]) > 0.0


def test_losses_increase_with_Q_magnitude(model):
    """Losses increase with |Q|."""
    losses = []
    for Q in [0, 25, 50, 75, 100]:
        r = model.predict({"Q_demand_MVAR": float(Q)})
        losses.append(float(r["P_total_loss_MW"]))
    for i in range(len(losses) - 1):
        assert losses[i] <= losses[i + 1]


def test_symmetric_losses(model):
    """Loss at +Q should approximately equal loss at -Q (symmetric VSC)."""
    r_pos = model.predict({"Q_demand_MVAR": 80.0})
    r_neg = model.predict({"Q_demand_MVAR": -80.0})
    # Should be ~equal within 1% (same current magnitude)
    diff = abs(float(r_pos["P_total_loss_MW"]) - float(r_neg["P_total_loss_MW"]))
    avg = (float(r_pos["P_total_loss_MW"]) + float(r_neg["P_total_loss_MW"])) / 2.0
    assert diff / avg < 0.01, f"Losses not symmetric: {diff:.4f} MW difference"


def test_temperature_increases_losses(model):
    """Higher T_j -> higher r_ce -> higher conduction losses."""
    r_cold = model.predict({"Q_demand_MVAR": 100.0, "T_j": 25.0})
    r_hot = model.predict({"Q_demand_MVAR": 100.0, "T_j": 150.0})
    assert float(r_hot["P_cond_MW"]) > float(r_cold["P_cond_MW"])


def test_r_ce_increases_with_temperature(model):
    """r_ce(T) must increase with T_j."""
    from model import STATCOMF1b
    import json
    with open(Path(__file__).parent.parent / "data" / "parameters.json") as f:
        p = json.load(f)
    m = STATCOMF1b(p)
    assert m.r_ce(150.0) > m.r_ce(25.0)


def test_q_independent_of_voltage(model):
    """VSC property: Q_out independent of V_pu."""
    r1 = model.predict({"Q_demand_MVAR": 80.0, "V_pu": 0.9})
    r2 = model.predict({"Q_demand_MVAR": 80.0, "V_pu": 1.1})
    assert float(r1["Q_out_MVAR"]) == float(r2["Q_out_MVAR"])


def test_clamping_at_Q_max(model):
    r = model.predict({"Q_demand_MVAR": 200.0})
    assert float(r["Q_out_MVAR"]) <= 100.0 + 1e-9


def test_clamping_at_Q_min(model):
    r = model.predict({"Q_demand_MVAR": -200.0})
    assert float(r["Q_out_MVAR"]) >= -100.0 - 1e-9


def test_current_zero_at_Q_zero(model):
    r = model.predict({"Q_demand_MVAR": 0.0})
    assert float(r["I_rms_A"]) == 0.0


# --- Vectorised ---

def test_vectorised(model):
    Q = np.linspace(-100, 100, 50)
    r = model.predict({"Q_demand_MVAR": Q})
    assert len(r["P_total_loss_MW"]) == 50


# --- Benchmark ---

def test_benchmark(model):
    Q = np.random.uniform(-100, 100, 1000)
    start = time.perf_counter()
    model.predict({"Q_demand_MVAR": Q})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed*1000:.2f} ms")
    assert elapsed < 1.0
