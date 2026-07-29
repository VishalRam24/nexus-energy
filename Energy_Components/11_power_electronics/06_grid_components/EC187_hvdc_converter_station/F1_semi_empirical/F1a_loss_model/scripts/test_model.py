"""EC187 — HVDC Converter Station — F1a Loss Model — Test Suite"""
import sys, time, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()


def test_predict_keys(model):
    r = model.predict({"P_transfer_MW": 500.0})
    for k in ["P_in_MW", "P_out_MW", "P_loss_MW", "efficiency",
              "I_dc_kA", "utilization", "Q_delivered_MVAR"]:
        assert k in r


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC187"
    assert info["fidelity"] == "F1a"


def test_P_loss_positive(model):
    """Losses always positive (no-load + variable)."""
    r = model.predict({"P_transfer_MW": 500.0})
    assert float(r["P_loss_MW"]) > 0.0


def test_P_loss_formula_rectifier(model):
    """P_loss = P_no_load + loss_factor * P."""
    P = 600.0
    expected = model._model.P_no_load + model._model.loss_factor * P
    r = model.predict({"P_transfer_MW": P, "direction": "rectifier"})
    assert abs(float(r["P_loss_MW"]) - expected) < 1e-9


def test_power_balance_rectifier(model):
    """P_in = P_out + P_loss for rectifier."""
    r = model.predict({"P_transfer_MW": 700.0, "direction": "rectifier"})
    balance = float(r["P_in_MW"]) - float(r["P_out_MW"]) - float(r["P_loss_MW"])
    assert abs(balance) < 1e-9


def test_power_balance_inverter(model):
    """P_in = P_out + P_loss for inverter."""
    r = model.predict({"P_transfer_MW": 700.0, "direction": "inverter"})
    balance = float(r["P_in_MW"]) - float(r["P_out_MW"]) - float(r["P_loss_MW"])
    assert abs(balance) < 1e-9


def test_efficiency_below_one(model):
    r = model.predict({"P_transfer_MW": 500.0})
    assert float(r["efficiency"]) < 1.0


def test_efficiency_above_zero(model):
    r = model.predict({"P_transfer_MW": 500.0})
    assert float(r["efficiency"]) > 0.0


def test_efficiency_increases_with_load(model):
    """At F1a, efficiency improves with load (fixed no-load loss amortized)."""
    r_low = model.predict({"P_transfer_MW": 100.0})
    r_high = model.predict({"P_transfer_MW": 800.0})
    assert float(r_high["efficiency"]) > float(r_low["efficiency"]), \
        "Higher load → higher efficiency (no-load loss amortized)"


def test_I_dc_scales_with_P(model):
    """I_dc = P / (V_dc); doubling P doubles I_dc."""
    r1 = model.predict({"P_transfer_MW": 300.0})
    r2 = model.predict({"P_transfer_MW": 600.0})
    ratio = float(r2["I_dc_kA"]) / float(r1["I_dc_kA"])
    assert abs(ratio - 2.0) < 1e-6, f"I_dc ratio={ratio:.6f}, expected 2.0"


def test_P_capped_at_rated(model):
    """Power transfer capped at P_rated."""
    P_rated = model._model.P_rated
    r = model.predict({"P_transfer_MW": P_rated * 2})
    assert float(r["utilization"]) <= 1.0 + 1e-9


def test_Q_delivered_capped(model):
    """Q_delivered capped at ±Q_capability."""
    Q_cap = model._model.Q_cap
    r = model.predict({"P_transfer_MW": 500.0, "Q_request_MVAR": Q_cap + 100.0})
    assert abs(float(r["Q_delivered_MVAR"]) - Q_cap) < 1e-9


def test_vectorized_P(model):
    P = np.linspace(0, 1000, 50)
    r = model.predict({"P_transfer_MW": P})
    assert r["P_loss_MW"].shape == (50,)
    assert np.all(r["P_loss_MW"] >= 0)


def test_benchmark(model):
    P = np.random.uniform(0, 1000, 1000)
    start = time.perf_counter()
    model.predict({"P_transfer_MW": P})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed*1000:.2f} ms")
    assert elapsed < 1.0
