"""EC187 — HVDC Converter Station — F1b Full Link — Test Suite"""
import sys, time, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()


def test_predict_keys(model):
    r = model.predict({"P_transfer_MW": 500.0})
    for k in ["P_AC_in_MW", "P_AC_out_MW", "P_loss_total_MW",
              "P_loss_rect_MW", "P_loss_line_MW", "P_loss_inv_MW",
              "I_dc_kA", "R_line_ohm", "link_efficiency",
              "Q_reactive_demand_MVAR"]:
        assert k in r, f"Missing key: {k}"


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC187"
    assert info["fidelity"] == "F1b"


def test_zero_power_near_zero_output(model):
    """Near-zero transfer → minimal output (only no-load losses)."""
    r = model.predict({"P_transfer_MW": 0.0})
    assert float(r["P_AC_out_MW"]) == 0.0


def test_efficiency_less_than_one(model):
    r = model.predict({"P_transfer_MW": 800.0})
    assert float(r["link_efficiency"]) < 1.0


def test_efficiency_greater_than_zero(model):
    r = model.predict({"P_transfer_MW": 800.0})
    assert float(r["link_efficiency"]) > 0.0


def test_total_loss_equals_sum_of_parts(model):
    """P_loss_total = P_loss_rect + P_loss_line + P_loss_inv (approximately)."""
    r = model.predict({"P_transfer_MW": 700.0})
    parts = float(r["P_loss_rect_MW"]) + float(r["P_loss_line_MW"]) + float(r["P_loss_inv_MW"])
    total = float(r["P_loss_total_MW"])
    assert abs(total - parts) < 0.5, f"Loss breakdown mismatch: {total:.2f} vs {parts:.2f}"


def test_power_balance(model):
    """P_AC_in = P_AC_out + P_loss_total."""
    r = model.predict({"P_transfer_MW": 600.0})
    balance = float(r["P_AC_in_MW"]) - float(r["P_AC_out_MW"]) - float(r["P_loss_total_MW"])
    assert abs(balance) < 0.5, f"Power balance error {balance:.4f} MW"


def test_dc_line_r_increases_with_temperature(model):
    """R_line at 60C > R_line at 20C."""
    R20 = model.model.dc_line_resistance(20.0)
    R60 = model.model.dc_line_resistance(60.0)
    assert R60 > R20


def test_hot_line_more_losses(model):
    """Hotter line → higher I^2*R → more line losses."""
    P_line_cold = float(model.model.line_losses_MW(700.0, T_line_C=10.0))
    P_line_hot  = float(model.model.line_losses_MW(700.0, T_line_C=70.0))
    assert P_line_hot > P_line_cold


def test_lcc_reactive_demand_positive(model):
    """LCC requires reactive power from the network."""
    r = model.predict({"P_transfer_MW": 500.0})
    assert float(r["Q_reactive_demand_MVAR"]) >= 0.0


def test_lcc_reactive_increases_at_full_load(model):
    """Higher power → higher LCC reactive demand (for LCC type)."""
    r_low  = model.predict({"P_transfer_MW": 100.0})
    r_high = model.predict({"P_transfer_MW": 900.0})
    assert float(r_high["Q_reactive_demand_MVAR"]) > float(r_low["Q_reactive_demand_MVAR"])


def test_dc_current_positive(model):
    r = model.predict({"P_transfer_MW": 500.0})
    assert float(r["I_dc_kA"]) > 0.0


def test_link_efficiency_acceptable_range(model):
    """Typical HVDC link efficiency ~96-98% at rated power."""
    r = model.predict({"P_transfer_MW": 1000.0})
    eta = float(r["link_efficiency"])
    assert 0.93 < eta < 1.0, f"Link efficiency {eta:.4f} out of expected range 0.93-1.0"


def test_vectorized(model):
    P = np.linspace(0, 1000, 50)
    r = model.predict({"P_transfer_MW": P})
    assert r["P_AC_out_MW"].shape == (50,)


def test_benchmark(model):
    P = np.random.uniform(0, 1000, 1000)
    start = time.perf_counter()
    model.predict({"P_transfer_MW": P})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed*1000:.2f} ms")
    assert elapsed < 5.0
